# Scalpel：Recovery-Aware Layer Pruning for Faster Vision-Language Models
<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/TsDLjZIblTgMjdDkJ0feU.png" alt="FelineBench benchmark visualization" width="420"/>
</div>


Scalpel 是用于多模态大模型的逐层结构化剪枝工具。它的恢复目标只有一个：

> 删除 Student 的当前第 $i$ 层后，只训练 Student 的第 $i-1$ 层，使 Student 的边界输出模拟 Teacher 原第 $i$ 层的输出。

这里的“第 $i$ 层”都是当前模型的 **0-based language-decoder layer index**。一轮剪枝严格对应一份删层前的 Teacher 和一份删层后的 Student。

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/dGVEz_s2Qvl0D-5ebWUa5.png" alt="Scalpel-VL overall architecture preview" width="400"/>
</div>

## 方法概览

每一轮按照下面顺序执行：

1. 用固定的 10×10 probe 集评估当前模型，临时绕过每个候选层。
2. 用最坏轴风险选择一个待删除的当前层 $i$。
3. 从当前模型物理删除第 $i$ 层，得到 Student；删除后层号重新连续编号。
4. 保留删层前模型作为冻结 Teacher，保留删层后的模型作为 Student。
5. 冻结 Student 的所有参数，只解冻 language-model.layers[i-1]。
6. 在同一批输入上比较：
   - Teacher 原第 $i$ 层输出 $h_i^T$；
   - Student 第 $i-1$ 层输出 $h_{i-1}^S$。
7. 对两侧隐藏态使用同一个冻结的 Teacher final norm 和 LM head，得到边界软分布，最小化 field-weighted boundary KL。
8. 导出恢复后的 Student，作为下一轮的当前模型。

因此，恢复过程不是把所有线性层加 LoRA，也不是让删层后的 Student 拟合固定的原始 28 层模型最终 logits。Teacher 是**本轮删层前的当前模型**，每轮都会随模型状态更新。

## 边界目标与 loss

设本轮删除当前层 $i$：

$$
h_i^T = F_i^T(h_{i-1}^T),\qquad
h_{i-1}^S = F_{i-1}^S(h_{i-2}^S).
$$

训练只改变 $F_{i-1}^S$ 的参数。恢复目标是让 $h_{i-1}^S$ 经过同一个冻结 logit lens 后，模拟 Teacher 的 $h_i^T$：

$$
q_i^T =
\mathrm{softmax}\left(
\frac{W\,\mathrm{Norm}(h_i^T)}{T}
\right),\qquad
q_{i-1}^S =
\mathrm{softmax}\left(
\frac{W\,\mathrm{Norm}(h_{i-1}^S)}{T}
\right).
$$

代码中的总 loss 只有一项：

$$
\mathcal L_{\mathrm{boundary}}
 =\frac{1}{B}\sum_{b=1}^{B}
 \frac{\sum_t w_{b,t}\,
 \mathrm{KL}\!\left(q^T_{b,t}\Vert q^S_{b,t}\right)}
 {\max(\sum_t w_{b,t},1)}.
$$

实现细节：

- 只保留 assistant response 的 token；prompt、图像占位符和 padding 不参与恢复 loss。
- $w_{b,t}$ 使用 utils/field_weights.py 的字段权重：默认字段 1.0、cats_visible 2.0、动作/身体/耳朵/尾巴/脸/毛发 3.0、JSON 格式字符 0.5。
- 温度默认为 $T=1$，KL 内部乘回 $T^2$ 保持蒸馏梯度尺度。
- 每个样本先按自身有效 token 权重归一化，再对 batch 求平均，避免长回答支配短回答。
- 训练日志记录 loss 和 boundary_weighted_kl；两者在当前实现中相同。
- 当前恢复训练没有 hard-label CE、最终层 CE、隐藏态 cosine loss，也没有 LoRA adapter。

## 层号和 Teacher/Student 对齐

假设删层前 Teacher 有 $N$ 层：

| 对象 | 层数 | 边界层 |
| --- | ---: | --- |
| 本轮 Teacher（删层前当前模型） | $N$ | 原第 $i$ 层 $h_i^T$ |
| 本轮 Student（物理删除第 $i$ 层） | $N-1$ | 第 $i-1$ 层 $h_{i-1}^S$ |
| 唯一可训练参数 | — | Student language_model.layers[i-1] |

deleted-layer i 传入的是 Teacher/删层前模型的当前层号。因为第 $i$ 层已经删除，Student 中第 $i-1$ 层仍然保留，正是唯一解冻的层。第 0 层不能删除后恢复，因为没有 $i-1$；当前探测默认从第 3 层开始。

代码会验证 Teacher 和 Student 的层数必须相差 1，并把训练范围写入 resolved_config.json：

~~~json
{
  "recovery_method": "previous_layer_boundary_kd_v1",
  "teacher_boundary_layer": 12,
  "student_boundary_layer": 11,
  "trainable_layers": [11],
  "train_scope": "student_language_layer_i_minus_1_only",
  "loss": "field_weighted_KL(q_teacher_i||q_student_i_minus_1)"
}
~~~

## 运行方式

以下命令均在项目根目录 /home/alex/soulgard/soulgard-vl 执行。模型目录必须是本地 Hugging Face Qwen3-VL checkpoint。
当前 checkout 将独立包放在 highway/Scalpel，而源码内部使用 highway 包名；若没有把 Scalpel 安装或映射为 highway，请先建立临时包别名：

~~~bash
export SCALPEL_ALIAS="$(mktemp -d)"
ln -s "$PWD/highway/Scalpel" "$SCALPEL_ALIAS/highway"
export PYTHONPATH="$SCALPEL_ALIAS:$PWD"
~~~

运行完成后可用 unlink "$SCALPEL_ALIAS/highway" && rmdir "$SCALPEL_ALIAS" 清理别名目录。

### 1. 准备固定 probe 集

~~~bash
python -m highway.prune_prepare \
  --model /path/to/base_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run
~~~

该步骤生成固定 probe manifest、数据指纹和恢复协议配置。固定 probe 只用于层选择和评估，不直接替代恢复训练数据。

### 2. 执行多轮剪枝

~~~bash
python -m highway \
  --project-root /home/alex/soulgard/soulgard-vl \
  --python "$(which python)" \
  --eval-script /path/to/eval.py \
  --reference-model /path/to/base_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --model-root /path/to/model_root \
  --rounds 9
~~~

每一轮会生成：

- round_NN/probe/：候选层 probe、风险和 selected_layer.json；
- round_NN/deletion.json：实际删除的当前层和原始层映射；
- round_NN/train/：边界训练日志、resolved_config.json、loss_summary.json；
- round_NN/eval_pre_recovery/ 和 eval_post_recovery/：恢复前后评估；
- model_root/round_NN/post_recovery_model/：下一轮使用的模型。

恢复命令由 orchestrator 自动拼出，等价于：

~~~bash
python -m highway.train_weighted_kd \
  --teacher-model /path/to/current_model_before_deletion \
  --student-model /path/to/pruned_student \
  --deleted-layer 12 \
  --train-data /path/to/train.json \
  --output-dir /path/to/train \
  --export-dir /path/to/recovered_student \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --temperature 1.0
~~~

这个示例表示：Teacher 原第 12 层与 Student 第 11 层对齐，只训练 Student 第 11 层。student-model 必须已经物理删除第 12 层；该脚本不会再次删层。

### 3. 单独物理删层

~~~bash
python -m highway.prune_layer \
  --model /path/to/current_model \
  --layer 12 \
  --output-model /path/to/pruned_model \
  --record /path/to/deletion.json
~~~

prune_layer.py 只负责结构化删除和保存，不负责恢复训练。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| prune_prepare.py | 固定 probe 划分、数据指纹和本轮恢复协议 |
| prune_probe.py | 对候选层做临时绕过 probe，按最坏轴风险选层 |
| prune_layer.py | 物理删除一个 language decoder layer，更新层数、attention cache index 和原始层映射 |
| train_weighted_kd.py | 边界恢复训练；冻结 Teacher，只解冻 Student 的 $i-1$ 层 |
| prune_highway.py | 可恢复的多轮 orchestrator；连接 probe、删层、边界训练和评估 |
| train_summarize.py | 汇总 loss.jsonl，只认 boundary_weighted_kl |
| utils/model_ops.py | Qwen3-VL 层访问、视觉输入缓存、边界 hook、物理删层和冻结/解冻 |
| utils/metrics.py | field-weighted boundary KL；掩码、样本归一化和温度处理 |
| utils/field_weights.py | 字段解析及 probe/恢复共用的 token 权重 |
| utils/recovery_config.py | previous_layer_boundary_kd_v1 协议标识 |
| utils/sft_dataset.py | 不依赖 LoRA 的图文样本和字段权重构造 |
| utils/training_log.py | 不依赖 LoRA 的 Trainer JSONL 日志回调 |
| utils/training_collator.py | 动态 padding、图像拼接和 loss 权重整理 |
| eval_generation.py | 生成式任务评估 |
| eval_logits.py | 诊断性的最终 logits CE/KL/top-1 指标，不参与恢复优化 |
| eval_summarize.py | 汇总评估目录中的指标 |
| tests/ | 层映射、边界 KL、训练日志和 probe 逻辑测试 |

## 输出审计

每个恢复训练目录至少包含：

- resolved_config.json：Teacher/Student 路径、删除层、边界层、唯一可训练层、loss 和字段权重；
- loss.jsonl：Trainer 的逐步 loss 与 boundary_weighted_kl；
- loss_summary.json：去重 step 后的首尾窗口统计；
- summary.json：训练指标和导出路径；
- checkpoints/：Trainer checkpoint。

模型导出时，highway_state.json 的 recoveries 记录本轮 Teacher、删除层、Student 边界层和恢复方法。Orchestrator 会拒绝没有 previous_layer_boundary_kd_v1 标记的旧训练结果，避免把旧的全量恢复协议混入新实验。

## 测试

~~~bash
python -m compileall -q highway/Scalpel
~~~

由于 highway/Scalpel 是独立源码目录，而测试使用 highway.* 包名，可通过临时包别名运行完整单测：

~~~bash
tmpdir="$(mktemp -d)"
ln -s /home/alex/soulgard/soulgard-vl/highway/Scalpel "$tmpdir/highway"
PYTHONPATH="$tmpdir:/home/alex/soulgard/soulgard-vl" \
python -m unittest discover -s "$tmpdir/highway/tests" -v
unlink "$tmpdir/highway"
rmdir "$tmpdir"
~~~

历史 highway/results/ 下若是旧的全量线性层恢复结果，不应直接续跑；请为边界协议使用新的 run-dir 和 model-root。
