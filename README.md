# ✂️Scalpel: Recovery-Aware Layer Pruning for Vision-Language Models

Scalpel 是面向多模态大模型的逐层结构化剪枝工具。它每轮只真正删除一层，并在删除后做一次轻量恢复训练，让更小的 student 尽量保持原模型在目标任务上的输出分布和字段准确率。

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/TsDLjZIblTgMjdDkJ0feU.png" alt="FelineBench benchmark visualization" width="420"/>
</div>

**模型路径：** https://huggingface.co/freeai-org/Scalpel-VL-1.6B

****

## 🚄方法流程

每一轮执行：

1. 在固定 probe 集上临时绕过候选层，计算任务准确率下降和 logits JS 漂移。
2. 用 `max(relative_hard_regret, normalized_js)` 作为该层风险，选择风险最低的当前层。
3. 物理删除该层，保存 `pre_recovery_model`。
4. 用固定 reference teacher 和删层后的 student 做 teacher-forcing 恢复训练。
5. Student 挂 LoRA，目标模块为 `all-linear`；训练完成后 merge LoRA 并导出 `post_recovery_model`。
6. `post_recovery_model` 作为下一轮要继续剪枝的当前模型；teacher 仍然是固定 reference。


## 🏋️训练
> 训练集为私有数据集，方法通用修改指令对格式即可

**Loss 图如下所示：**

<center>
<img width="650" alt="hard weighted CE loss curve" src="https://github.com/user-attachments/assets/bbe25e93-4be6-4f09-b64d-12ccbb7f1814">
<br>
<b>(a) CE Loss</b>
<br><br>

<img width="650" alt="soft weighted KL loss curve" src="https://github.com/user-attachments/assets/6c05d76f-7b5d-4cf7-8a30-54662e0988f6">
<br>
<b>(b) KL Loss</b>
</center>


**损失函数:**

$$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{KL}}$$

$$\mathcal{L}_{\mathrm{CE}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{CE}(z^{S}_{b,t}, y_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

$$\mathcal{L}_{\mathrm{KL}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{KL}(P^{T}_{b,t} || P^{S}_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

**超参：**

```text
epochs: 2
effective_batch_size: 16
learning_rate: 1e-4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: all-linear
temperature: 1.0
```

## 🚕运行方式

以下命令在项目根目录 `/home/alex/soulgard/soulgard-vl` 执行。当前源码位于 `highway/Scalpel`，内部包名为 `highway`；如果没有安装包，可先建立临时别名：

```bash
export SCALPEL_ALIAS="$(mktemp -d)"
ln -s "$PWD/highway/Scalpel" "$SCALPEL_ALIAS/highway"
export PYTHONPATH="$SCALPEL_ALIAS:$PWD"
```

准备固定 probe 集：

```bash
python -m highway.prune_prepare \
  --model /path/to/reference_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --repeats 10 \
  --samples-per-repeat 10 \
  --rounds 9 \
  --recovery-epochs 2
```

启动多轮剪枝：

```bash
python -m highway \
  --project-root /home/alex/soulgard/soulgard-vl \
  --python "$(which python)" \
  --eval-script /home/alex/soulgard/soulgard-vl/sft_scripts/eval_universal_json.py \
  --reference-model /path/to/reference_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --model-root /path/to/model_root \
  --rounds 9 \
  --recovery-batch-size 4 \
  --recovery-effective-batch-size 16 \
  --recovery-epochs 2 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

单独跑一轮恢复训练：

```bash
python -m highway.train_weighted_kd \
  --teacher-model /path/to/reference_model \
  --student-model /path/to/pre_recovery_model \
  --deleted-layer 12 \
  --train-data /path/to/train.json \
  --output-dir /path/to/round_XX/train \
  --export-dir /path/to/round_XX/post_recovery_model \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --temperature 1.0
```

`--deleted-layer` 只用于审计记录，loss 不读取第 `i` 层或第 `i-1` 层 hidden state。

## ✨输出结构

每个 round 主要文件：

| 路径 | 作用 |
| --- | --- |
| `round_NN/probe/selected_layer.json` | 本轮选择删除的当前层和原始层 |
| `round_NN/deletion.json` | 物理删层记录 |
| `round_NN/eval_pre_recovery/` | 恢复前评估 |
| `round_NN/train/loss.jsonl` | 逐步训练日志，含 `hard_weighted_ce` 和 `soft_weighted_kl` |
| `round_NN/train/adapter/` | 本轮 LoRA adapter |
| `round_NN/train/summary.json` | teacher/student 路径、LoRA 超参、loss 和训练指标 |
| `round_NN/eval_post_recovery/` | 恢复后评估 |
| `model_root/round_NN/post_recovery_model/` | merge LoRA 后的完整 student |

训练 summary 中的协议字段应类似：

```json
{
  "recovery_method": "final_logits_lora_kd_v1",
  "teacher_target": "final_logits",
  "student_target": "final_logits",
  "train_scope": "student_lora_all_linear",
  "loss": "field_weighted_ce + field_weighted_KL(teacher_final||student_final)"
}
```

## 📚文件说明

| 文件 | 作用 |
| --- | --- |
| `prune_prepare.py` | 固定 probe 划分、数据指纹和实验配置 |
| `prune_probe.py` | 临时绕过候选层，按任务后悔值和 JS 漂移选层 |
| `prune_layer.py` | 物理删除一个 language decoder layer |
| `train_weighted_kd.py` | 最终 logits CE+KL 恢复训练，LoRA 挂 all-linear |
| `prune_highway.py` | 多轮剪枝编排器 |
| `train_summarize.py` | 汇总 `loss.jsonl` 的 CE/KL 分量 |
| `utils/metrics.py` | probe 指标和字段加权 KD loss |
| `utils/kd_runtime.py` | 固定 teacher 加载和 student all-linear LoRA 挂载 |
| `utils/field_weights.py` | JSON 字段到 token 权重的映射 |
| `utils/model_ops.py` | Qwen3-VL 层访问、视觉输入缓存、临时/物理删层和模型保存 |
| `tests/` | 核心指标、层删除、训练日志和 probe 逻辑测试 |

## 🖱️测试

```bash
python -m compileall -q highway/Scalpel

tmpdir="$(mktemp -d)"
ln -s /home/alex/soulgard/soulgard-vl/highway/Scalpel "$tmpdir/highway"
PYTHONPATH="$tmpdir:/home/alex/soulgard/soulgard-vl" \
python -m unittest discover -s "$tmpdir/highway/tests" -v
unlink "$tmpdir/highway"
rmdir "$tmpdir"
```



## 📖Citation

```text
@misc{wu2026soulgardvl2bvisionlanguagemodeledgebased,
      title={SoulGard-VL-2B: A Vision-Language Model for Edge-Based Feline Behavior Understanding},
      author={YuHang Wu and HaoXian Liu and Jia Tao},
      year={2026},
      eprint={2608.22070},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2608.22070},
}
```
