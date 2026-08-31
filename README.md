# ✂️Scalpel: Recovery-Aware Layer Pruning for Vision-Language Models

Scalpel 是面向多模态大模型的逐层结构化剪枝工具。它每轮只真正删除一层，并在删除后做一次轻量恢复训练，让更小的 student 尽量保持原模型在目标任务上的输出分布和字段准确率。

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/TsDLjZIblTgMjdDkJ0feU.png" alt="FelineBench benchmark visualization" width="420"/>
</div>

**Scalpel-VL-1.6B 模型仓库：** https://huggingface.co/freeai-org/Scalpel-VL-1.6B

**Scalpel 完整介绍：** https://freeai-org.github.io/Scalpel/

****

## 🚄方法流程

每一轮执行：

1. 在固定 probe 集上临时绕过候选层，计算任务准确率下降和 logits 分布漂移。
2. 用 `max(relative_hard_regret, normalized_js)` 作为该层风险，选择风险最低的当前层。
3. 物理删除该层，保存 `pre_recovery_model`。
4. 用固定 reference teacher 和删层后的 student 做 teacher-forcing 恢复训练。
5. Student 挂 LoRA，目标模块为 `all-linear`；训练完成后 merge LoRA 并导出 `post_recovery_model`。
6. `post_recovery_model` 作为下一轮要继续剪枝的当前模型；teacher 仍然是固定 reference。

## 📏评估

逐步移除Transformer层后模型精度与推理速度变化。随着剪枝轮次增加，推理速度持续提升，精度损失整体可控。

| Round | 删除原始层 | 剩余层 | Acc. | $\Delta$ Accuracy | Speed     | Speed‑up |
|:-----:|:----------:|:------:|:----------:|:-----------------:|:---------:|:--------:|
| 01    | L05        | 27     | 84.00%     | -0.21 pp          | 0.845 it/s| +3.5%    |
| 02    | L08        | 26     | 83.85%     | -0.35 pp          | 0.853 it/s| +4.4%    |
| 03    | L09        | 25     | 83.84%     | -0.36 pp          | 0.897 it/s| +9.8%    |
| 04    | L23        | 24     | 83.76%     | -0.44 pp          | 0.941 it/s| +15.3%   |
| 05    | L04        | 23     | 83.87%     | -0.34 pp          | 0.956 it/s| +17.1%   |
| 06    | L18        | 22     | 83.86%     | -0.34 pp          | 1.045 it/s| +28.0%   |
| 07    | L15        | 21     | 83.56%     | -0.64 pp          | 1.073 it/s| +31.4%   |
| 08    | L13        | 20     | 83.82%     | -0.39 pp          | 1.122 it/s| +37.4%   |
| 09    | L25        | 19     | 83.85%     | -0.36 pp          | 1.172 it/s| +43.5%   |


$$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{KL}}$$


## 🏋️训练
> 训练集为私有数据集，本项目的输出数据格式为 JSON 格式，但方法通用


**Loss 图如下所示：**

<p align="center">
  <img width="520" alt="hard weighted CE loss curve" src="https://github.com/user-attachments/assets/bbe25e93-4be6-4f09-b64d-12ccbb7f1814">
  <br>
  <b>(a) CE Loss</b>
</p>


$$\mathcal{L}_{\mathrm{CE}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{CE}(z^{S}_{b,t}, y_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

$$\mathcal{L}_{\mathrm{KL}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{KL}(P^{T}_{b,t} || P^{S}_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

其中：

$$P^{T}_{b,t} = \mathrm{softmax}(z^{T}_{b,t}/T), \quad P^{S}_{b,t} = \mathrm{softmax}(z^{S}_{b,t}/T)$$

这里 `teacher_logits` 和 `student_logits` 都是完整 forward 后经过最终 LM head 得到的 logits。`labels == -100` 的 prompt、padding、图像占位符不参与 loss。

<p align="center">
  <img width="520" alt="hard weighted CE loss curve" src="https://github.com/user-attachments/assets/bbe25e93-4be6-4f09-b64d-12ccbb7f1814">
  <br>
  <b>(a) CE Loss</b>
</p>

<p align="center">
  <img width="520" alt="soft weighted KL loss curve" src="https://github.com/user-attachments/assets/6c05d76f-7b5d-4cf7-8a30-54662e0988f6">
  <br>
  <b>(b) KL Loss</b>
</p>


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
注意文件相对路径

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
  --eval-script /path/to/eval_universal_json.py \
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

<<<<<<< HEAD
## ✨输出结构
=======
### Scalpel 100M text mixture 实验

`data/train/*.parquet` 使用 token 预算 `English 65% / Chinese 20% /
Math 10% / Code 5%`。数据全部来自原生 instruction/response、question/answer、
problem/solution 或 messages 对；不使用文本 continuation 伪造指令。完整 chat
超过 1536 token 的样本在构建时整条排除，训练时禁止静默截断。新的 mixture
流程不重写原 Parquet，而是在读取时
构造可复现的虚拟 mixture：

- 固定 seed 后在每个大类内打乱，然后按 estimated token 贪心均衡到
  `t1 ... t10`。
- 每个 `ti` 内使用确定性的按比例交错顺序；每一步选择“相对累计目标最
  欠采样”的类别，使每个短窗口接近该分区的总体样本比例，同时每条样本
  仍恰好读取一次。
- `batch_size=1, grad_accum=16` 时，每 16 个样本构成一个有效 batch；
  `logging_steps=20` 时，每条 loss 记录平均同一个 320 样本混合窗口。
  日志同时写入 `mixture_window_samples` 和四个
  `mixture_<group>_fraction` 字段，用于核对该 loss 对应的数据组成。
- validation 每类固定抽 20 条，共 80 条；PPL 评估全部可监督 token。
- 内部 `leaderboard_score` 定义为四个大类 teacher-forced token
  accuracy 的 macro average 乘 100；它用于这次消融实验的可比较排名，不是
  Open LLM Leaderboard 的对外分数。
- 每轮删除一层，只在对应 `ti` 上恢复训练 1 epoch，累计 10 轮后
  每条 train 样本恰好被使用一次。

完整命令：

```bash
export PYTHONPATH="$PWD/highway:$PWD"
python -m Scalpel.mixture_experiment \
  --project-root "$PWD" \
  --python "$(which python)" \
  --reference-model "$PWD/B-RVPO/models/vl_local" \
  --train-data "$PWD/highway/Scalpel/data/train" \
  --val-data "$PWD/highway/Scalpel/data/validation" \
  --run-dir "$PWD/highway/Scalpel/results/qwen3vl2b_mixture_interleaved_prune10_seed20260828" \
  --model-root "$PWD/highway/Scalpel/results/qwen3vl2b_mixture_interleaved_prune10_seed20260828/models" \
  --rounds 10 --train-parts 10 --data-seed 20260828 \
  --test-samples-per-group 20 --max-length 1536
```

每轮会写入 `probe/layer_leaderboard.{csv,json}`、`train/loss.jsonl`、
`eval_pre_recovery/metrics.json` 和 `eval_post_recovery/metrics.json`；最终汇总为
`final/all_rounds.{csv,json}`。流程可重入，已完成的阶段不会重算。

## 输出结构
>>>>>>> 02706b2 (Scalpel-VL-Prune)

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
ln -s ../Scalpel "$tmpdir/highway"
PYTHONPATH="$tmpdir:/home/alex/soulgard/soulgard-vl" \
python -m unittest discover -s "$tmpdir/highway/tests" -v
unlink "$tmpdir/highway"
rmdir "$tmpdir"
```



## 📖Citation

```text
@misc{wu2026catellectvl2bvisionlanguagemodeledgebased,
      title={Catellect-VL-2B: A Vision-Language Model for Edge-Based Feline Behavior Understanding}, 
      author={YuHang Wu and HaoXian Liu and Jia Tao},
      year={2026},
      eprint={2608.22070},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2608.22070}, 
}
```
