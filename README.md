# ✂️Scalpel: Recovery-Aware Layer Pruning for Vision-Language Models

Scalpel 是面向多模态大模型的逐层结构化剪枝工具。它每轮只真正删除一层，并在删除后做一次轻量恢复训练，让更小的 student 尽量保持原模型在目标任务上的输出分布和字段准确率。

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/TsDLjZIblTgMjdDkJ0feU.png" alt="FelineBench benchmark visualization" width="420"/>
</div>

**🦊业务剪枝 Scalpel-VL-1.6B-Animal 模型仓库：** https://huggingface.co/freeai-org/Scalpel-VL-1.6B-Animal

**📗ScalpelBench：** 0.1B 大小的数据集，300k 条样本，覆盖英语、中文、数学、代码四大类别，目的是在剪枝的同时保留模型基座的能力，参考 https://huggingface.co/datasets/freeai-org/ScalpelBench

**🔗Scalpel 完整介绍：** https://freeai-org.github.io/Scalpel

****

## 📑目录

- [🚄Scalpel 方法流程](#scalpel-方法流程)
- [📏Scalpel 业务剪枝](#scalpel-业务剪枝)
- [🏋️Scalpel 通用剪枝](#scalpel-通用剪枝)
- [📚ScalpelBench：0.1B 混合数据恢复](#scalpelbench01b-混合数据恢复)
- [🚕运行方式](#运行方式)
  - [入口一：完整业务剪枝](#入口一完整业务剪枝)
  - [入口二：ScalpelBench 0.1B 混合恢复](#入口二scalpelbench-01b-混合恢复)
- [🔧 `prune_*` 文件怎么使用](#-prune_-文件怎么使用)
  - [`prune_prepare.py`：准备业务 probe](#prune_preparepy准备业务-probe)
  - [`prune_probe.py`：业务候选层排序](#prune_probepy业务候选层排序)
  - [`prune_text.py`：ScalpelBench 文本候选层排序](#prune_textpyscalpelbench-文本候选层排序)
  - [`prune_layer.py`：物理删除一层](#prune_layerpy物理删除一层)
  - [`prune_highway.py`：完整多轮编排](#prune_highwaypy完整多轮编排)
- [✨输出结构](#输出结构)
- [📚文件说明](#文件说明)
- [🖱️测试](#测试)
- [📖Citation](#citation)

## 🚄Scalpel 方法流程

每一轮执行：

1. 在固定 probe 集上临时绕过候选层，计算任务准确率下降和 logits 分布漂移。
2. 用 `max(relative_hard_regret, normalized_js)` 作为该层风险，选择风险最低的当前层。
3. 物理删除该层，保存 `pre_recovery_model`。
4. 用固定 reference teacher 和删层后的 student 做 teacher-forcing 恢复训练。
5. Student 挂 LoRA，目标模块为 `all-linear`；训练完成后 merge LoRA 并导出 `post_recovery_model`。
6. `post_recovery_model` 作为下一轮要继续剪枝的当前模型；teacher 仍然是固定 reference。

## 📏Scalpel 业务剪枝

业务数据涉暂不开放，只展示 Scalpel 的优势，结果显示，逐步移除Transformer层后模型精度与推理速度变化。随着剪枝轮次增加，推理速度持续提升，精度损失整体可控。

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

**业务 Bench Loss 图如下：**
Prune 10 层 Layer，速度快约 43.5%，精度掉点不到 0.5%。
<div align="center">
<img src="https://github.com/user-attachments/assets/36585efe-ab7c-48ae-8f09-0dec156bb1cd" alt="FelineBench benchmark visualization" width="500"/>
</div>



## 🏋️Scalpel 通用剪枝

模型在 Prune 的同时一定会损伤其泛化能力。因此我们需要通过 Scalpel 在每轮删层后使用固定 reference teacher 和删层后的 student 做恢复训练：student 仅训练 `all-linear` LoRA，完成后 merge adapter，导出新的完整模型供下一轮继续剪枝。

恢复目标使用完整 forward 后的最终 LM-head logits：

$$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{KL}}$$

$$\mathcal{L}_{\mathrm{CE}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{CE}(z^{S}_{b,t}, y_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

$$\mathcal{L}_{\mathrm{KL}} = \frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{t} w_{b,t}\,\mathrm{KL}(P^{T}_{b,t} || P^{S}_{b,t})}{\max(\sum_{t} w_{b,t}, 1)}$$

其中：

$$P^{T}_{b,t} = \mathrm{softmax}(z^{T}_{b,t}/T), \quad P^{S}_{b,t} = \mathrm{softmax}(z^{S}_{b,t}/T)$$

`labels == -100` 的 prompt、padding 和图像占位符不参与 loss。`--deleted-layer` 只用于审计记录，loss 不读取第 `i` 层或第 `i-1` 层 hidden state。

## 📚ScalpelBench：0.1B 混合数据恢复

为了进一步弥补模型在业务数据剪枝后损失的通用能力，我们同时发布了 [ScalpelBench](https://huggingface.co/datasets/freeai-org/ScalpelBench) 和对应的混合数据恢复流程。ScalpelBench 是约 **0.1B tokens** 的 instruction-response 语料，覆盖通用英文、中文、数学推理和代码生成。该实验用持续训练模拟 **mid-training-style recovery**，观察公开混合数据能否补偿结构化剪枝造成的能力损失；它是受控的恢复实验，不等同于从头预训练。

实验使用的顶层 token 预算为：

| 能力组 | Token 预算 |
| --- | ---: |
| English | 65% |
| Chinese | 20% |
| Math | 10% |
| Code | 5% |

数据只保留原生 instruction/response、question/answer、problem/solution 或 messages 对，不使用文本 continuation 伪造指令。`train` 和 `validation` 均以 Parquet 发布；数据来源、字段、限制和许可证请以 [ScalpelBench 数据集页面](https://huggingface.co/datasets/freeai-org/ScalpelBench) 为准。

```python
from datasets import load_dataset

dataset = load_dataset("freeai-org/ScalpelBench")
```


**ScalpelBench Loss 图如下：**
剪枝 7 层，依然拥有较强的泛化能力，并对比 Baseline 加速大约 26.77%。
<div align="center">
<img src="https://github.com/user-attachments/assets/46f9e025-0b85-4d04-9280-84945645cced" alt="FelineBench benchmark visualization" width="500"/>
</div>



## 🚕运行方式

以下命令假设当前目录是包含 `highway/Scalpel` 的项目根目录。业务 JSON/VL 流程内部使用 `highway.*` 包名，因此先建立临时包别名：

```bash
export SCALPEL_ALIAS="$(mktemp -d)"
ln -s "$PWD/highway/Scalpel" "$SCALPEL_ALIAS/highway"
export PYTHONPATH="$SCALPEL_ALIAS:$PWD"
```

### 入口一：完整业务剪枝

先用 `prune_prepare.py` 固定 probe 和数据指纹：

```bash
python -m highway.prune_prepare \
  --model /path/to/reference_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --repeats 10 \
  --samples-per-repeat 10 \
  --seed 20260726 \
  --rounds 9 \
  --recovery-epochs 2
```

然后使用 `prune_highway.py` 一次完成 baseline 评估、逐层 probe、物理删层、恢复训练和前后评估：

```bash
python -m highway \
  --project-root "$PWD" \
  --python "$(which python)" \
  --eval-script "$PWD/highway/Scalpel/eval_universal_json.py" \
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

`prune_highway.py` 是业务数据流程的推荐入口。相同的 `run-dir` 可以续跑；默认在 post-recovery 模型确认可用后删除体积较大的 `pre_recovery_model`，需要保留时加 `--retain-pre-models`。

### 入口二：ScalpelBench 0.1B 混合恢复

先将数据集下载到本地目录：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="freeai-org/ScalpelBench",
    repo_type="dataset",
    local_dir="/path/to/ScalpelBench",
)
```

然后从项目根目录启动 10 轮实验：

```bash
export PYTHONPATH="$PWD/highway:$PWD"

python -m Scalpel.mixture_experiment \
  --project-root "$PWD" \
  --python "$(which python)" \
  --reference-model /path/to/qwen3_vl_model \
  --train-data /path/to/ScalpelBench/train \
  --val-data /path/to/ScalpelBench/validation \
  --run-dir /path/to/mixture_run \
  --model-root /path/to/mixture_run/models \
  --rounds 10 \
  --train-parts 10 \
  --data-seed 20260828 \
  --test-samples-per-group 20 \
  --max-length 1536 \
  --recovery-batch-size 1 \
  --recovery-effective-batch-size 16 \
  --learning-rate 1e-4
```

只准备固定 probe 和 10 个训练分区、不启动模型计算时，加 `--prepare-only`。

## 🔧 `prune_*` 文件怎么使用

| 文件 | 用途 | 何时使用 | 主要输出 |
| --- | --- | --- | --- |
| `prune_prepare.py` | 固定业务验证集 probe，记录模型/数据 SHA-256 和实验配置 | 业务 JSON/VL 流程开始前运行一次 | `probe/probe_10x10.{jsonl,csv}`、`dataset_fingerprints.json` |
| `prune_probe.py` | 临时绕过业务模型的每个候选层，用生成准确率下降和 normalized JS 排序 | 已有固定 probe 和 reference baseline predictions 时 | `selected_layer.json`、`layer_metrics.csv`、逐层 predictions |
| `prune_text.py` | 在 ScalpelBench 文本 probe 上用 PPL、token accuracy 和宏平均能力分数排层 | 0.1B 混合恢复流程，或单独分析文本层风险时 | `selected_layer.json`、`layer_leaderboard.{csv,json}`、`progress.json` |
| `prune_layer.py` | 物理删除指定的当前 language decoder layer，并维护原始层编号映射 | probe 已选出目标层之后 | 完整删层模型、`deletion.json`、`highway_state.json` |
| `prune_highway.py` | 编排完整业务流程 | 希望自动完成多轮业务剪枝与恢复时 | 每轮 probe/train/eval、最终汇总和模型 |

### `prune_prepare.py`：准备业务 probe

它只生成不可变的采样清单和审计信息，不加载模型做推理。`--train-data` 和 `--val-data` 应为项目支持的 SFT JSON 文件，`--samples-per-repeat` 不能大于验证集样本数。

```bash
python -m highway.prune_prepare \
  --model /path/to/reference_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --repeats 10 \
  --samples-per-repeat 10
```

### `prune_probe.py`：业务候选层排序

`--reference-model` 是始终固定的 teacher，`--candidate-model` 是本轮准备继续剪枝的 student。`--baseline-predictions` 必须是 reference model 在完整验证集上的 `predictions.jsonl`；完整编排器会自动准备它。

```bash
python -m highway.prune_probe \
  --reference-model /path/to/reference_model \
  --candidate-model /path/to/current_model \
  --val-data /path/to/val.json \
  --probe-manifest /path/to/run/probe/probe_10x10.jsonl \
  --baseline-predictions /path/to/run/baseline/predictions.jsonl \
  --output-dir /path/to/run/round_01/probe \
  --min-layer 3 \
  --generation-batch-size 4 \
  --max-new-tokens 2048
```

只测试部分当前层时使用 `--candidate-layers 3,4,5`；这些数字是当前模型中的层索引，不是 `original_layer`。

### `prune_text.py`：ScalpelBench 文本候选层排序

该脚本使用 teacher-forced 评估，不进行自由生成。`--data` 可以是固定的 JSONL probe；如果同一个 `output-dir` 下存在签名一致的 `progress.json`，中断后会从下一条样本继续。

```bash
export PYTHONPATH="$PWD/highway:$PWD"

python -m Scalpel.prune_text \
  --model /path/to/current_model \
  --data /path/to/mixture_run/probe/fixed_4x20.jsonl \
  --output-dir /path/to/mixture_run/round_01/probe \
  --min-layer 3 \
  --max-length 1536 \
  --max-answer-tokens 128 \
  --logit-chunk-size 32
```

### `prune_layer.py`：物理删除一层

从 `selected_layer.json` 读取 `selected.current_layer` 传给 `--layer`。不要传 `selected.original_layer`。`--output-model` 必须是尚不存在的新目录。

```bash
python -m highway.prune_layer \
  --model /path/to/current_model \
  --layer 12 \
  --output-model /path/to/round_01/pre_recovery_model \
  --record /path/to/round_01/deletion.json
```

### `prune_highway.py`：完整多轮编排

通常直接使用上面的 `python -m highway ...`。它按顺序调用 `prune_probe.py`、`prune_layer.py` 和 `train_weighted_kd.py`，并在每轮保存 pre/post-recovery 指标。只有在调试单个阶段时，才需要手动调用前三个脚本。

单独跑一轮恢复训练：

```bash
python -m highway.train_weighted_kd \
  --teacher-model /path/to/reference_model \
  --student-model /path/to/pre_recovery_model \
  --deleted-layer 12 \
  --train-data /path/to/train.json \
  --output-dir /path/to/round_01/train \
  --export-dir /path/to/round_01/post_recovery_model \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --temperature 1.0
```

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
| `prune_text.py` | 在 ScalpelBench 固定文本 probe 上按 PPL 和能力保持度选层 |
| `prune_layer.py` | 物理删除一个 language decoder layer |
| `train_weighted_kd.py` | 最终 logits CE+KL 恢复训练，LoRA 挂 all-linear |
| `prune_highway.py` | 多轮剪枝编排器 |
| `mixture_experiment.py` | ScalpelBench 0.1B、10 分区、10 轮可恢复实验编排器 |
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
