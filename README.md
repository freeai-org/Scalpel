# ✂️ Scalpel: Recovery-Aware Layer Pruning for LLMs

Scalpel makes an LLM smaller, reduces GPU memory usage, and improves inference speed by roughly 20–40%. It is a layer-by-layer structured pruning toolkit that physically removes one layer per round, followed by lightweight recovery training. The goal is to keep the smaller student model close to the original model in output distribution and task accuracy.

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66d295c4f87ed8c2bc246a2d/TsDLjZIblTgMjdDkJ0feU.png" alt="Scalpel overview" width="420"/>
</div>

**🦜 General-purpose recovery model — Scalpel-VL-1.8B:**  
https://huggingface.co/freeai-org/Scalpel-VL-1.8B

**🦊 Domain-specific recovery model — Scalpel-VL-1.6B-Animal:**  
https://huggingface.co/freeai-org/Scalpel-VL-1.6B-Animal

**📗 ScalpelBench:** A 0.1B-token dataset with roughly 300K samples across English, Chinese, mathematics, and code. It is designed to preserve the general capabilities of the base model during pruning.  
https://huggingface.co/datasets/freeai-org/ScalpelBench

**🔗 Full Scalpel documentation:**  
https://freeai-org.github.io/Scalpel

<div>
  <video
    controls
    width="32%"
    src="https://github.com/user-attachments/assets/576a2832-7629-46d5-a964-1d38afa1240c">
  </video>
  <video
    controls
    width="32%"
    src="https://github.com/user-attachments/assets/404e1909-950b-4d98-b95d-8943dbe5b948">
  </video>
  <video
    controls
    width="32%"
    src="https://github.com/user-attachments/assets/7eaeff6f-1206-4064-8891-cfbe5623d259">
  </video>
</div>

1. **Scalpel-VL-1.8B** is obtained by pruning seven layers from Qwen3-VL-2B and applying recovery training. It has the highest throughput while retaining solid accuracy and generalization on the evaluation set.
2. **InternLM2-1.8B** comes second, but remains noticeably slower than Scalpel-VL-1.8B.
3. **Qwen3-VL-2B** is the slowest of the three.

---

## 📑 Table of Contents

- [Scalpel Workflow](#scalpel-workflow)
- [Domain-Specific Pruning](#domain-specific-pruning)
- [General-Purpose Pruning](#general-purpose-pruning)
- [ScalpelBench: 0.1B-Token Recovery Mixture](#scalpelbench-01b-token-recovery-mixture)
- [Running Scalpel](#running-scalpel)
  - [Entry Point 1: Full Domain-Specific Pruning](#entry-point-1-full-domain-specific-pruning)
  - [Entry Point 2: ScalpelBench Recovery](#entry-point-2-scalpelbench-recovery)
- [Using the `prune_*` Scripts](#using-the-prune_-scripts)
  - [`prune_prepare.py`: Prepare the Domain Probe](#prune_preparepy-prepare-the-domain-probe)
  - [`prune_probe.py`: Rank Candidate Layers](#prune_probepy-rank-candidate-layers)
  - [`prune_text.py`: Rank Layers with ScalpelBench](#prune_textpy-rank-layers-with-scalpelbench)
  - [`prune_layer.py`: Physically Remove a Layer](#prune_layerpy-physically-remove-a-layer)
  - [`prune_highway.py`: Run the Full Multi-Round Pipeline](#prune_highwaypy-run-the-full-multi-round-pipeline)
- [Output Structure](#output-structure)
- [File Reference](#file-reference)
- [Tests](#tests)
- [Citation](#citation)

## 🚄 Scalpel Workflow

Each pruning round follows the same procedure:

1. Temporarily bypass each candidate layer on a fixed probe set and measure the drop in task accuracy and the shift in the output-logit distribution.
2. Compute the layer risk using `max(relative_hard_regret, normalized_js)` and select the current layer with the lowest risk.
3. Physically remove that layer and save the result as `pre_recovery_model`.
4. Run teacher-forced recovery training using a fixed reference teacher and the pruned student.
5. Attach LoRA adapters to all linear modules in the student. After training, merge the adapters and export `post_recovery_model`.
6. Use `post_recovery_model` as the starting point for the next pruning round. The reference teacher remains fixed throughout the experiment.

## 📏 Domain-Specific Pruning

The domain dataset is not public at this time, but the results below illustrate the main behavior of Scalpel. As more Transformer layers are removed, inference speed improves steadily while the accuracy loss remains controlled.

| Round | Removed Original Layer | Remaining Layers | Accuracy | $\Delta$ Accuracy | Speed | Speed-up |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | L05 | 27 | 84.00% | -0.21 pp | 0.845 it/s | +3.5% |
| 02 | L08 | 26 | 83.85% | -0.35 pp | 0.853 it/s | +4.4% |
| 03 | L09 | 25 | 83.84% | -0.36 pp | 0.897 it/s | +9.8% |
| 04 | L23 | 24 | 83.76% | -0.44 pp | 0.941 it/s | +15.3% |
| 05 | L04 | 23 | 83.87% | -0.34 pp | 0.956 it/s | +17.1% |
| 06 | L18 | 22 | 83.86% | -0.34 pp | 1.045 it/s | +28.0% |
| 07 | L15 | 21 | 83.56% | -0.64 pp | 1.073 it/s | +31.4% |
| 08 | L13 | 20 | 83.82% | -0.39 pp | 1.122 it/s | +37.4% |
| 09 | L25 | 19 | 83.85% | -0.36 pp | 1.172 it/s | +43.5% |

**Domain benchmark loss curve**

After pruning ten layers, inference is approximately 43.5% faster while the accuracy drop remains below 0.5 percentage points.

<div align="center">
<img src="https://github.com/user-attachments/assets/36585efe-ab7c-48ae-8f09-0dec156bb1cd" alt="Domain benchmark loss curve" width="500"/>
</div>

## 🏋️ General-Purpose Pruning

Pruning inevitably affects the general capabilities of an LLM. To reduce this loss, Scalpel runs recovery training after each layer removal using a fixed reference teacher and the pruned student.

Only `all-linear` LoRA parameters are trained in the student. The adapter is merged after each round, producing a complete model for the next pruning step.

The recovery objective uses the final LM-head logits from a complete forward pass:

$$
\mathcal{L}_{\mathrm{Total}}
=
\mathcal{L}_{\mathrm{CE}}
+
\mathcal{L}_{\mathrm{KL}}
$$

$$
\mathcal{L}_{\mathrm{CE}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\frac{
\sum_{t} w_{b,t}\,
\mathrm{CE}(z^{S}_{b,t}, y_{b,t})
}{
\max(\sum_{t} w_{b,t}, 1)
}
$$

$$
\mathcal{L}_{\mathrm{KL}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\frac{
\sum_{t} w_{b,t}\,
\mathrm{KL}(P^{T}_{b,t} \parallel P^{S}_{b,t})
}{
\max(\sum_{t} w_{b,t}, 1)
}
$$

where:

$$
P^{T}_{b,t}
=
\mathrm{softmax}(z^{T}_{b,t}/T),
\qquad
P^{S}_{b,t}
=
\mathrm{softmax}(z^{S}_{b,t}/T)
$$

Prompt tokens, padding tokens, and image placeholders with `labels == -100` do not contribute to the loss.

The `--deleted-layer` argument is used only for experiment tracking. The loss function does not read the hidden state of layer $i$ or layer $i-1$.

## 📚 ScalpelBench: 0.1B-Token Recovery Mixture

To restore general capabilities after domain-specific pruning, we also release [ScalpelBench](https://huggingface.co/datasets/freeai-org/ScalpelBench) and the corresponding mixed-data recovery pipeline.

ScalpelBench contains approximately **0.1B tokens** of instruction-response data across general English, Chinese, mathematical reasoning, and code generation.

The experiment uses continued training as a form of **mid-training-style recovery**. Its purpose is to study whether a compact public data mixture can compensate for the capability loss caused by structured pruning. This is a controlled recovery experiment rather than pretraining from scratch.

The top-level token budget is:

| Capability Group | Token Budget |
|---|---:|
| English | 65% |
| Chinese | 20% |
| Math | 10% |
| Code | 5% |

The dataset keeps native instruction-response, question-answer, problem-solution, and message pairs. It does not convert plain text continuations into synthetic instructions.

Both `train` and `validation` are released in Parquet format. See the [ScalpelBench dataset page](https://huggingface.co/datasets/freeai-org/ScalpelBench) for source information, field definitions, limitations, and licensing details.

```python
from datasets import load_dataset

dataset = load_dataset("freeai-org/ScalpelBench")
```

**ScalpelBench loss curve**

After pruning seven layers, the model still retains useful general capabilities while running approximately 26.77% faster than the baseline.

<div align="center">
<img src="https://github.com/user-attachments/assets/46f9e025-0b85-4d04-9280-84945645cced" alt="ScalpelBench loss curve" width="500"/>
</div>

### Evaluation

MMLU is a four-choice benchmark, so random guessing gives an expected accuracy of 25%. Performance above this level indicates that Scalpel-VL-1.8B retains non-trivial general capabilities after pruning.

We evaluate Scalpel-VL-1.8B and InternLM2-1.8B under the same logit-based protocol and compare their category-level accuracy and inference speed.

| Model | Setting | STEM (%) ↑ | Humanities (%) ↑ | Social Sciences (%) ↑ | Other (%) ↑ | Avg. (%) ↑ | Speedup ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalpel-VL-1.8B | Zero-shot | 31.95 | 35.84 | 38.83 | 39.99 | 36.65 | **1.25×** |
| InternLM2-1.8B | Zero-shot | **37.18** | **41.85** | **51.90** | **49.91** | **45.21** | 1.00× |
| Scalpel-VL-1.8B | 5-shot | 34.86 | 35.81 | 42.08 | 40.79 | 38.39 | **1.16×** |
| InternLM2-1.8B | 5-shot | **39.50** | **41.25** | **50.21** | **50.06** | **45.26** | 1.00× |

These results suggest that recovery-aware pruning can provide a useful trade-off between model capability and inference efficiency. Better compact base models may make this approach even more effective.

## 🚕 Running Scalpel

The commands below assume that the current directory is the project root containing `highway/Scalpel`.

The domain-specific JSON/LLM workflow uses the `highway.*` package path internally, so create a temporary package alias first:

```bash
export SCALPEL_ALIAS="$(mktemp -d)"
ln -s "$PWD/highway/Scalpel" "$SCALPEL_ALIAS/highway"
export PYTHONPATH="$SCALPEL_ALIAS:$PWD"
```

### Entry Point 1: Full Domain-Specific Pruning

First, use `prune_prepare.py` to create a fixed probe set and record the dataset fingerprints:

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

Then use `prune_highway.py` to run baseline evaluation, layer probing, physical layer removal, recovery training, and pre/post-recovery evaluation:

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

`prune_highway.py` is the recommended entry point for the domain-specific workflow.

The same `run-dir` can be used to resume an interrupted run. By default, the large `pre_recovery_model` directory is removed after the corresponding post-recovery model has been verified. Add `--retain-pre-models` if you want to keep it.

### Entry Point 2: ScalpelBench Recovery

Download ScalpelBench to a local directory:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="freeai-org/ScalpelBench",
    repo_type="dataset",
    local_dir="/path/to/ScalpelBench",
)
```

Then launch the ten-round experiment from the project root:

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

Add `--prepare-only` to create the fixed probe and ten training partitions without starting model computation.

## 🔧 Using the `prune_*` Scripts

| File | Purpose | When to Use | Main Outputs |
|---|---|---|---|
| `prune_prepare.py` | Creates the fixed domain validation probe and records model/data SHA-256 fingerprints and experiment settings | Run once before starting the domain-specific JSON workflow | `probe/probe_10x10.{jsonl,csv}`, `dataset_fingerprints.json` |
| `prune_probe.py` | Temporarily bypasses each candidate layer and ranks it using generation-accuracy loss and normalized JS divergence | Use after preparing the fixed probe and reference baseline predictions | `selected_layer.json`, `layer_metrics.csv`, per-layer predictions |
| `prune_text.py` | Ranks layers on the ScalpelBench text probe using PPL, token accuracy, and macro capability scores | Use in the 0.1B recovery workflow or for standalone text-layer analysis | `selected_layer.json`, `layer_leaderboard.{csv,json}`, `progress.json` |
| `prune_layer.py` | Physically removes one current language-decoder layer and preserves the original-layer mapping | Use after the probe selects a target layer | Pruned model, `deletion.json`, `highway_state.json` |
| `prune_highway.py` | Orchestrates the full domain-specific workflow | Use for automated multi-round pruning and recovery | Per-round probe, training, evaluation, summaries, and models |

### `prune_prepare.py`: Prepare the Domain Probe

This script creates an immutable sampling manifest and audit metadata. It does not load the model or run inference.

Both `--train-data` and `--val-data` should point to supported SFT JSON files. The value of `--samples-per-repeat` must not exceed the number of validation samples.

```bash
python -m highway.prune_prepare \
  --model /path/to/reference_model \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --run-dir /path/to/run \
  --repeats 10 \
  --samples-per-repeat 10
```

### `prune_probe.py`: Rank Candidate Layers

`--reference-model` is the fixed teacher, while `--candidate-model` is the current student to be pruned in this round.

`--baseline-predictions` must point to the `predictions.jsonl` produced by the reference model on the full validation set. The full orchestrator prepares this file automatically.

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

Use `--candidate-layers 3,4,5` to evaluate only a subset of the current layers. These values refer to layer indices in the current model, not `original_layer`.

### `prune_text.py`: Rank Layers with ScalpelBench

This script uses teacher-forced evaluation rather than free-form generation.

`--data` may point to a fixed JSONL probe. If the same `output-dir` contains a compatible `progress.json`, an interrupted run resumes from the next unfinished sample.

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

### `prune_layer.py`: Physically Remove a Layer

Read `selected.current_layer` from `selected_layer.json` and pass it to `--layer`.

Do not pass `selected.original_layer`. The `--output-model` path must point to a new directory that does not already exist.

```bash
python -m highway.prune_layer \
  --model /path/to/current_model \
  --layer 12 \
  --output-model /path/to/round_01/pre_recovery_model \
  --record /path/to/round_01/deletion.json
```

### `prune_highway.py`: Run the Full Multi-Round Pipeline

In most cases, use the `python -m highway ...` command shown above.

The orchestrator calls `prune_probe.py`, `prune_layer.py`, and `train_weighted_kd.py` in order. It records pre-recovery and post-recovery metrics for every round.

The individual scripts are mainly useful when debugging a specific stage.

To run one recovery round manually:

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

## ✨ Output Structure

Main files produced in each round:

| Path | Description |
|---|---|
| `round_NN/probe/selected_layer.json` | Current-layer and original-layer indices selected for removal |
| `round_NN/deletion.json` | Physical layer-removal record |
| `round_NN/eval_pre_recovery/` | Evaluation before recovery |
| `round_NN/train/loss.jsonl` | Step-level training log containing `hard_weighted_ce` and `soft_weighted_kl` |
| `round_NN/train/adapter/` | LoRA adapter for the current round |
| `round_NN/train/summary.json` | Teacher/student paths, LoRA settings, loss configuration, and training metrics |
| `round_NN/eval_post_recovery/` | Evaluation after recovery |
| `model_root/round_NN/post_recovery_model/` | Complete student model after merging LoRA |

The protocol fields in the training summary should look similar to:

```json
{
  "recovery_method": "final_logits_lora_kd_v1",
  "teacher_target": "final_logits",
  "student_target": "final_logits",
  "train_scope": "student_lora_all_linear",
  "loss": "field_weighted_ce + field_weighted_KL(teacher_final||student_final)"
}
```

## 📚 File Reference

| File | Description |
|---|---|
| `prune_prepare.py` | Creates fixed probe splits, dataset fingerprints, and experiment settings |
| `prune_probe.py` | Temporarily bypasses candidate layers and selects a layer using task regret and JS divergence |
| `prune_text.py` | Selects layers on the fixed ScalpelBench text probe using PPL and capability retention |
| `prune_layer.py` | Physically removes one language-decoder layer |
| `train_weighted_kd.py` | Runs final-logit CE+KL recovery with all-linear LoRA |
| `prune_highway.py` | Multi-round pruning orchestrator |
| `mixture_experiment.py` | Resumable ten-part, ten-round ScalpelBench recovery orchestrator |
| `train_summarize.py` | Summarizes CE and KL components from `loss.jsonl` |
| `utils/metrics.py` | Probe metrics and field-weighted KD loss |
| `utils/kd_runtime.py` | Fixed-teacher loading and all-linear LoRA setup for the student |
| `utils/field_weights.py` | Maps JSON fields to token-level weights |
| `utils/model_ops.py` | Qwen3-VL layer access, input caching, temporary/physical layer removal, and model saving |
| `tests/` | Tests for metrics, layer removal, training logs, and probe behavior |

## 🖱️ Tests

```bash
python -m compileall -q highway/Scalpel

tmpdir="$(mktemp -d)"
ln -s ../Scalpel "$tmpdir/highway"
PYTHONPATH="$tmpdir:/home/alex/soulgard/soulgard-vl" \
python -m unittest discover -s "$tmpdir/highway/tests" -v
unlink "$tmpdir/highway"
rmdir "$tmpdir"
```

## 📖 Citation

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
