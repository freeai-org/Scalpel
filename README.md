# Scalpel: Recovery-Aware Layer Pruning for Faster Vision-Language Models

这套方案的目标是：在尽量少损伤业务能力的前提下，删除语言模型层数，让模型更小、更快。

## 文件命名约定

可执行脚本按职责使用统一前缀：

| 前缀 | 用途 | 示例 |
|---|---|---|
| `prune_` | 剪枝准备、候选层探测、物理删层和多轮编排 | `prune_highway.py` |
| `train_` | KD recovery 训练及训练日志汇总 | `train_weighted_kd.py` |
| `eval_` | 生成评估、logits 评估及评估结果汇总 | `eval_generation.py` |
| `test_` | 与上述模块对应的自动化测试 | `test_prune_probe.py` |

共享实现统一放在 `utils/`。本目录不再维护绘图脚本；`results/**/plots/`
中的图片仅作为已有实验产物保留。

## 运行前提

- 从仓库根目录运行，并让仓库根目录出现在 `PYTHONPATH` 中。
- Python 环境需要能导入 `torch`、`transformers`、`peft` 和项目依赖。
- `reference-model` 必须是可本地加载的 Qwen3-VL 完整模型。
- 训练集和验证集使用项目单图 SFT JSON 格式，每条样本包含 `image` 和
  `conversations`。
- `run-dir` 保存日志和指标，`model-root` 保存完整模型；两者应放在空间充足的磁盘。
- 主流程调用 `sft_scripts/eval_universal_json.py` 做真实生成评估，因此该脚本和
  数据中的图像路径必须在运行机器上可访问。

最小数据结构如下；assistant 的 `content` 是合法业务 JSON 字符串。绝对图像路径直接
使用，相对路径按仓库根目录下的 `datasets/`（或显式的 `datasets/`、`sft_data/`
前缀）解析：

```json
[
  {
    "image": "/path/to/image.jpg",
    "conversations": [
      {"role": "user", "content": "请分析图中的猫。"},
      {"role": "assistant", "content": "{\"cats_visible\": 0, \"cats\": []}"}
    ]
  }
]
```

## 完整方法流程

### 0. 固定实验输入

先运行 `prune_prepare.py`：

1. 读取训练集和验证集。
2. 按固定随机种子从验证集独立抽取 `repeats × samples_per_repeat` 个 probe 行；
   正式配置为 `10 × 10 = 100` 行，同一原始样本可以出现在不同 repeat 中。
3. 写入 `probe/probe_10x10.jsonl` 和 CSV 索引。
4. 记录模型配置、训练集、验证集的 SHA-256 指纹。
5. 记录 Python、PyTorch、Transformers、PEFT、CUDA 和 GPU 环境。

固定 manifest 在所有剪枝轮次复用，保证不同候选层和不同轮次看到相同 probe。

### 1. 建立 baseline 和执行预检

`prune_highway.py` 首先对原始 28 层 `reference-model` 跑完整验证集生成评估，
得到 baseline 的 `metrics.json`、`predictions.jsonl` 和字段级
`atomic_metrics.json`。

默认随后执行一次 preflight：

1. 只探测两个候选层，确认生成与 logits 风险计算可运行。
2. 临时物理删除 layer 12，确认模型能够保存和重新加载。
3. 对删层模型评估 1 条样本。
4. 跑 2 个 optimizer step 的 KD recovery。这里必须是 2 步，因为只有 1 步时
   warmup scheduler 可能使唯一一步的学习率为 0。

任一步失败都会在正式长实验前停止。

### 2. 每轮选择要删除的层

第 (k) 轮的当前 student 是上一轮的 `post_recovery_model`；第 1 轮则直接使用
原始 reference。默认保护当前模型最前面的 3 层，候选范围为
`current_layer = 3 ... N-1`。

对每个候选层依次执行：

1. 仅在上下文中临时绕过该层，不立即改写模型文件。
2. 在固定 probe 上做确定性 JSON 生成，计算业务字段宏平均准确率。
3. 在真实 assistant 监督位置比较固定 reference 与临时删层 candidate 的词表
   分布，计算字段加权 normalized Jensen-Shannon divergence。
4. 在每个 repeat 内计算剪枝风险，然后对 repeats 求均值。
5. 按平均风险、平均 hard regret、平均 JS、原始层号依次升序排序。

风险最低的层写入 `probe/selected_layer.json`。

### 3. 物理删层、恢复训练和评估

选中层后，每轮执行以下闭环：

1. `prune_layer.py` 从 `model.language_model.layers` 中物理删除当前层，修正层数配置
   和 attention 的 `layer_idx`，并导出 `pre_recovery_model`。
2. 对 `pre_recovery_model` 跑完整验证集生成评估，记录相对 baseline 的退化。
3. `train_weighted_kd.py` 使用原始 28 层 reference 作为固定 teacher，对删层 student
   做字段加权 CE + KL recovery。
4. 将 LoRA 合并进 student，导出 `post_recovery_model`。
5. 对 `post_recovery_model` 再跑完整验证集评估，记录恢复量和速度。
6. 默认删除已经不再需要的 `pre_recovery_model`，然后把 post 模型交给下一轮。

### 4. 完成与断点续跑

所有轮次结束后写出 `final/all_rounds.csv`、`final/all_rounds.json` 和
`final/experiment_complete.json`。

主流程通过结果文件判断步骤是否已经完成：已有完整评估或完整模型会直接复用；
训练目录存在 `checkpoint-*` 时会从最新 checkpoint 恢复。中断期间只写出一部分的
模型目录不会被当成成功结果，而会重命名为 `.incomplete_XX` 后重新导出。

## 剪枝层选择指标

生成准确率使用 10 个原子字段的宏平均：

```text
cat_presence, cat_count, location_on, vertical_position, action,
overall_body, ears, tail, face, fur_state
```

JSON 解析失败或漏掉真实存在的猫会在对应字段上计错。对 repeat (r) 和候选层
(l)，相对 hard regret 为：

$$
R_{r,l}
=
\operatorname{clip}_{[0,1]}
\left(
\frac{A^{ref}_r-A^{cand}_{r,l}}
{\max(A^{ref}_r,10^{-12})}
\right)
$$

改善不会产生负 regret，而是记为 0。词表分布风险使用按字段权重聚合、再除以
\(\log 2\) 归一化到 \([0,1]\) 的 Jensen-Shannon divergence，记为
\(J_{r,l}\)。单个 repeat 的风险采用 Chebyshev/minimax 标量化：

$$
Q_{r,l}=\max(R_{r,l},J_{r,l})
$$

候选层最终分数是 \(\bar Q_l=\frac{1}{K}\sum_r Q_{r,l}\)。这种设计要求生成
准确率和分布一致性两条轴都不能明显变坏，避免一个很好的平均项掩盖另一个坏项。

这里的 \(A^{ref}\) 和 JS reference 始终来自原始 28 层 baseline，而不是本轮删层前的
student；candidate 则是“当前累计剪枝模型再临时绕过候选层”。因此排序衡量的是候选
方案相对最初模型的累计业务/分布风险。若分数相同，依次用 hard regret、JS、原始层号
做稳定 tie-break。

## Loss 细节与符号

Recovery 训练使用固定 teacher 和可训练 student：

| 符号 | 含义 |
|---|---|
| \(x\) | 输入图像和用户 prompt |
| \(y\) | assistant 真实回答 token 序列 |
| \(B\) | batch size |
| \(T\) | causal shift 后参与 loss 的 token 位置数 |
| \(V\) | 词表大小 |
| \(z^s_{b,t}\in\mathbb{R}^V\) | student 在第 \(b\) 条样本、第 \(t\) 个预测位置的 logits |
| \(z^r_{b,t}\in\mathbb{R}^V\) | reference/teacher 在同一位置的 logits |
| \(p^s_{b,t}=\mathrm{softmax}(z^s_{b,t})\) | student token 分布 |
| \(p^r_{b,t}=\mathrm{softmax}(z^r_{b,t})\) | teacher token 分布 |
| \(y_{b,t}\) | 该位置真实 next token id |
| \(m_{b,t}=\mathbf{1}[y_{b,t}\ne -100]\) | supervised mask，只保留 assistant token |
| \(w_{b,t}\) | 字段权重，例如 action、body、ears、tail、face、fur_state 更高 |
| \(a_{b,t}=m_{b,t}w_{b,t}\) | 最终有效权重 |
| \(D_b=\max(\sum_t a_{b,t},1)\) | 第 \(b\) 条样本的归一化分母 |

对应代码：

```text
highway/train_weighted_kd.py
highway/utils/metrics.py::field_weighted_kd_loss
```

训练时先做 causal shift：

```python
shift_labels = labels[..., 1:]
shift_weights = loss_weights[..., 1:]
```

也就是模型第 \(t\) 个预测位置对应真实的 next token \(y_{b,t}\)。`labels == -100` 的位置不参与 loss。

字段加权 hard CE：

$$
L_{\mathrm{ce}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{1}{D_b}
\sum_{t=1}^{T}
a_{b,t}\left[-\log p^s_{b,t,y_{b,t}}\right]
$$

字段加权 soft KL：

$$
L_{\mathrm{kl}}
=
\frac{1}{B}\sum_{b=1}^{B}
\frac{1}{D_b}
\sum_{t=1}^{T}
a_{b,t}
\sum_{v=1}^{V}
p^r_{b,t,v}
\left(\log p^r_{b,t,v}-\log p^s_{b,t,v}\right)
$$

也就是：

$$
L_{\mathrm{kl}}=\mathrm{weighted}\ KL(p_{\mathrm{teacher}}\Vert p_{\mathrm{student}})
$$

最终 recovery loss：

$$
L_{\mathrm{total}} = L_{\mathrm{ce}} + L_{\mathrm{kl}}
$$

两项系数都固定为 1，没有额外的 `lambda_ce` 或 `lambda_kl`。实现先按每条样本的
有效字段权重和独立归一化，再对 batch 求平均，因此不同回答长度不会仅因 token 更多
而获得更大的样本权重。

这版正式实验里 recovery 的 KL 温度是 \(T_{\mathrm{temp}}=1\)，没有额外乘
\(T_{\mathrm{temp}}^2\)。`eval_logits.py` 的默认温度 2 只用于离线诊断，和 recovery
训练 Loss 不是同一个参数。

`teacher_model` 在 `torch.no_grad()` 下前向并且所有参数均被冻结，只提供目标分布。
Student 的原始参数同样不直接更新；训练脚本通过 `target_modules="all-linear"` 在
删层后的多模态模型上挂 LoRA，覆盖语言模型、视觉编码器和视觉 merger 中可支持的
线性层，但不训练 embedding、norm、原始 bias 和输出 `lm_head`。训练结束后才调用
`merge_and_unload()` 把 LoRA 增量合并到完整 student 模型。

字段权重来自 `highway/utils/field_weights.py`：

| 字段/内容 | 权重 |
|---|---:|
| JSON 格式字符 `{ } [ ] : , "` | `0.5` |
| 默认内容 | `1.0` |
| 环境相关字段 | `1.0` |
| `cats_visible` | `2.0` |
| `action`、`overall_body`、`ears`、`tail`、`face`、`fur_state`、`eyelid`、`mouth` | `3.0` |

字符级字段权重会通过 tokenizer offset 投影到 token 上；如果一个 token 覆盖多个字符，取该 token span 内的最大权重。

同一套 token 权重同时用于 hard CE 和 teacher KL。当前正式实验没有使用
hidden-state loss，也没有对生成结果反向传播；训练仍是 teacher-forcing。

## 主入口

正式长实验入口是：

```bash
python -m highway
```

等价于：

```bash
python -m highway.prune_highway
```

`highway/__main__.py` 只做一件事：把 `python -m highway` 转到 `prune_highway.py`。

下面先定义一次路径变量。`PYTHON_BIN` 必须指向安装了训练依赖的解释器：

```bash
export PROJECT_ROOT=/home/alex/soulgard/soulgard-vl
export PYTHON_BIN=/path/to/training-env/bin/python
export REFERENCE_MODEL=/path/to/reference-model
export TRAIN_JSON="$PROJECT_ROOT/sft_data/sft_cat_train.json"
export VAL_JSON="$PROJECT_ROOT/sft_data/sft_cat_val.json"
export RUN_DIR=/path/to/highway-run
export MODEL_ROOT=/path/to/highway-models
```

先生成固定 probe manifest 和实验元数据：

```bash
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m highway.prune_prepare \
  --model "$REFERENCE_MODEL" \
  --train-data "$TRAIN_JSON" \
  --val-data "$VAL_JSON" \
  --run-dir "$RUN_DIR" \
  --repeats 10 \
  --samples-per-repeat 10 \
  --seed 20260726 \
  --rounds 9 \
  --recovery-epochs 2
```

然后启动多轮剪枝 + recovery：

```bash
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway \
  --project-root "$PROJECT_ROOT" \
  --python "$PYTHON_BIN" \
  --eval-script "$PROJECT_ROOT/sft_scripts/eval_universal_json.py" \
  --reference-model "$REFERENCE_MODEL" \
  --train-data "$TRAIN_JSON" \
  --val-data "$VAL_JSON" \
  --run-dir "$RUN_DIR" \
  --model-root "$MODEL_ROOT" \
  --rounds 9 \
  --eval-batch-size 4 \
  --generation-batch-size 4 \
  --max-new-tokens 2048 \
  --recovery-batch-size 1 \
  --recovery-effective-batch-size 16 \
  --recovery-epochs 2 \
  --attention sdpa
```

## 主入口参数

`prune_prepare.py` 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--model` | 必填 | 原始 reference 模型目录，用于记录 config 指纹 |
| `--train-data` | 必填 | recovery 训练集 JSON |
| `--val-data` | 必填 | 生成评测/探测验证集 JSON |
| `--run-dir` | 必填 | 写入 probe manifest 和实验元数据的日志根目录 |
| `--repeats` | `10` | probe 重复采样轮数 |
| `--samples-per-repeat` | `10` | 每次 probe 采样多少验证样本 |
| `--seed` | `20260726` | 固定 probe 样本的随机种子 |
| `--rounds` | `9` | 写入元数据的计划剪枝轮数，应与主入口一致 |
| `--recovery-epochs` | `2.0` | 写入元数据的计划 recovery epoch，应与主入口一致 |

`prune_highway.py` 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--project-root` | 必填 | 仓库根目录；子进程会在这里运行，并把它放进 `PYTHONPATH` |
| `--python` | 必填 | 子进程使用的 Python 解释器 |
| `--eval-script` | 必填 | 业务 JSON 生成评测脚本，当前是 `sft_scripts/eval_universal_json.py` |
| `--reference-model` | 必填 | 固定 teacher/reference 模型；probe 和 KD 都以它作为高层目标 |
| `--train-data` | 必填 | recovery 训练数据 |
| `--val-data` | 必填 | baseline、pre、post 生成评测数据 |
| `--run-dir` | 必填 | 日志、probe、评测结果和 final 表格目录 |
| `--model-root` | 必填 | 每轮 `pre_recovery_model` / `post_recovery_model` 保存目录 |
| `--rounds` | `9` | 执行多少轮删除；每轮删除 1 个 decoder layer |
| `--eval-batch-size` | `4` | 完整生成评测 batch size |
| `--generation-batch-size` | `4` | probe 阶段临时删层生成 batch size |
| `--max-new-tokens` | `2048` | 生成评测最大输出 token 数 |
| `--recovery-batch-size` | `1` | recovery 训练的物理 batch size |
| `--recovery-effective-batch-size` | `16` | recovery batch size（期望的总 batchsize，可以用来求梯度累积步数）；代码会设 `grad_accum = effective / batch` |
| `--recovery-epochs` | `2.0` | 每轮 recovery 训练 epoch 数 |
| `--retain-pre-models` / `--no-retain-pre-models` | `False` | 是否保留每轮 recovery 前的 `pre_recovery_model` |
| `--attention` | `sdpa` | Transformers 加载模型时的 attention 实现 |
| `--post-baseline-preflight` / `--no-post-baseline-preflight` | `True` | baseline 后是否跑 smoke test，确认 probe、删层、评测、KD 都可用 |

`--recovery-effective-batch-size` 必须是 `--recovery-batch-size` 的整数倍。

## 输出结构

`run-dir` 保存日志和评测：

```text
run-dir/
  orchestrator_config.json
  dataset_fingerprints.json
  probe/
    probe_10x10.jsonl
    probe_10x10.csv
  baseline/
    metrics.json
    predictions.jsonl
    atomic_metrics.json
  preflight/
    post_baseline_complete.json
  round_01/
    probe/
      selected_layer.json
      layer_metrics.csv
      repeat_metrics.csv
      sample_soft_metrics.jsonl
    deletion.json
    eval_pre_recovery/
      metrics.json
      predictions.jsonl
      atomic_metrics.json
    train/
      summary.json
      resolved_config.json
      loss.jsonl
      loss_summary.json
      adapter/
      checkpoints/
    eval_post_recovery/
      metrics.json
      predictions.jsonl
      atomic_metrics.json
    round_summary.json
  final/
    all_rounds.csv
    all_rounds.json
    experiment_complete.json
```

`model-root` 保存模型：

```text
model-root/
  preflight/
  round_01/
    pre_recovery_model/
    post_recovery_model/
  round_02/
    pre_recovery_model/
    post_recovery_model/
```

`preflight/` 是正式长实验前的 smoke test：

| 子步骤 | 目的 |
|---|---|
| probe 两个层 | 确认 `prune_probe.py` 能生成和打分 |
| 删除 layer 12 | 确认 `prune_layer.py` 能导出可加载模型 |
| 评测 1 条样本 | 确认业务评测脚本能跑通 |
| KD 两步训练 | 确认 `train_weighted_kd.py` 的前向、反向、日志写入正常 |

## 层号约定

代码里有两种层号：

| 名称 | 含义 |
|---|---|
| `current_layer` | 当前模型里的层下标；删除过层之后会重新变短 |
| `original_layer` | 原始 28 层模型里的层号；写入 `highway_state.json`，用于审计删除历史 |

例如老命名里的 `round_03_layers_25` 表示：第 3 轮结束后模型还剩 25 个 decoder layer。

每个导出的模型目录中都有：

```text
highway_state.json
```

它记录：

| 字段 | 含义 |
|---|---|
| `original_layer_ids` | 当前模型每一层对应的原始层号 |
| `deleted_original_layers` | 已删除的原始层号列表 |
| `rounds` | 每轮删除记录 |

## 文件说明

下面覆盖 `highway/` 中的全部源码和配置文件。生成文件已经在“输出结构”中说明，
`results/` 下则是历史实验结果，不是运行依赖。

顶层文件：

| 文件 | 职责 | 主要输入 → 输出 | 是否直接运行 |
|---|---|---|---|
| `.gitignore` | 忽略本地缓存、临时模型和大体积运行产物 | Git 工作树规则 | 否 |
| `README.md` | 方法、Loss、文件、命令和历史结果说明 | 文档 | 否 |
| `__init__.py` | 声明 `highway` Python 包 | 包初始化 | 否 |
| `__main__.py` | 把 `python -m highway` 转发到主编排器 | 命令行参数 → `prune_highway.main()` | 是，推荐主入口 |
| `prune_prepare.py` | 固定 probe、保存数据/模型指纹和运行环境 | 模型 config、train/val JSON → `run-dir/probe/` 和元数据 | 是，正式实验第一步 |
| `prune_highway.py` | 编排 baseline、preflight、多轮 probe、删层、recovery、评测和续跑 | 全部实验路径/参数 → 每轮日志、模型、最终汇总 | 是，正式实验第二步 |
| `prune_probe.py` | 临时绕过候选层，计算 hard regret 与 normalized JS，选择风险最低层 | reference、当前模型、固定 probe、baseline 预测 → 层级指标和 `selected_layer.json` | 可单独调试 |
| `prune_layer.py` | 物理删除一个当前 decoder layer，修正 config 和层号映射 | 当前模型、当前层号 → 删层完整模型和删除记录 | 可单独调试 |
| `train_weighted_kd.py` | 以固定 reference 为 teacher，对删层 student 做字段加权 CE+KL LoRA recovery | teacher、student、train JSON → checkpoint、adapter、合并后的完整模型 | 可单独训练 |
| `train_summarize.py` | 汇总追加式 loss 日志，处理断点续训造成的重复 step | `loss.jsonl` → `loss_summary.json` | 可单独运行 |
| `eval_generation.py` | 对 baseline 和各轮 post 模型统一做完整验证集/固定 probe 生成复评 | 模型根目录、运行目录、val JSON → 各模型生成结果和跨轮 CSV | 可选离线复评 |
| `eval_logits.py` | 比较已保存模型的 teacher-forcing CE、KL、top-1 一致率 | baseline、各轮 post 模型、val JSON → 明细 JSONL 和汇总 CSV | 可选离线诊断 |
| `eval_summarize.py` | 从已有生成预测重新计算 10 个原子字段指标 | `eval-dir/predictions.jsonl` → `atomic_metrics.json`、`per_field.csv` | 可单独运行 |

`utils/` 支撑模块不应作为命令直接运行：

| 文件 | 被谁使用 | 职责 |
|---|---|---|
| `utils/__init__.py` | 全部模块 | 声明工具子包 |
| `utils/data.py` | prepare、probe、train、logits eval | 读取 SFT JSON、抽样、构造数据集、定位 causal-shift 后的监督 token |
| `utils/field_weights.py` | train、probe | 解析回答中的业务字段 span，并投影成 token 级权重 |
| `utils/io_utils.py` | 全部阶段 | JSON/JSONL/CSV I/O、路径规范化、随机种子、SHA-256、层状态读写 |
| `utils/metrics.py` | probe、train、logits eval | normalized JS、KL、logits 对比、字段加权 CE+KL loss |
| `utils/model_ops.py` | probe、prune、train、eval | 加载模型/processor、临时绕过层、物理删层、修正 config、保存模型 |
| `utils/task_metrics.py` | probe、eval summarize | 规范化业务 JSON、字段准确率、hard regret 和 pruning risk |
| `utils/training_collator.py` | train | 拼接多模态样本，padding `input_ids`、`labels`、图像输入和 loss 权重 |

`tests/` 文件：

| 文件 | 覆盖内容 |
|---|---|
| `tests/__init__.py` | 声明测试包 |
| `tests/test_core.py` | 抽样、层操作、状态隔离、CE/KL/JS、风险函数、collator、模型完成判定 |
| `tests/test_eval_generation.py` | probe manifest、checkpoint 解析、生成评测计划和跨轮汇总 |
| `tests/test_prune_probe.py` | probe 多模态输入、批生成 padding、失败后的单样本重试 |
| `tests/test_train_summarize.py` | loss 汇总、重复 step、非有限数和 Trainer 末尾统计记录 |

仓库根目录辅助脚本：

| 文件 | 作用 |
|---|---|
| `scripts/eval_round9_mmlu_ceval.py` | MMLU / C-Eval zero-shot logits 投影评测 |
| `scripts/direct_delete_original_layers.py` | 从某个模型直接删除多个原始层，不做 recovery，用于对照实验 |
| `scripts/upload_final_highway_model_to_hf.sh` | 使用 Hugging Face CLI 上传最终模型 |
| `scripts/upload_final_highway_model_to_hf_api.py` | 使用 Hugging Face Hub Python API 上传最终模型 |

## 单步调试入口

通常只需要运行上面的 `prune_prepare` 和 `python -m highway`。下面命令用于重跑或
定位某一个阶段；运行前先定义“主入口”一节中的路径变量。查看任意模块的完整参数：

```bash
PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m highway.prune_probe --help
```

### 1. 单独探测候选层

下面只探测当前模型的第 3、12、27 层；`candidate-model` 可替换为上一轮的
`post_recovery_model`：

```bash
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway.prune_probe \
  --reference-model "$REFERENCE_MODEL" \
  --candidate-model "$REFERENCE_MODEL" \
  --val-data "$VAL_JSON" \
  --probe-manifest "$RUN_DIR/probe/probe_10x10.jsonl" \
  --baseline-predictions "$RUN_DIR/baseline/predictions.jsonl" \
  --output-dir "$RUN_DIR/manual_probe" \
  --candidate-layers 3,12,27 \
  --generation-batch-size 4 \
  --max-new-tokens 2048 \
  --attention sdpa
```

主要输出是 `manual_probe/layer_metrics.csv`、`repeat_metrics.csv`、
`sample_soft_metrics.jsonl` 和 `selected_layer.json`。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--reference-model` | 必填 | 固定 reference；提供 logits 和对照目标 |
| `--candidate-model` | 必填 | 当前待删层模型；第 1 轮可与 reference 相同 |
| `--val-data` | 必填 | probe 行所引用的验证集 JSON |
| `--probe-manifest` | 必填 | `prune_prepare.py` 生成的固定 probe |
| `--baseline-predictions` | 必填 | 原始 reference 的验证集预测，用于取得 baseline 业务准确率 |
| `--output-dir` | 必填 | 候选层指标和选择结果目录 |
| `--device` | `cuda` | 推理设备 |
| `--dtype` | `bfloat16` | 模型 dtype，可选 `bfloat16`/`float16` |
| `--attention` | `sdpa` | attention 实现 |
| `--min-layer` | `3` | 自动枚举时保护前多少层 |
| `--candidate-layers` | 空 | 手动候选当前层号，逗号分隔；空表示从 `min-layer` 自动枚举 |
| `--max-probe-rows` | `0` | 调试时限制 probe 行数；`0` 为完整 manifest |
| `--generation-batch-size` | `4` | 临时删层生成 batch size |
| `--max-new-tokens` | `2048` | 每条 probe 的最大生成长度 |
| `--js-chunk-size` | `16` | 逐块计算 JS 的监督 token 数，越小越省显存 |

### 2. 单独物理删除一层

```bash
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway.prune_layer \
  --model "$REFERENCE_MODEL" \
  --layer 12 \
  --output-model "$MODEL_ROOT/manual/pre_recovery_model" \
  --record "$RUN_DIR/manual/deletion.json" \
  --attention sdpa
```

`--layer` 是当前模型下标，不是原始层号。为防止误覆盖，`--output-model` 已存在时会
报错；需要换一个新目录或先人工确认旧目录是否可删除。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--model` | 必填 | 要删层的输入完整模型 |
| `--layer` | 必填 | 要删除的当前层下标 |
| `--output-model` | 必填 | 删除后的完整模型新目录 |
| `--record` | 必填 | 删除层号、映射和模型路径记录 JSON |
| `--device` | `cuda` | 操作模型的设备 |
| `--dtype` | `bfloat16` | 可选 `bfloat16`/`float16` |
| `--attention` | `sdpa` | attention 实现 |

### 3. 单独做 Recovery 训练

```bash
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway.train_weighted_kd \
  --teacher-model "$REFERENCE_MODEL" \
  --student-model "$MODEL_ROOT/manual/pre_recovery_model" \
  --train-data "$TRAIN_JSON" \
  --output-dir "$RUN_DIR/manual/train" \
  --adapter-dir "$RUN_DIR/manual/train/adapter" \
  --export-dir "$MODEL_ROOT/manual/post_recovery_model" \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 1e-4 \
  --attention sdpa
```

训练日志在 `manual/train/loss.jsonl`，Trainer checkpoint 在
`manual/train/checkpoints/checkpoint-*`，LoRA 在 `adapter/`，合并后的完整模型在
`post_recovery_model/`。续训时加
`--resume-from-checkpoint "$RUN_DIR/manual/train/checkpoints/checkpoint-N"`。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--teacher-model` | 必填 | 原始 reference 模型 |
| `--student-model` | 必填 | 删除层后的 `pre_recovery_model` |
| `--train-data` | 必填 | recovery 训练数据 |
| `--output-dir` | 必填 | 训练日志和 checkpoint 目录 |
| `--adapter-dir` | 必填 | LoRA adapter 保存目录 |
| `--export-dir` | 必填 | 合并 LoRA 后的完整模型导出目录 |
| `--epochs` | `2.0` | recovery 训练 epoch 数 |
| `--max-length` | `2560` | 文本最大长度 |
| `--image-max-pixels` | `262144` | 图像最大像素数 |
| `--batch-size` | `1` | 物理 batch size |
| `--grad-accum` | `16` | 梯度累积步数 |
| `--learning-rate` | `1e-4` | LoRA 学习率 |
| `--lora-rank` | `8` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--logging-steps` | `5` | 每多少 optimizer step 追加一次 loss 日志 |
| `--save-steps` | `100` | 每多少 optimizer step 保存 Trainer checkpoint |
| `--max-samples` | `0` | 调试时限制训练样本数，`0` 表示不限制 |
| `--max-steps` | `-1` | 调试时限制训练步数，`-1` 表示不限制 |
| `--attention` | `sdpa` | teacher/student attention 实现 |
| `--device` | `cuda` | 训练设备 |
| `--seed` | `42` | Trainer 和数据顺序随机种子 |
| `--resume-from-checkpoint` | 空 | 从 Trainer checkpoint 续训 |
| `--skip-export` | `False` | 只训练不导出完整模型，适合 smoke test |

### 4. 汇总训练 Loss 或已有生成预测

```bash
PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m highway.train_summarize \
  --loss-log "$RUN_DIR/manual/train/loss.jsonl" \
  --output "$RUN_DIR/manual/train/loss_summary_manual.json"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m highway.eval_summarize \
  --eval-dir "$RUN_DIR/baseline"
```

第二条命令要求目标目录已经有 `predictions.jsonl`，并在同一目录写
`atomic_metrics.json` 和 `per_field.csv`。`train_summarize --output` 可省略，默认写在
loss 日志旁的 `loss_summary.json`。

### 5. 对所有 Post-Recovery 模型统一做生成复评

主编排器已经在每轮执行 pre/post 生成评估。下面是可选的离线统一复评，可同时跑
完整验证集和固定 probe，并计算相对 baseline/上一轮的变化：

```bash
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway.eval_generation \
  --project-root "$PROJECT_ROOT" \
  --python "$PYTHON_BIN" \
  --eval-script "$PROJECT_ROOT/sft_scripts/eval_universal_json.py" \
  --base-model "$REFERENCE_MODEL" \
  --checkpoint-root "$MODEL_ROOT" \
  --source-run-dir "$RUN_DIR" \
  --val-json "$VAL_JSON" \
  --output-dir "$RUN_DIR/offline_generation_eval" \
  --accepted-rounds 9 \
  --batch-size 4 \
  --max-new-tokens 2048
```

脚本从 `$MODEL_ROOT/round_XX/post_recovery_model` 找模型，从
`$RUN_DIR/probe/probe_10x10.jsonl` 找固定 probe，最终写
`generation_eval_summary.csv`。先加 `--dry-run` 可只生成计划；加 `--force` 可覆盖已有
评估，`--skip-full` 或 `--skip-probe` 可跳过对应数据集。

### 6. 对所有 Post-Recovery 模型做 Logits 诊断

```bash
PYTHONPATH="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -m highway.eval_logits \
  --base-model "$REFERENCE_MODEL" \
  --checkpoint-root "$MODEL_ROOT" \
  --source-run-dir "$RUN_DIR" \
  --val-json "$VAL_JSON" \
  --output-dir "$RUN_DIR/offline_logits_eval" \
  --accepted-rounds 9 \
  --max-length 2048 \
  --max-tokens 64 \
  --temperature 2 \
  --attention sdpa
```

输出为 `logit_eval_plan.json`、各比较的明细 JSONL/summary JSON 和
`logit_eval_summary.csv`。默认完整集比较 baseline→每轮，并在固定 probe 上比较
上一轮→本轮；`--include-full-stage` 会额外在完整验证集上比较上一轮→本轮。
`--max-tokens` 只限制每条样本参与诊断的监督 token 数，不限制生成长度。

### 7. 运行自动化测试

```bash
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m unittest discover \
  -s highway/tests -v
```

这些是轻量单元测试，不会下载模型或启动完整 GPU 实验；真实模型仍建议先保留默认
preflight，再开始九轮长实验。

## 正式实验路径

实验运行在 `soul`：

```text
run id:
highway-5round-10x10-weighted-kd-20260726

原始 28 层模型:
/home/yuhang/models/soulgard-vl-bestloss-step6215-batch4-rank64

训练集:
/home/yuhang/soulgard-vl/sft_data/sft_cat_train.json

验证集:
/home/yuhang/soulgard-vl/sft_data/sft_cat_val.json

训练/评测日志:
/mnt/disks/alg-cat-vlm/highway-runs/highway-5round-10x10-weighted-kd-20260726

模型目录:
/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726
```

`highway-5round-...` 是实验最初创建时沿用的历史 run id，后来在同一目录扩展到了
9 轮；实际轮数应以 `round_01` 至 `round_09`、最终汇总和下面的删除记录为准，不要
从目录名推断。

最终模型参数量：

```text
1,674,508,032 params
约 1.6745B
BF16
```

## 删除层记录

九轮结果如下：

| Round | 删除的原始层 | 剩余层数 | post 模型路径 |
|---|---:|---:|---|
| round_01 | 5 | 27 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_01/post_recovery_model` |
| round_02 | 8 | 26 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_02/post_recovery_model` |
| round_03 | 9 | 25 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_03/post_recovery_model` |
| round_04 | 23 | 24 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_04/post_recovery_model` |
| round_05 | 4 | 23 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_05/post_recovery_model` |
| round_06 | 18 | 22 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_06/post_recovery_model` |
| round_07 | 15 | 21 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_07/post_recovery_model` |
| round_08 | 13 | 20 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_08/post_recovery_model` |
| round_09 | 25 | 19 | `/mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_09/post_recovery_model` |

最终 round09 剩余原始层：

```text
[0, 1, 2, 3, 6, 7, 10, 11, 12, 14, 16, 17, 19, 20, 21, 22, 24, 26, 27]
```

已删除原始层：

```text
[5, 8, 9, 23, 4, 18, 15, 13, 25]
```

## MMLU / C-Eval 评测

评测口径：

```text
MMLU:  cais/mmlu, test split, 14042 题
C-Eval: ceval/ceval-exam, test split, 12342 题
设置: zero-shot
prompt: 官方 completion 风格
打分: 最后位置 A/B/C/D logits 投影
chat template: 不使用
```

结果：

| Round | MMLU | C-Eval | Avg |
|---|---:|---:|---:|
| round_01 | 50.48% | 50.67% | 50.58% |
| round_02 | 43.23% | 44.78% | 44.01% |
| round_03 | 36.98% | 35.29% | 36.14% |
| round_04 | 36.43% | 35.04% | 35.74% |
| round_05 | 29.40% | 28.55% | 28.98% |
| round_06 | 29.90% | 26.75% | 28.33% |
| round_07 | 24.17% | 23.84% | 24.01% |
| round_08 | 23.69% | 23.94% | 23.82% |
| round_09 | 23.29% | 23.81% | 23.55% |

MMLU / C-Eval 脚本示例：

```bash
export HF_HOME=/mnt/disks/alg-cat-vlm/data/hf_home
export HF_DATASETS_CACHE=/mnt/disks/alg-cat-vlm/data/hf_datasets_cache
export TRANSFORMERS_CACHE=/mnt/disks/alg-cat-vlm/data/hf_home/transformers

CUDA_VISIBLE_DEVICES=0 /home/yuhang/miniconda3/envs/soulgard/bin/python -u \
  scripts/eval_round9_mmlu_ceval.py \
  --model /path/to/post_recovery_model \
  --datasets mmlu ceval \
  --nshot 0 \
  --batch-size 128 \
  --score-mode next_token \
  --output-dir /mnt/disks/alg-cat-vlm/data/eval_results/your_run_name
```

`scripts/eval_round9_mmlu_ceval.py` 关键参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--model` | round09 路径 | 待评测模型 |
| `--data-root` | `/mnt/disks/alg-cat-vlm/data` | 数据和缓存根目录 |
| `--output-dir` | 自动生成 | 评测输出目录 |
| `--datasets` | `mmlu ceval` | 选择评测集 |
| `--nshot` | `0` | few-shot 数；正式结果用 zero-shot |
| `--batch-size` | `128` | logits 打分 batch size |
| `--score-mode` | `next_token` | 使用最后位置 A/B/C/D logits 投影 |
| `--use-chat-template` | `False` | 是否套 chat template；正式结果不使用 |
| `--resume` | `False` | 断点续评 |

## Direct Delete 对照实验

对照目标：从 round01 出发，直接删除 round02 到 round09 那些原始层，不做 recovery，观察纯删除的伤害。

```bash
PYTHONPATH=/home/yuhang/soulgard-vl CUDA_VISIBLE_DEVICES=0 \
/home/yuhang/miniconda3/envs/soulgard/bin/python -u \
  scripts/direct_delete_original_layers.py \
  --model /mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/round_01/post_recovery_model \
  --delete-original-layers 8,9,23,4,18,15,13,25 \
  --output-model /mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/direct_delete_round01_drop02to09_no_recovery/model \
  --record /mnt/disks/alg-cat-vlm/models/highway/highway-5round-10x10-weighted-kd-20260726/direct_delete_round01_drop02to09_no_recovery/direct_delete_record.json
```

结果：

| 实验 | MMLU | C-Eval | Avg |
|---|---:|---:|---:|
| round01 recovered | 50.48% | 50.67% | 50.58% |
| round02 recovered | 43.23% | 44.78% | 44.01% |
| direct drop 02-09, no recovery | 22.89% | 23.23% | 23.06% |
| round09 recovered | 23.29% | 23.81% | 23.55% |

结论：从 round01 一次性删到 round09 的层数、不做 recovery，会直接掉到接近随机；round09 recovery 比 no-recovery 略好，但没有救回纯文本选择题能力。
