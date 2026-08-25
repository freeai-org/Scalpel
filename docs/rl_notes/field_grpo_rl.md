# SoulGard-VL Field-GRPO / Compact-RL 实施 README

这份文档说明如何在 SoulGard-VL 现有 SFT 主线之后，加一段轻量 RL。目标不是泛泛地做 PPO，也不是优化整个 JSON，而是围绕项目真正关心的字段做后训练：

- 单帧：提高姿势行为 6 项指标均值。
- 多帧：同样思路，但额外保护帧数、猫数量和逐帧数组长度。
- compact/stage3：保证紧凑序列化输出严格可还原，同时尽量不掉姿势行为字段。

当前建议优先做 **Field-GRPO**，暂时不要先上标准 actor-critic PPO。原因是这个任务有现成可自动计算的离散指标，不需要先训练 Reward Model，也不值得为一个结构化小任务额外维护 critic。

## 0. 当前模型和阶段确认

单帧 RL 起点模型：

```text
/mnt/d/models/soulgard-vl-bestloss-step6215
```

这个模型是当前单帧主线的 stage1+stage2 后完整合并模型：

```text
stage1: natural warmup
stage2: weighted JSON SFT
export: /mnt/d/models/soulgard-vl-bestloss-step6215
```

对应 adapter 是：

```text
sft_scripts/output/adapter_warm1_json4
```

stage3 compact 起点通常是上面的 stage2 模型，再继续训 compact adapter：

```text
result13:
base model: /mnt/d/models/soulgard-vl-bestloss-step6215
adapter:    sft_scripts/output/adapter_warm1_json4_compact8k_r16_e5
merged:     /mnt/d/models/soulgard-vl_serialize
```

## 1. 优化目标

### 1.1 单帧主指标

单帧姿势行为只看这 6 项：

```text
action
overall_body
ears
tail
face
fur_state
```

主分数是宏平均：

```text
Avg = (action + overall_body + ears + tail + face + fur_state) / 6
```

不要把下面这些字段放进主 reward：

```text
location_on
vertical_position
nearby_anchors
nearby_beings
lighting
summary
environment_anomalies
interactions
```

它们可以作为辅助统计或格式检查，但不要让它们主导 RL。

### 1.2 硬约束

以下不是“加权平均项”，而是 gate：

```text
parse_ok
cat_presence_correct
cat_count_correct
```

如果任意一个失败，行为字段 reward 应该归零或大幅降权。

单帧 gate：

```text
G_single =
  1[parse_ok]
* 1[cat_presence_correct]
* 1[cat_count_correct]
```

最终单帧 reward：

```text
R_single =
  G_single * field_avg
- unknown_penalty
- format_penalty
- length_penalty
```

### 1.3 多帧额外硬约束

多帧和单帧类似，但还要保护：

```text
frame_count_correct
per-frame array length valid
```

多帧 gate：

```text
G_multi =
  1[parse_ok]
* 1[cat_presence_correct]
* 1[cat_count_correct]
* 1[frame_count_correct]
* 1[array_length_valid]
```

多帧字段 reward 推荐先用逐帧准确率，而不是数组级 exact match。原因是数组级太稀疏，6 帧里错 1 帧和全错都会被算 0，不适合 RL 初期。

## 2. 为什么用 GRPO 而不是 PPO

标准 PPO 需要：

```text
actor
old actor
reference
critic/value model
reward model 或 reward function
GAE
```

这个项目暂时不需要这么重。我们有确定的自动指标，可以直接对同一张图采样多条输出，然后做组内比较。

Field-GRPO 的核心：

```text
同一张图 x
采样 G 条回答 y_1 ... y_G
每条回答计算 reward R_i
用组内均值/标准差归一化成 advantage A_i
用 A_i 更新当前 actor
```

公式：

```text
A_i = (R_i - mean(R_1...R_G)) / (std(R_1...R_G) + eps)
```

训练 loss 可以理解为：

```text
L_grpo = - A_i * log pi_theta(y_i | x)
```

实际训练时还要加 reference KL 和 SFT anchor：

```text
L_total =
  L_grpo
+ beta * KL(pi_theta || pi_ref)
+ lambda_sft * L_sft
```

其中：

```text
pi_theta: 当前 LoRA actor
pi_ref:   冻结的 /mnt/d/models/soulgard-vl-bestloss-step6215
L_sft:    原始答案的 teacher-forcing CE，防止格式和猫识别被 RL 冲坏
```

## 3. 单帧 Field-GRPO 怎么做

### 3.1 数据文件

训练数据来源：

```text
sft_data/sft_cat_train.json
```

建议先不要全量跑。先切三份：

```text
RL train: 2000-5000 条
RL dev:   300-500 条
Eval-416: sft_data/sft_cat_val_doubao_lite.json
Eval-full:sft_data/sft_cat_val.json
```

切分原则：

1. `Eval-416` 和 `Eval-full` 绝对不能进 RL train。
2. RL train 优先有猫样本，但要保留一定无猫负样本。
3. 有猫样本里优先保留 6 个行为字段覆盖较丰富的样本。
4. `未知` 和非 `未知` 要尽量均衡，避免模型被训练成总猜已知或总猜未知。

建议比例：

```text
有猫样本: 75%-85%
无猫样本: 15%-25%
```

如果只优化有猫字段，不放无猫样本，模型可能提高字段 Avg 但增加假阳性。

### 3.2 每条训练样本需要保存的信息

一条 RL 样本至少需要：

```json
{
  "image": "...",
  "prompt": "...",
  "ground_truth_json": {...}
}
```

可以直接复用 SFT 数据格式：

```text
sample["image"]
sample["conversations"][0]["content"]  # user prompt
sample["conversations"][1]["content"]  # GT JSON 字符串
```

### 3.3 采样设置

每张图采样多条输出：

```text
group_size G: 4 或 8
temperature: 0.7-1.0
top_p: 0.9-0.95
max_new_tokens: 2048
```

建议第一轮：

```text
G=4
temperature=0.8
top_p=0.9
```

如果 G 太小，组内 reward 差异不明显；如果 G 太大，生成开销会明显上升。

### 3.4 reward 计算

先解析输出 JSON。如果解析失败：

```text
R = -1.0
```

如果解析成功，先算猫门槛：

```text
cat_presence_correct = pred cats_visible 是否和 GT 同为有猫/无猫
cat_count_correct    = pred cats_visible 是否等于 GT cats_visible
```

如果猫门槛失败：

```text
R = -0.5
```

如果猫门槛通过，并且 GT 有猫，计算 6 字段：

```text
field_avg =
  (
    action_correct
  + overall_body_correct
  + ears_correct
  + tail_correct
  + face_correct
  + fur_state_correct
  ) / 6
```

推荐基础 reward：

```text
R = field_avg
```

如果 GT 无猫，行为字段不存在，reward 只看无猫预测是否正确：

```text
R = 1.0 if cat_count_correct else -0.5
```

### 3.5 unknown 校准

小模型容易预测 `未知`，所以 reward 必须显式处理。

不要简单写：

```text
出现未知就罚
```

正确做法是区分两类：

```text
GT 已知，预测未知: known_to_unknown
GT 未知，预测已知: unknown_to_known
```

建议：

```text
unknown_penalty =
  0.15 * known_to_unknown_rate
+ 0.05 * unknown_to_known_rate
```

如果你确认当前模型非常爱输出 `未知`，可以提高第一项：

```text
unknown_penalty =
  0.25 * known_to_unknown_rate
+ 0.05 * unknown_to_known_rate
```

但不要把 `unknown_to_known` 完全不罚。因为看不清时硬猜耳朵、尾巴、脸，也会伤真实部署。

### 3.6 最终 reward 示例

```text
if parse_failed:
    R = -1.0
elif cat_presence_wrong or cat_count_wrong:
    R = -0.5
elif gt_has_no_cat:
    R = 1.0
else:
    R = field_avg
        - 0.15 * known_to_unknown_rate
        - 0.05 * unknown_to_known_rate
        - 0.02 * extra_text_penalty
```

最后把 reward clip 到一个稳定范围：

```text
R = clip(R, -1.0, 1.0)
```

## 4. 训练流程

### 4.1 初始化

```text
actor = /mnt/d/models/soulgard-vl-bestloss-step6215 + 新 LoRA
reference = /mnt/d/models/soulgard-vl-bestloss-step6215 冻结
```

LoRA 参数建议：

```text
rank: 8 或 16
alpha: 32
dropout: 0.05
target_modules: all-linear
```

先用 rank 8，跑通后再试 rank 16。

### 4.2 每个 step 做什么

一个 batch 内有 N 张图。每张图采样 G 条回答：

```text
N images * G responses
```

每条回答：

1. 解析 JSON。
2. 计算 reward。
3. 对同一张图的 G 条 reward 做组内归一化。
4. 计算这些生成 token 的 logprob。
5. 用 advantage 加权更新 actor。
6. 加 reference KL。
7. 混入少量 SFT CE。

### 4.3 KL 约束

KL 用来防止 actor 跑飞：

```text
KL = log pi_theta(y_t | s_t) - log pi_ref(y_t | s_t)
```

建议初始：

```text
beta = 0.02
```

如果发现：

```text
cat_count 掉了
parse_failed 增加
输出变长或多余解释
```

提高 beta：

```text
beta = 0.05
```

如果发现 reward 几乎不涨，模型学不动，可以降低：

```text
beta = 0.01
```

### 4.4 SFT anchor

每个 RL step 混一点原始 GT CE：

```text
lambda_sft = 0.05 - 0.2
```

建议第一轮：

```text
lambda_sft = 0.1
```

SFT anchor 的作用：

1. 保住 JSON 格式。
2. 保住猫数量。
3. 防止模型只钻 reward 空子。

### 4.5 训练步数

第一轮不要大跑。

建议：

```text
RL train samples: 2000
group_size: 4
max_steps: 200-500
eval every: 50 steps
```

如果 416 验证集上以下指标没有明显坏掉，再扩：

```text
parse_rate >= 0.99
cat_recognition 不低于原模型 1 个百分点
cat_count 不低于原模型 1 个百分点
6-field Avg 提升
known_to_unknown 下降
```

## 5. 评估方式

### 5.1 单帧必须汇报

必须汇报：

```text
parse_rate
cat_recognition_accuracy
cat_count_accuracy
action
overall_body
ears
tail
face
fur_state
Avg
known_to_unknown_rate
unknown_to_known_rate
```

不要只报 strict behavior，因为 strict behavior 太苛刻，也不符合当前优化目标。

### 5.2 通过标准

一版 Field-GRPO 才算有价值，需要满足：

```text
6-field Avg 上升
cat_recognition 不明显下降
cat_count 不明显下降
parse_rate 不下降
known_to_unknown_rate 下降
unknown_to_known_rate 不明显上升
```

如果 Avg 涨了，但猫数量掉了，不算成功。

如果 Avg 涨了，但模型把未知都硬猜成已知，也不算成功。

## 6. 多帧 Field-GRPO

多帧版本可以复用同一套思想，但 reward 改成逐帧。

### 6.1 数据

训练来源：

```text
sft_data/sft_multi_cat_train.json
```

评估：

```text
sft_data/sft_multi_cat_test100.json
```

如果做英文 schema：

```text
sft_data/sft_multi_cat_train_en.json
sft_data/sft_multi_cat_test100_en.json
```

### 6.2 多帧 gate

```text
if parse_failed:
    R = -1.0
elif frame_count_wrong:
    R = -0.8
elif cat_presence_or_count_wrong:
    R = -0.8
elif any per-frame array length != frame_count:
    R = -0.8
else:
    R = field_frame_avg - unknown_penalty
```

多帧 frame count 现在基本都是对的，所以一定要保护住。

### 6.3 多帧字段

中文多帧当前 schema 没有 `face/fur_state`，主要字段是：

```text
位置
邻近锚点
体态
耳
尾
动作
```

但如果你的目标仍然只围绕行为姿态，建议先看：

```text
体态
耳
尾
动作
```

不要让 `邻近锚点` 把训练目标带偏，除非这就是当前业务目标。

逐帧 reward：

```text
field_frame_avg =
  mean over fields and frames of 1[pred == gt]
```

数组级 exact match 可以作为辅助：

```text
R += 0.1 * field_array_avg
```

但不要只用数组级。

## 7. Compact / Stage3 的 RL

### 7.1 当前 compact 状态

result13：

```text
train data:  sft_data/sft_train_field_score_balanced_8000.json
base model:  /mnt/d/models/soulgard-vl-bestloss-step6215
merged:      /mnt/d/models/soulgard-vl_serialize
parse_rate:  1.0000
Avg:         0.7509
```

result14：

```text
train data:  sft_data/sft_train_field_diverse_10000.json
merged:      /mnt/d/models/soulgard-vl_serialize_result14
Avg:         0.7658
```

compact 格式本身是固定槽位：

```text
top-level: 7 slots separated by |
cat:       19 slots separated by ,
list:      # or ;
object:    ~
presence:  ^
```

程序端通过 `deserialize_schema_compact()` 还原 JSON。

### 7.2 compact RL 的目标

compact RL 不是单纯提高字段分数，而是：

```text
严格可反序列化
猫数量不掉
6 字段 Avg 不掉或上涨
生成更短更稳定
尽量不依赖 repair
```

当前 `eval_compact.py` 有 repair/canonicalize 逻辑。RL 时要区分：

```text
strict parse ok:      高奖励
repair 后 parse ok:   小奖励或轻罚
parse failed:         重罚
```

因为真实场景里“必须靠 repair 才能还原”仍然有风险。

### 7.3 compact reward

推荐：

```text
if strict_deserialize_ok:
    R_format = 1.0
elif repaired_deserialize_ok:
    R_format = 0.3
else:
    R_format = -1.0
```

然后看还原后的 JSON：

```text
if cat_presence_wrong or cat_count_wrong:
    R_task = -0.5
else:
    R_task = field_avg - unknown_penalty
```

最终：

```text
R_compact =
  0.4 * R_format
+ 0.6 * R_task
- 0.05 * length_over_budget
```

如果你最担心格式问题，把格式权重调大：

```text
R_compact =
  0.6 * R_format
+ 0.4 * R_task
```

### 7.4 compact 数据

优先从 result13 的 8K 开始：

```text
sft_data/sft_train_field_score_balanced_8000.json
```

这个文件已经是 compact prompt + compact assistant。

为了 reward 计算，需要拿到 GT JSON。有两种方式：

1. 从 compact assistant 反序列化回 GT JSON。
2. 使用中间 JSON 文件：

```text
sft_data/sft_train_field_score_balanced_8000_json_prompt.json
```

建议优先用方式 2，调试更直观。

compact eval 固定：

```text
sft_data/sft_cat_val_doubao_lite.json
```

### 7.5 compact GRPO 起点

可以从两个起点试：

```text
/mnt/d/models/soulgard-vl_serialize
/mnt/d/models/soulgard-vl_serialize_result14
```

如果目标是“格式安全”，建议先从 result14 起点，因为 Avg 更高。

如果目标是排查 8K compact 是否足够，先从 result13 起点。

### 7.6 compact 通过标准

compact RL 通过标准：

```text
strict_parse_rate >= 原模型
repair_needed_rate 下降
cat_recognition 不下降
cat_count 不下降
6-field Avg 不下降，最好上涨
generated tokens 不增加
```

如果 Avg 涨了，但 strict parse 下降，不接受。

如果 strict parse 保持，但 Avg 明显下降，也不接受。

## 8. 最小实验路线

### 8.1 单帧 Field-GRPO 最小闭环

第一轮目标：验证 RL 是否能提升 6-field Avg，并减少 known_to_unknown。

配置：

```text
base/ref model: /mnt/d/models/soulgard-vl-bestloss-step6215
train:          sft_data/sft_cat_train.json 抽 2000
dev:            train 里另抽 300
eval:           sft_data/sft_cat_val_doubao_lite.json
group_size:     4
temperature:    0.8
top_p:          0.9
max_steps:      200
beta_kl:        0.02
lambda_sft:     0.1
```

观察：

```text
Avg 是否上涨
tail/face/overall_body 是否上涨
known_to_unknown 是否下降
cat_count 是否稳定
parse_rate 是否稳定
```

### 8.2 compact Format-GRPO 最小闭环

第一轮目标：验证能否减少格式风险，同时不掉 Avg。

配置：

```text
base/ref model: /mnt/d/models/soulgard-vl_serialize_result14
train:          sft_data/sft_train_field_diverse_10000.json 抽 2000
eval:           sft_data/sft_cat_val_doubao_lite.json
group_size:     4
temperature:    0.8
top_p:          0.9
max_new_tokens: 192
max_steps:      200
beta_kl:        0.03
lambda_sft:     0.1
```

观察：

```text
strict_parse_rate
repair_needed_rate
cat_recognition
cat_count
6-field Avg
generated tokens / image
```

## 9. 常见失败模式

### 9.1 Reward hacking

现象：

```text
模型总输出最常见字段
模型减少未知但开始乱猜
模型输出变短但字段缺失
```

处理：

```text
提高 KL beta
提高 SFT anchor
加 unknown_to_known 惩罚
加 strict schema penalty
```

### 9.2 猫数量掉了

现象：

```text
6-field Avg 涨，但 cat_count 掉
```

处理：

```text
把 cat_count 作为 gate，不要只是一个加分项
无猫样本比例提高到 20%-25%
提高 KL beta
```

### 9.3 输出格式坏了

现象：

```text
parse_failed 增加
JSON 多解释文字
compact 槽位数量不稳定
```

处理：

```text
parse_failed = -1.0
提高 lambda_sft
compact 增加 strict_parse reward
降低 temperature
```

### 9.4 只会预测未知

现象：

```text
GT 已知时，模型预测未知增加
```

处理：

```text
提高 known_to_unknown penalty
RL train 里补充已知 ears/tail/face 样本
不要只抽模糊样本
```

### 9.5 过度乱猜未知

现象：

```text
GT 未知时，模型预测具体姿态增加
```

处理：

```text
提高 unknown_to_known penalty
保留模糊/遮挡样本
单独统计 ears/tail/face 的 unknown calibration
```

## 10. 推荐文件规划

如果后续实现代码，建议不要塞进现有 SFT 脚本，而是新建：

```text
sft_scripts/rl_prepare_field_grpo_data.py
sft_scripts/rl_train_field_grpo.py
sft_scripts/rl_eval_field_metrics.py
sft_scripts/rl_train_compact_grpo.py
sft_scripts/utils/rl_rewards.py
sft_scripts/utils/rl_grpo.py
```

其中：

```text
rl_rewards.py
  compute_single_field_reward()
  compute_compact_reward()
  unknown_calibration_stats()

rl_grpo.py
  group_advantages()
  sequence_logprobs()
  reference_kl()
```

结果目录建议：

```text
results/result22_field_grpo_single/
results/result23_compact_grpo/
```

不要覆盖已有 result13/result14/result21。

## 11. 最终建议

优先级：

```text
1. 单帧 Field-GRPO 小闭环
2. compact Format+Field-GRPO 小闭环
3. 多帧 Field-GRPO
4. 如果 GRPO 有收益，再考虑 PPO-lite
```

暂时不建议直接做完整 PPO。这个项目最需要的是“指标对齐”和“格式安全”，不是开放式偏好 RLHF。Field-GRPO 正好利用了现有 evaluator，工程更轻，也更容易解释为什么有效。
