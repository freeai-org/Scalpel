# PPO、TD 误差、GAE：大模型 RLHF 后训练通俗版

这份笔记专门回答几个容易卡住的问题：

1. PPO 的目的到底是什么？
2. PPO 是不是“很多 prompt 里挑最好的 prompt”？
3. Reward Model、Critic、Reference、Actor 分别是什么？
4. TD 误差到底在算什么？
5. GAE 为什么是 TD 误差的反向累加？
6. Critic 网络默认长什么样？

默认场景是大模型 RLHF：

$$
x \sim \mathcal{D}_{\text{prompt}}, \qquad
y = (y_1, y_2, \ldots, y_T) \sim \pi_\theta(\cdot \mid x)
$$

PPO 训练的不是 prompt，而是训练模型参数，让模型以后面对类似 prompt 时，更倾向于生成高 reward 的 response。

## 1. 先说结论：PPO 的目的是什么

PPO 的目标不是：

```text
从很多 prompt 里找到最好的 prompt。
```

也不是简单地：

```text
对一个 prompt 生成多个回答，然后只记住最好的那个回答。
```

PPO 的真正目标是：

```text
在一批 prompt 上，让当前大模型自己生成回答；
用 Reward Model 判断回答好坏；
再调整模型的 token 概率；
让模型以后更容易生成高 reward 的回答，
更不容易生成低 reward 的回答。
```

用一句话说：

> PPO 是让模型根据 reward 反馈，微调自己的生成概率分布。

它优化的是期望 reward：

$$
\max_{\theta}\;
\mathbb{E}_{x \sim \mathcal{D}_{\text{prompt}},\; y \sim \pi_\theta(\cdot \mid x)}
\left[
R(x, y)
\right]
$$

也就是：

- prompt `x` 来自训练 prompt 分布；
- response `y` 是当前模型自己采样出来的；
- reward 来自 RM、规则、verifier 或环境；
- 训练目标是让模型生成的 response 整体更好。

## 2. Prompt、Response、Token 在 PPO 里分别是什么

对大模型来说：

```text
状态 state s_t = prompt + 已经生成的前缀 y_<t
动作 action a_t = 当前要生成的 token y_t
轨迹 trajectory = 整个 response y_1 ... y_T
```

比如 prompt 是：

```text
用一句话解释光合作用。
```

模型生成：

```text
植物通过光能把二氧化碳和水转化为有机物。
```

那每一步是：

```text
s_1 = prompt
a_1 = "植物"

s_2 = prompt + "植物"
a_2 = "通过"

s_3 = prompt + "植物通过"
a_3 = "光能"
...
```

PPO 不是直接说“这个回答好，所以整个回答复制下来”。它会具体到每个生成 token：

```text
这个 token 在这个前缀下，应该更容易出现，还是更不容易出现？
```

这个“应该更容易还是更不容易”，靠 advantage 来判断。

## 3. RLHF-PPO 里的四个模型

| 名称 | 符号 | 作用 | 是否训练 |
|---|---|---|---|
| Actor / Policy | `πθ` | 当前大模型，负责生成回答 | 训练 |
| Reference Model | `πref` | 冻结的 SFT 模型，用于 KL 约束 | 冻结 |
| Reward Model | `Rφ` | 给完整回答打分 | PPO 阶段冻结 |
| Critic / Value Model | `Vψ` | 预测每个前缀未来能拿多少 reward | 训练 |

### 3.1 Actor 是什么

Actor 就是你最终想优化的大模型。

它输入：

```text
prompt + 已生成前缀
```

输出：

```text
下一个 token 的概率分布
```

例如：

```text
πθ("植物" | prompt) = 0.20
πθ("动物" | prompt) = 0.03
πθ("阳光" | prompt) = 0.08
```

PPO 更新后，如果某条回答 reward 高，那么对应 token 的概率会被适度提高。

### 3.2 Reference Model 是什么

Reference model 通常是 PPO 开始前的 SFT 模型，冻结不动。

它的作用是防止模型为了刷 reward 跑飞。

如果 PPO 只最大化 reward，模型可能学会奇怪模式：

```text
重复安全套话
过度讨好
输出 reward model 偏爱的模板
语言能力变差
```

所以每一步都会看：

```text
当前 policy 和 reference 在这个 token 上差了多少？
```

常见 token-level KL 近似：

$$
\mathrm{KL}_t \approx
\log \pi_\theta(y_t \mid s_t)
-
\log \pi_{\text{ref}}(y_t \mid s_t)
$$

带 KL 惩罚的 reward：

$$
r_t = r_{\text{raw},t} - \beta \,\mathrm{KL}_t
$$

直觉：

> RM 负责告诉模型“往好回答走”，Reference 负责告诉模型“别走太远”。

### 3.3 Reward Model 是什么

Reward Model 输入：

```text
prompt + 完整 response
```

输出：

```text
一个标量分数
```

例如：

```text
Rφ(prompt, response) = 7.8
```

RM 通常是用人类偏好数据训练出来的：

```text
(prompt, chosen_response, rejected_response)
```

训练目标是让：

```text
R(prompt, chosen) > R(prompt, rejected)
```

PPO 阶段 RM 通常冻结，不参与反向更新。

你说 HF 上有别人训练好的 RM，这个理解对。实际工程里可以直接加载一个 reward model 或 verifier。

### 3.4 Critic / Value Model 是什么

Critic 预测的是：

```text
从当前前缀状态 s_t 开始，未来整段生成大概能拿多少总 reward。
```

公式：

$$
V_\psi(s_t)
\approx
\mathbb{E}
\left[
\sum_{k=t}^{T} \gamma^{k-t} r_k
\;\middle|\;
s_t
\right]
$$

它不是判断当前 token 的概率，也不是 reward model。

区别：

```text
Reward Model:
看完整回答，给最终分数。

Critic:
看当前前缀，预测未来总收益。
```

比如：

```text
s_t = prompt + "植物通过光能"
```

Critic 会估计：

```text
从这个前缀继续写完，最终大概能拿 7.2 分。
```

## 4. Critic 网络默认长什么样

大模型 RLHF 里的 Critic 通常有两种实现。

### 4.1 最常见：Transformer Backbone + Value Head

结构大概是：

```text
输入 token
-> Transformer backbone
-> 每个位置的 hidden state h_t
-> linear value head
-> 每个位置一个 scalar value V(s_t)
```

也就是：

$$
V_\psi(s_t) = W_v h_t + b_v
$$

如果 hidden size 是 2048，那么 value head 很可能就是：

```text
Linear(2048, 1)
```

输出 shape：

```text
[batch, seq_len, 1]
```

每个 token 位置都有一个 value。

### 4.2 Critic 可以和 Actor 共享 backbone 吗

可以，常见两种：

**方式 A：Actor 和 Critic 分开**

```text
actor_model: CausalLM
critic_model: CausalLM backbone + value head
```

优点：

- 训练逻辑清楚；
- actor/critic 互不干扰。

缺点：

- 显存大；
- forward 成本高。

**方式 B：Actor 和 Critic 共享 backbone，不同 head**

```text
shared transformer
-> lm_head 输出 token logits
-> value_head 输出 scalar values
```

优点：

- 省显存；
- 省计算。

缺点：

- policy 和 value 更新会互相影响；
- 工程上要处理 loss 权重。

实际 RLHF 框架里经常是：

```text
actor 是一个模型
critic 是另一个带 value head 的模型
reference 是冻结 SFT
reward model 是单独模型
```

例如 TRL 里常见的思路就是给 causal LM 加一个 value head。

### 4.3 Critic 初始化从哪里来

没有唯一默认。

常见选择：

- 从 SFT model 初始化 backbone，再加 value head；
- 从 reward model 初始化；
- 从 actor 初始化一份副本，再加 value head；
- 小模型场景直接随机初始化 value head。

关键是：

> Critic 要学会预测当前前缀未来的总 reward。

它不是提前就会，需要 PPO 过程中用 value loss 训练出来。

## 5. PPO 一轮到底在做什么

一轮 PPO 可以拆成 6 步。

### 5.1 第一步：采样 prompt

从 prompt 数据集中取一批 prompt：

```text
x_1, x_2, ..., x_B
```

这些 prompt 是训练任务，不是要被优化的对象。

### 5.2 第二步：Actor 生成回答

用当前 policy 生成 response：

$$
y \sim \pi_{\text{old}}(\cdot \mid x)
$$

注意这里是 `π_old`，也就是本轮更新前的模型。

同时记录每个 token 的旧 logprob：

$$
\log p^{\text{old}}_t =
\log \pi_{\text{old}}(y_t \mid s_t)
$$

为什么要记录？

因为后面 PPO 要比较：

```text
新模型对这个 token 的概率
vs
旧模型对这个 token 的概率
```

### 5.3 第三步：RM 打分 + KL 惩罚

Reward Model 给完整回答打分：

```text
R = Rφ(x, y)
```

大模型里 RM 通常只给整段回答一个分数，所以这是稀疏奖励。

然后每个 token 再扣 KL：

$$
r_t = -\beta \,\mathrm{KL}_t,\qquad t < T
$$

$$
r_T = R_\phi(x, y) - \beta \,\mathrm{KL}_T
$$

其中：

$$
\mathrm{KL}_t \approx
\log \pi_{\text{old}}(y_t \mid s_t)
-
\log \pi_{\text{ref}}(y_t \mid s_t)
$$

有些实现用 `πθ`，有些 rollout 阶段用 `π_old` 近似；核心都是惩罚偏离 reference。

### 5.4 第四步：Critic 输出 value

Critic 对每个前缀状态输出：

$$
V_\psi(s_1), V_\psi(s_2), \ldots, V_\psi(s_T)
$$

它的含义是：

```text
从这个前缀继续生成，预计最终总收益是多少。
```

### 5.5 第五步：用 TD + GAE 算 advantage

这一步是你最卡的部分。先记一句话：

> Advantage 用来判断某个 token 动作比 critic 原本预期更好还是更差。

如果：

```text
A_t > 0
```

说明：

```text
在状态 s_t 生成 token y_t，比 critic 原本预期更好。
```

PPO 会提高这个 token 的概率。

如果：

```text
A_t < 0
```

说明：

```text
这个 token 导向的未来结果比预期差。
```

PPO 会降低这个 token 的概率。

### 5.6 第六步：更新 Actor 和 Critic

Actor 用 PPO clipped loss 更新。

Critic 用 value loss 更新。

更新完以后，新的 actor 再去采样下一批 response，进入下一轮。

## 6. TD 误差到底是什么

TD 误差公式：

$$
\delta_t =
r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

拆开看：

$$
r_t + \gamma V_\psi(s_{t+1})
$$

叫 TD target，意思是：

```text
当前这一步真实拿到的奖励
+ 下一步状态预计还能拿到的未来奖励
```

它是对 `V(s_t)` 的一个更好的训练目标。

所以：

$$
\delta_t =
\underbrace{r_t + \gamma V_\psi(s_{t+1})}_{\text{TD target}}
-
\underbrace{V_\psi(s_t)}_{\text{critic prediction}}
$$

也就是：

```text
δ_t = 新证据认为应该是多少 - critic 原来以为是多少
```

### 6.1 δ 为正是什么意思

```text
δ_t > 0
```

说明：

```text
实际情况比 critic 预期更好。
```

critic 低估了这个状态。

这个 token 对未来结果可能是正贡献。

### 6.2 δ 为负是什么意思

```text
δ_t < 0
```

说明：

```text
实际情况比 critic 预期更差。
```

critic 高估了这个状态。

这个 token 对未来结果可能是负贡献。

### 6.3 TD 误差不是 reward

TD 误差不是 reward 本身。

reward 是外部反馈：

```text
这个回答得 8 分。
```

TD 误差是：

```text
这个结果相对 critic 原来的预期，是超预期还是低于预期。
```

PPO 真正在更新 actor 时更关心的是 advantage，而不是裸 reward。

原因是：

```text
同样 8 分，如果 critic 原本预期 9 分，那其实低于预期；
同样 8 分，如果 critic 原本预期 5 分，那就是超预期。
```

## 7. 为什么需要 GAE

大模型 RM 通常只给完整回答一个总分。

比如：

```text
整条回答 reward = 8
```

但 actor 需要知道：

```text
第 1 个 token 该不该增强？
第 2 个 token 该不该增强？
第 3 个 token 该不该增强？
...
```

这就是 credit assignment，也就是把整条回答的好坏分摊给每个 token。

GAE 的作用：

> 用一串 TD 误差，给每个 token 算一个平滑的 advantage。

公式：

$$
A_t =
\delta_t
+ \gamma\lambda\,\delta_{t+1}
+ (\gamma\lambda)^2 \delta_{t+2}
+ \cdots
$$

递推写法：

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

从最后一个 token 往前算。

### 7.1 λ 控制什么

`λ` 是偏差和方差的折中。

如果：

```text
λ = 0
```

那么：

$$
A_t = \delta_t
$$

只看一步 TD，方差小，但太短视。

如果：

```text
λ = 1
```

接近看完整未来回报，偏差小，但方差大。

常用：

```text
λ = 0.95
```

意思是：

```text
未来 TD 也看，但越远权重越低。
```

## 8. 一个正确、不绕的数值例子

为了避免 EOS 和状态下标混乱，我们这样定义：

```text
生成 4 个 token：y1, y2, y3, y4
状态 s1 是生成 y1 前的前缀
状态 s2 是生成 y2 前的前缀
状态 s5 是生成完 y4 后的终止状态
```

每个 token 是一次 action：

```text
s1 --y1--> s2
s2 --y2--> s3
s3 --y3--> s4
s4 --y4--> s5 terminal
```

最终 RM 分数放在最后一步 `y4` 上。

设：

```text
γ = 1.0
λ = 0.95
β = 0.1
RM score = 8.0
```

每步 KL 惩罚：

| t | token | raw reward | KL | total reward r_t |
|---|---|---:|---:|---:|
| 1 | 植物 | 0 | 0.2 | -0.02 |
| 2 | 通过 | 0 | 0.1 | -0.01 |
| 3 | 光能 | 0 | 0.3 | -0.03 |
| 4 | 转化 | 8.0 | 0.2 | 7.98 |

所以：

```text
r1 = -0.02
r2 = -0.01
r3 = -0.03
r4 = 7.98
```

Critic 预测：

| state | 含义 | V(s) |
|---|---|---:|
| s1 | prompt | 6.0 |
| s2 | prompt + 植物 | 6.3 |
| s3 | prompt + 植物通过 | 6.7 |
| s4 | prompt + 植物通过光能 | 7.0 |
| s5 | 终止态 | 0 |

### 8.1 计算 TD 误差

公式：

$$
\delta_t =
r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

因为这里 `γ=1`：

$$
\delta_t =
r_t + V_\psi(s_{t+1}) - V_\psi(s_t)
$$

逐步算：

$$
\delta_4 =
r_4 + V_\psi(s_5) - V_\psi(s_4)
= 7.98 + 0 - 7.0
= 0.98
$$

最后一步超预期，因为最终 reward 比 critic 在 s4 预期的 7.0 更好。

$$
\delta_3 =
r_3 + V_\psi(s_4) - V_\psi(s_3)
= -0.03 + 7.0 - 6.7
= 0.27
$$

$$
\delta_2 =
r_2 + V_\psi(s_3) - V_\psi(s_2)
= -0.01 + 6.7 - 6.3
= 0.39
$$

$$
\delta_1 =
r_1 + V_\psi(s_2) - V_\psi(s_1)
= -0.02 + 6.3 - 6.0
= 0.28
$$

这里所有 TD 都是正的，说明这条回答整体比 critic 原来预期更好。

### 8.2 用 GAE 反向算 advantage

公式：

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

这里：

```text
γλ = 1.0 * 0.95 = 0.95
```

从最后一步开始：

$$
A_4 = \delta_4 = 0.98
$$

$$
A_3 =
\delta_3 + 0.95 A_4
= 0.27 + 0.95 \times 0.98
= 1.201
$$

$$
A_2 =
\delta_2 + 0.95 A_3
= 0.39 + 0.95 \times 1.201
= 1.531
$$

$$
A_1 =
\delta_1 + 0.95 A_2
= 0.28 + 0.95 \times 1.531
= 1.734
$$

得到：

| t | token | δ_t | A_t | PPO 倾向 |
|---|---|---:|---:|---|
| 1 | 植物 | 0.28 | 1.734 | 提高概率 |
| 2 | 通过 | 0.39 | 1.531 | 提高概率 |
| 3 | 光能 | 0.27 | 1.201 | 提高概率 |
| 4 | 转化 | 0.98 | 0.980 | 提高概率 |

为什么前面的 token advantage 也高？

因为前面的 token 帮模型走到了一条最终 reward 高的路径。

GAE 把后面的好结果反向传给前面的 token。

### 8.3 如果 reward 低会怎样

如果最终 RM score 只有 3.0，那么：

$$
r_4 = 3.0 - 0.02 = 2.98
$$

$$
\delta_4 = 2.98 + 0 - 7.0 = -4.02
$$

这个巨大的负 TD 会通过 GAE 往前传。

那么前面 token 的 advantage 也可能变成负数。

PPO 就会降低这条生成路径上 token 的概率。

## 9. PPO policy loss 到底怎么用 advantage

PPO 记录旧模型生成时的概率：

$$
\log p^{\text{old}}_t =
\log \pi_{\text{old}}(y_t \mid s_t)
$$

更新时重新算当前新模型的概率：

$$
\log p^\theta_t =
\log \pi_\theta(y_t \mid s_t)
$$

概率比：

$$
\rho_t(\theta)
=
\exp
\left(
\log \pi_\theta(y_t \mid s_t)
-
\log \pi_{\text{old}}(y_t \mid s_t)
\right)
$$

如果：

$$
\rho_t > 1
$$

说明新模型更喜欢这个 token。

如果：

$$
\rho_t < 1
$$

说明新模型更不喜欢这个 token。

PPO clipped loss：

$$
L_{\text{policy}}(\theta)
=
-
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta) A_t,\;
\mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

直觉：

- `A_t > 0`：提高这个 token 概率；
- `A_t < 0`：降低这个 token 概率；
- `clip`：不允许一次更新改太猛。

### 9.1 为什么要 clip

因为 PPO 用的是刚才旧模型采样的数据。

如果新模型一下子变化太大：

```text
新模型已经不是采样这批数据的那个模型了
```

训练会不稳定。

所以 PPO 限制：

$$
\rho_t \in [1-\epsilon, 1+\epsilon]
$$

当 $\epsilon = 0.2$ 时：

$$
\rho_t \in [0.8, 1.2]
$$

常见：

```text
ε = 0.2
```

## 10. Value loss 怎么训练 Critic

Critic 要学会预测 future return。

常见 value target：

$$
\mathrm{Return}_t = A_t + V_{\text{old}}(s_t)
$$

value loss：

$$
L_{\text{value}}(\psi)
=
\mathbb{E}_t
\left[
\left(
V_\psi(s_t) - \mathrm{Return}_t
\right)^2
\right]
$$

也可以从 TD target 角度理解：

```text
让 V(s_t) 靠近 r_t + γ V(s_{t+1})
```

但实际 PPO + GAE 里常用 `A_t + old_value_t` 作为 return target。

Critic 的目的不是直接让回答更好。

它的目的是：

```text
提供 baseline，降低 policy gradient 的方差。
```

没有 critic 也可以做 RL，但方差会大。

GRPO 就是一个例子：它不用 critic，而是用同 prompt 多个回答的组内平均 reward 当 baseline。

## 11. PPO 总 loss

典型 PPO 总 loss：

$$
L_{\text{total}}
=
L_{\text{policy}}
+ c_v L_{\text{value}}
- c_e H(\pi_\theta)
$$

有些实现还会显式加 KL loss：

$$
L_{\text{total}}
=
L_{\text{policy}}
+ c_v L_{\text{value}}
- c_e H(\pi_\theta)
+ \beta \,\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})
$$

也有些实现把 KL 放进 reward 里，不再额外加一遍。

大模型 PPO 常见训练信号：

```text
policy loss: 训练 actor
value loss: 训练 critic
KL penalty: 防止偏离 reference
entropy bonus: 保持一定探索
```

## 12. PPO 不是“选最好回答”这么简单

你可能会问：

> 如果一个 prompt 生成多个回答，直接选 reward 最高的回答做 SFT 不行吗？

这确实是一种方法，类似 rejection sampling / best-of-N / rejection sampling fine-tuning。

但 PPO 不只是这样。

PPO 的特点是：

```text
它用 reward 和 advantage 来连续调整 token 概率，
而不是只把最高分答案当标签。
```

它会利用：

- 这条回答比预期好多少；
- 每个 token 对未来回报的贡献；
- 当前 policy 和 old policy 的概率变化；
- 当前 policy 和 reference 的 KL 距离。

所以 PPO 比“挑最好的回答做 SFT”更像真正的强化学习。

## 13. PPO 和 GRPO、DPO 的区别

### 13.1 PPO

```text
当前模型在线生成回答
RM 打分
Critic 预测 value
TD + GAE 算 advantage
PPO clip 更新 actor
```

特点：

- online；
- on-policy；
- 有 critic；
- 有 TD；
- 有 GAE；
- 系统最重。

### 13.2 GRPO

```text
同一个 prompt 生成多个回答
每个回答打 reward
用组内平均 reward 做 baseline
```

advantage：

$$
A_i =
\frac{
r_i - \mathrm{mean}(r_{\text{group}})
}{
\mathrm{std}(r_{\text{group}}) + \epsilon
}
$$

特点：

- online；
- 通常 on-policy；
- 不需要 critic；
- 不需要 TD；
- 不需要 GAE；
- 适合数学、代码、可验证任务。

### 13.3 DPO

```text
固定偏好对：
(prompt, chosen, rejected)
```

loss：

$$
L_{\text{DPO}}(\theta)
=
-
\log \sigma
\left(
\beta
\left[
\log \pi_\theta(y_w \mid x)
-
\log \pi_\theta(y_l \mid x)
-
\log \pi_{\text{ref}}(y_w \mid x)
+
\log \pi_{\text{ref}}(y_l \mid x)
\right]
\right)
$$

特点：

- offline；
- 不需要在线采样；
- 不需要 RM；
- 不需要 critic；
- 不需要 TD；
- 不需要 GAE。

## 14. 一句话理解 TD、GAE、Critic

TD 误差：

```text
这一步之后，结果比 critic 原来预期更好还是更差？
```

GAE：

```text
把后面发生的好坏，按衰减权重传回前面的 token。
```

Critic：

```text
预测当前前缀未来能拿多少 reward，给 actor 更新提供 baseline。
```

Advantage：

```text
这个 token 动作相对 critic 预期，是加分动作还是减分动作。
```

PPO：

```text
根据 advantage 调整 token 概率，但用 clip 和 KL 防止模型一次改太猛。
```

## 15. 面试回答模板

### Q1：PPO 的目的是什么？

PPO 的目的是让大模型在 prompt 分布上最大化期望 reward。它不是挑最好的 prompt，而是让当前 policy 生成 response，用 reward model 打分，再通过 advantage 调整每个 token 的生成概率，让高 reward 路径更容易出现，低 reward 路径更不容易出现。

### Q2：TD 误差是什么？

TD 误差是：

$$
\delta_t =
r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

它表示基于当前奖励和下一状态价值形成的新目标，和 critic 原本预测之间的差。如果 δ 为正，说明 critic 低估了这个状态；如果 δ 为负，说明 critic 高估了这个状态。

### Q3：GAE 是什么？

GAE 是对 TD 误差的反向加权累加：

$$
A_t =
\delta_t
+ \gamma\lambda \delta_{t+1}
+ (\gamma\lambda)^2 \delta_{t+2}
+ \cdots
$$

它用于给每个 token 估计 advantage，把最终 reward 的好坏分摊回生成路径上的每个 token。

### Q4：Critic 网络长什么样？

大模型 PPO 里的 critic 通常是 transformer backbone 加 value head。每个 token 位置的 hidden state 经过一个线性层输出一个标量：

$$
V_\psi(s_t) = W_v h_t + b_v
$$

这个标量表示从当前前缀继续生成，预计未来能获得多少总 reward。

### Q5：Reward Model 和 Critic 有什么区别？

Reward Model 看完整 prompt-response，输出最终偏好分数，PPO 阶段通常冻结。Critic 看当前前缀状态，预测未来总 reward，PPO 阶段会训练。RM 是外部裁判，Critic 是 actor 更新时用的 baseline 估计器。

## 16. 最短记忆版

```text
Actor:
负责生成回答。

RM:
看完整回答，给最终分。

Critic:
看每个前缀，预测未来能拿多少分。

TD:
新目标 - critic 旧预测。

GAE:
把一串 TD 误差反向累加，得到每个 token 的 advantage。

PPO:
advantage 为正就提高 token 概率；
advantage 为负就降低 token 概率；
clip + KL 防止模型跑飞。
```

PPO 的精神是：

> 不把回答当死标签模仿，而是根据 reward 反馈，稳定地移动整个生成分布。

## 17. 完整例子：一个 prompt 如何通过 PPO 更新模型

这一节完整走一遍大模型 PPO。为了看清楚流程，我们用很短的回答和简化数字。

### 17.1 输入 prompt

假设训练 prompt 是：

```text
请用一句话解释光合作用。
```

当前 actor / policy 是：

$$
\pi_{\theta_{\text{old}}}
$$

它生成了一个回答：

```text
植物 利用 光 制造 能量
```

把它拆成 4 个 token：

```text
y_1 = 植物
y_2 = 利用
y_3 = 光
y_4 = 制造能量
```

状态和动作对应：

| step | state `s_t` | action `y_t` |
|---:|---|---|
| 1 | prompt | 植物 |
| 2 | prompt + 植物 | 利用 |
| 3 | prompt + 植物 利用 | 光 |
| 4 | prompt + 植物 利用 光 | 制造能量 |

PPO 更新的对象不是这一个 prompt，而是模型参数。这个 prompt 只是本轮训练 batch 里的一个样本。

### 17.2 Rollout 时记录 old logprob

actor 生成时，会记录旧模型对每个 token 的 logprob：

$$
\log \pi_{\text{old}}(y_t \mid s_t)
$$

假设记录如下：

| step | token | old prob | old logprob |
|---:|---|---:|---:|
| 1 | 植物 | 0.30 | -1.204 |
| 2 | 利用 | 0.25 | -1.386 |
| 3 | 光 | 0.40 | -0.916 |
| 4 | 制造能量 | 0.20 | -1.609 |

这些 old logprob 后面算 PPO ratio 会用。

### 17.3 Reference model 计算 KL 惩罚

reference model 是冻结的 SFT 模型：

$$
\pi_{\text{ref}}
$$

它对同样 token 的概率如下：

| step | token | ref prob | ref logprob |
|---:|---|---:|---:|
| 1 | 植物 | 0.35 | -1.050 |
| 2 | 利用 | 0.30 | -1.204 |
| 3 | 光 | 0.35 | -1.050 |
| 4 | 制造能量 | 0.18 | -1.715 |

token-level KL 近似：

$$
\mathrm{KL}_t
\approx
\log \pi_{\text{old}}(y_t \mid s_t)
-
\log \pi_{\text{ref}}(y_t \mid s_t)
$$

代入：

| step | token | old logprob | ref logprob | KL approx |
|---:|---|---:|---:|---:|
| 1 | 植物 | -1.204 | -1.050 | -0.154 |
| 2 | 利用 | -1.386 | -1.204 | -0.182 |
| 3 | 光 | -0.916 | -1.050 | 0.134 |
| 4 | 制造能量 | -1.609 | -1.715 | 0.106 |

真实实现里 KL 可能会用更稳定的估计方式；这里用这个近似只是为了理解。

KL 惩罚的作用是：

> 如果 actor 偏离 reference 太多，就扣 reward，防止模型为了刷 RM 分跑飞。

### 17.4 RM 怎么发挥作用

Reward Model 输入完整 prompt 和完整 response：

```text
prompt:
请用一句话解释光合作用。

response:
植物利用光制造能量。
```

RM 输出一个标量分数。假设：

$$
R_\phi(x, y) = 7.5
$$

这表示 RM 认为这个回答整体还不错。

注意：

```text
RM 不给每个 token 单独打分。
RM 给的是整条回答的总分。
```

所以大模型 PPO 里的 reward 通常是稀疏的：

```text
前面 token 原始 reward = 0
最后一步放入 RM score
```

设 KL 系数：

$$
\beta = 0.1
$$

每步 reward：

$$
r_t = r_{\text{raw},t} - \beta \,\mathrm{KL}_t
$$

其中：

$$
r_{\text{raw},1}=r_{\text{raw},2}=r_{\text{raw},3}=0
$$

$$
r_{\text{raw},4}=R_\phi(x,y)=7.5
$$

代入得到：

| step | token | raw reward | KL approx | total reward `r_t` |
|---:|---|---:|---:|---:|
| 1 | 植物 | 0 | -0.154 | 0.0154 |
| 2 | 利用 | 0 | -0.182 | 0.0182 |
| 3 | 光 | 0 | 0.134 | -0.0134 |
| 4 | 制造能量 | 7.5 | 0.106 | 7.4894 |

这里 RM 发挥作用的地方是最后一步：

$$
r_4 = R_\phi(x,y) - \beta \,\mathrm{KL}_4
$$

如果 RM 给低分，比如 `2.0`，那么最后一步 reward 会很低，GAE 会把这个坏结果往前传，让整条生成路径的 token 概率下降。

如果 RM 给高分，比如 `9.0`，最后一步 reward 会很高，GAE 会把好结果往前传，让整条生成路径的 token 概率上升。

### 17.5 Critic 预测每个前缀的价值

Critic 看每个前缀，输出未来总 reward 的预测：

$$
V_\psi(s_t)
$$

假设 critic 输出：

| state | 含义 | value |
|---:|---|---:|
| `s_1` | prompt | 6.0 |
| `s_2` | prompt + 植物 | 6.2 |
| `s_3` | prompt + 植物 利用 | 6.4 |
| `s_4` | prompt + 植物 利用 光 | 6.7 |
| `s_5` | 终止态 | 0 |

这些 value 的含义是：

```text
critic 觉得从这个前缀继续写完，大概能拿多少总 reward。
```

### 17.6 计算 TD 误差

设：

$$
\gamma = 1.0
$$

TD 误差：

$$
\delta_t =
r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

逐步算：

$$
\delta_4 =
7.4894 + 0 - 6.7
= 0.7894
$$

$$
\delta_3 =
-0.0134 + 6.7 - 6.4
= 0.2866
$$

$$
\delta_2 =
0.0182 + 6.4 - 6.2
= 0.2182
$$

$$
\delta_1 =
0.0154 + 6.2 - 6.0
= 0.2154
$$

这些 TD 误差都是正的，说明这条生成路径比 critic 原本预期更好。

### 17.7 用 GAE 把最终好坏传回每个 token

设：

$$
\lambda = 0.95
$$

因为：

$$
\gamma\lambda = 0.95
$$

GAE 递推：

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

从最后一步往前算：

$$
A_4 = \delta_4 = 0.7894
$$

$$
A_3 =
\delta_3 + 0.95 A_4
= 0.2866 + 0.95 \times 0.7894
= 1.0365
$$

$$
A_2 =
\delta_2 + 0.95 A_3
= 0.2182 + 0.95 \times 1.0365
= 1.2039
$$

$$
A_1 =
\delta_1 + 0.95 A_2
= 0.2154 + 0.95 \times 1.2039
= 1.3591
$$

得到：

| step | token | advantage | 含义 |
|---:|---|---:|---|
| 1 | 植物 | 1.3591 | 应该提高概率 |
| 2 | 利用 | 1.2039 | 应该提高概率 |
| 3 | 光 | 1.0365 | 应该提高概率 |
| 4 | 制造能量 | 0.7894 | 应该提高概率 |

这就是 GAE 的核心作用：

> RM 只给整条回答打分，但 GAE 把这个整体好坏传回每个 token。

### 17.8 更新时重新计算 new logprob

现在开始训练更新。当前要优化的新 actor 是：

$$
\pi_\theta
$$

它会对刚才同样的 token 重新算 logprob：

| step | token | old logprob | new logprob |
|---:|---|---:|---:|
| 1 | 植物 | -1.204 | -1.150 |
| 2 | 利用 | -1.386 | -1.330 |
| 3 | 光 | -0.916 | -0.880 |
| 4 | 制造能量 | -1.609 | -1.560 |

概率比：

$$
\rho_t(\theta)
=
\exp
\left(
\log \pi_\theta(y_t \mid s_t)
-
\log \pi_{\text{old}}(y_t \mid s_t)
\right)
$$

代入：

| step | token | new-old | ratio |
|---:|---|---:|---:|
| 1 | 植物 | 0.054 | 1.055 |
| 2 | 利用 | 0.056 | 1.058 |
| 3 | 光 | 0.036 | 1.037 |
| 4 | 制造能量 | 0.049 | 1.050 |

因为这些 advantage 都是正的，ratio 大于 1 表示新模型正在提高这些 token 的概率，这是 PPO 希望看到的方向。

### 17.9 PPO clipped loss 怎么限制更新

设：

$$
\epsilon = 0.2
$$

允许 ratio 大概在：

$$
[1-\epsilon, 1+\epsilon] = [0.8, 1.2]
$$

policy loss：

$$
L_{\text{policy}}(\theta)
=
-
\mathbb{E}_t
\left[
\min
\left(
\rho_t A_t,\;
\mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t
\right)
\right]
$$

本例中 ratio 都在 `[0.8, 1.2]` 内，所以不会触发裁剪。

如果某一步 ratio 变成：

$$
\rho_t = 1.8
$$

说明新模型对这个 token 的概率提高太猛，PPO 会把它裁剪成：

$$
\mathrm{clip}(1.8, 0.8, 1.2) = 1.2
$$

这样就限制了一次更新的幅度。

### 17.10 Critic 怎么更新

Critic 的目标是学会更准确地预测 future return。

常见 return target：

$$
\mathrm{Return}_t = A_t + V_{\text{old}}(s_t)
$$

本例：

| state | old value | advantage | return target |
|---:|---:|---:|---:|
| `s_1` | 6.0 | 1.3591 | 7.3591 |
| `s_2` | 6.2 | 1.2039 | 7.4039 |
| `s_3` | 6.4 | 1.0365 | 7.4365 |
| `s_4` | 6.7 | 0.7894 | 7.4894 |

value loss：

$$
L_{\text{value}}(\psi)
=
\mathbb{E}_t
\left[
\left(
V_\psi(s_t) - \mathrm{Return}_t
\right)^2
\right]
$$

这个 loss 会让 critic 下次看到类似前缀时，value 预测更接近实际结果。

### 17.11 这一步到底更新了什么

本轮 PPO 更新会做两件事。

第一，更新 actor：

```text
因为这条回答 RM 分数高，advantage 为正，
所以提高 “植物 / 利用 / 光 / 制造能量” 这条路径的概率。
```

但因为有 clip 和 KL：

```text
概率只能小步提高，不能一下子暴涨。
```

第二，更新 critic：

```text
critic 原来觉得这些前缀大概 6 分多，
但实际 rollout + RM + KL 后发现可以到 7 分多，
所以 critic 的 value 预测要上调。
```

### 17.12 如果 RM 给低分会发生什么

如果 RM 认为这个回答很差：

$$
R_\phi(x,y)=2.0
$$

那么最后一步 reward 会变成低分，TD 误差会变负，GAE 往前传以后，多数 token 的 advantage 也会变负。

这时 PPO 会做相反更新：

```text
降低这条回答路径上 token 的概率。
```

也就是以后模型更不容易生成类似回答。

### 17.13 这个例子里 RM 的核心作用

RM 的作用可以总结为：

```text
RM 把完整回答变成一个可优化的标量 reward。
```

然后：

```text
TD 把 reward 和 critic 预期做比较；
GAE 把整条回答的好坏传回每个 token；
PPO 根据 advantage 调整 token 概率；
KL/reference 防止 actor 为了刷 RM 分跑偏。
```

所以一条完整链路是：

```text
prompt
-> actor 生成 response
-> RM 给完整 response 打分
-> reference 计算 KL 惩罚
-> critic 预测每个前缀的 value
-> TD/GAE 算每个 token 的 advantage
-> PPO loss 更新 actor
-> value loss 更新 critic
```
