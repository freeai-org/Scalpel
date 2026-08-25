# PPO 从 0 开始：Ref=Actor 时，Actor 和 Critic 到底怎么更新参数

这份笔记只讲大模型 RLHF / RLAIF 里的 PPO，目标是把一轮训练从 0 拆开：

```text
第 0 步 actor 和 ref 是不是一样？
是的，一开始 actor = ref = SFT。

那 PPO 到底更新谁？
只更新 actor 和 critic。

ref 干什么？
ref 冻结不动，只当作 SFT 锚点，用来算 KL 惩罚。

RM 干什么？
RM 冻结不动，只给完整回答打 reward 分数。

critic 干什么？
critic 预测每个前缀未来能拿多少 reward，用来算 advantage，并用 value loss 更新自己。
```

一句话版本：

$$
\pi_{\theta_0} = \pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}
$$

PPO 开始后：

$$
\theta \leftarrow \theta - \eta_{\pi}\nabla_{\theta}L_{\mathrm{actor}}
$$

$$
\psi \leftarrow \psi - \eta_v\nabla_{\psi}L_{\mathrm{value}}
$$

其中：

```text
actor 参数 θ 会动；
critic 参数 ψ 会动；
ref 参数不动；
RM 参数不动。
```

---

## 1. PPO 的目的不是挑 prompt

大模型 PPO 的目标不是：

```text
从很多 prompt 里挑出最好的 prompt。
```

也不是：

```text
一个 prompt 生成很多回答，只把最高分那个回答拿去 SFT。
```

它真正优化的是：

$$
\max_{\theta}
\mathbb{E}_{x \sim \mathcal{D}_{\mathrm{prompt}},\; y \sim \pi_{\theta}(\cdot \mid x)}
\left[
R(x,y)
\right]
$$

含义是：

```text
从 prompt 数据集采样 x；
当前 actor 自己生成回答 y；
RM / verifier / 环境给这个回答打分；
根据 reward 调整 actor 的 token 概率；
让以后面对类似 prompt 时，更容易生成高 reward 的回答。
```

如果加上 ref 的 KL 约束，更贴近 RLHF-PPO 的目标是：

$$
\max_{\theta}
\mathbb{E}_{x \sim \mathcal{D}_{\mathrm{prompt}},\; y \sim \pi_{\theta}(\cdot \mid x)}
\left[
R_{\phi}(x,y)
-
\beta
\sum_{t=1}^{T}
\left(
\log \pi_{\theta}(y_t \mid s_t)
-
\log \pi_{\mathrm{ref}}(y_t \mid s_t)
\right)
\right]
$$

这里：

$$
s_t = (x, y_{<t})
$$

$$
y_t \sim \pi_{\theta}(\cdot \mid s_t)
$$

直觉是：

```text
RM 负责拉着模型往高质量回答走；
ref 负责拉着模型别离 SFT 太远；
PPO 负责把整条回答的 reward 分摊到每个 token 的概率更新上。
```

---

## 2. 大模型生成怎么对应强化学习

强化学习里通常写：

$$
a_t \sim \pi_{\theta}(\cdot \mid s_t)
$$

大模型生成里：

$$
s_t = (x, y_{<t})
$$

$$
a_t = y_t
$$

也就是：

| RL 概念 | 大模型 PPO 里的含义 |
|---|---|
| state $s_t$ | prompt + 已经生成的回答前缀 |
| action $a_t$ | 当前生成的 token |
| policy $\pi_{\theta}$ | actor 大模型的 next-token 分布 |
| trajectory $\tau$ | 一整条 response |
| reward | RM / verifier / 环境给的分数，加上 KL 惩罚 |
| value $V_{\psi}(s_t)$ | critic 预测当前前缀未来总 reward |

举个小例子：

```text
prompt x:
高速上前车突然急刹，自动驾驶系统应该怎么做？

actor 生成 response y:
减速，保持车距，必要时刹停。
```

可以拆成 token 级动作：

| 时间步 | state $s_t$ | action $y_t$ |
|---:|---|---|
| 1 | prompt | `减速` |
| 2 | prompt + `减速` | `保持车距` |
| 3 | prompt + `减速，保持车距` | `必要时刹停` |

PPO 关心的是：

```text
这条回答最后 reward 高不高？
如果高，哪些 token 的概率应该提高？
如果低，哪些 token 的概率应该降低？
这个提高/降低不能太猛，否则会偏离 ref。
```

---

## 3. 从 0 开始：PPO 前已经有什么

在 PPO 真正开始之前，通常已经有这些东西。

| 对象 | 符号 | 怎么来 | PPO 期间是否更新 | 作用 |
|---|---|---|---|---|
| SFT 模型 | $\pi_{\mathrm{SFT}}$ | 用监督数据训练 | 不直接训练 | actor/ref 初始化来源 |
| Actor | $\pi_{\theta}$ | 复制 SFT | 更新 | 生成回答，最终要部署 |
| Ref | $\pi_{\mathrm{ref}}$ | 复制同一个 SFT | 冻结 | 算 KL，防止跑偏 |
| RM | $R_{\phi}$ | 偏好数据训练，或加载 HF 上现成 RM | 冻结 | 给完整回答打分 |
| Critic | $V_{\psi}$ | 常用 SFT/RM backbone + value head | 更新 | 预测每个前缀的未来 reward |

第 0 步最重要的等式：

$$
\pi_{\theta_0} = \pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}
$$

第一批 rollout 时，还有一个 old actor：

$$
\pi_{\theta_{\mathrm{old}}} = \pi_{\theta_0}
$$

所以第 0 轮第一批数据里：

$$
\pi_{\theta_{\mathrm{old}}}
=
\pi_{\theta_0}
=
\pi_{\mathrm{ref}}
=
\pi_{\mathrm{SFT}}
$$

但是它们的命运不一样：

```text
actor：后面会被 PPO 更新。
old actor：只是本轮 rollout 的快照，通常保存 old_logprob 就够了。
ref：永远冻结在 PPO 开始前的 SFT。
RM：冻结，只打分。
critic：会被 value loss 更新。
```

第一轮更新之后：

$$
\pi_{\theta_1} \neq \pi_{\mathrm{ref}}
$$

从这时开始，KL 惩罚才真正体现出“别离原 SFT 太远”的作用。

---

## 4. Critic 默认是什么样子的

critic 不是 RM。

RM 是：

$$
R_{\phi}(x,y)
$$

输入完整 prompt + 完整 response，输出一个整体分数。

critic 是：

$$
V_{\psi}(s_t)
$$

输入 prompt + 当前 response 前缀，输出一个 value 标量。

常见大模型 critic 结构：

```text
Transformer backbone
    +
value head
```

对每个 token 位置，取 hidden state：

$$
h_t = \mathrm{Transformer}_{\psi}(x, y_{\le t})
$$

接一个线性头：

$$
V_{\psi}(s_t) = w_v^{\top}h_t + b_v
$$

所以 critic 的输出形状通常可以理解成：

```text
每个 response token 位置一个 value 标量。
```

工程里常见三种实现：

| 实现 | 长什么样 | 怎么更新 |
|---|---|---|
| actor 和 critic 两套模型 | 最容易理解 | actor loss 只更新 actor，value loss 只更新 critic |
| actor 和 critic 共享 backbone | 省显存 | 共享层同时吃 policy loss 和 value loss |
| actor 模型上挂 value head | TRL 常见风格 | LM head 做 actor，value head 做 critic |

为了先把逻辑看清楚，下面默认：

```text
actor 参数叫 θ。
critic 参数叫 ψ。
二者先当成两套网络理解。
```

critic 的任务不是判断回答好不好，而是预测：

```text
已经生成到这个前缀了，后面继续生成到结束，大概能拿多少总 reward？
```

它一开始通常不准，需要在 PPO 中被训练。

---

## 5. 一轮 PPO 的完整流程

假设现在是第 $k$ 轮 PPO iteration。

先固定本轮采样用的 actor：

$$
\theta_{\mathrm{old}} \leftarrow \theta_k
$$

这一轮分两大段：

```text
rollout：生成数据，不更新参数。
update：用这批数据更新 actor 和 critic。
```

---

## 6. Rollout 阶段：不更新参数，只收集训练材料

### 6.1 采样 prompt

从 prompt 数据集采样：

$$
x \sim \mathcal{D}_{\mathrm{prompt}}
$$

例如：

```text
x = 高速上前车突然急刹，自动驾驶系统应该怎么做？
```

### 6.2 old actor 生成回答

用当前 actor 生成回答。因为这是本轮采样时的 actor，所以也叫 old actor：

$$
y_t \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t)
$$

假设生成：

```text
y = 减速，保持车距，必要时刹停。
```

### 6.3 保存 old logprob

对每个已经生成出来的 token，记录生成当时的 logprob：

$$
\ell_t^{\mathrm{old}}
=
\log \pi_{\theta_{\mathrm{old}}}(y_t \mid s_t)
$$

这些 old logprob 后面会作为 PPO ratio 的分母。

重要点：

```text
old_logprob 是常量。
update 阶段不会对 old_logprob 反传。
```

### 6.4 critic 给每个前缀打 old value

critic 对每个 state 预测 value：

$$
v_t^{\mathrm{old}} = V_{\psi_k}(s_t)
$$

它表示：

```text
critic 在 rollout 当时认为：
从当前前缀继续生成到结束，未来大概能拿多少总 reward。
```

这些 old value 后面用来算 TD error 和 GAE。

### 6.5 ref 给同一批 token 算 ref logprob

ref 不生成新回答。

ref 只是看 actor 已经生成好的 token，然后计算：

$$
\ell_t^{\mathrm{ref}}
=
\log \pi_{\mathrm{ref}}(y_t \mid s_t)
$$

也就是：

```text
同一个 token，在冻结 SFT 模型看来概率是多少？
```

如果第 0 轮刚开始：

$$
\pi_{\theta_{\mathrm{old}}} = \pi_{\mathrm{ref}}
$$

所以：

$$
\ell_t^{\mathrm{old}} = \ell_t^{\mathrm{ref}}
$$

### 6.6 RM 给完整回答打分

RM 输入完整 prompt-response：

$$
r_{\mathrm{RM}} = R_{\phi}(x,y)
$$

例如：

```text
RM(prompt, response) = 1.0
```

RM 在 PPO 里一般不更新。

它只是给一个标量分数。

真正让 actor 参数改变的是后面的 policy gradient。

---

## 7. 构造 token-level reward

RM 给的是整条回答的分数，但 PPO 要更新每个 token 的概率。

所以通常会构造每个 token 的 reward。

先算 KL 惩罚：

$$
r_t^{\mathrm{KL}}
=
-
\beta
\left(
\ell_t^{\mathrm{old}}
-
\ell_t^{\mathrm{ref}}
\right)
$$

然后把 RM 分数加在最后一个 token 上：

$$
r_t = r_t^{\mathrm{KL}}, \quad t < T
$$

$$
r_T = r_T^{\mathrm{KL}} + r_{\mathrm{RM}}
$$

所以：

```text
每个 token 都有 KL 惩罚；
完整回答的 RM 分数一般放在最后；
TD/GAE 再把最后的好坏往前传到前面的 token。
```

第 0 轮刚开始时：

$$
\ell_t^{\mathrm{old}} = \ell_t^{\mathrm{ref}}
$$

所以：

$$
r_t^{\mathrm{KL}} = 0
$$

这时 reward 主要来自 RM。

等 actor 更新过一次以后：

$$
\ell_t^{\mathrm{old}} \neq \ell_t^{\mathrm{ref}}
$$

KL 惩罚就开始约束 actor。

---

## 8. TD error：现实比 critic 预期好多少

critic 在 rollout 时给了旧预测：

$$
v_t^{\mathrm{old}} = V_{\psi_k}(s_t)
$$

但现在我们已经拿到了 token reward：

$$
r_t
$$

也知道下一步 state 的 value：

$$
v_{t+1}^{\mathrm{old}}
$$

TD error 定义为：

$$
\delta_t
=
r_t
+
\gamma v_{t+1}^{\mathrm{old}}
-
v_t^{\mathrm{old}}
$$

最后一步：

$$
v_{T+1}^{\mathrm{old}} = 0
$$

直觉：

```text
critic 原本以为 s_t 的价值是 v_t。
现在看到了一步 reward r_t，又看到下一步价值 v_{t+1}。
所以新的局部估计是 r_t + γ v_{t+1}。
两者相减，就是 critic 这一步猜错了多少。
```

如果：

$$
\delta_t > 0
$$

说明：

```text
实际比 critic 原先预期更好。
这个 state/action 后续表现比想象中强。
```

如果：

$$
\delta_t < 0
$$

说明：

```text
实际比 critic 原先预期更差。
这个 state/action 后续表现不如预期。
```

---

## 9. GAE：把未来的 TD error 往前累计

单步 TD error 只看一步。

但一个 token 的好坏，可能要等后面几个 token 才体现出来。

所以 PPO 常用 GAE：

$$
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}
$$

边界：

$$
A_T = \delta_T
$$

展开就是：

$$
A_t
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
$$

这里 $A_t$ 叫 advantage。

它回答的问题是：

```text
在 state s_t 选择 token y_t，
比 critic 原来预期的平均水平好多少？
```

如果：

$$
A_t > 0
$$

actor 应该提高这个 token 在这个上下文下的概率。

如果：

$$
A_t < 0
$$

actor 应该降低这个 token 在这个上下文下的概率。

注意这里不是直接用 RM 分数更新每个 token。

更准确地说：

```text
RM 给整条回答一个终局信号；
critic 给每个前缀一个 baseline；
TD error 比较现实和 baseline；
GAE 把后面的好坏往前传播；
advantage 决定每个 token 概率该升还是降。
```

---

## 10. Return：critic 的训练目标

有了 advantage 以后，可以构造 critic 的 target return：

$$
G_t = A_t + v_t^{\mathrm{old}}
$$

也可以理解成：

```text
old value 加上这次实际看到的修正量。
```

critic 训练目标是让当前 value 接近这个 target：

$$
L_{\mathrm{value}}(\psi)
=
\frac{1}{2}
\mathbb{E}_t
\left[
\left(
V_{\psi}(s_t) - G_t
\right)^2
\right]
$$

更新 critic：

$$
\psi
\leftarrow
\psi
-
\eta_v
\nabla_{\psi}
L_{\mathrm{value}}(\psi)
$$

对单个 token 来说，梯度方向很直观：

$$
\nabla_{\psi}
\frac{1}{2}
\left(
V_{\psi}(s_t)-G_t
\right)^2
=
\left(
V_{\psi}(s_t)-G_t
\right)
\nabla_{\psi}V_{\psi}(s_t)
$$

如果：

$$
V_{\psi}(s_t) < G_t
$$

说明 critic 预测低了，更新会把 value 往上推。

如果：

$$
V_{\psi}(s_t) > G_t
$$

说明 critic 预测高了，更新会把 value 往下拉。

---

## 11. Actor 的 PPO loss：让好 token 更可能、坏 token 更不可能

update 阶段会重新用当前 actor 计算同一批 token 的 logprob：

$$
\ell_t^{\mathrm{new}}(\theta)
=
\log \pi_{\theta}(y_t \mid s_t)
$$

然后算 PPO ratio：

$$
\rho_t(\theta)
=
\frac{
\pi_{\theta}(y_t \mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(y_t \mid s_t)
}
=
\exp
\left(
\ell_t^{\mathrm{new}}(\theta)
-
\ell_t^{\mathrm{old}}
\right)
$$

这个 ratio 表示：

```text
同一个 token，在当前 actor 下的概率，
相比 rollout 当时的 old actor，变成了多少倍。
```

PPO clipped objective：

$$
L_{\mathrm{actor}}(\theta)
=
-
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\mathrm{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)
A_t
\right)
\right]
$$

为什么前面有负号？

因为优化器默认最小化 loss。

PPO 原本想最大化：

$$
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\mathrm{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)
A_t
\right)
\right]
$$

所以训练代码里常写成负号。

actor 更新：

$$
\theta
\leftarrow
\theta
-
\eta_{\pi}
\nabla_{\theta}
L_{\mathrm{actor}}(\theta)
$$

如果先不看 clip，对单个 token：

$$
L_t(\theta) = -\rho_t(\theta)A_t
$$

它的梯度近似可以理解成：

$$
\nabla_{\theta}L_t
\approx
-
A_t
\nabla_{\theta}
\log \pi_{\theta}(y_t \mid s_t)
$$

所以：

```text
A_t > 0：
最小化 loss 会提高 log πθ(y_t | s_t)，也就是提高这个 token 的概率。

A_t < 0：
最小化 loss 会降低 log πθ(y_t | s_t)，也就是降低这个 token 的概率。
```

clip 的作用是：

```text
如果概率已经被提高太多，别再继续猛推；
如果概率已经被压低太多，别再继续猛压。
```

也就是 PPO 里的 proximal：

```text
每次别离 old actor 太远。
```

---

## 12. 一个从 0 开始的完整数值例子

为了看清楚参数怎么更新，我们用一个极简例子。

prompt：

```text
高速上前车突然急刹，自动驾驶系统应该怎么做？
```

第 0 步：

$$
\pi_{\theta_0} = \pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}
$$

本轮 rollout 时：

$$
\pi_{\theta_{\mathrm{old}}} = \pi_{\theta_0}
$$

actor 生成 3 个简化 token：

| t | token $y_t$ |
|---:|---|
| 1 | `减速` |
| 2 | `保持车距` |
| 3 | `必要时刹停` |

### 12.1 记录 old logprob 和 ref logprob

因为这是第 0 轮第一批数据，actor/ref/old actor 完全一样。

| t | token | $\ell_t^{\mathrm{old}}$ | $\ell_t^{\mathrm{ref}}$ |
|---:|---|---:|---:|
| 1 | `减速` | -1.20 | -1.20 |
| 2 | `保持车距` | -0.90 | -0.90 |
| 3 | `必要时刹停` | -1.50 | -1.50 |

所以 KL reward：

$$
r_t^{\mathrm{KL}}
=
-
\beta
\left(
\ell_t^{\mathrm{old}}-\ell_t^{\mathrm{ref}}
\right)
=0
$$

### 12.2 critic 给 old value

假设 rollout 时 critic 输出：

| t | state | $v_t^{\mathrm{old}}$ |
|---:|---|---:|
| 1 | prompt | 0.20 |
| 2 | prompt + `减速` | 0.25 |
| 3 | prompt + `减速，保持车距` | 0.10 |
| 4 | 结束 | 0 |

这些 value 表示 critic 当时的预期：

```text
在每个前缀继续生成到结束，大概能拿多少总 reward。
```

### 12.3 RM 给完整回答打分

RM 看完整回答：

```text
高速上前车突然急刹，自动驾驶系统应该怎么做？
减速，保持车距，必要时刹停。
```

假设 RM 输出：

$$
r_{\mathrm{RM}}=1.00
$$

第 0 轮 KL 为 0，所以 token reward 是：

| t | token | $r_t^{\mathrm{KL}}$ | RM reward | 总 $r_t$ |
|---:|---|---:|---:|---:|
| 1 | `减速` | 0 | 0 | 0 |
| 2 | `保持车距` | 0 | 0 | 0 |
| 3 | `必要时刹停` | 0 | 1.00 | 1.00 |

也就是：

$$
r_1=0,\quad r_2=0,\quad r_3=1.00
$$

### 12.4 算 TD error

取：

$$
\gamma=1
$$

TD error：

$$
\delta_t = r_t + \gamma v_{t+1}^{\mathrm{old}} - v_t^{\mathrm{old}}
$$

第 3 步：

$$
\delta_3
=
1.00 + 0 - 0.10
=
0.90
$$

第 2 步：

$$
\delta_2
=
0 + 0.10 - 0.25
=
-0.15
$$

第 1 步：

$$
\delta_1
=
0 + 0.25 - 0.20
=
0.05
$$

解释：

```text
第 3 步明显比 critic 预期好，因为最后 RM 给了 1.00。
第 2 步单看一步是负的，因为 critic 原本估计 0.25，但下一步只剩 0.10。
第 1 步略正，因为从 0.20 走到了下一步 0.25。
```

### 12.5 用 GAE 往前传

取：

$$
\lambda=0.95
$$

从后往前算：

$$
A_3 = \delta_3 = 0.90
$$

$$
A_2
=
\delta_2 + \gamma\lambda A_3
=
-0.15 + 1 \times 0.95 \times 0.90
=
0.705
$$

$$
A_1
=
\delta_1 + \gamma\lambda A_2
=
0.05 + 1 \times 0.95 \times 0.705
=
0.71975
$$

得到：

| t | token | $A_t$ |
|---:|---|---:|
| 1 | `减速` | 0.71975 |
| 2 | `保持车距` | 0.705 |
| 3 | `必要时刹停` | 0.90 |

虽然第 2 步单步 TD error 是负的，但 GAE 后它仍然是正的。

原因是：

```text
第 2 步后面接到了一个高 reward 的结尾；
GAE 把未来好的结果往前传了一部分。
```

所以这三个 token 在这条轨迹里都应该被提高概率。

### 12.6 构造 critic 的 target return

$$
G_t = A_t + v_t^{\mathrm{old}}
$$

所以：

| t | token | $A_t$ | $v_t^{\mathrm{old}}$ | $G_t$ |
|---:|---|---:|---:|---:|
| 1 | `减速` | 0.71975 | 0.20 | 0.91975 |
| 2 | `保持车距` | 0.705 | 0.25 | 0.955 |
| 3 | `必要时刹停` | 0.90 | 0.10 | 1.00 |

critic 的训练目标就是：

```text
以后看到这些前缀时，value 不要再预测成 0.20、0.25、0.10；
应该更接近 0.91975、0.955、1.00。
```

critic loss：

$$
L_{\mathrm{value}}
=
\frac{1}{2}
\left[
(V_{\psi}(s_1)-0.91975)^2
+
(V_{\psi}(s_2)-0.955)^2
+
(V_{\psi}(s_3)-1.00)^2
\right]
$$

然后更新：

$$
\psi
\leftarrow
\psi
-
\eta_v
\nabla_{\psi}L_{\mathrm{value}}
$$

这会把 critic 对这几个前缀的 value 往上推。

### 12.7 actor 第一次更新

第 0 轮第一次 update 刚开始时，current actor 还没变：

$$
\ell_t^{\mathrm{new}} = \ell_t^{\mathrm{old}}
$$

所以 ratio：

$$
\rho_t
=
\exp(\ell_t^{\mathrm{new}}-\ell_t^{\mathrm{old}})
=
1
$$

PPO actor objective 的每个 token 项是：

$$
\rho_t A_t = A_t
$$

因为三个 $A_t$ 都是正的，所以梯度方向是：

```text
提高 `减速` 在 prompt 后面的概率；
提高 `保持车距` 在 prompt + `减速` 后面的概率；
提高 `必要时刹停` 在 prompt + `减速，保持车距` 后面的概率。
```

用公式看，不考虑 clip 时：

$$
L_t(\theta) = -\rho_t(\theta)A_t
$$

第 0 次 update 初始 $\rho_t=1$，所以近似：

$$
\nabla_{\theta}L_t
\approx
-
A_t
\nabla_{\theta}
\log \pi_{\theta}(y_t \mid s_t)
$$

因为优化器做的是：

$$
\theta
\leftarrow
\theta
-
\eta_{\pi}
\nabla_{\theta}L_t
$$

代进去直觉就是：

$$
\theta
\leftarrow
\theta
+
\eta_{\pi}
A_t
\nabla_{\theta}
\log \pi_{\theta}(y_t \mid s_t)
$$

如果 $A_t>0$，就沿着提高这个 token logprob 的方向更新。

假设更新后，同一批 token 的 logprob 变成：

| t | token | 更新前 old logprob | 更新后 actor logprob |
|---:|---|---:|---:|
| 1 | `减速` | -1.20 | -1.12 |
| 2 | `保持车距` | -0.90 | -0.83 |
| 3 | `必要时刹停` | -1.50 | -1.35 |

logprob 变大，代表概率变大。

比如：

$$
\exp(-1.20)=0.301
$$

$$
\exp(-1.12)=0.326
$$

所以 `减速` 的概率被提高了。

### 12.8 clip 怎么限制 actor 不要一步走太远

设：

$$
\epsilon=0.2
$$

clip 范围是：

$$
[1-\epsilon,1+\epsilon] = [0.8,1.2]
$$

如果某个正 advantage token 的概率被提高到原来的 1.5 倍：

$$
\rho_t=1.5,\quad A_t>0
$$

那 PPO 会用：

$$
\mathrm{clip}(\rho_t,0.8,1.2)=1.2
$$

防止继续把它推得太猛。

如果某个负 advantage token 的概率被压到原来的 0.5 倍：

$$
\rho_t=0.5,\quad A_t<0
$$

PPO 也会 clip，防止压得太猛。

所以 PPO 不是无限相信 RM 的打分。

它会说：

```text
这批样本告诉我应该往这个方向动；
但是每一轮只许动一点点。
```

---

## 13. 第二轮开始：actor 已经不等于 ref

第一轮 actor 更新后：

$$
\pi_{\theta_1} \neq \pi_{\mathrm{ref}}
$$

第二轮 rollout 时：

$$
\pi_{\theta_{\mathrm{old}}} = \pi_{\theta_1}
$$

但 ref 仍然是最初的 SFT：

$$
\pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}
$$

假设第二轮又生成了类似 token，old logprob 和 ref logprob 是：

| t | token | $\ell_t^{\mathrm{old}}$ | $\ell_t^{\mathrm{ref}}$ |
|---:|---|---:|---:|
| 1 | `减速` | -1.12 | -1.20 |
| 2 | `保持车距` | -0.83 | -0.90 |
| 3 | `必要时刹停` | -1.35 | -1.50 |

取：

$$
\beta=0.1
$$

KL reward：

$$
r_t^{\mathrm{KL}}
=
-
0.1
\left(
\ell_t^{\mathrm{old}}
-
\ell_t^{\mathrm{ref}}
\right)
$$

于是：

| t | token | $\ell_t^{\mathrm{old}}-\ell_t^{\mathrm{ref}}$ | $r_t^{\mathrm{KL}}$ |
|---:|---|---:|---:|
| 1 | `减速` | 0.08 | -0.008 |
| 2 | `保持车距` | 0.07 | -0.007 |
| 3 | `必要时刹停` | 0.15 | -0.015 |

如果 RM 仍然给：

$$
r_{\mathrm{RM}}=1.00
$$

那么：

$$
r_1=-0.008
$$

$$
r_2=-0.007
$$

$$
r_3=1.00-0.015=0.985
$$

这就是 ref 的作用：

```text
actor 确实因为高 reward 想提高这些 token 概率；
但它越偏离初始 SFT，KL 惩罚越大；
最终 PPO 会在 reward 和 KL 之间折中。
```

---

## 14. Actor 和 Critic 的更新顺序到底是什么

一轮实际训练通常像这样：

```text
1. 用当前 actor 生成一批 response。
2. 记录 old_logprob。
3. 用 ref 计算同一批 token 的 ref_logprob。
4. 用 critic 记录 old value。
5. 用 RM 给完整 response 打分。
6. 构造 token-level reward。
7. 用 reward 和 old value 算 TD error。
8. 用 TD error 反向累计 GAE，得到 advantage。
9. 用 advantage 和 old_logprob 更新 actor。
10. 用 return target 更新 critic。
11. 这批 rollout 可以被切成 mini-batch，重复 PPO epoch 若干次。
12. 下一轮重新用更新后的 actor 采样新 response。
```

其中参数更新只有两类：

actor：

$$
\theta
\leftarrow
\theta
-
\eta_{\pi}
\nabla_{\theta}
L_{\mathrm{actor}}(\theta)
$$

critic：

$$
\psi
\leftarrow
\psi
-
\eta_v
\nabla_{\psi}
L_{\mathrm{value}}(\psi)
$$

如果 actor 和 critic 共享 backbone，代码里可能会把 loss 加在一起：

$$
L_{\mathrm{total}}
=
L_{\mathrm{actor}}
+
c_v L_{\mathrm{value}}
-
c_{\mathrm{ent}}H(\pi_{\theta})
$$

这里 $H(\pi_{\theta})$ 是 entropy bonus，用来鼓励一定探索。

但逻辑仍然是：

```text
actor 部分根据 advantage 更新 token 概率；
critic 部分根据 return target 更新 value 预测；
ref/RM 不更新。
```

---

## 15. 为什么不能只更新 actor，不要 critic

理论上可以用整条 reward 直接做 REINFORCE：

$$
\nabla_{\theta}J(\theta)
\approx
R(x,y)
\sum_{t=1}^{T}
\nabla_{\theta}
\log \pi_{\theta}(y_t \mid s_t)
$$

但这样方差很大。

一条回答得分高，可能不是每个 token 都好。

一条回答得分低，也可能前面做得对，后面才崩。

critic 提供 baseline：

$$
A_t = G_t - V_{\psi}(s_t)
$$

意思是：

```text
不是看绝对 reward 高不高；
而是看这个 token 之后的结果，比 critic 对这个前缀的预期好不好。
```

这会让训练更稳定。

critic 越准，advantage 越像“这个 token 真正该背多少功劳或责任”。

---

## 16. RM 在里面到底怎么发挥作用

RM 不直接给 actor 反传梯度。

RM 的作用链条是：

```text
完整回答
  -> RM 打分 r_RM
  -> 和 KL 一起构造 token reward r_t
  -> TD error δ_t
  -> GAE advantage A_t
  -> actor loss 更新 token logprob
  -> value loss 更新 critic
```

也就是：

$$
R_{\phi}(x,y)
\rightarrow
r_t
\rightarrow
\delta_t
\rightarrow
A_t
\rightarrow
L_{\mathrm{actor}}
$$

以及：

$$
R_{\phi}(x,y)
\rightarrow
r_t
\rightarrow
\delta_t
\rightarrow
A_t
\rightarrow
G_t
\rightarrow
L_{\mathrm{value}}
$$

所以 RM 是 reward 来源。

actor 真正更新靠的是：

$$
\nabla_{\theta}
\log \pi_{\theta}(y_t \mid s_t)
$$

critic 真正更新靠的是：

$$
\nabla_{\psi}
V_{\psi}(s_t)
$$

---

## 17. 最容易混的几个点

### 17.1 ref 和 actor 一开始确实一样

是的。

第 0 步：

$$
\pi_{\theta_0} = \pi_{\mathrm{ref}}
$$

区别只是：

```text
actor 后面会训练；
ref 从头到尾冻结。
```

### 17.2 old actor 不是 ref

第 0 轮第一批时，old actor 和 ref 数值一样。

但概念上不同：

```text
old actor：本轮数据是哪个 policy 采样出来的。
ref：PPO 开始前的 SFT 锚点。
```

第 5 轮时：

$$
\pi_{\theta_{\mathrm{old}}} = \pi_{\theta_5}
$$

但：

$$
\pi_{\mathrm{ref}} = \pi_{\theta_0}
$$

### 17.3 PPO 是在线还是离线

大模型 PPO 通常是 on-policy 或近似 on-policy。

原因是：

```text
response 是当前 actor 自己采样出来的；
更新几步之后，下一轮要重新用新的 actor 采样。
```

但一批 rollout 内部会被复用多个 PPO epoch。

所以更准确地说：

```text
采样层面是 on-policy；
mini-batch 更新阶段短时间复用 old rollout。
```

### 17.4 critic 不是 reward model

RM：

```text
看完整回答，输出最终质量分。
```

critic：

```text
看当前前缀，预测未来总 reward。
```

RM 可以来自 HF 现成模型。

critic 通常需要跟着 PPO 一起训练。

### 17.5 actor loss 和 value loss 更新的不是同一个东西

actor loss 更新：

$$
\pi_{\theta}(y_t \mid s_t)
$$

也就是 token 概率分布。

value loss 更新：

$$
V_{\psi}(s_t)
$$

也就是前缀价值预测。

---

## 18. 一句话总图

完整 PPO-RLHF 可以压缩成这条链：

```text
SFT 模型复制两份：
actor = SFT
ref   = SFT

actor 生成回答；
ref 算 KL；
RM 给完整回答打分；
critic 给每个前缀估 value；
reward + value 算 TD error；
TD error 通过 GAE 得到 advantage；
advantage 决定 actor 对每个 token 概率升还是降；
return target 决定 critic 的 value 预测往上还是往下；
actor 和 critic 更新；
ref 和 RM 冻结；
下一轮用新的 actor 重新采样。
```

最核心的两个更新公式：

$$
\boxed{
\theta
\leftarrow
\theta
-
\eta_{\pi}
\nabla_{\theta}
\left[
-
\mathbb{E}_t
\min
\left(
\rho_t(\theta)A_t,
\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
}
$$

$$
\boxed{
\psi
\leftarrow
\psi
-
\eta_v
\nabla_{\psi}
\left[
\frac{1}{2}
\mathbb{E}_t
\left(
V_{\psi}(s_t)-G_t
\right)^2
\right]
}
$$

其中：

$$
\rho_t(\theta)
=
\exp
\left(
\log \pi_{\theta}(y_t \mid s_t)
-
\log \pi_{\theta_{\mathrm{old}}}(y_t \mid s_t)
\right)
$$

$$
\delta_t
=
r_t
+
\gamma v_{t+1}^{\mathrm{old}}
-
v_t^{\mathrm{old}}
$$

$$
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}
$$

$$
G_t
=
A_t
+
v_t^{\mathrm{old}}
$$

记住这句就够了：

```text
actor 学“这个 token 在这个上下文里以后该更容易还是更不容易出现”；
critic 学“这个前缀未来大概能拿多少 reward”；
ref 负责提醒 actor 别离最初 SFT 太远；
RM 负责告诉这整条回答最终好不好。
```
