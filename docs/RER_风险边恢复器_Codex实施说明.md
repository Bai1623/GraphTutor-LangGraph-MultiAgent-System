# 可信度引导的时空物理风险边恢复模块 RER：Codex 实施说明

> 目标：在当前 `MTC-RGAT-DDQN` 第二块框架中，只替换原来的固定加权恢复风险公式，不改动整体主流程、TrustMLP、风险校准、鲁棒邻域、双通道 RGAT 和 DDQN 的主线。
>
> 新模块名称建议：**RER: Risk Edge Recovery Network，风险边恢复网络**。
>
> 论文表述名称建议：**可信度引导的时空物理风险边恢复模块**。

---

## 1. 为什么要改这一块

原方案中恢复风险定义为：

\[
S_{ij}^{rec}
=
\omega_t M_{ij}
+
\omega_s S_{ij}^{spa}
+
\omega_k \bar{S}_{ij}^{kin}
\]

这个公式逻辑合理，但容易被理解成“历史项 + 空间项 + 物理项”的人工固定加权融合，方法创新感不足。

借鉴 V2X-INCOP 的思想：

```text
当前协同信息中断 / 缺失
        ↓
从历史协同信息中恢复当前缺失信息
        ↓
用干净/无中断信息作为 teacher 监督恢复模块
```

迁移到本文任务中：

```text
当前风险边 S̄_ij 受扰动影响而不可靠
        ↓
从历史风险边、空间相似风险边、物理动能风险中恢复当前风险边
        ↓
用干净风险场 S_clean 作为 teacher 监督恢复风险 S_rec 和校准风险 S_hat
```

因此，本模块不恢复 LiDAR 特征，不恢复原始车辆状态，而是恢复对 DDQN 决策最关键的车辆交互风险边。

---

## 2. 总体流程保持不变

原有总流程保持：

```text
受扰动风险 S̄_ij
    ↓
可信度 C_ij
    ↓
恢复风险 S_rec
    ↓
校准风险 S_hat
    ↓
鲁棒邻域 N_rob
    ↓
双通道 RGAT
    ↓
DDQN
```

只改中间这一块：

```text
历史风险记忆 M_ij + 空间风险补全 S_spa + 动能风险 S_kin
```

不要再直接固定加权，而是输入一个轻量 MLP 恢复器：

\[
S_{ij}^{rec}
=
f_{RER}
(
\mathcal{H}_{ij}^{t},
S_{ij}^{spa},
\bar{S}_{ij}^{kin},
C_{ij},
\Delta\bar{S}_{ij}
)
\]

其中：

\[
\mathcal{H}_{ij}^{t}
=
[
\bar{S}_{ij}(t-1),
\bar{S}_{ij}(t-2),
...,
\bar{S}_{ij}(t-L)
]
\]

表示风险边短时历史序列。

建议初版取：

```text
L = 3
```

理由：V2X-INCOP 中历史步长超过 3 后收益变小，而且引入更多历史帧会增加噪声和计算成本。本文是小核心创新点，不需要做太重。

---

## 3. 推荐最终公式

### 3.1 历史风险边编码

轻量版本使用 MLP，不建议初版用大 GRU 或 Transformer。

\[
h_{ij}^{his}
=
MLP_{his}
(
\mathcal{H}_{ij}^{t}
)
\]

其中：

```text
输入: [S_bar(t-1), S_bar(t-2), S_bar(t-3)]
输出: h_his，维度建议 16 或 32
```

### 3.2 当前风险跳变量

继续使用原方案里的风险跳变证据：

\[
\Delta\bar{S}_{ij}
=
|
\bar{S}_{ij}(t)-M_{ij}(t-1)
|
\]

含义：当前风险与历史记忆差距越大，说明当前风险边可能存在异常跳变。

### 3.3 轻量风险边恢复器

最终恢复公式改为：

\[
S_{ij}^{rec}
=
MLP_{rec}
([
h_{ij}^{his},
S_{ij}^{spa},
\bar{S}_{ij}^{kin},
C_{ij},
\Delta\bar{S}_{ij}
])
\]

为了保证输出范围稳定，建议输出层使用 Sigmoid：

\[
S_{ij}^{rec}\in[0,1]
\]

如果项目里风险值不是 0 到 1，需要先统一风险归一化。

---

## 4. 模块输入输出定义

### 4.1 输入

RER 每个时间步需要以下张量：

| 符号 | 代码变量建议 | 形状 | 含义 |
|---|---|---|---|
| \(\mathcal{H}_{ij}^{t}\) | `risk_hist` | `[N, N, L]` 或 `[B, N, N, L]` | 最近 L 帧风险边序列 |
| \(S_{ij}^{spa}\) | `S_spa` | `[N, N]` 或 `[B, N, N]` | 空间风险补全项 |
| \(\bar{S}_{ij}^{kin}\) | `S_kin` | `[N, N]` 或 `[B, N, N]` | 动能/物理风险 |
| \(C_{ij}\) | `C` | `[N, N]` 或 `[B, N, N]` | TrustMLP 输出的风险边可信度 |
| \(\Delta\bar{S}_{ij}\) | `delta_S` | `[N, N]` 或 `[B, N, N]` | 当前风险与历史记忆差异 |

### 4.2 输出

| 符号 | 代码变量建议 | 形状 | 含义 |
|---|---|---|---|
| \(S_{ij}^{rec}\) | `S_rec` | `[N, N]` 或 `[B, N, N]` | 恢复风险边 |

---

## 5. 推荐代码结构

建议新增文件：

```text
GRL_Simulation/risk/risk_edge_recovery.py
```

或如果项目已有 `risk/` 目录：

```text
risk/risk_edge_recovery.py
```

模块类名建议：

```python
class RiskEdgeRecoveryMLP(nn.Module):
    """Confidence-guided spatiotemporal physical risk-edge recovery module."""
```

---

## 6. PyTorch 参考实现

```python
import torch
import torch.nn as nn


class RiskEdgeRecoveryMLP(nn.Module):
    """
    轻量风险边恢复器。

    输入：
        risk_hist: [B, N, N, L] 或 [N, N, L]
        S_spa:     [B, N, N] 或 [N, N]
        S_kin:     [B, N, N] 或 [N, N]
        C:         [B, N, N] 或 [N, N]
        delta_S:   [B, N, N] 或 [N, N]

    输出：
        S_rec:     [B, N, N] 或 [N, N]
    """
    def __init__(self, hist_len=3, hist_hidden=16, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_len, hist_hidden),
            nn.ReLU(),
            nn.Linear(hist_hidden, hist_hidden),
            nn.ReLU(),
        )

        # [h_his, S_spa, S_kin, C, delta_S]
        rec_in_dim = hist_hidden + 4
        self.rec_mlp = nn.Sequential(
            nn.Linear(rec_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, risk_hist, S_spa, S_kin, C, delta_S):
        # risk_hist: [..., L]
        h_his = self.hist_encoder(risk_hist)

        # 统一扩展最后一维
        x = torch.cat([
            h_his,
            S_spa.unsqueeze(-1),
            S_kin.unsqueeze(-1),
            C.unsqueeze(-1),
            delta_S.unsqueeze(-1),
        ], dim=-1)

        S_rec = self.rec_mlp(x).squeeze(-1)
        return S_rec
```

---

## 7. 风险历史缓存实现

建议新增一个简单历史缓存，负责维护最近 L 帧 `S_bar`。

```python
class RiskHistoryBuffer:
    def __init__(self, num_vehicles, hist_len=3, device="cpu"):
        self.num_vehicles = num_vehicles
        self.hist_len = hist_len
        self.device = device
        self.buffer = torch.zeros(num_vehicles, num_vehicles, hist_len, device=device)
        self.initialized = False

    def update(self, S_bar):
        """
        S_bar: [N, N]
        当前帧进入历史缓存。缓存顺序建议：[..., 0] 是 t-1，[..., 1] 是 t-2。
        """
        if not self.initialized:
            for k in range(self.hist_len):
                self.buffer[..., k] = S_bar
            self.initialized = True
        else:
            self.buffer = torch.roll(self.buffer, shifts=1, dims=-1)
            self.buffer[..., 0] = S_bar

    def get(self):
        return self.buffer

    def reset(self):
        self.buffer.zero_()
        self.initialized = False
```

注意：

```text
在每个 episode 开始时必须 reset。
在每个 step 结束后 update(S_bar)。
如果当前 step 先计算 S_rec，则 get() 应该返回 t-1 到 t-L 的历史，不要把当前帧提前塞进去。
```

---

## 8. 与 TrustMLP 的前向流程衔接

原流程：

```python
C = trust_mlp(edge_evidence)
S_rec = omega_t * M + omega_s * S_spa + omega_k * S_kin
S_hat = C * S_bar + (1 - C) * S_rec
```

新流程：

```python
C = trust_mlp(edge_evidence)  # [N, N]

risk_hist = risk_history_buffer.get()  # [N, N, L]
delta_S = torch.abs(S_bar - M_prev)

# 建议默认 detach C，避免 RER 的 L_rec 反向把 C 推乱。
S_rec = risk_edge_recovery(
    risk_hist=risk_hist,
    S_spa=S_spa,
    S_kin=S_kin,
    C=C.detach(),
    delta_S=delta_S,
)

S_hat = C * S_bar + (1.0 - C) * S_rec
S_hat = torch.maximum(S_hat, rho * S_kin)

risk_history_buffer.update(S_bar.detach())
```

为什么 `C.detach()`：

```text
C 的主要含义是“当前风险边可信度”。
S_rec 的直接监督会逼近 S_clean，如果不 detach，RER 的恢复损失可能反向改变 C 的语义。
默认 detach 更稳定。
TrustMLP 通过 L_gate 和 L_cal 更新，RER 通过 L_rec 和 L_phy 更新。
```

后期如果训练稳定，可以做联合微调，把 `C.detach()` 改成 `C`。

---

## 9. 训练目标：结合 V2X-INCOP 摘要第四层

V2X-INCOP 的摘要第四层思想是：

```text
用知识蒸馏给预测模型提供显式监督；
用课程学习稳定训练并提升不同中断条件下的泛化能力。
```

迁移到本文：

```text
teacher: 干净风险场 S_clean
student: RER 输出的 S_rec 和校准后的 S_hat
curriculum: 扰动强度从轻到重逐步增加
```

---

## 10. 损失函数设计

### 10.1 恢复风险监督损失

\[
\mathcal{L}_{rec}
=
\sum_{i,j}
w_{ij}
(S_{ij}^{rec}-S_{ij}^{clean})^2
\]

作用：直接监督 RER，让它学会从历史、空间、物理、可信度证据中恢复接近 clean risk 的风险边。

### 10.2 校准风险损失

\[
\mathcal{L}_{cal}
=
\sum_{i,j}
w_{ij}
(\hat{S}_{ij}-S_{ij}^{clean})^2
\]

作用：保证最终进入后续图构建和 RGAT 的校准风险场接近 clean risk。

### 10.3 可信度软门控损失

构造软门控参考：

\[
C_{ij}^{ref}
=
\frac{
\exp(-|\bar{S}_{ij}-S_{ij}^{clean}|/\tau)
}{
\exp(-|\bar{S}_{ij}-S_{ij}^{clean}|/\tau)
+
\exp(-|S_{ij}^{rec}-S_{ij}^{clean}|/\tau)
}
\]

解释：

```text
如果当前风险 S_bar 更接近 S_clean，则 C_ref 接近 1；
如果恢复风险 S_rec 更接近 S_clean，则 C_ref 接近 0；
如果二者差不多，则 C_ref 接近 0.5。
```

门控损失：

\[
\mathcal{L}_{gate}
=
\sum_{i,j}
w_{ij}
(C_{ij}-C_{ij}^{ref})^2
\]

建议计算 `C_ref` 时使用 `S_rec.detach()`，避免门控标签反向影响 RER：

```python
err_cur = torch.abs(S_bar - S_clean)
err_rec = torch.abs(S_rec.detach() - S_clean)
score_cur = torch.exp(-err_cur / tau)
score_rec = torch.exp(-err_rec / tau)
C_ref = score_cur / (score_cur + score_rec + eps)
```

### 10.4 物理安全下界损失

\[
\mathcal{L}_{phy}
=
\sum_{i,j}
[
\max(0,\rho\bar{S}_{ij}^{kin}-S_{ij}^{rec})
]^2
\]

作用：如果动能/TTC 等物理风险已经很高，恢复风险不能被预测得过低。

### 10.5 时间平滑损失，可选

\[
\mathcal{L}_{smooth}
=
\sum_{i,j}
(S_{ij}^{rec}(t)-S_{ij}^{rec}(t-1))^2
\]

该项不建议一开始加得太大，防止把真实高风险突变也抹平。

建议：

```text
lambda_smooth = 0.01 或先不用
```

### 10.6 总损失

\[
\mathcal{L}
=
\lambda_{rec}\mathcal{L}_{rec}
+
\lambda_{cal}\mathcal{L}_{cal}
+
\lambda_{gate}\mathcal{L}_{gate}
+
\lambda_{phy}\mathcal{L}_{phy}
+
\lambda_{smooth}\mathcal{L}_{smooth}
\]

推荐初始权重：

```yaml
lambda_rec: 1.0
lambda_cal: 1.0
lambda_gate: 1.0
lambda_phy: 0.2
lambda_smooth: 0.0
```

风险边权重：

\[
w_{ij}=1+\mu S_{ij}^{clean}
\]

推荐：

```yaml
mu: 2.0
```

如果高风险边太少，可以加高风险额外权重：

```python
w = 1.0 + mu * S_clean + mu_high * (S_clean > high_risk_threshold).float()
```

推荐：

```yaml
high_risk_threshold: 0.6
mu_high: 1.0
```

---

## 11. 训练流程建议

### 11.1 阶段一：构造训练样本

每个样本需要保存：

```text
S_clean      # 干净状态计算出的风险场，teacher
S_bar        # 扰动状态计算出的风险场
S_kin        # 动能/物理风险子场
S_spa        # 空间风险补全项
M_prev       # 上一时刻历史风险记忆
risk_hist    # 最近 L 帧风险边序列
edge_evidence # TrustMLP 输入 e_ij
```

如果当前项目还没保存 `risk_hist`，需要在采样时增加历史缓存。

---

### 11.2 阶段二：warmup，可选但推荐

前几个 epoch 先让 RER 学会基本恢复，降低联合训练不稳定。

做法：

```text
1. TrustMLP 正常输出 C，但 RER 输入可以先使用 C.detach()。
2. 损失只使用 L_rec + lambda_phy L_phy。
3. warmup_epoch 建议 5~10。
```

也可以先不用 TrustMLP 的 C，而用一个简单启发式可信度：

\[
C_{ij}^{init}=\exp(-|\bar{S}_{ij}-M_{ij}|/\tau_d)
\]

但为了少改代码，推荐直接用 TrustMLP 输出并 detach。

---

### 11.3 阶段三：联合训练 TrustMLP + RER

前向过程：

```python
C = trust_mlp(edge_evidence)
S_rec = rer(risk_hist, S_spa, S_kin, C.detach(), delta_S)
S_hat = C * S_bar + (1 - C) * S_rec
S_hat = torch.maximum(S_hat, rho * S_kin)
```

损失：

```python
loss = (
    lambda_rec * L_rec
    + lambda_cal * L_cal
    + lambda_gate * L_gate
    + lambda_phy * L_phy
)
```

默认推荐：

```text
RER 由 L_rec、L_cal、L_phy 更新；
TrustMLP 由 L_gate、L_cal 更新；
C 输入 RER 时 detach，保持 C 的语义稳定。
```

---

### 11.4 阶段四：扰动强度渐进式训练

借鉴 V2X-INCOP 的 curriculum learning，不要一开始就给很强扰动。

推荐三阶段：

```yaml
curriculum:
  stage_1:
    epochs: 0-30
    dropout_prob: 0.05
    delay_steps: [0, 1]
    noise_level: low

  stage_2:
    epochs: 31-70
    dropout_prob: 0.10
    delay_steps: [1, 2]
    noise_level: medium

  stage_3:
    epochs: 71-100
    dropout_prob: 0.20
    delay_steps: [2, 3]
    noise_level: high
```

如果训练时间有限，可以简化为：

```text
前 30% epoch：轻扰动
中间 40% epoch：中扰动
最后 30% epoch：重扰动
```

---

## 12. 训练脚本伪代码

```python
for epoch in range(num_epochs):
    perturb_cfg = get_curriculum_perturbation(epoch)

    for batch in dataloader:
        S_clean = batch["S_clean"]
        S_bar = batch["S_bar"]
        S_kin = batch["S_kin"]
        S_spa = batch["S_spa"]
        M_prev = batch["M_prev"]
        risk_hist = batch["risk_hist"]
        edge_evidence = batch["edge_evidence"]

        C = trust_mlp(edge_evidence)
        delta_S = torch.abs(S_bar - M_prev)

        S_rec = rer(
            risk_hist=risk_hist,
            S_spa=S_spa,
            S_kin=S_kin,
            C=C.detach(),
            delta_S=delta_S,
        )

        S_hat = C * S_bar + (1.0 - C) * S_rec
        S_hat = torch.maximum(S_hat, rho * S_kin)

        # weights for high-risk edges
        w = 1.0 + mu * S_clean
        if use_high_risk_weight:
            w = w + mu_high * (S_clean > high_risk_threshold).float()

        L_rec = (w * (S_rec - S_clean).pow(2)).mean()
        L_cal = (w * (S_hat - S_clean).pow(2)).mean()

        err_cur = torch.abs(S_bar - S_clean)
        err_rec = torch.abs(S_rec.detach() - S_clean)
        score_cur = torch.exp(-err_cur / gate_temperature)
        score_rec = torch.exp(-err_rec / gate_temperature)
        C_ref = score_cur / (score_cur + score_rec + 1e-8)
        L_gate = (w * (C - C_ref).pow(2)).mean()

        L_phy = torch.relu(rho * S_kin - S_rec).pow(2).mean()

        if epoch < warmup_epochs:
            loss = L_rec + lambda_phy * L_phy
        else:
            loss = (
                lambda_rec * L_rec
                + lambda_cal * L_cal
                + lambda_gate * L_gate
                + lambda_phy * L_phy
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
```

---

## 13. 推理阶段流程

推理时没有：

```text
S_clean
C_ref
L_rec / L_cal / L_gate
```

只执行：

```python
C = trust_mlp(edge_evidence)
risk_hist = risk_history_buffer.get()
delta_S = torch.abs(S_bar - M_prev)

S_rec = rer(
    risk_hist=risk_hist,
    S_spa=S_spa,
    S_kin=S_kin,
    C=C.detach(),
    delta_S=delta_S,
)

S_hat = C * S_bar + (1.0 - C) * S_rec
S_hat = torch.maximum(S_hat, rho * S_kin)

risk_history_buffer.update(S_bar.detach())
```

然后继续进入：

```text
S_keep → TopK → N_rob → 双通道 RGAT → DDQN
```

---

## 14. 与原模块的替换位置

在原文档中，主要替换以下位置：

### 原第 9 节：时空-物理恢复风险

原公式：

\[
S_{ij}^{rec}
=
\omega_t M_{ij}
+
\omega_s S_{ij}^{spa}
+
\omega_k \bar{S}_{ij}^{kin}
\]

替换为：

\[
S_{ij}^{rec}
=
MLP_{rec}
([
h_{ij}^{his},
S_{ij}^{spa},
\bar{S}_{ij}^{kin},
C_{ij},
\Delta\bar{S}_{ij}
])
\]

其中：

\[
h_{ij}^{his}=MLP_{his}(\mathcal{H}_{ij}^{t})
\]

### 原第 11 节：可信度 MLP 训练方式

保留原来的：

```text
C_ij = σ(MLP(e_ij))
S_hat = C_ij S_bar + (1-C_ij) S_rec
L_cal + L_gate
```

新增：

```text
L_rec: 监督 RER 输出的 S_rec 接近 S_clean
L_phy: 约束 S_rec 不低于物理动能风险下界
curriculum learning: 扰动强度从轻到重
```

### 原第 16 节：训练流程

阶段一训练 TrustMLP 时，改成：

```text
阶段一：联合训练 TrustMLP + RER
1. 干净状态计算 S_clean
2. 扰动状态计算 S_bar
3. 构造 edge_evidence
4. TrustMLP 输出 C
5. RER 输出 S_rec
6. 得到 S_hat
7. 用 L_rec + L_cal + L_gate + L_phy 训练
```

---

## 15. 配置文件建议

在 yaml 中加入：

```yaml
risk_edge_recovery:
  enabled: true
  hist_len: 3
  hist_hidden: 16
  hidden_dim: 64
  dropout: 0.0
  use_c_detach: true

loss:
  lambda_rec: 1.0
  lambda_cal: 1.0
  lambda_gate: 1.0
  lambda_phy: 0.2
  lambda_smooth: 0.0
  mu_risk_weight: 2.0
  use_high_risk_weight: true
  high_risk_threshold: 0.6
  mu_high: 1.0
  gate_temperature: 0.1
  rho_phy: 0.8

training:
  warmup_epochs: 5
  curriculum: true
```

---

## 16. 消融实验建议

为了证明 RER 有效，至少做下面三个消融：

```text
1. FixedFusion：原固定加权版本
   S_rec = ω_t M + ω_s S_spa + ω_k S_kin

2. RER-no-C：去掉 C 输入
   S_rec = MLP([h_his, S_spa, S_kin, delta_S])

3. RER-full：完整版本
   S_rec = MLP([h_his, S_spa, S_kin, C, delta_S])
```

如果时间允许，再加：

```text
4. RER-no-phy：去掉物理下界损失 L_phy
5. RER-no-curriculum：不使用扰动强度渐进训练
```

评价指标：

```text
risk_mse
high_risk_mse
collision_rate
arrival_count
average_speed
robustness_drop_under_disturbance
```

重点展示：

```text
RER-full 在中重度扰动下 high_risk_mse 更低，碰撞率更低，arrival_count 更高。
```

---

## 17. 论文表述模板

可以这样写：

> 受中断感知协同感知中“利用历史协同信息恢复当前缺失信息”的思想启发，本文将该思想从感知特征层迁移到风险图决策层，提出可信度引导的时空物理风险边恢复模块。不同于直接恢复原始车辆状态，本文以车辆交互风险边作为恢复对象，将短时历史风险边序列、空间相似风险补全、动能物理风险以及风险边可信度共同作为输入，通过轻量 MLP 自适应生成恢复风险 \(S_{ij}^{rec}\)。训练阶段，利用仿真环境中可获得的干净风险场 \(S_{ij}^{clean}\) 作为 teacher signal，对恢复风险和校准风险进行显式监督；同时引入物理安全下界约束，避免高动能风险边被错误恢复为低风险。最终，恢复风险通过可信度门控与当前扰动风险自适应融合，生成校准风险场 \(\hat{S}_{ij}\)，用于后续鲁棒邻域构建和双通道 RGAT 决策传播。

---

## 18. 最终落地结论

本次修改只做一件事：

```text
把 S_rec 从固定加权公式改成轻量 RER-MLP。
```

保留：

```text
TrustMLP 输出 C_ij
S_hat = C S_bar + (1-C) S_rec
A_rob 构建
双通道 RGAT
DDQN
```

这样改的好处：

```text
1. 方法创新比固定融合更强；
2. 复杂度不高，适合硕士论文小核心创新；
3. 和 V2X-INCOP 的历史信息恢复思想能自然对应；
4. 和现有代码框架兼容，不需要大改整体结构；
5. 可以通过 risk_mse、high_risk_mse 和扰动强度实验清楚验证有效性。
```
