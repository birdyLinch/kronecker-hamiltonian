# Kronecker Hamiltonian Model

端到端等变分子 Hamiltonian 预测模型，支持快速对角化。

## 核心优势：无 PBC 的长程相互作用

传统 k 空间方法（如平面波 DFT）依赖周期性边界条件（PBC）来处理长程相互作用。模拟非周期体系（孤立分子、界面、缺陷）时，需要引入大真空盒避免周期像之间的虚假相互作用，造成巨大的计算浪费。

本模型通过 Kronecker 结构天然绕开了这一限制：

```
子系统 A 的原子 ──┐                   ┌── 子系统 B 的原子
                  ▼                   ▼
            同一个 GNN（共享权重）
                  │                   │
            a_k 向量 [dim_A]     b_k 向量 [dim_B]
                  └──────┬────────────┘
                         ▼
              H = D + Σ_k (a_k a_k^T) ⊗ I  +  I ⊗ Σ_k (b_k b_k^T)
                         │
                   全局特征值 ε_{ij} = ε_i^A + ε_j^B
```

- **长程耦合**：A 和 B 的低秩向量由同一 GNN 产生，共享的 K 个潜在模式隐式编码了跨子系统的长程关联，无需显式列出所有原子对
- **无需 PBC**：不依赖 Fourier 变换，直接在实空间构建 Hamiltonian，适用于任意非周期体系
- **无需大真空盒**：k 空间方法需要盒子足够大使周期像不相互作用；本方法只需定义两个子系统，无额外开销
- **计算高效**：对角化复杂度从暴力的 O((dim_A · dim_B)³) 降至 O(dim_A³ + dim_B³)

## 架构总览

```
原子坐标 + 原子类型
    │
    ▼
┌─────────────────────────────────────────┐
│  Part 1: 等变消息传递 (e3nn)             │
│  node_irreps = "16x0e + 8x1o + 4x2e"   │
│  每层: h_i += Σ_j  TP(h_j, Y(r_ij))    │
└──────────────────┬──────────────────────┘
                   │ node_feat [N, irreps.dim]
         ┌─────────┴─────────┐
         ▼                   ▼
┌────────────────┐  ┌─────────────────────┐
│  Part 2:       │  │  Part 3:            │
│  低秩向量头    │  │  对角局域项头       │
│  u_k ∈ irreps  │  │  onsite energy      │
│  [N, K, bdim]  │  │  [N, basis_dim]     │
└───────┬────────┘  └──────────┬──────────┘
        └──────────┬───────────┘
                   ▼
┌──────────────────────────────────────────┐
│  Part 4: Kronecker H 构建                │
│                                          │
│  H = D + Σ_k (u_k ⊗ u_k^T)             │
│                                          │
│  D: block-diagonal onsite项              │
│  低秩项: rank-K 更新                     │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  Part 5: 对角化                          │
│  full   → torch.linalg.eigh  O(N³)      │
│  lobpcg → 最低k个            O(N·k²)    │
│  lanczos→ Krylov方法         O(N·k²)    │
└──────────────────┬───────────────────────┘
                   ▼
        ε, ψ → 能量预测头 → E_pred
```

## 快速开始

```python
from model import KroneckerHamiltonianModel

config = {
    'atom_types':        10,
    'atom_embed_dim':    32,
    'node_irreps':       "16x0e + 8x1o + 4x2e",
    'edge_sh_lmax':      2,
    'n_mp_layers':       3,
    'K':                 4,
    'vector_irreps':     "1x0e + 1x1o",   # basis_dim = 4
    'basis_dim':         4,
    'hamiltonian_mode':  'sum_outer',
    'diag_mode':         'full',
    'diag_k':            10,
    'cutoff':            5.0,
}

model = KroneckerHamiltonianModel(config)
out = model(pos, atom_types)

out['energy']       # 预测能量（标量）
out['eigenvalues']  # 特征值
out['eigenvectors'] # 特征向量（波函数）
out['H']            # Hamiltonian 矩阵
```

## vector_irreps 配置指南

| vector_irreps          | basis_dim | 物理意义              | H 对称性    |
|------------------------|-----------|-----------------------|-------------|
| `"1x0e"`               | 1         | 纯标量轨道            | 完全对称    |
| `"1x1o"`               | 3         | p 型轨道              | SO(3) 等变  |
| `"1x0e + 1x1o"`        | 4         | s + p 型              | SO(3) 等变  |
| `"1x0e + 1x1o + 1x2e"` | 9         | s + p + d 型          | SO(3) 等变  |

basis_dim 必须等于 vector_irreps.dim。

## 对角化模式选择

| 分子大小  | 推荐模式  | 说明                        |
|-----------|-----------|-----------------------------|
| N < 50    | `full`    | torch.linalg.eigh，稳定     |
| N < 500   | `lobpcg`  | 最低 k 个，O(N·k²)          |
| N > 500   | `lanczos` | Krylov，内存友好            |

## 加速效果（Kronecker vs 暴力对角化）

H_full = H_A ⊗ I_B + I_A ⊗ H_B，复杂度对比：Kronecker O(d³+d³) vs 暴力 O(d⁶)。

以 `basis_dim=4` 为例（Apple M 系列 CPU，单精度）：

| atoms/subsystem | H_A 尺寸 | H_full 尺寸   | 暴力 (ms) | Kronecker (ms) | 加速  |
|-----------------|----------|---------------|-----------|----------------|-------|
| 2               | 8×8      | 64×64         | 0.15      | 0.02           | 8x    |
| 5               | 20×20    | 400×400       | 7.43      | 0.05           | 143x  |
| 10              | 40×40    | 1600×1600     | 214.86    | 0.32           | 669x  |
| 20              | 80×80    | 6400×6400     | 不可行    | 1.06           | —     |
| 50              | 200×200  | 40000×40000   | OOM       | 6.84           | —     |
| 100             | 400×400  | 160000×160000 | OOM       | 21.28          | —     |

10 个原子/子系统时暴力需 **215ms/step**，Kronecker 只需 **0.32ms**（**669x**）。20 个原子以上暴力完全不可行。

## 训练技巧

1. **梯度裁剪**：对角化梯度可能很大，建议 `clip_grad_norm_(max_norm=1.0)`
2. ~~**Off-diagonal 惩罚**~~：**不适用于 Kronecker 模型**。低秩向量 v_k⊗v_k^T 的非对角项正是跨子系统耦合机制的载体，惩罚它等于杀掉 Kronecker 效果。
3. **只监督能量**：不需要 H 的标签，损失函数只需 MSE(E_pred, E_true)
4. **简并处理**：训练时加微小扰动 `perturb=1e-6` 防止特征值简并导致梯度爆炸。注意不要用更大的值（如 1e-3）——过大的扰动会使训练样本的谱指纹过于独特，导致记忆化和过拟合。
5. **Huber 损失**（δ=0.1）：比 MAE 更好。MAE 在零点梯度为零，会导致 KronHamCore 训练出现"梯度停滞 plateau"；Huber 在零附近是二次的，消除了这一现象。
6. **对特征值不加 LayerNorm**：特征值的数值量级本身携带物理信息（E_AB ∝ q_A×q_B/r），归一化会抹除这一信号，实验确认 MAE 从 0.046 退化为 0.056。
7. **早停**：KronHamModel 在 1000 个训练样本下约在 ep80 达到最优，ep100 后轻微过拟合，建议监控验证集并保存最佳检查点。

## 环境安装

```bash
# 创建并激活 conda 环境
conda create -n KronHam python=3.11 -y
conda activate KronHam

# 安装依赖
pip install -r requirements.txt
```

## 运行测试

```bash
conda activate KronHam
python test_kronecker.py
```

## Long-Range Interaction Experiment (Coulomb Validation)

A controlled benchmark proving that the Kronecker spectral coupling mechanism captures
long-range interactions **with zero A–B edges** in the GNN graph.

### Setup

```
Subsystem A  (n=5 atoms, σ=0.5 Å)        Subsystem B  (n=5 atoms, σ=0.5 Å)
        ● ● ● ● ●      ←── sep=8 Å ───→      ● ● ● ● ●
     intra-cluster ~1 Å                      intra-cluster ~1 Å

GNN cutoff = 4 Å  →  zero A–B edges in message-passing graph
Target: E_total = E_AA + E_BB + E_AB    (pairwise 1/r Coulomb, random charges q ∈ [−1,1])
```

Each atom carries a continuous scalar charge. The inter-cluster energy is:

```
E_AB = Σ_{i∈A, j∈B}  q_i · q_j / |r_i − r_j|
```

A standard GNN with cutoff=4 Å **cannot see any A–B pair** — it must predict E_AB
blindly from intra-cluster structure alone, which is impossible.

### Models

| Model | Backbone | Kronecker | Notes |
|-------|----------|-----------|-------|
| `LocalGNN` | ScalarMPNN | ✗ | Baseline — atom energy sum, no A–B coupling |
| `KronHamModel` | ScalarMPNN | ✓ | Scalar features, Kronecker spectral head |
| `KronHamModelE3NN` | FlexEquivMP (e3nn) | ✓ | Equivariant backbone, currently L=0 only |

### Results  (100 epochs · 1000 training samples · Huber loss δ=0.1)

```
std(E_AB, test) = 0.1803   ←  irreducible error floor for any model that cannot see E_AB
```

| Model | Params | Test MAE | vs LocalGNN |
|-------|--------|----------|-------------|
| LocalGNN (no Kronecker) | 56,129 | 0.1859 | — |
| **KronHamModel (scalar + Kronecker)** | **74,666** | **0.0459** | **4.1× better** |
| KronHamModelE3NN (e3nn L=0 + Kronecker) | 54,762 | 0.1364 | 1.4× better |

KronHamModel is **4× better** than LocalGNN and breaks well below the `std(E_AB)` floor,
confirming it genuinely predicts the long-range term.

### Why Kronecker works: spectral encoding of E_AB

```
KronHamCore builds:  H = diag(d) + Σ_k  v_k ⊗ v_k^T
where d, v_k come from the node features of subsystem A or B independently.

Eigenvalues of H:  ε_{ij} = ε_i^A + ε_j^B

Because d and v_k encode the charge distribution of each subsystem,
the global eigenvalue spectrum encodes cross-subsystem correlations:

    ε_i^A  ∝  charge geometry of A
    ε_j^B  ∝  charge geometry of B
    ε_{ij} = ε_i^A + ε_j^B  ∝  q_A × q_B  ∝  E_AB

The energy MLP reads 3·k_states+4 spectral features and recovers E_AB
without any A–B edges.
```

### Spectral features (`KronHamCore._spectral_features`)

| Feature | Shape | Description |
|---------|-------|-------------|
| `evals_global[:k]` | [k] | k smallest eigenvalues of combined A+B spectrum |
| `evals_A[:k]` | [k] | k smallest eigenvalues of subsystem A alone |
| `evals_B[:k]` | [k] | k smallest eigenvalues of subsystem B alone |
| `gap.min()` | scalar | min(ε_i^A − ε_j^B) — cross-subsystem spectral gap |
| `gap.max()` | scalar | max(ε_i^A − ε_j^B) |
| `gap.abs().mean()` | scalar | mean absolute cross-gap |
| `gap.std().clamp(1e-8)` | scalar | spectral spread — clamped to avoid NaN at degeneracy |

**Total feat_dim = 3·k_states + 4** (default k_states=8 → 28 features)

### Architecture details

#### ScalarMPNN (backbone of `KronHamModel`)

```python
# Distance encoding:  GaussianRBF(20) × cosine_envelope(r, cutoff)
# Message:  h_i ← MLP(cat(h_i,  Σ_j  rbf_feat(d_ij) · h_j))
# Residual connection at each layer
```

Why RBF matters: mapping raw distance (1 scalar) → TP weights via `fc([1,32,32,256])`
is a very hard regression. GaussianRBF(20) provides smooth distance basis functions that
let the FC network learn weight profiles easily.
**Without RBF: e3nn L=0 MAE ≈ 0.77; with RBF: 0.14 (5× improvement).**

#### FlexEquivMP (backbone of `KronHamModelE3NN`)

```python
# TP weights:  fc(GaussianRBF(n_rbf=20, r_max=cutoff) × cosine_envelope)
# Message:  Linear(Σ_j  TP(h_j, Y_l(r̂_ij), W(d_ij))) + h_i
# node_irreps = "16x0e"  (L=0 sanity-check mode)
```

Note: L=0 equivariant MP = linear aggregation + residual, which is strictly weaker than
ScalarMPNN's nonlinear self+message mixing. This is why e3nn L=0 lags 3× behind the
scalar model despite carrying identical information.

### Hard-won training lessons

#### 1. Huber loss (δ=0.1) — not MAE or MSE

| Loss | Problem | Symptom |
|------|---------|---------|
| MSE | Over-weights outliers | Slow convergence on long-tailed E_AB |
| MAE (L1) | Zero gradient at minimum | KronHamCore plateau: loss stuck at 0.16 for 20+ epochs |
| **Huber (δ=0.1)** | **None** | **Smooth quadratic near zero, linear in tails** |

#### 2. Per-group AdamW learning rates (equivariant models)

```python
optimizer = AdamW([
    {'params': scalar_and_kron_params,  'lr': lr * 1.0},   # scalar head, KronHamCore
    {'params': e3linear_params,         'lr': lr * 0.5},   # E3nn Linear layers
    {'params': tp_fc_params,            'lr': lr * 0.3},   # TP weight networks
], weight_decay=1e-4)
```

Without per-group LRs, the scalar head trains much faster than the TP-fc networks,
causing L>0 channels to never align before LR decays → non-monotone test MAE.

#### 3. Linear warmup → cosine annealing

```python
# 10-epoch linear warmup: start_factor=0.1 → 1.0
# Then CosineAnnealingLR(T_max=n_epochs-10, eta_min=lr/50)
```

Warmup gives all parameter groups (especially TP-fc) time to orient before LR decays.

#### 4. Do NOT apply LayerNorm to eigenvalues

Eigenvalue magnitudes directly encode E_AB scale (∝ charge products / distance).
Normalizing destroys this physical signal.

**Ablation result:**
```
With LayerNorm on spectral features:   MAE = 0.056  (regression)
Without LayerNorm (current):           MAE = 0.046  (best)
```

#### 5. gap.std() is useful — clamp, don't remove

The spectral spread `gap.std()` encodes how spread the eigenvalue cross-differences are,
which correlates with charge heterogeneity. Removing it causes regression.
However `∂std/∂x_i = (x_i − mean)/(n·std)` → NaN when std→0 (degenerate eigenvalues).

**Fix:** `.clamp(min=1e-8)` — safe, preserves gradient signal.

```
Remove gap.std() entirely:         MAE = 0.082  (regression)
Keep gap.std().clamp(min=1e-8):    MAE = 0.046  (best)
```

#### 6. perturb=1e-6 — do not increase

`KronHamCore` adds `perturb * randn` to matrix diagonal before eigvalsh to avoid
degenerate eigenvalue NaN. 1e-6 is enough to prevent NaN while being small enough
that it does not change the eigenvalue fingerprint of each training sample.

**What happens with perturb=1e-3:**
Eigenvalues become more spread and distinguishable → network memorizes train-sample
fingerprints → overfitting (test MAE rises after ep80, ep100 worse than ep80).

#### 7. Ablation: what does NOT work for KronHamModel

| Idea | Result | Why |
|------|--------|-----|
| Off-diagonal penalty `λ‖H−diag(H)‖²` | Kills Kronecker | v_k⊗v_k^T non-diagonal terms ARE the coupling |
| LayerNorm on spectral features | +22% MAE | Erases physical eigenvalue scale |
| perturb=1e-3 | Overfitting | Over-stabilizes spectrum → memorization |
| Remove gap.std() | +79% MAE | Spectral spread is a useful feature |
| MSE loss | Slow, noisy | Heavy tail in E_AB distribution |
| MAE loss | Plateau | Zero gradient near minimum in KronHamCore |

### e3nn L=0 ablation: commit-by-commit story

Both `KronHamModel` and `KronHamModelE3NN` share the same `KronHamCore`. Every change
to KronHamCore affects both. This table traces what each commit did to e3nn L=0 MAE:

| Commit | What changed | e3nn L=0 MAE | Δ | Note |
|--------|-------------|:------------:|:---:|------|
| (before `4471e6a`) | raw dist (1 scalar) → TP weights | ~0.77 | — | FC maps 1 number → 256 weights |
| `4471e6a` | ✅ GaussianRBF(20) + cosine envelope + per-group AdamW + warmup | **0.136** | **−82%** | Single biggest improvement |
| `10a89a1` | ⚠️ +eval_norm +perturb=1e-3; gap.std→clamp(1e-8) | 0.143 | +5% | LayerNorm hurts |
| `5065313` | ✅ −eval_norm (reverted) | **0.129** | −10% | **Best e3nn run** |
| `479b5f4` | — perturb 1e-3→1e-6 | ~0.129 | ~0% | perturb has no effect on e3nn |
| `b251592` | ❌ −gap.std() (removed entirely) | 0.221 | **+71%** | Largest single regression |
| `38c9559` | ✅ +gap.std().clamp(1e-8) (restored) | 0.136 | −39% | Recovered to baseline |

#### Lessons specific to e3nn L=0

**1. GaussianRBF(20) is the single biggest win (−82%)**

The TP weight network `fc([n_in, 32, 32, n_weights])` must produce smooth radial
profiles. With raw distance (1 scalar), the FC has to invent smooth functions from
a point input — nearly impossible. With RBF(20), the inputs already span the radial
frequency range; the FC just needs a linear combination.

```
raw dist (1 input):    MAE = 0.77
GaussianRBF(20):       MAE = 0.136   ← 5.7× improvement
```

**2. LayerNorm on eigenvalues consistently hurts (both models)**

Eigenvalue magnitudes encode E_AB scale (∝ q_A × q_B / r). Normalising erases this.
The energy MLP then has to infer the absolute energy scale from normalised ratios alone —
it can learn the shape of E_AB but not its magnitude.

```
With LayerNorm:    MAE = 0.143   (+5%)
Without:           MAE = 0.129   (best)
```

**3. gap.std() removal is catastrophic for e3nn (even more than for scalar)**

`gap.std()` = spread of cross-subsystem eigenvalue differences = how heterogeneous
charges are across A and B. This is the single most informative scalar summary of E_AB.

```
With gap.std().clamp(1e-8):   MAE = 0.129
Without gap.std():             MAE = 0.221   (+71% regression)
```

Fix: `.clamp(min=1e-8)` prevents `∂std/∂x → 0/0` at degeneracy while preserving
the informative signal everywhere else.

**4. perturb has no effect on e3nn L=0 (unlike scalar)**

`perturb=1e-3` caused the scalar model to overfit (stronger backbone memorised the
more-distinguishable eigenvalue fingerprints). e3nn L=0's weaker linear backbone
cannot memorise regardless — so perturb size is irrelevant for it.

```
perturb=1e-3:   e3nn MAE = 0.129,  scalar MAE = 0.073 (overfitting)
perturb=1e-6:   e3nn MAE = 0.129,  scalar MAE = 0.046 (clean)
```

This asymmetry reveals that model capacity determines whether perturb matters:
weak model (e3nn L=0) → immune to memorisation → perturb irrelevant.
Strong model (ScalarMPNN) → can memorise → perturb must stay small.

### Closing the e3nn gap: self-interaction + channel width + MACE-style fc + EMA

A systematic ablation revealing and addressing the three root causes of the 3× gap.

#### Step 1 — Nonlinearity: self-interaction ablation (16x0e, uvw)

Added `self_interaction` parameter to `FlexEquivMP`:

| Variant | Self-interaction | MAE | Δ vs linear |
|---------|-----------------|-----|------------|
| `none` | `h ← Linear(aggr) + h` (linear only) | 0.1364 | — |
| `nequip` | `h_L0 ← SiLU(Linear(h_L0_combined))` | 0.1239 | −9% |
| `scalar_mix` | `h_L0 ← h_L0 + MLP(cat(h_self, h_aggr))` | 0.1208 | −11% |

**Finding:** nonlinearity helps ~10%, but the gap to scalar (0.046) remains 2.6×.
Conclusion: nonlinearity is not the bottleneck alone.

#### Step 2 — Channel width: 64x0e + uvu

The real bottleneck: during 3 MP layers, ScalarMPNN uses 64 hidden dims while
FlexEquivMP uses only 16. Moving to 64x0e requires switching TP mode:

| TP mode | Weight count | Constraint |
|---------|-------------|-----------|
| `uvw` | mul_in × mul_out = 64² = 4096 | None (too expensive) |
| `uvu` | mul_in = 64 | mul_in == mul_out (per-channel gate) |

`uvu` for L=0 reduces to per-channel gating (`out[u] = w_u(r) × h[u]`) —
architecturally identical to the scalar model's `gate ⊙ h_j`, but with unconstrained `w`.

| Model | ep80 MAE | ep100 MAE | Overfit? |
|-------|----------|-----------|---------|
| 16x0e ScalarMix | 0.121 | 0.121 | no |
| 64x0e uvu ScalarMix (no fixes) | **0.083** | 0.113 | **yes** |

64x0e hits 0.083 at ep80 (2× gap, down from 3×), but then overfits. Train loss
collapses to 0.0002 while test MAE rises.

#### Step 3 — MACE-style fc: LayerNorm inside radial MLP

Root cause of overfitting: the TP weight predictor (`fc`) uses `Linear → SiLU` without
normalization. As channel width grows (16→64), the TP weight scale becomes
poorly conditioned, causing the bilinear coupling to blow up.

**Fix (from MACE `RadialMLP`):** `Linear → LayerNorm → SiLU` per hidden layer.

```python
# Before (FullyConnectedNet from e3nn)
fc = Linear(20,32) → SiLU → Linear(32,32) → SiLU → Linear(32, weight_numel)

# After (MACE-style)
fc = Linear(20,32) → LayerNorm(32) → SiLU → Linear(32,32) → LayerNorm(32) → SiLU → Linear(32, weight_numel)
```

| Model | ep80 MAE | ep100 MAE | train@100 | Overfit? |
|-------|----------|-----------|-----------|---------|
| no LayerNorm | 0.083 | 0.113 | 0.0002 | yes |
| + LayerNorm | 0.102 | **0.092** | 0.0003 | no |

LayerNorm fixes overfitting but convergence is slower (0.092 vs 0.083 best).

#### Step 4 — EMA: Exponential Moving Average of weights (MACE/NequIP style)

Standard in MACE and NequIP: maintain a shadow EMA copy of model weights
(`ema_w ← 0.99 × ema_w + 0.01 × w`) and evaluate on those averaged weights.
EMA smooths over the co-adaptation transients in the bilinear TP coupling.

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
# after each step: ema_model.update_parameters(model)
# evaluate on: ema_model
```

| Model | ep80 | ep100 | train@100 |
|-------|------|-------|-----------|
| + LayerNorm, no EMA | 0.102 | 0.092 | 0.0003 |
| + LayerNorm + EMA | 0.097 | **0.085** | 0.0011 |

EMA improves ep100 by 8% and the model is **still converging** at ep100 (not plateaued).

#### Current best results (100 epochs, 1000 training samples)

| Model | Params | MAE | vs LocalGNN |
|-------|--------|-----|------------|
| LocalGNN (no Kronecker) | 56,129 | 0.1859 | — |
| **KronHamModel (scalar + Kronecker)** | **74,666** | **0.0459** | **4.1×** |
| e3nn 16x0e, linear | 54,762 | 0.1364 | 1.4× |
| e3nn 16x0e, NequIP SiLU | 55,578 | 0.1239 | 1.5× |
| e3nn 16x0e, ScalarMix | 57,162 | 0.1208 | 1.5× |
| e3nn 64x0e, uvu, ScalarMix | 90,666 | 0.1130 | 1.6× |
| e3nn 64x0e + MACE fc + EMA | 90,666 | **0.0852** | **2.2×** |
| **e3nn 64x0e, NormSAGE (L=0 only, equiv to scalar_mix)** | **90,666** | **0.0477** | **3.9×** |
| e3nn 64x0e + sigmoid gate + MLP update (`e3nn-scalarmpnn`) | 90,666 | TBD | TBD |

Gap reduced from 3.0× → 1.9× (still converging at ep100).

`e3nn-scalarmpnn` is the architectural equivalence test: L=0 e3nn with sigmoid-bounded
TP weights and `h ← h + MLP(cat(h, aggr))` update — making it structurally identical
to ScalarMPNN. If the gap closes to ~1×, it proves the gap was entirely caused by the
two design choices (unbounded weights + linear update), not e3nn's framework overhead.

`e3nn-normsage` is the invariant NormSAGE self-interaction variant: in the L=0-only
configuration used here it is architecturally equivalent to `scalar_mix` and empirically
achieves MAE ≈ 0.0477 on the Coulomb benchmark (3.9× better than LocalGNN, essentially
matching the scalar Kronecker model).

---

### Why e3nn is hard to train: the bilinear coupling problem

This is the fundamental reason e3nn models need careful engineering:

```
Scalar model message:
  gate = sigmoid(MLP(rbf))    # bounded ∈ (0,1) — one learned quantity
  msg  = gate ⊙ h_j           # gate × features

e3nn model message (uvu, L=0):
  w   = fc(rbf)               # unbounded — jointly learned
  msg = w ⊙ h_j               # two simultaneously-optimized quantities × each other
```

The message is **bilinear** in two co-trained quantities `w(θ_fc)` and `h(θ_model)`.
Their gradients are coupled:

```
∂L/∂θ_fc ∝ ∂L/∂aggr · h_j        ← conditioned on current h
∂L/∂h_j  ∝ ∂L/∂aggr · w(r_ij)   ← conditioned on current w
```

If `h_j` is miscalibrated early in training → bad gradient to `fc`.
If `fc` produces bad `w` → corrupted updates to `h_j`.
Both fight each other → non-monotone MAE curves, slow convergence, overfitting.

**Three consequences observed:**
1. Non-monotone MAE early (ep40 > ep20) — co-adaptation transients
2. Overfitting at 64x0e without LayerNorm — `w` scale blows up as mul grows
3. EMA helps — averages over the bilinear co-adaptation oscillations

**What MACE does to tame bilinear coupling:**
- `LayerNorm` inside `fc` → keeps `w` well-conditioned regardless of `h` magnitude
- `EMA (decay=0.99)` → smooths over transients
- Bessel basis (vs Gaussian RBF) → orthogonal, smoother gradients for `fc`
- `ReduceLROnPlateau` → backs off when coupling oscillates

**Scalar model avoids this entirely:** `sigmoid` bounds the gate to (0,1), breaking
the unbounded feedback loop. This is a structural advantage of scalar architectures
for purely invariant tasks.

### Step 5 — Architectural equivalence: sigmoid gate + MLP update (`e3nn-scalarmpnn`)

The root-cause analysis predicts the L=0 e3nn gap is 100% caused by two design choices:

| Design choice | e3nn default | ScalarMPNN |
|--------------|-------------|-----------|
| TP weight bound | unbounded `fc(rbf)` | `sigmoid(fc(rbf))` ∈ (0,1) |
| Node update | `Linear(aggr) + h` | `h + MLP(cat(h_self, aggr))` |

`e3nn-scalarmpnn` fixes both simultaneously:

```python
# In FlexEquivMP.forward:
tp_weights = torch.sigmoid(self.fc(rbf_feat))        # bounded gate, like ScalarMPNN
msg        = self.tp(node_feat[src], edge_sh, tp_weights)
aggr       = scatter_add(msg, dst)
h_new      = h + self_net(cat(h, aggr))              # MLP update, no Linear(aggr)
```

For a `64x0e` model this is architecturally identical to ScalarMPNN:
- `Y⁰₀(r̂) = const` → direction ignored, pure distance model
- Sigmoid gate ∈ (0,1) → same bounded message as ScalarMPNN
- `h + MLP(cat(h, aggr))` → exact same update function

**Hypothesis:** `e3nn-scalarmpnn` should train like `scalar` (MAE ~0.05), closing
the gap entirely. Residual differences (LN inside fc, Y⁰₀ scale factor, `inv_norm →
to_hidden` projection before KronHamCore) should not affect trainability.

### Run the experiment

```bash
python train_coulomb.py                                    # all models (full comparison)
python train_coulomb.py --models e3nn-64                   # just the 64x0e variant
python train_coulomb.py --models scalar e3nn-64            # compare scalar vs best e3nn
python train_coulomb.py --models scalar e3nn-scalarmpnn    # equivalence test ← new
python train_coulomb.py --models e3nn-64 --epochs 200      # longer run
```

Available model keys: `local`, `scalar`, `e3nn-none`, `e3nn-nequip`, `e3nn-scalarmix`, `e3nn-64`, `e3nn-scalarmpnn`

## 文件结构

```
kronecker_hamiltonian/
├── model.py            # 完整模型（5个部分，e3nn 等变骨干）
├── model_v2.py         # 模型 v2（早期 e3nn 版本，已弃用）
├── model_coulomb.py    # Coulomb 实验模型（LocalGNN / KronHamModel / KronHamModelE3NN）
│                       #   · ScalarMPNN: GaussianRBF(20) + cosine envelope 距离编码
│                       #   · FlexEquivMP: e3nn TP，相同 RBF 权重生成
│                       #   · KronHamCore: H=diag+Σ v_k⊗v_k^T，3k+4 谱特征
├── train.py            # 训练脚本 + 使用示例
├── train_coulomb.py    # 长程交互对比实验（含 Huber 损失 / 分组 AdamW / warmup）
├── test_kronecker.py   # 测试套件（8项测试）
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```

### Key git commits (Coulomb experiment)

```
(HEAD)   Add e3nn-scalarmpnn: sigmoid gate + MLP update (architectural equivalence test)
87a33b4  MACE-style fc LayerNorm + EMA + argparse CLI: close e3nn gap to 1.9×
b6c07e1  Add self-interaction ablation to FlexEquivMP (none/nequip/scalar_mix)
38c9559  Restore gap.std() with clamp — confirmed useful spectral feature
b251592  Remove gap.std() from spectral features — NaN at degeneracy        [regression]
479b5f4  Revert perturb 1e-3 → 1e-6: larger value causes overfitting
5065313  Revert eval_norm: LayerNorm on eigenvalues erases physical scale   [regression]
10a89a1  Fix three KronHamCore numerical bugs + trim to L=0 sanity check
4471e6a  Fix FlexEquivMP + optimizer for equivariant Coulomb experiment     ← scalar best (MAE=0.046)
```
