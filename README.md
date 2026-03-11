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

### Open question: e3nn L=0 vs scalar gap

```
KronHamModel (ScalarMPNN):         MAE = 0.0459   ← 1.0×
KronHamModelE3NN (FlexEquivMP L=0):MAE = 0.1364   ← 3.0× worse
```

Both models pass the same information (scalar charges, intra-cluster geometry) into the
same KronHamCore. The gap comes purely from the message-passing architecture:

| Architecture | Aggregation | Self-interaction |
|---|---|---|
| ScalarMPNN | MLP(cat(h_i, Σ_j w(d_ij) h_j)) | Nonlinear mixing of self + message |
| FlexEquivMP L=0 | Linear(Σ_j TP_L0(h_j, w(d_ij))) + h_i | Linear aggregation + residual |

L=0 equivariant = linear operation; scalar MPNN = nonlinear. Same information,
weaker expressiveness → 3× MAE gap.

**Next step:** close this gap (nonlinear mixing in FlexEquivMP) before adding L>0 channels.
Adding L>0 with a weak backbone just adds noise.

### Run the experiment

```bash
python train_coulomb.py   # full training + comparison (~3 minutes on CPU)
```

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
38c9559  Restore gap.std() with clamp — confirmed useful spectral feature
b251592  Remove gap.std() from spectral features — NaN at degeneracy        [regression]
479b5f4  Revert perturb 1e-3 → 1e-6: larger value causes overfitting
5065313  Revert eval_norm: LayerNorm on eigenvalues erases physical scale   [regression]
10a89a1  Fix three KronHamCore numerical bugs + trim to L=0 sanity check
4471e6a  Fix FlexEquivMP + optimizer for equivariant Coulomb experiment     ← best result (MAE=0.046)
```
