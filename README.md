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
2. **Off-diagonal 惩罚**：`λ * ||H - diag(H)||²` 使 H 趋向对角，加速收敛
3. **只监督能量**：不需要 H 的标签，损失函数只需 MSE(E_pred, E_true)
4. **简并处理**：训练时加微小扰动 `perturb=1e-6` 防止特征值简并导致梯度爆炸

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

## 长程相互作用实验（Coulomb 验证）

用纯 PyTorch 模型在一个受控数据集上验证 Kronecker 结构能否捕获长程相互作用。

### 实验设置

```
子系统 A（5个原子，σ=0.5Å）          子系统 B（5个原子，σ=0.5Å）
       ●●●●●          ←── 8 Å ───→          ●●●●●
   团簇内距离 ~1Å                         团簇内距离 ~1Å

GNN cutoff = 4 Å  →  A、B 之间 **零条边**
目标: E_total = E_AA + E_BB + E_AB   (1/r Coulomb)
```

- `LocalGNN`：MPNN 骨干 + 原子能量求和，无法"看到" E_AB（完全没有 A-B 边）
- `KronHamModel`：MPNN 骨干 + Kronecker 核，通过谱结构隐式编码 A-B 耦合
- `KronHamV2`：e3nn 等变骨干 + Kronecker 核

### 结果（100 epoch，1000 训练样本）

| 模型                      | 参数量  | Test MAE | vs LocalGNN |
|---------------------------|---------|----------|-------------|
| LocalGNN（无 Kronecker）  | 56,129  | 0.2139   | —           |
| **KronHamModel（纯 PyTorch）** | 74,666 | **0.0756** | **2.8x 更好** |
| KronHamV2（e3nn 骨干）    | 33,957  | 1.4641   | 更差        |

`std(E_AB) = 0.18`（LocalGNN 的理论下界，不能预测 E_AB 时的不可约误差）

### 结论

**① Kronecker 谱耦合捕获了长程 E_AB**

虽然 A-B 之间无任何 GNN 边，KronHamModel 依然把 MAE 从 0.21 降到 0.076：

```
ε_i^A 编码了子系统 A 的电荷分布谱
ε_j^B 编码了子系统 B 的电荷分布谱
ε_{ij} = ε_i^A + ε_j^B  ∝  q_A × q_B  ∝  E_AB
能量 MLP 从中学到了跨子系统的关联
```

**② e3nn 等变性对标量 Coulomb 任务没有帮助**

e3nn 版（KronHamV2）收敛很差，原因是架构上的输入特征不匹配：

| 模型 | 电荷输入 | 说明 |
|------|----------|------|
| 纯 PyTorch | `nn.Linear(1, hidden)` — 直接嵌入连续标量电荷 | ✓ 正确 |
| e3nn (model_v2) | `nn.Embedding(atom_types, dim)` — 仅整数原子类型 | ✗ 没有电荷信息 |

核心教训：**正确的物理输入比等变架构更重要**。给错误输入的 e3nn 比给正确输入的纯 PyTorch 差很多。等变性提供了归纳偏置（旋转对称），但无法弥补缺失的物理量（电荷数值）。

运行实验：

```bash
python train_coulomb.py   # 完整训练 + 对比（~10 分钟）
```

## 文件结构

```
kronecker_hamiltonian/
├── model.py            # 完整模型（5个部分）
├── model_v2.py         # 模型 v2（e3nn 等变骨干）
├── model_coulomb.py    # Coulomb demo（纯 PyTorch，无 e3nn）
├── train.py            # 训练脚本 + 使用示例
├── train_coulomb.py    # 长程交互对比实验
├── test_kronecker.py   # 测试套件（8项测试）
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```
