# Kronecker Hamiltonian Model

端到端等变分子 Hamiltonian 预测模型，支持快速对角化。

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

## 文件结构

```
kronecker_hamiltonian/
├── model.py           # 完整模型（5个部分）
├── model_v2.py        # 模型 v2
├── train.py           # 训练脚本 + 使用示例
├── test_kronecker.py  # 测试套件（7项测试，仅依赖 torch）
├── requirements.txt   # 依赖列表
└── README.md          # 本文件
```
