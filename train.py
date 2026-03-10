"""
训练脚本示例
============
演示如何用 KroneckerHamiltonianModel 训练分子能量预测。

运行:
    python train.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import KroneckerHamiltonianModel


# ─────────────────────────────────────────────
# 合成数据集（替换成你的真实数据）
# ─────────────────────────────────────────────

def make_dummy_batch(batch_size=4, n_atoms_range=(4, 8), n_atom_types=10):
    """
    生成随机分子批次。
    真实使用时替换成 ASE/pymatgen 读取的数据。
    返回 list of (pos, atom_types, energy)
    """
    batch = []
    for _ in range(batch_size):
        N = torch.randint(*n_atoms_range, (1,)).item()
        pos = torch.randn(N, 3) * 2.0
        atom_types = torch.randint(0, n_atom_types, (N,))
        # 合成能量：用距离矩阵的某个函数
        dist = torch.cdist(pos, pos)
        energy = -dist[dist > 0].pow(-1).sum() * 0.1 + torch.randn(1).item() * 0.01
        batch.append((pos, atom_types, energy))
    return batch


# ─────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────

class HamiltonianLoss(nn.Module):
    """
    组合损失:
      L = L_energy + λ_offdiag * L_offdiag + λ_smooth * L_smooth

    L_energy:   MSE(E_pred, E_true)
    L_offdiag:  ||H - diag(H)||_F²  鼓励 H 趋向对角（加速对角化）
    L_smooth:   特征值差分正则（防止特征值塌缩）
    """
    def __init__(
        self,
        lambda_offdiag: float = 0.01,
        lambda_smooth:  float = 0.001,
    ):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag
        self.lambda_smooth  = lambda_smooth
        self.mse = nn.MSELoss()

    def forward(self, pred_energy, true_energy, H, eigenvalues):
        # 能量损失
        L_e = self.mse(pred_energy, true_energy)

        # Off-diagonal 惩罚
        H_diag = torch.diag(torch.diag(H))
        L_od = (H - H_diag).pow(2).mean()

        # 特征值间距正则（防止所有特征值相等）
        if eigenvalues.shape[0] > 1:
            gaps = eigenvalues[1:] - eigenvalues[:-1]
            L_sm = torch.exp(-gaps).mean()  # 鼓励间距大
        else:
            L_sm = torch.tensor(0.0)

        total = L_e + self.lambda_offdiag * L_od + self.lambda_smooth * L_sm
        return total, {'energy': L_e.item(), 'offdiag': L_od.item(), 'smooth': L_sm.item()}


# ─────────────────────────────────────────────
# 训练循环
# ─────────────────────────────────────────────

def train():
    config = {
        'atom_types':        10,
        'atom_embed_dim':    32,
        # node_irreps 控制等变表达能力
        # 格式: "{mul}x{l}{parity}"
        # 例: "32x0e" 纯标量; "16x0e + 8x1o" 标量+向量
        'node_irreps':       "16x0e + 8x1o + 4x2e",
        'edge_sh_lmax':      2,
        'n_mp_layers':       3,
        # K: 低秩分量数，越大表达能力越强但对角化越慢
        'K':                 4,
        # vector_irreps: u_k 的等变阶次，决定 basis_dim
        # "1x0e"         → scalar only,  basis_dim=1
        # "1x1o"         → vector only,  basis_dim=3
        # "1x0e + 1x1o"  → scalar+vec,   basis_dim=4
        # "1x0e + 1x1o + 1x2e" → s+v+d,  basis_dim=9
        'vector_irreps':     "1x0e + 1x1o",
        'basis_dim':         4,
        'hamiltonian_mode':  'sum_outer',
        'diag_mode':         'full',     # 小分子用 full；大分子换 lobpcg
        'diag_k':            10,
        'cutoff':            5.0,
    }

    model = KroneckerHamiltonianModel(config)
    criterion = HamiltonianLoss(lambda_offdiag=0.01, lambda_smooth=0.001)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    print("开始训练...")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(50):
        batch = make_dummy_batch(batch_size=4)
        total_loss = 0.0

        optimizer.zero_grad()
        for pos, atom_types, true_energy in batch:
            out = model(pos, atom_types)
            true_e = torch.tensor(true_energy, dtype=torch.float32)

            loss, details = criterion(
                out['energy'], true_e,
                out['H'], out['eigenvalues']
            )
            loss = loss / len(batch)  # 平均梯度
            loss.backward()
            total_loss += loss.item()

        # 梯度裁剪（对角化的梯度可能很大）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | "
                  f"E: {details['energy']:.4f} | "
                  f"OD: {details['offdiag']:.4f} | "
                  f"Sm: {details['smooth']:.4f}")

    print("\n训练完成！")


# ─────────────────────────────────────────────
# 使用示例：不同 vector_irreps 配置
# ─────────────────────────────────────────────

def demo_irreps_configs():
    """
    展示不同 irreps 配置下 H 的维度和对称性
    """
    configs = [
        ("纯标量 (l=0)",    "1x0e",              1),
        ("向量   (l=1)",    "1x1o",              3),
        ("标量+向量",        "1x0e + 1x1o",       4),
        ("到 l=2",          "1x0e + 1x1o + 1x2e", 9),
    ]

    N = 4
    pos = torch.randn(N, 3)
    atom_types = torch.randint(0, 5, (N,))

    print("vector_irreps 配置对比:")
    print(f"{'名称':<15} {'basis_dim':<12} {'H 维度':<15} {'等变阶次'}")
    print("-" * 60)

    for name, vir, bdim in configs:
        node_ir = Irreps(vir)
        total_dim = N * bdim
        l_max = max(ir.l for _, ir in node_ir)
        print(f"{name:<15} {bdim:<12} {total_dim}×{total_dim:<10}  l ≤ {l_max}")


if __name__ == '__main__':
    demo_irreps_configs()
    print()
    train()
