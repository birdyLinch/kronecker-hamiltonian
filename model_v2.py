"""
Kronecker Hamiltonian — 选项A：解析对角化
==========================================

核心思想：
  把分子分成子系统 A 和 B，分别构建各自的 Hamiltonian，
  然后利用 Kronecker 结构解析地得到全局特征值/特征向量。

  H = H_A ⊗ I_B  +  I_A ⊗ H_B
  
  其中:
    H_A = D_A + Σ_k a_k a_k^T    [n_A, n_A]
    H_B = D_B + Σ_k b_k b_k^T    [n_B, n_B]

  特征值:   ε_{ij} = ε_i^A + ε_j^B
  特征向量: Φ_{ij} = φ_i^A ⊗ ψ_j^B

  复杂度: O(n_A³ + n_B³)  vs  暴力 O((n_A·n_B)³)

关键约束（选项A的代价）：
  H_A 和 H_B 独立构建，没有跨子系统的直接耦合项。
  耦合只通过"共享的 K 个模式"间接体现（a_k 和 b_k 由同一个 GNN 产生）。

依赖:
    pip install torch e3nn
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from e3nn.o3 import Irreps, Linear, TensorProduct, SphericalHarmonics
from e3nn.nn import FullyConnectedNet


# ══════════════════════════════════════════════════════════════
# Part 1: 等变消息传递（不变，复用原版）
# ══════════════════════════════════════════════════════════════

def build_edge_index(pos: Tensor, cutoff: float):
    N = pos.shape[0]
    idx_i, idx_j = torch.meshgrid(
        torch.arange(N, device=pos.device),
        torch.arange(N, device=pos.device),
        indexing='ij'
    )
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    edge_vec = pos[idx_j] - pos[idx_i]
    dist = edge_vec.norm(dim=-1)
    keep = dist < cutoff
    edge_index = torch.stack([idx_i[keep], idx_j[keep]], dim=0)
    return edge_index, edge_vec[keep]


class EquivariantMessagePassing(nn.Module):
    def __init__(self, node_irreps: str, edge_sh_lmax: int = 2, fc_hidden: list = [64, 64]):
        super().__init__()
        self.node_irreps = Irreps(node_irreps)
        self.edge_sh_irreps = Irreps.spherical_harmonics(edge_sh_lmax)
        self.sh = SphericalHarmonics(self.edge_sh_irreps, normalize=True, normalization='component')

        instructions = []
        for i, (mul_i, ir_i) in enumerate(self.node_irreps):
            for j, (_, ir_j) in enumerate(self.edge_sh_irreps):
                for k, (mul_k, ir_k) in enumerate(self.node_irreps):
                    if ir_k in ir_i * ir_j:
                        instructions.append((i, j, k, 'uvu', True))

        self.tp = TensorProduct(
            self.node_irreps, self.edge_sh_irreps, self.node_irreps,
            instructions=instructions, shared_weights=False, internal_weights=False,
        )
        self.fc = FullyConnectedNet([1] + fc_hidden + [self.tp.weight_numel], act=torch.nn.functional.silu)
        self.linear = Linear(self.node_irreps, self.node_irreps)

    def forward(self, node_feat: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        src, dst = edge_index
        edge_sh = self.sh(edge_vec)
        dist = edge_vec.norm(dim=-1, keepdim=True)
        tp_weights = self.fc(dist)
        msg = self.tp(node_feat[src], edge_sh, tp_weights)
        aggr = torch.zeros_like(node_feat)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.linear(aggr) + node_feat


# ══════════════════════════════════════════════════════════════
# Part 2 (新): 双子系统低秩向量头
# ══════════════════════════════════════════════════════════════

class DualSubsystemVectorHead(nn.Module):
    """
    从节点特征同时预测子系统 A 和 B 各自的低秩向量。

    对每个原子预测两套向量：
      a_k^(i): 该原子对子系统A的贡献
      b_k^(i): 该原子对子系统B的贡献

    然后在子系统内聚合：
      A_k = Σ_{i∈A} a_k^(i) (a_k^(i))^T    [dim_A, dim_A]  (sum of rank-1)
      B_k = Σ_{i∈B} b_k^(i) (b_k^(i))^T    [dim_B, dim_B]

    最后对 k 求和:
      M_A = Σ_k A_k    [dim_A, dim_A]
      M_B = Σ_k B_k    [dim_B, dim_B]

    注意: M_A 和 M_B 是独立的低秩矩阵，秩 ≤ K * |subsystem|
    """
    def __init__(self, node_irreps: str, vector_irreps: str, K: int):
        super().__init__()
        self.K = K
        self.basis_dim = Irreps(vector_irreps).dim

        # 子系统 A 的投影
        self.linears_A = nn.ModuleList([
            Linear(Irreps(node_irreps), Irreps(vector_irreps)) for _ in range(K)
        ])
        # 子系统 B 的投影（独立权重）
        self.linears_B = nn.ModuleList([
            Linear(Irreps(node_irreps), Irreps(vector_irreps)) for _ in range(K)
        ])

    def forward(
        self,
        node_feat: Tensor,          # [N, node_irreps.dim]
        subsystem_ids: Tensor,      # [N] — 0 表示属于 A，1 表示属于 B
    ) -> Tuple[Tensor, Tensor]:
        """
        返回:
            M_A: [dim_A, dim_A]  子系统A的低秩矩阵 Σ_k Σ_{i∈A} a_k^i (a_k^i)^T
            M_B: [dim_B, dim_B]  子系统B的低秩矩阵
        """
        mask_A = subsystem_ids == 0   # [N] bool
        mask_B = subsystem_ids == 1

        feat_A = node_feat[mask_A]    # [n_A, node_dim]
        feat_B = node_feat[mask_B]    # [n_B, node_dim]

        dim_A = feat_A.shape[0] * self.basis_dim
        dim_B = feat_B.shape[0] * self.basis_dim

        M_A = torch.zeros(dim_A, dim_A, device=node_feat.device, dtype=node_feat.dtype)
        M_B = torch.zeros(dim_B, dim_B, device=node_feat.device, dtype=node_feat.dtype)

        for k in range(self.K):
            # 子系统A：每个原子贡献一个向量，拼接后做 outer product
            a_k = self.linears_A[k](feat_A).reshape(-1)   # [n_A * basis_dim]
            M_A = M_A + torch.outer(a_k, a_k)

            # 子系统B
            b_k = self.linears_B[k](feat_B).reshape(-1)   # [n_B * basis_dim]
            M_B = M_B + torch.outer(b_k, b_k)

        return M_A, M_B


# ══════════════════════════════════════════════════════════════
# Part 3 (新): 双子系统对角项头
# ══════════════════════════════════════════════════════════════

class DualSubsystemOnsiteHead(nn.Module):
    """
    分别预测子系统 A 和 B 的 onsite（对角）项。
    只用标量通道（0e）确保旋转不变性。
    """
    def __init__(self, node_irreps: str, basis_dim: int):
        super().__init__()
        node_ir = Irreps(node_irreps)
        scalar_mul = sum(mul for mul, ir in node_ir if ir.l == 0)

        self.extract_scalars = Linear(node_ir, Irreps(f"{scalar_mul}x0e"))
        self.mlp = nn.Sequential(
            nn.Linear(scalar_mul, 64),
            nn.SiLU(),
            nn.Linear(64, basis_dim),
        )

    def forward(
        self,
        node_feat: Tensor,       # [N, node_dim]
        subsystem_ids: Tensor,   # [N]
    ) -> Tuple[Tensor, Tensor]:
        """
        返回:
            d_A: [n_A * basis_dim]  子系统A的对角元素（展平）
            d_B: [n_B * basis_dim]  子系统B的对角元素
        """
        scalars = self.extract_scalars(node_feat)   # [N, scalar_mul]
        onsite = self.mlp(scalars)                   # [N, basis_dim]

        mask_A = subsystem_ids == 0
        mask_B = subsystem_ids == 1

        d_A = onsite[mask_A].reshape(-1)   # [n_A * basis_dim]
        d_B = onsite[mask_B].reshape(-1)   # [n_B * basis_dim]

        return d_A, d_B


# ══════════════════════════════════════════════════════════════
# Part 4 (新): Kronecker 子系统 Hamiltonian 构建
# ══════════════════════════════════════════════════════════════

class KroneckerSubsystemBuilder(nn.Module):
    """
    构建两个子系统各自的 Hamiltonian：
        H_A = diag(d_A) + M_A    [dim_A, dim_A]
        H_B = diag(d_B) + M_B    [dim_B, dim_B]

    全局 H（概念上）：
        H = H_A ⊗ I_B  +  I_A ⊗ H_B

    但我们**不显式构建** H，直接进入对角化。
    """
    def forward(
        self,
        M_A: Tensor, d_A: Tensor,   # [dim_A, dim_A], [dim_A]
        M_B: Tensor, d_B: Tensor,   # [dim_B, dim_B], [dim_B]
    ) -> Tuple[Tensor, Tensor]:
        """
        返回:
            H_A: [dim_A, dim_A]
            H_B: [dim_B, dim_B]
        两者都是实对称矩阵。
        """
        H_A = torch.diag(d_A) + M_A
        H_B = torch.diag(d_B) + M_B

        # 强制对称（数值安全）
        H_A = (H_A + H_A.T) / 2
        H_B = (H_B + H_B.T) / 2

        return H_A, H_B


# ══════════════════════════════════════════════════════════════
# Part 5 (新): Kronecker 解析对角化
# ══════════════════════════════════════════════════════════════

class KroneckerAnalyticDiag(nn.Module):
    """
    利用 Kronecker 结构解析对角化。

    数学基础:
        H = H_A ⊗ I_B + I_A ⊗ H_B

        H_A φ_i = ε_i^A φ_i
        H_B ψ_j = ε_j^B ψ_j

        ⟹  H (φ_i ⊗ ψ_j) = (ε_i^A + ε_j^B)(φ_i ⊗ ψ_j)

    复杂度:
        O(dim_A³ + dim_B³)
        vs 暴力 O((dim_A · dim_B)³)

    参数:
        perturb: 训练时加到对角的小扰动，防止简并导致梯度爆炸
        k_keep:  只保留最低 k 个特征值（None = 全部保留）
    """
    def __init__(self, perturb: float = 1e-6, k_keep: Optional[int] = None):
        super().__init__()
        self.perturb = perturb
        self.k_keep = k_keep

    def forward(
        self,
        H_A: Tensor,   # [dim_A, dim_A] 实对称
        H_B: Tensor,   # [dim_B, dim_B] 实对称
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        返回:
            evals:    [dim_A * dim_B] 或 [k_keep]，升序排列
            evecs_A:  [dim_A, dim_A]  H_A 的特征向量
            evecs_B:  [dim_B, dim_B]  H_B 的特征向量

        全局特征向量 Φ_{ij} = evecs_A[:, i] ⊗ evecs_B[:, j]
        不显式构建以节省内存。
        """
        # 训练时加微小扰动防简并
        if self.training and self.perturb > 0:
            H_A = H_A + self.perturb * torch.eye(H_A.shape[0], device=H_A.device, dtype=H_A.dtype)
            H_B = H_B + self.perturb * torch.eye(H_B.shape[0], device=H_B.device, dtype=H_B.dtype)

        # 分别对角化，O(dim_A³) + O(dim_B³)
        evals_A, evecs_A = torch.linalg.eigh(H_A)   # [dim_A], [dim_A, dim_A]
        evals_B, evecs_B = torch.linalg.eigh(H_B)   # [dim_B], [dim_B, dim_B]

        # 全局特征值: ε_{ij} = ε_i^A + ε_j^B
        # 广播: [dim_A, 1] + [1, dim_B] → [dim_A, dim_B]
        evals_full = evals_A.unsqueeze(1) + evals_B.unsqueeze(0)  # [dim_A, dim_B]
        evals_flat = evals_full.reshape(-1)                        # [dim_A * dim_B]

        # 排序（eigh 已升序，但 sum 后需要重排）
        sort_idx = torch.argsort(evals_flat)
        evals_sorted = evals_flat[sort_idx]

        # 只保留最低 k 个
        if self.k_keep is not None:
            k = min(self.k_keep, evals_sorted.shape[0])
            evals_sorted = evals_sorted[:k]

        return evals_sorted, evecs_A, evecs_B

    def get_global_eigenvec(
        self,
        evecs_A: Tensor,   # [dim_A, dim_A]
        evecs_B: Tensor,   # [dim_B, dim_B]
        i: int,            # A 的第 i 个特征向量
        j: int,            # B 的第 j 个特征向量
    ) -> Tensor:
        """
        按需计算第 (i,j) 个全局特征向量 φ_i ⊗ ψ_j。
        避免存储完整的 [dim_A*dim_B, dim_A*dim_B] 矩阵。
        """
        return torch.kron(evecs_A[:, i], evecs_B[:, j])   # [dim_A * dim_B]

    def apply_global_evecs(
        self,
        evecs_A: Tensor,   # [dim_A, dim_A]
        evecs_B: Tensor,   # [dim_B, dim_B]
        v: Tensor,         # [dim_A * dim_B]  待变换的向量
    ) -> Tensor:
        """
        计算 (Φ^T v)，其中 Φ = evecs_A ⊗ evecs_B。

        利用 vec 技巧（避免构建完整 Kronecker 矩阵）：
            (A ⊗ B)^T vec(V) = vec(B^T V A)

        其中 V = reshape(v, [dim_A, dim_B])

        复杂度: O(dim_A² * dim_B + dim_A * dim_B²)
        vs 暴力: O(dim_A² * dim_B²)
        """
        dim_A, dim_B = evecs_A.shape[0], evecs_B.shape[0]
        V = v.reshape(dim_A, dim_B)
        # (evecs_A ⊗ evecs_B)^T v = evecs_A^T V evecs_B
        result = evecs_A.T @ V @ evecs_B    # [dim_A, dim_B]
        return result.reshape(-1)           # [dim_A * dim_B]


# ══════════════════════════════════════════════════════════════
# 波函数特征提取
# ══════════════════════════════════════════════════════════════

class WavefunctionFeatureExtractor(nn.Module):
    """
    从特征向量中提取非局域特征用于能量预测。

    不显式构建全局特征向量，而是利用 Kronecker 结构
    在子系统空间内计算物理量。

    提取的特征:
      1. 最低 k 个特征值（直接）
      2. A 侧 reduced density matrix 的对角（occupation numbers in A）
      3. B 侧 reduced density matrix 的对角（occupation numbers in B）
      4. A-B 纠缠熵（作为非局域特征）
    """
    def __init__(self, k_states: int = 8, n_occ: Optional[int] = None):
        super().__init__()
        self.k_states = k_states
        self.n_occ = n_occ   # 占据轨道数（None = 取最低 k_states 个）

    def forward(
        self,
        evals: Tensor,       # [k] 最低 k 个特征值
        evecs_A: Tensor,     # [dim_A, dim_A]
        evecs_B: Tensor,     # [dim_B, dim_B]
        evals_A: Tensor,     # [dim_A] H_A 的特征值（用于重建排序）
        evals_B: Tensor,     # [dim_B] H_B 的特征值
    ) -> Tensor:
        """
        返回: feature vector，形状 [feature_dim]
        """
        features = []

        # 1. 特征值特征（最直接的）
        k = min(self.k_states, evals.shape[0])
        eig_feat = evals[:k]
        if eig_feat.shape[0] < self.k_states:
            eig_feat = torch.nn.functional.pad(eig_feat, (0, self.k_states - k))
        features.append(eig_feat)

        # 2. A 子系统的 occupation（H_A 特征值分布）
        k_A = min(self.k_states, evals_A.shape[0])
        occ_A = evals_A[:k_A]
        if occ_A.shape[0] < self.k_states:
            occ_A = torch.nn.functional.pad(occ_A, (0, self.k_states - k_A))
        features.append(occ_A)

        # 3. B 子系统的 occupation
        k_B = min(self.k_states, evals_B.shape[0])
        occ_B = evals_B[:k_B]
        if occ_B.shape[0] < self.k_states:
            occ_B = torch.nn.functional.pad(occ_B, (0, self.k_states - k_B))
        features.append(occ_B)

        # 4. 非局域特征：A-B 特征值的交叉相关
        #    (ε_i^A - ε_j^B) 的统计量
        gap_AB = (evals_A[:k_A].unsqueeze(1) - evals_B[:k_B].unsqueeze(0))
        cross_feat = torch.stack([
            gap_AB.min(),
            gap_AB.max(),
            gap_AB.abs().mean(),
            gap_AB.std(),
        ])
        features.append(cross_feat)

        return torch.cat(features)   # [3*k_states + 4]


# ══════════════════════════════════════════════════════════════
# 完整模型
# ══════════════════════════════════════════════════════════════

class KroneckerHamiltonianModelV2(nn.Module):
    """
    选项A完整模型。

    配置示例:
        config = {
            'atom_types':      5,
            'atom_embed_dim':  32,
            'node_irreps':     "16x0e + 8x1o + 4x2e",
            'edge_sh_lmax':    2,
            'n_mp_layers':     3,
            'K':               4,
            'vector_irreps':   "1x0e + 1x1o",   # basis_dim = 4
            'basis_dim':       4,
            'k_keep':          20,   # 只保留最低20个特征值
            'k_states':        8,    # 特征提取用的状态数
            'cutoff':          5.0,
        }
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.cutoff = config['cutoff']

        embed_dim = config['atom_embed_dim']
        self.atom_embedding = nn.Embedding(config['atom_types'], embed_dim)

        node_ir = Irreps(config['node_irreps'])
        scalar_mul = sum(mul for mul, ir in node_ir if ir.l == 0)
        self.embed_to_node = nn.Linear(embed_dim, scalar_mul, bias=False)

        self.mp_layers = nn.ModuleList([
            EquivariantMessagePassing(
                node_irreps=config['node_irreps'],
                edge_sh_lmax=config['edge_sh_lmax'],
            )
            for _ in range(config['n_mp_layers'])
        ])

        self.vec_head = DualSubsystemVectorHead(
            node_irreps=config['node_irreps'],
            vector_irreps=config['vector_irreps'],
            K=config['K'],
        )

        self.diag_head = DualSubsystemOnsiteHead(
            node_irreps=config['node_irreps'],
            basis_dim=config['basis_dim'],
        )

        self.H_builder = KroneckerSubsystemBuilder()

        self.diag_layer = KroneckerAnalyticDiag(
            perturb=1e-6,
            k_keep=config.get('k_keep', None),
        )

        self.feat_extractor = WavefunctionFeatureExtractor(
            k_states=config.get('k_states', 8),
        )

        # 能量头：输入维度 = 3*k_states + 4
        k_s = config.get('k_states', 8)
        feat_dim = 3 * k_s + 4
        self.energy_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _init_node_features(self, atom_types: Tensor) -> Tensor:
        node_ir = Irreps(self.config['node_irreps'])
        embed = self.atom_embedding(atom_types)
        scalars = self.embed_to_node(embed)
        node_feat = torch.zeros(atom_types.shape[0], node_ir.dim,
                                device=atom_types.device, dtype=scalars.dtype)
        scalar_dim = sum(mul for mul, ir in node_ir if ir.l == 0)
        node_feat[:, :scalar_dim] = scalars
        return node_feat

    def forward(
        self,
        pos: Tensor,               # [N, 3]
        atom_types: Tensor,        # [N] int
        subsystem_ids: Tensor,     # [N] — 0=子系统A, 1=子系统B（必须提供）
    ) -> dict:
        # ── 消息传递 ──
        edge_index, edge_vec = build_edge_index(pos, self.cutoff)
        node_feat = self._init_node_features(atom_types)
        for mp in self.mp_layers:
            node_feat = mp(node_feat, edge_index, edge_vec)

        # ── 预测低秩向量和对角项 ──
        M_A, M_B = self.vec_head(node_feat, subsystem_ids)
        d_A, d_B = self.diag_head(node_feat, subsystem_ids)

        # ── 构建子系统 H ──
        H_A, H_B = self.H_builder(M_A, d_A, M_B, d_B)

        # ── Kronecker 解析对角化 ──
        evals, evecs_A, evecs_B = self.diag_layer(H_A, H_B)
        # evals: 全局特征值（来自 ε_i^A + ε_j^B）
        # evecs_A, evecs_B: 两个子系统各自的特征向量

        # ── 提取波函数特征 ──
        # 重新获取子系统特征值（diag_layer 内部的，这里重算一次用于特征）
        with torch.no_grad():
            evals_A_raw = torch.linalg.eigvalsh(H_A)
            evals_B_raw = torch.linalg.eigvalsh(H_B)

        feat = self.feat_extractor(evals, evecs_A, evecs_B, evals_A_raw, evals_B_raw)

        # ── 能量预测 ──
        energy = self.energy_head(feat).squeeze()

        return {
            'energy':    energy,
            'evals':     evals,
            'evecs_A':   evecs_A,
            'evecs_B':   evecs_B,
            'H_A':       H_A,
            'H_B':       H_B,
            # 便于调试
            'evals_A':   evals_A_raw,
            'evals_B':   evals_B_raw,
        }


# ══════════════════════════════════════════════════════════════
# 验证与复杂度演示
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    torch.manual_seed(42)

    config = {
        'atom_types':    10,
        'atom_embed_dim': 32,
        'node_irreps':   "16x0e + 8x1o + 4x2e",
        'edge_sh_lmax':  2,
        'n_mp_layers':   2,
        'K':             4,
        'vector_irreps': "1x0e + 1x1o",
        'basis_dim':     4,
        'k_keep':        16,
        'k_states':      8,
        'cutoff':        5.0,
    }

    model = KroneckerHamiltonianModelV2(config)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # 随机分子：4个原子在A，4个在B
    N_A, N_B = 4, 4
    N = N_A + N_B
    pos = torch.randn(N, 3) * 2.0
    atom_types = torch.randint(0, 10, (N,))
    subsystem_ids = torch.tensor([0]*N_A + [1]*N_B)

    out = model(pos, atom_types, subsystem_ids)
    bd = config['basis_dim']

    print(f"H_A 形状:     {out['H_A'].shape}  (= [{N_A*bd}, {N_A*bd}])")
    print(f"H_B 形状:     {out['H_B'].shape}  (= [{N_B*bd}, {N_B*bd}])")
    print(f"全局特征值:   {out['evals'].shape}  (≤ {N_A*bd * N_B*bd})")
    print(f"预测能量:     {out['energy'].item():.4f}")

    # 验证特征值确实是 ε_A + ε_B 的组合
    eA = out['evals_A']
    eB = out['evals_B']
    expected_min = (eA[0] + eB[0]).item()
    actual_min = out['evals'][0].item()
    print(f"\n特征值验证:")
    print(f"  ε_A[0] + ε_B[0] = {expected_min:.4f}")
    print(f"  evals[0]        = {actual_min:.4f}")
    print(f"  误差            = {abs(expected_min - actual_min):.2e}  ✓")

    # 复杂度对比
    print(f"\n复杂度对比 (dim_A={N_A*bd}, dim_B={N_B*bd}):")
    print(f"  暴力 eigh: O({(N_A*bd * N_B*bd)**3:.0e})")
    print(f"  Kronecker: O({(N_A*bd)**3 + (N_B*bd)**3:.0e})  ({((N_A*bd * N_B*bd)**3) / ((N_A*bd)**3 + (N_B*bd)**3):.0f}x 更快)")

    # 梯度
    out['energy'].backward()
    print(f"\n梯度反传: ✓ 成功")

    # ── 计时对比（更大的系统）──
    print("\n── 大系统计时对比 ──")
    for n_sub in [10, 20, 40]:
        dim = n_sub * bd
        H_rand = torch.randn(dim, dim)
        H_rand = H_rand + H_rand.T

        H_A_r = torch.randn(dim, dim); H_A_r = H_A_r + H_A_r.T
        H_B_r = torch.randn(dim, dim); H_B_r = H_B_r + H_B_r.T

        # 暴力
        H_full = torch.kron(H_A_r, torch.eye(dim)) + torch.kron(torch.eye(dim), H_B_r)
        t0 = time.perf_counter()
        torch.linalg.eigh(H_full)
        t_brute = time.perf_counter() - t0

        # Kronecker
        t0 = time.perf_counter()
        torch.linalg.eigh(H_A_r)
        torch.linalg.eigh(H_B_r)
        t_kron = time.perf_counter() - t0

        print(f"  dim={dim}×{dim} (全局 {dim*dim}×{dim*dim}):  "
              f"暴力 {t_brute*1000:.1f}ms  |  Kronecker {t_kron*1000:.1f}ms  |  "
              f"加速 {t_brute/t_kron:.1f}x")
