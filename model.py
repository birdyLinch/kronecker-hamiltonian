"""
Equivariant Kronecker Hamiltonian Model
========================================
Pipeline:
  原子坐标 + 原子类型
    → Part 1: 等变消息传递 (e3nn)  →  per-atom node features (irreps)
    → Part 2: 低秩向量头           →  K 组 u_k (可配置 irrep)
    → Part 3: 对角局域项头         →  per-atom onsite energy
    → Part 4: Kronecker H 构建    →  H = D + Σ_k (u_k ⊗ u_k^T)
    → Part 5: 对角化               →  eigenvalues ε, eigenvectors ψ

依赖:
    pip install torch e3nn torch-scatter torch-sparse

用法示例见文件底部 __main__ 块。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# e3nn imports
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear, TensorProduct, SphericalHarmonics


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def build_edge_index(pos: Tensor, cutoff: float):
    """
    暴力 O(N²) 构建边列表（小分子够用；大分子换 torch_cluster.radius_graph）
    返回: edge_index [2, E], edge_vec [E, 3]
    """
    N = pos.shape[0]
    idx_i, idx_j = torch.meshgrid(
        torch.arange(N, device=pos.device),
        torch.arange(N, device=pos.device),
        indexing='ij'
    )
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    edge_vec = pos[idx_j] - pos[idx_i]          # [E, 3]
    dist = edge_vec.norm(dim=-1)
    keep = dist < cutoff
    edge_index = torch.stack([idx_i[keep], idx_j[keep]], dim=0)  # [2, E]
    return edge_index, edge_vec[keep]


def make_irreps_str(l_max: int, parity: bool = True) -> str:
    """生成 '1x0e + 1x1o + 1x2e + ...' 形式的 irreps 字符串"""
    parts = []
    for l in range(l_max + 1):
        p = 'e' if (l % 2 == 0) else 'o'
        if not parity:
            p = 'e'
        parts.append(f"1x{l}{p}")
    return " + ".join(parts)


# ─────────────────────────────────────────────
# Part 1: 等变消息传递层
# ─────────────────────────────────────────────

class EquivariantMessagePassing(nn.Module):
    """
    单层等变消息传递。
    使用 e3nn TensorProduct: node_irreps ⊗ edge_sh_irreps → node_irreps
    
    参数:
        node_irreps: 节点特征的 Irreps，如 "32x0e + 16x1o + 8x2e"
        edge_sh_lmax: 球谐函数最大阶数
        fc_hidden: 径向网络隐藏层
    """
    def __init__(
        self,
        node_irreps: str,
        edge_sh_lmax: int = 2,
        fc_hidden: list = [64, 64],
    ):
        super().__init__()
        self.node_irreps = Irreps(node_irreps)
        self.edge_sh_irreps = Irreps.spherical_harmonics(edge_sh_lmax)

        # 球谐函数：边方向 → 不变/等变特征
        self.sh = SphericalHarmonics(
            self.edge_sh_irreps,
            normalize=True,
            normalization='component'
        )

        # TensorProduct: h_j ⊗ Y_ij → message
        self.tp = TensorProduct(
            self.node_irreps,
            self.edge_sh_irreps,
            self.node_irreps,
            instructions=self._build_tp_instructions(),
            shared_weights=False,   # 权重由径向网络给出
            internal_weights=False,
        )

        # 径向网络：把 ||r_ij|| 映射到 TensorProduct 的权重
        n_tp_weights = self.tp.weight_numel
        self.fc = FullyConnectedNet(
            [1] + fc_hidden + [n_tp_weights],
            act=torch.nn.functional.silu,
        )

        # 聚合后的线性变换
        self.linear = Linear(self.node_irreps, self.node_irreps)

    def _build_tp_instructions(self):
        """
        构建 TP instructions：对每对 (node_irrep_i, edge_sh_irrep_j)
        找到所有输出 irrep 并配对。
        使用 'uvu' 模式（channelwise）。
        """
        instructions = []
        node_ir = self.node_irreps
        sh_ir = self.edge_sh_irreps

        for i, (mul_i, ir_i) in enumerate(node_ir):
            for j, (_, ir_j) in enumerate(sh_ir):
                for k, (mul_k, ir_k) in enumerate(node_ir):
                    if ir_k in ir_i * ir_j:
                        instructions.append((i, j, k, 'uvu', True))
        return instructions

    def forward(
        self,
        node_feat: Tensor,          # [N, node_irreps.dim]
        edge_index: Tensor,         # [2, E]
        edge_vec: Tensor,           # [E, 3]
    ) -> Tensor:
        src, dst = edge_index        # src→dst

        # 球谐函数
        edge_sh = self.sh(edge_vec)  # [E, sh_irreps.dim]

        # 径向权重（用距离的归一化）
        dist = edge_vec.norm(dim=-1, keepdim=True)  # [E, 1]
        tp_weights = self.fc(dist)                   # [E, n_tp_weights]

        # 消息 = TP(h_src, Y_edge)
        msg = self.tp(node_feat[src], edge_sh, tp_weights)  # [E, node_irreps.dim]

        # 聚合（sum over neighbors）
        aggr = torch.zeros_like(node_feat)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        # 残差 + 线性
        out = self.linear(aggr) + node_feat
        return out


# ─────────────────────────────────────────────
# Part 2: 低秩向量头（预测 u_k）
# ─────────────────────────────────────────────

class LowRankVectorHead(nn.Module):
    """
    从节点特征预测 K 组 低秩向量 u_k，每个 u_k 的 irrep 可独立配置。

    参数:
        node_irreps:   输入节点特征的 irreps
        vector_irreps: u_k 的 irreps，如 "1x0e" (scalar) 或 "1x1o" (vector) 或 "1x0e + 1x1o"
        K:             秩数（低秩分量数）
        basis_dim:     Hilbert 空间维度（H 的大小 = basis_dim × basis_dim）

    输出:
        u_vectors: [N, K, basis_dim]  —— 每个原子贡献 K 个向量
                   （通常 basis_dim = irreps_of_u.dim）
    """
    def __init__(
        self,
        node_irreps: str,
        vector_irreps: str,
        K: int,
        basis_dim: int,
    ):
        super().__init__()
        self.K = K
        self.basis_dim = basis_dim
        self.vector_irreps = Irreps(vector_irreps)

        assert self.vector_irreps.dim == basis_dim, (
            f"vector_irreps.dim ({self.vector_irreps.dim}) must equal basis_dim ({basis_dim})"
        )

        # 为每个 k 独立地从 node_irreps → vector_irreps
        self.linears = nn.ModuleList([
            Linear(Irreps(node_irreps), self.vector_irreps)
            for _ in range(K)
        ])

    def forward(self, node_feat: Tensor) -> Tensor:
        # node_feat: [N, node_irreps.dim]
        vectors = []
        for k in range(self.K):
            u_k = self.linears[k](node_feat)   # [N, basis_dim]
            vectors.append(u_k)
        return torch.stack(vectors, dim=1)     # [N, K, basis_dim]


# ─────────────────────────────────────────────
# Part 3: 对角局域项头（onsite energy）
# ─────────────────────────────────────────────

class DiagonalOnsiteHead(nn.Module):
    """
    预测每个原子的对角贡献（onsite energy），组成 H 的对角块。

    参数:
        node_irreps: 输入 irreps
        basis_dim:   每个原子对应的轨道数（对角块大小）

    输出:
        onsite: [N, basis_dim]  —— 对角元素
    """
    def __init__(self, node_irreps: str, basis_dim: int):
        super().__init__()
        # 只用标量通道（0e）来预测对角，因为对角项是实数/不变量
        node_ir = Irreps(node_irreps)
        scalar_mul = sum(mul for mul, ir in node_ir if ir.l == 0)

        self.extract_scalars = Linear(
            node_ir,
            Irreps(f"{scalar_mul}x0e")
        )
        self.mlp = nn.Sequential(
            nn.Linear(scalar_mul, 64),
            nn.SiLU(),
            nn.Linear(64, basis_dim),
        )

    def forward(self, node_feat: Tensor) -> Tensor:
        scalars = self.extract_scalars(node_feat)   # [N, scalar_mul]
        onsite = self.mlp(scalars)                  # [N, basis_dim]
        return onsite


# ─────────────────────────────────────────────
# Part 4: Kronecker Hamiltonian 构建
# ─────────────────────────────────────────────

class KroneckerHamiltonianBuilder(nn.Module):
    """
    从低秩向量和对角项构建全局 Hamiltonian。

    两种模式：
    ──────────────────────────────────────────
    mode='sum_outer':
        H = diag(d) + Σ_k  U_k U_k^T
        其中 U_k = concat([u_k^(atom_1), ..., u_k^(atom_N)])  ∈ R^{N*basis_dim}
        
    mode='kronecker':
        将分子分成两个子系统 A, B（各有 n_A, n_B 个原子）
        H = D_A ⊗ I_B + I_A ⊗ D_B + Σ_k  A_k ⊗ B_k
        其中 A_k, B_k 分别由 subsystem_A/B 的原子的 u_k 聚合而成
        特征值 = ε_A_i + ε_B_j（若只有局域项，可解析求解）
    ──────────────────────────────────────────
    """
    def __init__(self, mode: str = 'sum_outer'):
        super().__init__()
        assert mode in ('sum_outer', 'kronecker')
        self.mode = mode

    def forward(
        self,
        u_vectors: Tensor,              # [N, K, basis_dim]
        onsite: Tensor,                 # [N, basis_dim]
        subsystem_ids: Optional[Tensor] = None,  # [N] 0/1 for kronecker mode
    ) -> Tensor:
        """
        返回: H  [total_dim, total_dim]
        """
        N, K, basis_dim = u_vectors.shape

        if self.mode == 'sum_outer':
            return self._build_sum_outer(u_vectors, onsite, N, basis_dim)
        else:
            return self._build_kronecker(u_vectors, onsite, subsystem_ids, basis_dim)

    def _build_sum_outer(
        self, u_vectors: Tensor, onsite: Tensor, N: int, basis_dim: int
    ) -> Tensor:
        """
        H = D + Σ_k U_k U_k^T
        D: block-diagonal，每个原子贡献一个 basis_dim×basis_dim 对角块
        U_k: [N*basis_dim] 拼接后的向量
        """
        total_dim = N * basis_dim

        # 对角项：block diagonal
        D = torch.diag(onsite.reshape(-1))   # [total_dim, total_dim]

        # 低秩项
        H_lr = torch.zeros(total_dim, total_dim,
                           dtype=u_vectors.dtype, device=u_vectors.device)
        for k in range(u_vectors.shape[1]):
            u_k = u_vectors[:, k, :].reshape(-1)    # [N*basis_dim]
            H_lr += torch.outer(u_k, u_k)

        # Hermitian 化（确保数值对称）
        H = D + H_lr
        H = (H + H.T) / 2
        return H

    def _build_kronecker(
        self,
        u_vectors: Tensor,
        onsite: Tensor,
        subsystem_ids: Tensor,
        basis_dim: int,
    ) -> Tensor:
        """
        H = (D_A ⊗ I_B) + (I_A ⊗ D_B) + Σ_k (A_k ⊗ B_k)
        
        D_A: 子系统A的对角（onsite聚合），dim n_A*basis_dim
        A_k: 子系统A聚合后的秩1矩阵
        
        总维度: (n_A * basis_dim) × (n_B * basis_dim)
        → 等价于 (n_A * n_B * basis_dim²) × (n_A * n_B * basis_dim²)
        这里简化为 A, B 各自用聚合向量表示。
        """
        mask_A = subsystem_ids == 0
        mask_B = subsystem_ids == 1

        u_A = u_vectors[mask_A]    # [n_A, K, basis_dim]
        u_B = u_vectors[mask_B]    # [n_B, K, basis_dim]
        d_A = onsite[mask_A]       # [n_A, basis_dim]
        d_B = onsite[mask_B]       # [n_B, basis_dim]

        n_A, K, bd = u_A.shape
        n_B = u_B.shape[0]
        dim_A = n_A * bd
        dim_B = n_B * bd

        # 聚合成矩阵
        def outer_sum(u: Tensor) -> Tensor:
            # u: [n, K, bd] → Σ_k (n*bd × n*bd)
            n = u.shape[0]
            M = torch.zeros(n * bd, n * bd, device=u.device, dtype=u.dtype)
            for k in range(K):
                v = u[:, k, :].reshape(-1)
                M += torch.outer(v, v)
            return M

        M_A = outer_sum(u_A)  # [dim_A, dim_A]
        M_B = outer_sum(u_B)  # [dim_B, dim_B]

        D_A = torch.diag(d_A.reshape(-1))
        D_B = torch.diag(d_B.reshape(-1))
        I_A = torch.eye(dim_A, device=u_A.device, dtype=u_A.dtype)
        I_B = torch.eye(dim_B, device=u_B.device, dtype=u_B.dtype)

        # H = D_A⊗I_B + I_A⊗D_B + M_A⊗M_B（这里 M_A⊗M_B 是 Kronecker product）
        H = (torch.kron(D_A, I_B)
             + torch.kron(I_A, D_B)
             + torch.kron(M_A, M_B))

        H = (H + H.T) / 2
        return H


# ─────────────────────────────────────────────
# Part 5: 对角化层
# ─────────────────────────────────────────────

class DiagonalizationLayer(nn.Module):
    """
    对 Hamiltonian 做完整或部分对角化。

    模式:
        full:    torch.linalg.eigh → 全部特征值/向量
        lobpcg:  LOBPCG → 最低 k 个（大分子推荐）
        lanczos: 手动 Lanczos（数值稳定版）→ 最低 k 个

    注意: eigh 在特征值简并时梯度可能不稳定，
          建议训练时加微小扰动或只对能量做监督。
    """
    def __init__(self, mode: str = 'full', k: int = 10, perturb: float = 1e-6):
        super().__init__()
        assert mode in ('full', 'lobpcg', 'lanczos')
        self.mode = mode
        self.k = k
        self.perturb = perturb

    def forward(self, H: Tensor):
        """
        输入:  H [D, D]  (实对称)
        输出:  eigenvalues [D or k], eigenvectors [D, D or k]
        """
        if self.mode == 'full':
            return self._eigh(H)
        elif self.mode == 'lobpcg':
            return self._lobpcg(H)
        else:
            return self._lanczos(H)

    def _eigh(self, H: Tensor):
        # 加微小扰动防止简并梯度爆炸
        if self.perturb > 0 and self.training:
            noise = self.perturb * torch.randn_like(H)
            H = H + (noise + noise.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def _lobpcg(self, H: Tensor):
        """
        LOBPCG: 只求最低 k 个特征值（sparse-friendly）。
        需要 H 是正定的（可以加 shift）。
        """
        D = H.shape[0]
        k = min(self.k, D - 1)

        # 随机初始猜测
        X0 = torch.randn(D, k, device=H.device, dtype=H.dtype)
        X0, _ = torch.linalg.qr(X0)

        eigenvalues, eigenvectors = torch.lobpcg(H, k=k, X=X0, largest=False)
        return eigenvalues, eigenvectors

    def _lanczos(self, H: Tensor):
        """
        简化 Lanczos（k步）。
        完整实现应加重正交化（re-orthogonalization）。
        """
        D = H.shape[0]
        k = min(self.k, D)

        # 初始向量
        v = torch.randn(D, device=H.device, dtype=H.dtype)
        v = v / v.norm()

        V = []      # Krylov 基
        alphas = []
        betas = []

        v_prev = torch.zeros_like(v)
        beta = 0.0

        for j in range(k):
            V.append(v)
            w = H @ v - beta * v_prev
            alpha = (w @ v).item()
            w = w - alpha * v

            # 完全重正交化（Gram-Schmidt）
            for vi in V:
                w = w - (w @ vi) * vi

            beta = w.norm().item()
            alphas.append(alpha)

            if beta < 1e-10:
                break
            v_prev = v
            v = w / beta
            betas.append(beta)

        # 构建三对角矩阵
        m = len(alphas)
        T = torch.zeros(m, m, device=H.device, dtype=H.dtype)
        for i in range(m):
            T[i, i] = alphas[i]
        for i in range(len(betas)):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

        # 对角化小矩阵
        evals, evecs_T = torch.linalg.eigh(T)

        # 变换回原空间
        V_mat = torch.stack(V, dim=1)         # [D, m]
        eigenvectors = V_mat @ evecs_T         # [D, m]

        return evals, eigenvectors


# ─────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────

class KroneckerHamiltonianModel(nn.Module):
    """
    完整的端到端模型。

    配置示例:
        config = {
            'atom_types':      5,          # 原子种类数
            'atom_embed_dim':  32,         # 初始 embedding 维度
            'node_irreps':     "32x0e + 16x1o + 8x2e",
            'edge_sh_lmax':    2,
            'n_mp_layers':     3,          # 消息传递层数
            'K':               4,          # 低秩分量数
            'vector_irreps':   "1x0e + 1x1o",  # u_k 的 irreps → basis_dim = 4
            'basis_dim':       4,          # 必须等于 vector_irreps.dim
            'hamiltonian_mode': 'sum_outer',  # 或 'kronecker'
            'diag_mode':       'full',     # 或 'lobpcg'/'lanczos'
            'diag_k':          10,
            'cutoff':          5.0,        # 截断半径 (Å)
        }
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.cutoff = config['cutoff']

        # 原子类型嵌入 → 初始节点特征（只用标量通道初始化）
        embed_dim = config['atom_embed_dim']
        self.atom_embedding = nn.Embedding(config['atom_types'], embed_dim)

        # 把 embed_dim 标量升到 node_irreps
        node_ir = Irreps(config['node_irreps'])
        scalar_mul = sum(mul for mul, ir in node_ir if ir.l == 0)
        self.embed_to_node = nn.Linear(embed_dim, scalar_mul, bias=False)

        # 等变消息传递层
        self.mp_layers = nn.ModuleList([
            EquivariantMessagePassing(
                node_irreps=config['node_irreps'],
                edge_sh_lmax=config['edge_sh_lmax'],
            )
            for _ in range(config['n_mp_layers'])
        ])

        # 低秩向量头
        self.vec_head = LowRankVectorHead(
            node_irreps=config['node_irreps'],
            vector_irreps=config['vector_irreps'],
            K=config['K'],
            basis_dim=config['basis_dim'],
        )

        # 对角项头
        self.diag_head = DiagonalOnsiteHead(
            node_irreps=config['node_irreps'],
            basis_dim=config['basis_dim'],
        )

        # H 构建
        self.H_builder = KroneckerHamiltonianBuilder(
            mode=config['hamiltonian_mode']
        )

        # 对角化
        self.diag_layer = DiagonalizationLayer(
            mode=config['diag_mode'],
            k=config.get('diag_k', 10),
        )

        # 能量预测头（用特征值 + 非局域特征）
        total_dim_estimate = 10  # 取最低10个特征值作特征
        self.energy_head = nn.Sequential(
            nn.Linear(total_dim_estimate, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def _init_node_features(self, atom_types: Tensor, node_ir: Irreps) -> Tensor:
        """
        初始化节点特征：embed → 填充到完整 irreps 维度（高阶项初始化为0）
        """
        embed = self.atom_embedding(atom_types)    # [N, embed_dim]
        scalars = self.embed_to_node(embed)         # [N, scalar_mul]

        # 填充高阶 irreps 为 0
        node_feat = torch.zeros(
            atom_types.shape[0], node_ir.dim,
            device=atom_types.device, dtype=scalars.dtype
        )
        scalar_dim = sum(mul for mul, ir in node_ir if ir.l == 0)
        node_feat[:, :scalar_dim] = scalars
        return node_feat

    def forward(
        self,
        pos: Tensor,                             # [N, 3]
        atom_types: Tensor,                      # [N] int
        subsystem_ids: Optional[Tensor] = None,  # [N] 0/1，仅 kronecker 模式需要
    ):
        """
        返回:
            energy:      scalar 预测能量
            eigenvalues: [n_eigs]
            eigenvectors:[total_dim, n_eigs]
            H:           [total_dim, total_dim]
        """
        node_ir = Irreps(self.config['node_irreps'])

        # 构建图
        edge_index, edge_vec = build_edge_index(pos, self.cutoff)

        # 初始节点特征
        node_feat = self._init_node_features(atom_types, node_ir)

        # 等变消息传递
        for mp in self.mp_layers:
            node_feat = mp(node_feat, edge_index, edge_vec)

        # 预测低秩向量和对角项
        u_vectors = self.vec_head(node_feat)    # [N, K, basis_dim]
        onsite = self.diag_head(node_feat)       # [N, basis_dim]

        # 构建 H
        H = self.H_builder(u_vectors, onsite, subsystem_ids)  # [D, D]

        # 对角化
        eigenvalues, eigenvectors = self.diag_layer(H)

        # 能量预测：用最低 k 个特征值作特征
        k_feat = min(10, eigenvalues.shape[0])
        eig_feat = eigenvalues[:k_feat]
        if eig_feat.shape[0] < 10:
            # 补零到固定长度
            eig_feat = torch.cat([
                eig_feat,
                torch.zeros(10 - k_feat, device=eig_feat.device, dtype=eig_feat.dtype)
            ])
        energy = self.energy_head(eig_feat).squeeze()

        return {
            'energy': energy,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'H': H,
        }


# ─────────────────────────────────────────────
# 主程序：验证前向传播
# ─────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)

    config = {
        'atom_types':        10,
        'atom_embed_dim':    32,
        'node_irreps':       "16x0e + 8x1o + 4x2e",   # 16+24+36 = 76 dim
        'edge_sh_lmax':      2,
        'n_mp_layers':       2,
        'K':                 4,
        'vector_irreps':     "1x0e + 1x1o",   # dim = 1 + 3 = 4
        'basis_dim':         4,
        'hamiltonian_mode':  'sum_outer',
        'diag_mode':         'full',
        'diag_k':            8,
        'cutoff':            5.0,
    }

    model = KroneckerHamiltonianModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 随机分子：5个原子
    N = 5
    pos = torch.randn(N, 3) * 2.0
    atom_types = torch.randint(0, config['atom_types'], (N,))

    out = model(pos, atom_types)

    total_dim = N * config['basis_dim']
    print(f"\nH 形状:          {out['H'].shape}  (预期 [{total_dim}, {total_dim}])")
    print(f"特征值形状:      {out['eigenvalues'].shape}")
    print(f"特征向量形状:    {out['eigenvectors'].shape}")
    print(f"预测能量:        {out['energy'].item():.4f}")
    print(f"\n最低5个特征值:  {out['eigenvalues'][:5].detach().numpy()}")

    # 验证 H 对称性
    H = out['H']
    sym_err = (H - H.T).abs().max().item()
    print(f"\nH 对称性误差:   {sym_err:.2e}  (应接近0)")

    # 验证梯度可以传播
    loss = out['energy']
    loss.backward()
    print("梯度反传:        ✓ 成功")
