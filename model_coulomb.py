"""
model_coulomb.py
================
Pure PyTorch (no e3nn) Coulomb interaction demo for Kronecker Hamiltonian.

Demonstrates that the Kronecker structure can capture long-range A-B interactions
that a standard local GNN cannot (due to cutoff barrier).

Architecture:
  LocalGNN     — MPNN backbone + per-atom energy sum  (baseline, blind to long-range)
  KronHamModel — MPNN backbone + KronHamCore          (captures E_AB via spectra)

Dataset:
  Two atomic clusters A and B, separated by >> cutoff.
  Atoms have scalar charges. Energy = Coulomb sum:
    E_total = E_AA + E_BB + E_AB   (1/r potential)
  LocalGNN can fit E_AA + E_BB but NOT E_AB (cutoff barrier).
  KronHamModel can implicitly learn E_AB through the Kronecker spectral structure.

Why KronHam captures E_AB without any A-B edges:
  H_A eigenvalues encode A's charge-distribution spectrum.
  H_B eigenvalues encode B's charge-distribution spectrum.
  Combined eigenvalues ε_ij = ε_i^A + ε_j^B contain cross-system products,
  which correlate with E_AB = Σ q_i^A q_j^B / r_ij.
  The energy MLP learns this correlation implicitly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List


# ══════════════════════════════════════════════════════════════
# Dataset generation
# ══════════════════════════════════════════════════════════════

def compute_coulomb_energies(
    pos: Tensor,      # [N, 3]
    charges: Tensor,  # [N]
    n_A: int,
) -> Dict[str, Tensor]:
    """Compute E_AA, E_BB, E_AB, E_total via 1/r Coulomb potential."""
    n_B = pos.shape[0] - n_A

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # [N, N, 3]
    dist = diff.norm(dim=-1).clamp(min=1e-3)            # [N, N]
    Q    = charges.unsqueeze(0) * charges.unsqueeze(1)  # [N, N]
    coulomb = Q / dist                                   # [N, N]

    E_AA = coulomb[:n_A, :n_A].triu(diagonal=1).sum()
    E_BB = coulomb[n_A:, n_A:].triu(diagonal=1).sum()
    E_AB = coulomb[:n_A, n_A:].sum()

    return {
        'E_total': E_AA + E_BB + E_AB,
        'E_local': E_AA + E_BB,
        'E_AB':    E_AB,
    }


def generate_sample(
    n_A: int = 5,
    n_B: int = 5,
    separation: float = 8.0,
    cluster_std: float = 0.5,
    charge_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Generate one sample: two clusters A and B separated along x-axis.

    A: centred at origin, B: centred at (separation, 0, 0).
    Charges: uniform in [-charge_scale, +charge_scale].
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    pos_A = torch.randn(n_A, 3, generator=rng) * cluster_std
    pos_B = torch.randn(n_B, 3, generator=rng) * cluster_std
    pos_B[:, 0] += separation

    pos     = torch.cat([pos_A, pos_B], dim=0)
    charges = (torch.rand(n_A + n_B, generator=rng) * 2 - 1) * charge_scale

    subsystem_ids = torch.cat([
        torch.zeros(n_A, dtype=torch.long),
        torch.ones(n_B,  dtype=torch.long),
    ])

    energies = compute_coulomb_energies(pos, charges, n_A)
    return {
        'pos':           pos,
        'charges':       charges,
        'subsystem_ids': subsystem_ids,
        'n_A':           n_A,
        'n_B':           n_B,
        **energies,
    }


def generate_dataset(
    n_samples:    int   = 1000,
    n_A:          int   = 5,
    n_B:          int   = 5,
    separation:   float = 8.0,
    cluster_std:  float = 0.5,
    charge_scale: float = 1.0,
    seed:         int   = 0,
) -> List[Dict[str, Tensor]]:
    return [
        generate_sample(n_A, n_B, separation, cluster_std, charge_scale, seed + i)
        for i in range(n_samples)
    ]


# ══════════════════════════════════════════════════════════════
# GNN building blocks
# ══════════════════════════════════════════════════════════════

class GaussianRBF(nn.Module):
    """Scalar distances → Gaussian radial basis functions."""

    def __init__(self, n_rbf: int = 20, r_max: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, r_max, n_rbf)
        self.register_buffer('centers', centers)
        self.width = (r_max / n_rbf) ** 2

    def forward(self, dist: Tensor) -> Tensor:
        """dist: [E] → [E, n_rbf]"""
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / self.width)


def build_edges(pos: Tensor, cutoff: float):
    """All pairs within cutoff. Returns (edge_index [2,E], dist [E], rbf_envelope [E])."""
    N = pos.shape[0]
    idx = torch.arange(N, device=pos.device)
    ii, jj = torch.meshgrid(idx, idx, indexing='ij')
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]

    edge_vec = pos[jj] - pos[ii]
    dist     = edge_vec.norm(dim=-1)
    keep     = dist < cutoff

    dist_k = dist[keep]
    # Cosine envelope for smooth cutoff
    envelope = 0.5 * (1.0 + torch.cos(math.pi * dist_k / cutoff))

    return (
        torch.stack([ii[keep], jj[keep]], dim=0),  # [2, E']
        dist_k,                                     # [E']
        envelope,                                   # [E']
    )


class ScalarMPNNLayer(nn.Module):
    """One distance-gated scalar message-passing layer."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(n_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),          # gate ∈ (0,1)
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: Tensor, edge_index: Tensor, rbf: Tensor) -> Tensor:
        src, dst = edge_index
        gate = self.edge_net(rbf)                    # [E, hidden]
        msg  = gate * h[src]                         # [E, hidden]

        aggr = torch.zeros_like(h)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        return h + self.update_net(torch.cat([h, aggr], dim=-1))


class ScalarMPNN(nn.Module):
    """
    Pure PyTorch scalar MPNN.

    Input:  charges [N], positions [N, 3]
    Output: node features [N, hidden]
    """

    def __init__(
        self,
        hidden:   int   = 64,
        n_rbf:    int   = 20,
        cutoff:   float = 4.0,
        n_layers: int   = 3,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.rbf    = GaussianRBF(n_rbf, r_max=cutoff)

        self.charge_embed = nn.Linear(1, hidden)
        self.layers       = nn.ModuleList([
            ScalarMPNNLayer(hidden, n_rbf) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, charges: Tensor, pos: Tensor) -> Tensor:
        """Returns node features [N, hidden]."""
        edge_index, dist, envelope = build_edges(pos, self.cutoff)
        rbf = self.rbf(dist) * envelope.unsqueeze(-1)   # smooth RBF

        h = F.silu(self.charge_embed(charges.unsqueeze(-1)))
        for layer in self.layers:
            h = layer(h, edge_index, rbf)
        return self.out_norm(h)


# ══════════════════════════════════════════════════════════════
# Kronecker Hamiltonian core  (backbone-agnostic, pure nn.Linear)
# ══════════════════════════════════════════════════════════════

class KronHamCore(nn.Module):
    """
    Backbone-agnostic Kronecker Hamiltonian.

    Given node features from any backbone, builds:
      H_A = diag(d_A) + Σ_k outer(v_Ak, v_Ak)   [dim_A × dim_A]
      H_B = diag(d_B) + Σ_k outer(v_Bk, v_Bk)   [dim_B × dim_B]

    where dim_X = n_X * basis_dim (each atom contributes basis_dim basis functions).

    Analytically diagonalises via Kronecker structure:
      ε_ij = ε_i^A + ε_j^B        (no O((dim_A·dim_B)³) brute-force diag needed)

    Extracts spectral features → MLP → E_kron.
    """

    def __init__(
        self,
        hidden:    int   = 64,
        basis_dim: int   = 4,
        K:         int   = 4,
        k_states:  int   = 8,
        perturb:   float = 1e-6,
    ):
        super().__init__()
        self.basis_dim = basis_dim
        self.K         = K
        self.k_states  = k_states
        self.perturb   = perturb

        # K low-rank vector heads per subsystem
        self.vec_A = nn.ModuleList([nn.Linear(hidden, basis_dim) for _ in range(K)])
        self.vec_B = nn.ModuleList([nn.Linear(hidden, basis_dim) for _ in range(K)])

        # Diagonal (onsite) heads
        self.diag_A = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, basis_dim),
        )
        self.diag_B = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, basis_dim),
        )

        # Spectral feature → energy MLP
        # Features: k_states global evals + k_states A evals + k_states B evals + 4 cross stats
        feat_dim = 3 * k_states + 4
        self.energy_mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64),      nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _build_H(
        self,
        feat:      Tensor,          # [n_sub, hidden]
        vec_heads: nn.ModuleList,
        diag_head: nn.Module,
    ) -> Tensor:
        """Build one subsystem Hamiltonian [dim, dim]."""
        d = diag_head(feat).reshape(-1)    # [n_sub * basis_dim]
        H = torch.diag(d)
        for head in vec_heads:
            v = head(feat).reshape(-1)     # [n_sub * basis_dim]
            H = H + torch.outer(v, v)
        return (H + H.T) / 2               # enforce symmetry

    def _spectral_features(self, evals_A: Tensor, evals_B: Tensor) -> Tensor:
        k = self.k_states

        # Global Kronecker eigenvalues: ε_ij = ε_i^A + ε_j^B
        evals_global = (evals_A.unsqueeze(1) + evals_B.unsqueeze(0)).reshape(-1)
        evals_global = evals_global.sort().values

        def pad_or_trim(x: Tensor) -> Tensor:
            n = x.shape[0]
            if n >= k:
                return x[:k]
            return F.pad(x, (0, k - n))

        feat_g = pad_or_trim(evals_global)
        feat_A = pad_or_trim(evals_A)
        feat_B = pad_or_trim(evals_B)

        # Cross-spectrum statistics (capture A-B spectral interaction)
        gap = evals_A[:k].unsqueeze(1) - evals_B[:k].unsqueeze(0)  # [k, k]
        cross = torch.stack([gap.min(), gap.max(), gap.abs().mean(), gap.std()])

        return torch.cat([feat_g, feat_A, feat_B, cross])   # [3k+4]

    def forward(
        self,
        node_feat:     Tensor,   # [N, hidden]
        subsystem_ids: Tensor,   # [N]  0=A, 1=B
    ) -> Dict[str, Tensor]:
        feat_A = node_feat[subsystem_ids == 0]   # [n_A, hidden]
        feat_B = node_feat[subsystem_ids == 1]   # [n_B, hidden]

        H_A = self._build_H(feat_A, self.vec_A, self.diag_A)
        H_B = self._build_H(feat_B, self.vec_B, self.diag_B)

        if self.training and self.perturb > 0:
            I_A = torch.eye(H_A.shape[0], device=H_A.device, dtype=H_A.dtype)
            I_B = torch.eye(H_B.shape[0], device=H_B.device, dtype=H_B.dtype)
            H_A = H_A + self.perturb * I_A
            H_B = H_B + self.perturb * I_B

        evals_A = torch.linalg.eigvalsh(H_A)    # [dim_A], sorted ↑
        evals_B = torch.linalg.eigvalsh(H_B)    # [dim_B], sorted ↑

        feat    = self._spectral_features(evals_A, evals_B)
        E_kron  = self.energy_mlp(feat).squeeze()

        return {'E_kron': E_kron, 'H_A': H_A, 'H_B': H_B,
                'evals_A': evals_A, 'evals_B': evals_B}


# ══════════════════════════════════════════════════════════════
# LocalGNN — baseline (provably blind to long-range E_AB)
# ══════════════════════════════════════════════════════════════

class LocalGNN(nn.Module):
    """
    Baseline model: MPNN backbone + per-atom energy sum.

    When cluster separation >> cutoff, there are zero A-B edges,
    so the model's prediction is identically independent of how
    A's charges correlate with B's charges.
    """

    def __init__(
        self,
        hidden:   int   = 64,
        n_rbf:    int   = 20,
        cutoff:   float = 4.0,
        n_layers: int   = 3,
    ):
        super().__init__()
        self.backbone    = ScalarMPNN(hidden, n_rbf, cutoff, n_layers)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        charges:       Tensor,   # [N]
        pos:           Tensor,   # [N, 3]
        subsystem_ids: Tensor,   # [N]  (unused — kept for API consistency)
    ) -> Dict[str, Tensor]:
        h = self.backbone(charges, pos)
        E = self.energy_head(h).sum()
        return {'energy': E}


# ══════════════════════════════════════════════════════════════
# KronHamModel — full model (local short-range + Kronecker long-range)
# ══════════════════════════════════════════════════════════════

class KronHamModel(nn.Module):
    """
    Full model: E_total = E_local + E_kron

    E_local: per-atom sum from MPNN backbone (captures intra-cluster interactions)
    E_kron:  Kronecker spectral energy        (captures inter-cluster A-B interactions)

    Both use the same shared backbone weights.
    """

    def __init__(
        self,
        hidden:    int   = 64,
        n_rbf:     int   = 20,
        cutoff:    float = 4.0,
        n_layers:  int   = 3,
        basis_dim: int   = 4,
        K:         int   = 4,
        k_states:  int   = 8,
    ):
        super().__init__()
        self.backbone    = ScalarMPNN(hidden, n_rbf, cutoff, n_layers)
        self.local_head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.kron_core   = KronHamCore(hidden, basis_dim, K, k_states)

    def forward(
        self,
        charges:       Tensor,   # [N]
        pos:           Tensor,   # [N, 3]
        subsystem_ids: Tensor,   # [N]
    ) -> Dict[str, Tensor]:
        h = self.backbone(charges, pos)

        E_local = self.local_head(h).sum()
        kron    = self.kron_core(h, subsystem_ids)
        E_kron  = kron['E_kron']

        return {
            'energy':  E_local + E_kron,
            'E_local': E_local,
            **kron,
        }


# ══════════════════════════════════════════════════════════════
# Quick sanity check
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(0)

    sample = generate_sample(n_A=5, n_B=5, separation=8.0, seed=42)
    print(f"E_total = {sample['E_total']:.4f}")
    print(f"E_local = {sample['E_local']:.4f}  (E_AA + E_BB)")
    print(f"E_AB    = {sample['E_AB']:.4f}")

    charges = sample['charges']
    pos     = sample['pos']
    sids    = sample['subsystem_ids']

    # LocalGNN
    m_local = LocalGNN(hidden=32, cutoff=4.0)
    out = m_local(charges, pos, sids)
    print(f"\nLocalGNN energy: {out['energy'].item():.4f}")
    print(f"Params: {sum(p.numel() for p in m_local.parameters()):,}")

    # KronHamModel
    m_kron = KronHamModel(hidden=32, cutoff=4.0, basis_dim=4, K=4, k_states=8)
    out = m_kron(charges, pos, sids)
    print(f"\nKronHamModel energy:  {out['energy'].item():.4f}")
    print(f"  E_local: {out['E_local'].item():.4f}")
    print(f"  E_kron:  {out['E_kron'].item():.4f}")
    print(f"  H_A:     {out['H_A'].shape}  (dim_A = 5*4 = 20)")
    print(f"  H_B:     {out['H_B'].shape}")
    print(f"Params: {sum(p.numel() for p in m_kron.parameters()):,}")

    # Gradient check
    out['energy'].backward()
    print(f"\nGradient flow: OK")
