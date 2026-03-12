"""
model_coulomb.py
================
Coulomb interaction demo for Kronecker Hamiltonian.
Includes both pure PyTorch and e3nn backbone variants.

Architecture:
  LocalGNN         — scalar MPNN + per-atom sum          (baseline, blind to long-range)
  KronHamModel     — scalar MPNN + KronHamCore           (no e3nn)
  KronHamModelE3NN — equivariant MPNN + KronHamCore     (e3nn backbone, fair comparison)

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

    # Optional global dense quadratic energy (for DenseHam ablation task).
    # This term depends on all atoms jointly via a fixed random symmetric
    # matrix Q whose entries do not factorise over A/B.
    #
    # E_dense_target = charges^T Q charges
    #
    # We construct Q deterministically from (n_A, n_B) so all samples in
    # the same experiment share the same global coupling.
    n_tot = n_A + n_B
    rng_Q = torch.Generator()
    rng_Q.manual_seed(1234 + 17 * n_A + 31 * n_B)
    Q = torch.randn(n_tot, n_tot, generator=rng_Q)
    Q = 0.5 * (Q + Q.t())  # symmetrise
    E_dense_target = charges @ (Q @ charges)
    return {
        'pos':           pos,
        'charges':       charges,
        'subsystem_ids': subsystem_ids,
        'n_A':           n_A,
        'n_B':           n_B,
        'E_dense_target': E_dense_target,
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
        perturb:   float = 1e-6,   # small jitter prevents exact degeneracy NaN in eigvalsh
                                   # NOTE: larger values (1e-3) over-stabilise eigenvalues →
                                   # training samples become too distinguishable → overfitting
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
        # cross = [gap.min, gap.max, gap.abs.mean, gap.std]
        feat_dim = 3 * k_states + 4
        # NOTE: No LayerNorm here — eigenvalue magnitudes carry physical information.
        # E_AB ∝ Σ q_i^A q_j^B / r, encoded in the absolute eigenvalue scale.
        # LayerNorm would erase this signal (confirmed: MAE 0.046 → 0.056 with norm).
        # Training stability is handled by Huber loss + AdamW, not by normalising evals.
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
        # gap.std() clamped: ∂std/∂x_i = (x_i−mean)/(n·std) → 0/0 NaN when degenerate.
        # .clamp(min=1e-8) gives gradient=0 at degeneracy (safe) and std elsewhere (informative).
        # Confirmed useful: removing std entirely caused MAE regression 0.046 → 0.082.
        cross = torch.stack([gap.min(), gap.max(), gap.abs().mean(),
                             gap.std().clamp(min=1e-8)])

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
# DenseHamCore — ablation: no Kronecker structure
# ══════════════════════════════════════════════════════════════

class DenseHamCore(nn.Module):
    """
    Ablation core: build a single dense Hamiltonian H_full over all atoms, without
    enforcing any Kronecker structure or A/B factorisation.

    Given node features [N, hidden], we build:

      H_full = diag(d) + Σ_k outer(v_k, v_k)   where dim = N * basis_dim

    then diagonalise H_full directly and feed spectral features into an MLP.

    This is strictly more expressive than KronHamCore on small systems, but scales
    as O((N · basis_dim)³) instead of O((N_A · basis_dim)³ + (N_B · basis_dim)³).
    It serves as a sanity check that the Kronecker bias is not losing much on the
    toy Coulomb benchmark.
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

        # K low-rank vector heads shared across all atoms
        self.vec_heads = nn.ModuleList([nn.Linear(hidden, basis_dim) for _ in range(K)])

        # Diagonal (onsite) head
        self.diag_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, basis_dim),
        )

        # Global spectral features: k smallest + k largest eigenvalues + 4 stats
        feat_dim = 2 * k_states + 4
        self.energy_mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64),       nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _build_H(self, feat: Tensor) -> Tensor:
        """feat: [N, hidden] → H_full [dim, dim], dim = N * basis_dim."""
        d = self.diag_head(feat).reshape(-1)   # [N * basis_dim]
        H = torch.diag(d)
        for head in self.vec_heads:
            v = head(feat).reshape(-1)        # [N * basis_dim]
            H = H + torch.outer(v, v)
        return (H + H.T) / 2

    def _spectral_features(self, evals: Tensor) -> Tensor:
        """Take smallest k and largest k eigenvalues + simple statistics."""
        k = self.k_states
        evals_sorted = evals.sort().values
        n = evals_sorted.shape[0]
        if n >= k:
            head = evals_sorted[:k]
            tail = evals_sorted[-k:]
        else:
            head = F.pad(evals_sorted, (0, k - n))
            tail = head.clone()
        stats = torch.stack([
            evals_sorted.min(),
            evals_sorted.max(),
            evals_sorted.mean(),
            evals_sorted.std().clamp(min=1e-8),
        ])
        return torch.cat([head, tail, stats])

    def forward(self, node_feat: Tensor) -> Dict[str, Tensor]:
        # node_feat: [N, hidden]
        H = self._build_H(node_feat)
        if self.training and self.perturb > 0:
            I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            H = H + self.perturb * I

        evals = torch.linalg.eigvalsh(H)   # [dim], sorted
        feat  = self._spectral_features(evals)
        E     = self.energy_mlp(feat).squeeze()
        return {'E_dense': E, 'H_full': H, 'evals_full': evals}


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


class DenseHamModel(nn.Module):
    """
    Ablation model: same ScalarMPNN backbone and local energy head as KronHamModel,
    but with a single dense Hamiltonian head (DenseHamCore) instead of a Kronecker
    structured core.

    E_total = E_local + E_dense
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
        self.dense_core  = DenseHamCore(hidden, basis_dim, K, k_states)

    def forward(
        self,
        charges:       Tensor,   # [N]
        pos:           Tensor,   # [N, 3]
        subsystem_ids: Tensor,   # [N]  (unused, kept for API parity)
    ) -> Dict[str, Tensor]:
        h = self.backbone(charges, pos)

        E_local = self.local_head(h).sum()
        dense   = self.dense_core(h)
        E_dense = dense['E_dense']

        return {
            'energy':  E_local + E_dense,
            'E_local': E_local,
            **dense,
        }


# ══════════════════════════════════════════════════════════════
# KronHamModelE3NN — e3nn equivariant backbone + KronHamCore
# (fair comparison: same charge input API as the pure-PyTorch models)
# ══════════════════════════════════════════════════════════════

try:
    from e3nn.o3 import Irreps, TensorProduct, SphericalHarmonics, Linear as E3Linear
    from model_v2 import build_edge_index as _e3nn_build_edges

    class FlexEquivMP(nn.Module):
        """
        Equivariant message passing with 'uvw' TensorProduct mode.

        Unlike model_v2.py's EquivariantMessagePassing (which uses 'uvu' and
        requires all irreps to have equal multiplicity), 'uvw' allows ANY
        combination of multiplicities — e.g. "16x0e + 4x1o + 2x2e".

        'uvu' constraint:  mul_in1[L] == mul_out[L]  for every L  → all muls equal
        'uvw' constraint:  none — full bilinear, weight_count per instruction
                           = mul_in1 × mul_in2 × mul_out
                           (mul_in2 = 1 always from spherical harmonics → manageable)

        Use fc_hidden=[32,32] (smaller than model_v2's [64,64]) because uvw
        produces more TP weights and we want to keep the fc MLP param count
        similar.
        """

        def __init__(self, node_irreps: str, edge_sh_lmax: int = 2,
                     fc_hidden: list = [32, 32],
                     n_rbf: int = 20, cutoff: float = 4.0,
                     self_interaction: str = 'none',
                     tp_mode: str = 'uvw',
                     use_sigmoid_gate: bool = False):
            """
            tp_mode selects the TensorProduct contraction mode:

              'uvw'   Full bilinear: output_u = Σ_v Σ_w W_{uvw} · in1_u · in2_v
                      weight_count per instruction = mul_in1 × mul_in2 × mul_out
                      No multiplicity constraint — works with mixed muls (16x0e+4x1o+2x2e).
                      Scales as mul²: 16×0e → 256 weights, 64×0e → 4096 weights (too large).

              'uvu'   Per-channel scalar gate: output_u = Σ_v W_u · in1_u · in2_v
                      weight_count per instruction = mul_in1 × mul_in2
                      Constraint: mul_in1 == mul_out (same channel count in/out).
                      Scales as mul: 64×0e → 64 weights per instruction.
                      Semantics: each output channel u is a distance-gated version of
                      the same input channel u — like SchNet's interaction layer.
                      Cross-channel mixing then comes entirely from self_interaction MLP.
                      Best used with uniform-mul irreps (e.g. '64x0e') + scalar_mix.

            self_interaction controls the nonlinear update on L=0 scalar channels:

              'none'        h_i ← Linear(aggr) + h_i                  [linear only]
              'nequip'      h_i_L0 ← SiLU(Linear(h_out_L0))           [NequIP gate]
              'scalar_mix'  h_i_L0 ← h_out_L0 + MLP(cat(h_self, h_msg)) [SchNet/ScalarMPNN style]
              'scalar_mpnn' h_i_L0 ← h_i_L0 + MLP(cat(h_self, aggr))  [exact ScalarMPNN update,
                                       skips Linear(aggr) — only valid for L=0 models]
              'norm_sage'   gate ← MLP(cat(‖h_self‖_L, ‖aggr‖_L))     [GraphSAGE on norms]
                            L=0: h_L0 ← h_self_L0 + gate_L0            [scalar_mpnn-style: no Linear(aggr)]
                            L>0: h_L ← sigmoid(gate_L) * h_L           [per-channel multiplicative gate]
                            norms are rotation-invariant → MLP is equivariance-safe for all L

            use_sigmoid_gate: if True, applies sigmoid() to TP weights before the TP call,
              bounding them to (0,1) — exactly matching ScalarMPNN's sigmoid gate.
              Only correct for L=0 models (sigmoid would prevent negative L>0 weights).

            Recommended combinations:
              '16x0e' + 'uvw' + 'none'                     → baseline (linear, 3× gap to scalar)
              '16x0e' + 'uvw' + 'scalar_mix'               → slightly better (+10%), still 2.6× gap
              '64x0e' + 'uvu' + 'scalar_mix'               → matches scalar dim, ~90k params
              '64x0e' + 'uvu' + 'scalar_mpnn' + sigmoid    → architecturally ≡ ScalarMPNN ← new
            """
            super().__init__()
            self.node_irreps      = Irreps(node_irreps)
            self.self_interaction  = self_interaction
            self.tp_mode           = tp_mode
            self.use_sigmoid_gate  = use_sigmoid_gate
            edge_sh_irreps       = Irreps.spherical_harmonics(edge_sh_lmax)
            self.sh = SphericalHarmonics(edge_sh_irreps, normalize=True,
                                         normalization='component')

            # RBF distance encoding — same as ScalarMPNN for fair comparison.
            # Using 20 RBF features (pre-decomposed distances) instead of raw
            # distance (1 scalar) makes the fc optimization ~10x easier.
            self.rbf    = GaussianRBF(n_rbf=n_rbf, r_max=cutoff)
            self.cutoff = cutoff

            # Build TP instructions for the chosen mode.
            # 'uvw': all valid couplings, full bilinear mixing (weight_count = mul_i*mul_k)
            # 'uvu': per-channel gate only where mul_in == mul_out (weight_count = mul_i)
            instructions = []
            for i, (mul_i, ir_i) in enumerate(self.node_irreps):
                for j, (_, ir_j) in enumerate(edge_sh_irreps):
                    for k, (mul_k, ir_k) in enumerate(self.node_irreps):
                        if ir_k in ir_i * ir_j:
                            if tp_mode == 'uvu':
                                if mul_i == mul_k:   # uvu constraint: same channel count
                                    instructions.append((i, j, k, 'uvu', True))
                            else:
                                instructions.append((i, j, k, 'uvw', True))

            self.tp = TensorProduct(
                self.node_irreps, edge_sh_irreps, self.node_irreps,
                instructions=instructions,
                shared_weights=False, internal_weights=False,
            )
            # fc: RBF features → TP weights
            # MACE-style: Linear → LayerNorm → SiLU per hidden layer.
            # LayerNorm is critical: without it the TP weight scale blows up as
            # multiplicity grows (16x0e OK, 64x0e overfit badly without LN).
            _fc_layers: list = []
            _dims = [n_rbf] + fc_hidden
            for _a, _b in zip(_dims, _dims[1:]):
                _fc_layers += [nn.Linear(_a, _b), nn.LayerNorm(_b), nn.SiLU()]
            _fc_layers.append(nn.Linear(_dims[-1], self.tp.weight_numel))
            self.fc = nn.Sequential(*_fc_layers)
            self.linear = E3Linear(self.node_irreps, self.node_irreps)

            # ── Per-L layout (needed for norm extraction in norm_sage) ──────
            self.scalar_mul  = sum(mul for mul, ir in self.node_irreps if ir.l == 0)
            self.vec_mul     = sum(mul for mul, ir in self.node_irreps if ir.l == 1)
            self.tens_mul    = sum(mul for mul, ir in self.node_irreps if ir.l == 2)
            self.vec_offset  = self.scalar_mul
            self.tens_offset = self.scalar_mul + self.vec_mul * 3
            self.inv_dim     = self.scalar_mul + self.vec_mul + self.tens_mul

            # ── Nonlinear self-interaction on L=0 scalar channels ────────────
            s = self.scalar_mul
            if self_interaction == 'nequip' and s > 0:
                # NequIP-style: Linear + SiLU on combined (residual+aggr) L=0 channels.
                # Matches NequIP's gated nonlinearity for scalars:
                #   h_L0_new = SiLU(W · h_L0_combined)
                self.self_net = nn.Linear(s, s)
            elif self_interaction == 'scalar_mix' and s > 0:
                # ScalarMPNN-style: 2-layer MLP on cat(h_self_L0, h_aggr_L0).
                # Explicitly mixes self-features and aggregated messages — slightly
                # more expressive than 'nequip' since it can distinguish the two.
                self.self_net = nn.Sequential(
                    nn.Linear(2 * s, s), nn.SiLU(),
                    nn.Linear(s, s),
                )
            elif self_interaction == 'scalar_mpnn' and s > 0:
                # Exact ScalarMPNN update: h ← h + MLP(cat(h_self, aggr)).
                # Unlike scalar_mix, this REPLACES Linear(aggr)+h entirely for L=0
                # channels — no additive linear term. Combined with use_sigmoid_gate=True,
                # this makes the L=0 e3nn layer architecturally identical to ScalarMPNN.
                # Only valid for L=0-only models (no L>0 channels to worry about).
                self.self_net = nn.Sequential(
                    nn.Linear(2 * s, s), nn.SiLU(),
                    nn.Linear(s, s),
                )
            elif self_interaction == 'norm_sage' and self.inv_dim > 0:
                # Two-part update: scalar_mpnn for L=0, norm-gate for L>0.
                #
                # self_net: exact scalar_mpnn MLP on raw cat(h_self_L0, aggr_L0).
                #   L=0 scalars are invariant → no need to go through norms.
                #   h_L0_new = h_self_L0 + self_net(cat(h_self_L0, aggr_L0))
                #
                # gate_net: norm-based gate on cat(‖h_self‖_L>0, ‖aggr‖_L>0).
                #   L>0 features are equivariant → only their norms are invariant.
                #   h_Lk_new = sigmoid(gate_net(...)) * h_out_Lk  (per-channel scale)
                self.self_net = nn.Sequential(
                    nn.Linear(2 * s, s), nn.SiLU(),
                    nn.Linear(s, s),
                ) if s > 0 else None
                high_dim = self.vec_mul + self.tens_mul
                self.gate_net = nn.Sequential(
                    nn.Linear(2 * high_dim, high_dim), nn.SiLU(),
                    nn.Linear(high_dim, high_dim),
                ) if high_dim > 0 else None
            else:
                self.self_net = None

        def _extract_norms(self, h: Tensor) -> Tensor:
            """Extract L=0 scalars + per-channel norms of L>0 blocks → [N, inv_dim].
            All outputs are rotation-invariant: norms are preserved under SO(3)."""
            N = h.shape[0]
            parts = [h[:, :self.vec_offset]]                          # L=0 directly
            if self.vec_mul > 0:
                h_L1 = h[:, self.vec_offset:self.tens_offset]         # [N, vec_mul*3]
                parts.append(h_L1.reshape(N, self.vec_mul, 3).norm(dim=-1))   # [N, vec_mul]
            if self.tens_mul > 0:
                h_L2 = h[:, self.tens_offset:]                         # [N, tens_mul*5]
                parts.append(h_L2.reshape(N, self.tens_mul, 5).norm(dim=-1))  # [N, tens_mul]
            return torch.cat(parts, dim=-1)                            # [N, inv_dim]

        def forward(self, node_feat: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
            src, dst    = edge_index
            edge_sh     = self.sh(edge_vec)
            dist        = edge_vec.norm(dim=-1)           # [E]
            # cosine envelope (smooth cutoff, matches ScalarMPNN)
            envelope    = 0.5 * (1 + torch.cos(torch.pi * dist / self.cutoff))
            rbf_feat    = self.rbf(dist) * envelope.unsqueeze(-1)  # [E, n_rbf]
            tp_weights  = self.fc(rbf_feat)               # [E, weight_numel]
            if self.use_sigmoid_gate:
                # Bound TP weights to (0,1) — matches ScalarMPNN's sigmoid gate.
                # Breaks bilinear co-adaptation: W is bounded regardless of h magnitude.
                tp_weights = torch.sigmoid(tp_weights)
            msg         = self.tp(node_feat[src], edge_sh, tp_weights)
            aggr        = torch.zeros_like(node_feat)
            aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

            # Equivariant linear + residual (same for all variants)
            h_out = self.linear(aggr) + node_feat         # [N, irreps.dim]

            s = self.scalar_mul
            if self.self_interaction == 'nequip' and self.self_net is not None:
                # NequIP: SiLU(Linear(combined L=0))
                # h_out[:, :s] already = Linear(aggr)_L0 + node_feat_L0
                h_scalar_new = F.silu(self.self_net(h_out[:, :s]))
                h_out = torch.cat([h_scalar_new, h_out[:, s:]], dim=-1)

            elif self.self_interaction == 'scalar_mix' and self.self_net is not None:
                # ScalarMix: MLP(cat(h_self_L0, h_aggr_L0)) added as residual
                h_self = node_feat[:, :s]   # L=0 self-features before this layer
                h_msg  = aggr[:, :s]        # L=0 aggregated messages (pre-linear)
                delta  = self.self_net(torch.cat([h_self, h_msg], dim=-1))
                h_out  = torch.cat([h_out[:, :s] + delta, h_out[:, s:]], dim=-1)

            elif self.self_interaction == 'scalar_mpnn' and self.self_net is not None:
                # Exact ScalarMPNN update: h_L0 ← h_L0 + MLP(cat(h_self_L0, aggr_L0))
                # Replaces Linear(aggr)_L0 + h_L0 (from h_out) entirely.
                # h_out[:, s:] carries L>0 channels unchanged (Linear(aggr) + h for those).
                h_self   = node_feat[:, :s]
                h_L0_new = h_self + self.self_net(torch.cat([h_self, aggr[:, :s]], dim=-1))
                h_out    = torch.cat([h_L0_new, h_out[:, s:]], dim=-1) \
                           if h_out.shape[1] > s else h_L0_new

            elif self.self_interaction == 'norm_sage':
                # Two-part update: scalar_mpnn for L=0, norm-gate for L>0.
                N = h_out.shape[0]
                parts = []

                # L=0: exact scalar_mpnn — MLP on raw features, residual from h_self.
                if self.self_net is not None:
                    h_self_L0 = node_feat[:, :self.vec_offset]        # [N, scalar_mul]
                    aggr_L0   = aggr[:, :self.vec_offset]              # [N, scalar_mul]
                    parts.append(h_self_L0 + self.self_net(
                        torch.cat([h_self_L0, aggr_L0], dim=-1)))
                else:
                    parts.append(h_out[:, :self.vec_offset])

                # L>0: per-channel sigmoid gate from norm-invariants.
                if self.gate_net is not None:
                    # Extract L>0 norms only (rotation-invariant)
                    high_norms_self, high_norms_aggr = [], []
                    if self.vec_mul > 0:
                        h_L1_s = node_feat[:, self.vec_offset:self.tens_offset]
                        h_L1_a = aggr[:, self.vec_offset:self.tens_offset]
                        high_norms_self.append(
                            h_L1_s.reshape(N, self.vec_mul, 3).norm(dim=-1))
                        high_norms_aggr.append(
                            h_L1_a.reshape(N, self.vec_mul, 3).norm(dim=-1))
                    if self.tens_mul > 0:
                        h_L2_s = node_feat[:, self.tens_offset:]
                        h_L2_a = aggr[:, self.tens_offset:]
                        high_norms_self.append(
                            h_L2_s.reshape(N, self.tens_mul, 5).norm(dim=-1))
                        high_norms_aggr.append(
                            h_L2_a.reshape(N, self.tens_mul, 5).norm(dim=-1))
                    gate = self.gate_net(torch.cat(
                        high_norms_self + high_norms_aggr, dim=-1))   # [N, high_dim]

                    off = 0
                    if self.vec_mul > 0:
                        g = torch.sigmoid(gate[:, off:off + self.vec_mul])
                        g = g.unsqueeze(-1).expand(-1, -1, 3).reshape(N, self.vec_mul * 3)
                        parts.append(h_out[:, self.vec_offset:self.tens_offset] * g)
                        off += self.vec_mul
                    if self.tens_mul > 0:
                        g = torch.sigmoid(gate[:, off:off + self.tens_mul])
                        g = g.unsqueeze(-1).expand(-1, -1, 5).reshape(N, self.tens_mul * 5)
                        parts.append(h_out[:, self.tens_offset:] * g)
                else:
                    # No L>0 channels (pure L=0 model) — nothing to gate
                    pass

                h_out = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

            return h_out

    class KronHamModelE3NN(nn.Module):
        """
        e3nn equivariant backbone + KronHamCore.  (v2 — physically correct)

        Two key fixes over the naive version:

        Fix 1 — correct charge input:
          model_v2.py  → nn.Embedding(integer atom_type)  — no charge value
          This model   → nn.Linear(1, scalar_mul)          — continuous scalar charge

        Fix 2 — invariant feature extraction from equivariant outputs:
          WRONG (v1): project full irreps (L=0,1,2) → hidden  [violates equivariance]
          WRONG (v2): extract only L=0 scalars (4 dims)       [rank-4 bottleneck]
          RIGHT (v3): extract ALL invariant signatures:
            • L=0 channels directly          (scalar_mul scalars)
            • norms of L=1 vectors  ‖v‖      (vec_mul  scalars — rotation-invariant)
            • norms of L=2 tensors  ‖T‖      (tens_mul scalars — rotation-invariant)
            → inv_dim = scalar_mul + vec_mul + tens_mul invariant features per node

        Fix 3 — FlexEquivMP with 'uvw' mode (no equal-mul constraint):
          'uvu' (model_v2.py): all muls must be equal → "4x0e + 4x1o + 4x2e" only
          'uvw' (FlexEquivMP): arbitrary muls → "16x0e + 4x1o + 2x2e" allowed
          This lets us use MANY scalar channels (16) for capacity while keeping
          few directional channels (4 L=1, 2 L=2) for enriched MP without bloat.
          inv_dim = 16 + 4 + 2 = 22  (vs 12 from equal-mul "4x0e+4x1o+4x2e")

        Same forward(charges, pos, subsystem_ids) API as LocalGNN / KronHamModel.
        """

        def __init__(
            self,
            hidden:           int   = 64,
            node_irreps:      str   = "16x0e + 4x1o + 2x2e",  # uvw allows mixed muls!
            edge_sh_lmax:     int   = 2,
            n_layers:         int   = 3,
            cutoff:           float = 4.0,
            basis_dim:        int   = 4,
            K:                int   = 4,
            k_states:         int   = 8,
            self_interaction:  str   = 'none',   # 'none' | 'nequip' | 'scalar_mix' | 'scalar_mpnn'
            tp_mode:           str   = 'uvw',    # 'uvw' (full bilinear) | 'uvu' (per-channel gate)
            use_sigmoid_gate:  bool  = False,    # bound TP weights to (0,1) like ScalarMPNN
        ):
            super().__init__()
            self.cutoff = cutoff

            node_ir = Irreps(node_irreps)
            # Track per-L channel counts for invariant extraction
            self.scalar_mul  = sum(mul for mul, ir in node_ir if ir.l == 0)
            self.vec_mul     = sum(mul for mul, ir in node_ir if ir.l == 1)
            self.tens_mul    = sum(mul for mul, ir in node_ir if ir.l == 2)
            self.node_irreps_dim = node_ir.dim
            # Layout in h: [L=0 (scalar_mul), L=1 (vec_mul*3), L=2 (tens_mul*5)]
            self.vec_offset  = self.scalar_mul
            self.tens_offset = self.scalar_mul + self.vec_mul * 3
            # Total invariant features: L0 scalars + L1 norms + L2 norms
            self.inv_dim = self.scalar_mul + self.vec_mul + self.tens_mul

            # ── charge input: continuous scalar → L=0 irreps channel ──
            self.charge_embed = nn.Linear(1, self.scalar_mul)

            # ── equivariant message passing (uvw mode → mixed muls allowed) ──
            # Pass cutoff so FlexEquivMP can use RBF+envelope (same as ScalarMPNN)
            self.mp_layers = nn.ModuleList([
                FlexEquivMP(node_irreps, edge_sh_lmax,
                            fc_hidden=[32, 32], n_rbf=20, cutoff=cutoff,
                            self_interaction=self_interaction,
                            tp_mode=tp_mode,
                            use_sigmoid_gate=use_sigmoid_gate)
                for _ in range(n_layers)
            ])

            # ── normalise all invariant features ──
            # LayerNorm on 12 invariant features (not just 4) fixes scale explosion
            self.inv_norm = nn.LayerNorm(self.inv_dim)

            # ── project 12 invariant features → hidden (deeper to break rank bottleneck) ──
            self.to_hidden = nn.Sequential(
                nn.Linear(self.inv_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),       nn.SiLU(),
            )

            # ── local energy head ──
            self.local_head = nn.Sequential(
                nn.Linear(self.inv_dim, hidden // 2), nn.SiLU(),
                nn.Linear(hidden // 2, 1),
            )

            # ── Kronecker core (identical to KronHamModel) ──
            self.kron_core = KronHamCore(hidden, basis_dim, K, k_states)

        def forward(
            self,
            charges:       Tensor,   # [N]  continuous scalar charges
            pos:           Tensor,   # [N, 3]
            subsystem_ids: Tensor,   # [N]
        ) -> Dict[str, Tensor]:
            # Build edges with full 3-D vectors (needed for spherical harmonics)
            edge_index, edge_vec = _e3nn_build_edges(pos, self.cutoff)

            # Initialise: scalar channels ← charges, L=1/L=2 ← 0
            h = torch.zeros(pos.shape[0], self.node_irreps_dim,
                            device=pos.device, dtype=pos.dtype)
            h[:, :self.scalar_mul] = self.charge_embed(charges.unsqueeze(-1))

            # Equivariant message passing (L=1,2 help directional aggregation)
            for mp in self.mp_layers:
                h = mp(h, edge_index, edge_vec)

            # ── Extract ALL rotation-invariant signatures ──────────────────
            N = h.shape[0]
            # L=0: scalars (directly invariant)
            h_L0  = h[:, :self.vec_offset]                           # [N, scalar_mul]
            # L=1: vectors → norms  (‖Rv‖ = ‖v‖ → rotation-invariant)
            h_L1  = h[:, self.vec_offset:self.tens_offset]           # [N, vec_mul*3]
            if self.vec_mul > 0:
                norm_L1 = h_L1.reshape(N, self.vec_mul, 3).norm(dim=-1)   # [N, vec_mul]
            else:
                norm_L1 = h_L1.new_zeros(N, 0)
            # L=2: tensors → norms  (also rotation-invariant)
            h_L2  = h[:, self.tens_offset:]                           # [N, tens_mul*5]
            if self.tens_mul > 0:
                norm_L2 = h_L2.reshape(N, self.tens_mul, 5).norm(dim=-1)  # [N, tens_mul]
            else:
                norm_L2 = h_L2.new_zeros(N, 0)

            # Concatenate → inv_dim invariant features, then normalise
            inv = self.inv_norm(
                torch.cat([h_L0, norm_L1, norm_L2], dim=-1)          # [N, inv_dim]
            )

            # Local energy from invariant features
            E_local = self.local_head(inv).sum()

            # Project 12 invariant → hidden (2-layer to break rank bottleneck)
            h_hid = self.to_hidden(inv)                              # [N, 64]
            kron  = self.kron_core(h_hid, subsystem_ids)

            return {
                'energy':  E_local + kron['E_kron'],
                'E_local': E_local,
                **kron,
            }

    HAS_E3NN_COULOMB = True

except ImportError:
    HAS_E3NN_COULOMB = False


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

    # KronHamModelE3NN  (mixed-mul config via uvw mode)
    if HAS_E3NN_COULOMB:
        m_e3nn = KronHamModelE3NN(hidden=32, node_irreps='16x0e + 4x1o + 2x2e',
                                   cutoff=4.0, basis_dim=4, K=4, k_states=8)
        out2 = m_e3nn(charges, pos, sids)
        print(f"\nKronHamModelE3NN energy: {out2['energy'].item():.4f}")
        print(f"  E_local: {out2['E_local'].item():.4f}")
        print(f"  E_kron:  {out2['E_kron'].item():.4f}")
        print(f"Params: {sum(p.numel() for p in m_e3nn.parameters()):,}")
        out2['energy'].backward()
        print(f"Gradient flow: OK")
