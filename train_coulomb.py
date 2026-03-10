"""
train_coulomb.py
================
Train and compare three models on the long-range Coulomb dataset:

  1. LocalGNN          — scalar MPNN, per-atom sum       (no e3nn, no Kronecker)
  2. KronHamModel      — scalar MPNN + Kronecker core    (no e3nn, +Kronecker)
  3. KronHamModelE3NN  — equivariant MPNN + Kronecker    (e3nn backbone, +Kronecker)

All three share the same:
  • forward(charges, pos, subsystem_ids) → {'energy': ...} API
  • continuous scalar charge input: nn.Linear(1, ·) — NOT nn.Embedding(atom_type)
  • KronHamCore (models 2 & 3 only)

Key question: does e3nn's rotational equivariance help for this scalar task?

Physics setup:
  Cluster A (5 atoms, σ=0.5 Å)  ←── 8 Å ───→  Cluster B (5 atoms, σ=0.5 Å)
  GNN cutoff = 4 Å  →  ZERO A-B edges in any model.
  E_total = E_AA + E_BB + E_AB   (1/r Coulomb, all atoms carry scalar charges)

Expected outcome:
  LocalGNN:         MAE ≈ std(E_AB)        cannot learn long-range at all
  KronHamModel:     MAE << std(E_AB)       Kronecker spectral coupling helps
  KronHamModelE3NN: similar to KronHamModel  equivariance has limited benefit
                    for a scalar energy task with isotropic (Gaussian) clusters
"""

import time
import torch
import torch.nn as nn
import numpy as np

from model_coulomb import (
    generate_dataset,
    LocalGNN,
    KronHamModel,
    HAS_E3NN_COULOMB,
)

if HAS_E3NN_COULOMB:
    from model_coulomb import KronHamModelE3NN


# ══════════════════════════════════════════════════════════════
# Normaliser
# ══════════════════════════════════════════════════════════════

class Normaliser:
    """Shift-scale normalisation fitted on training targets."""

    def __init__(self, values: list):
        arr = np.array(values, dtype=np.float32)
        self.mean = float(arr.mean())
        self.std  = max(float(arr.std()), 1e-6)

    def encode(self, x: float) -> float:
        return (x - self.mean) / self.std

    def decode(self, x: float) -> float:
        return x * self.std + self.mean


# ══════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════

def train_epoch(model, dataset, optimiser, norm: Normaliser, device):
    model.train()
    indices    = torch.randperm(len(dataset)).tolist()
    total_loss = 0.0
    BATCH      = 32

    for start in range(0, len(dataset), BATCH):
        batch = [dataset[i] for i in indices[start: start + BATCH]]
        optimiser.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for s in batch:
            pred   = model(s['charges'].to(device),
                           s['pos'].to(device),
                           s['subsystem_ids'].to(device))['energy']
            target = torch.tensor(norm.encode(s['E_total'].item()),
                                  dtype=torch.float32, device=device)
            loss   = loss + (pred - target) ** 2

        (loss / len(batch)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += (loss / len(batch)).item()

    return total_loss / max(1, len(dataset) // BATCH)


@torch.no_grad()
def evaluate(model, dataset, norm: Normaliser, device):
    model.eval()
    errs = []
    for s in dataset:
        pred = model(s['charges'].to(device),
                     s['pos'].to(device),
                     s['subsystem_ids'].to(device))['energy'].item()
        errs.append(abs(norm.decode(pred) - s['E_total'].item()))
    return float(np.mean(errs))


def run(model, train_data, test_data, norm, n_epochs, lr, device, label):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*58}")
    print(f"  {label}  ({n_params:,} params)")
    print(f"{'─'*58}")

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr / 20)

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        loss = train_epoch(model, train_data, opt, norm, device)
        sched.step()
        if ep % (n_epochs // 5) == 0 or ep == 1:
            mae = evaluate(model, test_data, norm, device)
            print(f"  epoch {ep:3d}/{n_epochs}  loss={loss:.4f}  "
                  f"test_MAE={mae:.4f}  ({time.time()-t0:.0f}s)")

    return evaluate(model, test_data, norm, device)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    device = 'cpu'

    # ── dataset ─────────────────────────────────────────────
    N_A, N_B     = 5, 5
    SEPARATION   = 8.0    # Å  >> cutoff → zero A-B edges
    CLUSTER_STD  = 0.5    # Å  tight clusters, all intra pairs within cutoff
    CUTOFF       = 4.0    # Å
    CHARGE_SCALE = 1.0
    N_EPOCHS     = 100
    LR           = 3e-3

    print("=" * 58)
    print(" Kronecker Hamiltonian — Long-Range Coulomb Experiment")
    print("=" * 58)
    print(f"\nGeometry: n_A={N_A}, n_B={N_B}, sep={SEPARATION} Å, cutoff={CUTOFF} Å")
    print("Generating dataset …")

    train_data = generate_dataset(1000, N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=0)
    test_data  = generate_dataset(200,  N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=1000)

    e_totals = [s['E_total'].item() for s in train_data]
    e_abs    = [s['E_AB'].item()    for s in train_data]
    norm     = Normaliser(e_totals)

    print(f"\nDataset statistics (train, N={len(train_data)}):")
    print(f"  E_total : std = {np.std(e_totals):.3f}")
    print(f"  E_AB    : std = {np.std(e_abs):.3f}  ← LocalGNN irreducible error floor")

    # ── shared backbone config ───────────────────────────────
    scalar_cfg = dict(hidden=64, n_rbf=20, cutoff=CUTOFF, n_layers=3)
    kron_cfg   = dict(basis_dim=4, K=4, k_states=8)

    results = {}

    # 1. LocalGNN
    m1 = LocalGNN(**scalar_cfg).to(device)
    results['LocalGNN (no e3nn, no Kronecker)'] = run(
        m1, train_data, test_data, norm, N_EPOCHS, LR, device,
        label='LocalGNN (no e3nn, no Kronecker)',
    )

    # 2. KronHamModel — pure PyTorch
    m2 = KronHamModel(**scalar_cfg, **kron_cfg).to(device)
    results['KronHamModel (no e3nn, +Kronecker)'] = run(
        m2, train_data, test_data, norm, N_EPOCHS, LR, device,
        label='KronHamModel (no e3nn, +Kronecker)',
    )

    # 3. KronHamModelE3NN — equivariant backbone
    if HAS_E3NN_COULOMB:
        m3 = KronHamModelE3NN(
            hidden=64, node_irreps='4x0e + 4x1o + 4x2e',
            edge_sh_lmax=2, n_layers=3,
            cutoff=CUTOFF, **kron_cfg,
        ).to(device)
        results['KronHamModelE3NN (e3nn, +Kronecker)'] = run(
            m3, train_data, test_data, norm, N_EPOCHS, LR, device,
            label='KronHamModelE3NN (e3nn backbone, +Kronecker)',
        )
    else:
        print("\n[skip] e3nn not installed — KronHamModelE3NN not tested")

    # ── summary ──────────────────────────────────────────────
    std_e_ab  = float(np.std([s['E_AB'].item() for s in test_data]))
    mae_local = results['LocalGNN (no e3nn, no Kronecker)']

    print(f"\n{'='*58}")
    print(" Results Summary")
    print(f"{'='*58}")
    print(f"  std(E_AB) test = {std_e_ab:.4f}  ← LocalGNN cannot beat this\n")
    print(f"  {'Model':<42} {'MAE':>7}  {'vs LocalGNN':>11}")
    print(f"  {'-'*62}")
    for name, mae in results.items():
        vs = f"{mae_local/mae:.1f}x better" if mae < mae_local else "—"
        print(f"  {name:<42} {mae:>7.4f}  {vs:>11}")

    print()
    # Kronecker gain
    mae_kron = results.get('KronHamModel (no e3nn, +Kronecker)', None)
    if mae_kron and mae_kron < mae_local * 0.80:
        print("✓ Kronecker spectral coupling captures long-range E_AB")

    # e3nn vs pure-PyTorch comparison
    mae_e3nn = results.get('KronHamModelE3NN (e3nn, +Kronecker)', None)
    if mae_e3nn is not None and mae_kron is not None:
        ratio = mae_kron / mae_e3nn
        if   ratio > 1.15:
            verdict = "✓ e3nn backbone is better  (orientation info helps)"
        elif ratio < 0.85:
            verdict = "≈ e3nn backbone is WORSE  (overhead > benefit for scalar task)"
        else:
            verdict = "≈ e3nn and plain PyTorch are similar  (equivariance not decisive)"
        print(f"{verdict}")
        print("  (expected: Coulomb energy is scalar/isotropic → equivariance"
              " less critical than for tensor properties like forces / polarisability)")


if __name__ == '__main__':
    main()
