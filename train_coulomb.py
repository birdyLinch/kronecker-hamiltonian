"""
train_coulomb.py
================
Train and compare models on the long-range Coulomb dataset:

  1. LocalGNN                              — scalar MPNN, per-atom sum  (no Kronecker)
  2. KronHamModel                          — scalar MPNN + Kronecker core
  3. KronHamModelE3NN (L=0, linear only)  — e3nn, no self-interaction   [baseline]
  4. KronHamModelE3NN (L=0, NequIP SiLU)  — e3nn + SiLU(Linear(h_L0))
  5. KronHamModelE3NN (L=0, ScalarMix)    — e3nn + MLP(cat(h_self, h_msg))

Models 3-5 are an ablation of the self-interaction style in FlexEquivMP.
All use node_irreps='16x0e' (pure scalars), so equivariance is trivially satisfied
and the only variable is how nonlinearly the scalar channels are updated.

Key question: which self-interaction style closes the 3× gap between
e3nn L=0 and the scalar KronHamModel?

  'none'       h_i ← Linear(Σ_j TP(h_j,Y,w)) + h_i          [linear, known 3× gap]
  'nequip'     h_i_L0 ← SiLU(Linear(h_out_L0))               [NequIP gated scalar]
  'scalar_mix' h_i_L0 ← h_out_L0 + MLP(cat(h_self, h_aggr))  [ScalarMPNN-style mix]

Physics setup:
  Cluster A (5 atoms, σ=0.5 Å)  ←── 8 Å ───→  Cluster B (5 atoms, σ=0.5 Å)
  GNN cutoff = 4 Å  →  ZERO A-B edges in any model.
  E_total = E_AA + E_BB + E_AB   (1/r Coulomb, all atoms carry scalar charges)
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from model_coulomb import (
    generate_dataset,
    LocalGNN,
    KronHamModel,
    HAS_E3NN_COULOMB,
)

if HAS_E3NN_COULOMB:
    from model_coulomb import KronHamModelE3NN

ALL_MODELS = ['local', 'scalar', 'e3nn-none', 'e3nn-nequip', 'e3nn-scalarmix', 'e3nn-64', 'e3nn-scalarmpnn',
              'e3nn-sigmoid-only', 'e3nn-mlpupdate-only',
              'e3nn-normsage', 'e3nn-normsage-mixed']


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

def train_epoch(model, dataset, optimiser, norm: Normaliser, device, ema_model=None):
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
            # Huber loss (δ=0.1 in normalised space ≈ 0.21 eV original):
            #   quadratic for |err| < δ  → smooth gradient near zero, no plateau
            #   linear  for |err| > δ  → robust to outliers, asymptotically MAE
            loss   = loss + torch.nn.functional.huber_loss(pred, target, delta=0.1)

        (loss / len(batch)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if ema_model is not None:
            ema_model.update_parameters(model)
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


def make_optimizer(model, lr: float, weight_decay: float):
    """
    Per-parameter-group AdamW.

    For plain scalar models (LocalGNN, KronHamModel) all params share one group.

    For equivariant models (KronHamModelE3NN), we use three groups:
      • TP fc networks  (predict TP weights from RBF) — large weight count, needs
                         finer steps  → lr × 0.3
      • E3Linear layers  (equivariant linear, constrained)     → lr × 0.5
      • Everything else  (charge_embed, KronHamCore, scalars)  → lr × 1.0

    This avoids the non-monotone test_MAE artifact where:
      the global LR decays before L>0 TP channels have aligned,
      causing temporary destabilisation of the L=0 learning.
    """
    if not hasattr(model, 'mp_layers'):
        # scalar model — single group
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # equivariant model — split into 3 groups
    tp_fc_params, e3lin_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'mp_layers' in name and '.fc.' in name:
            tp_fc_params.append(p)     # RBF → TP-weight fc networks
        elif 'mp_layers' in name and '.linear.' in name:
            e3lin_params.append(p)     # E3Linear layers
        else:
            other_params.append(p)     # scalars, KronHamCore, charge_embed, etc.

    return torch.optim.AdamW([
        {'params': other_params,  'lr': lr,        'name': 'scalar/kron'},
        {'params': e3lin_params,  'lr': lr * 0.5,  'name': 'E3Linear'},
        {'params': tp_fc_params,  'lr': lr * 0.3,  'name': 'TP-fc'},
    ], weight_decay=weight_decay)


def run(model, train_data, test_data, norm, n_epochs, lr, device, label,
        weight_decay=1e-4, warmup_epochs=10, ema_decay=0.99):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*58}")
    print(f"  {label}  ({n_params:,} params)")
    print(f"{'─'*58}")

    opt = make_optimizer(model, lr, weight_decay)

    # EMA: shadow copy of weights, evaluated at test time (MACE/NequIP style)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    # Linear warmup (0.1×lr → lr) then cosine decay (lr → lr/50)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs - warmup_epochs, eta_min=lr / 50)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        loss = train_epoch(model, train_data, opt, norm, device, ema_model=ema_model)
        sched.step()
        if ep % (n_epochs // 5) == 0 or ep == 1:
            mae = evaluate(ema_model, test_data, norm, device)
            print(f"  epoch {ep:3d}/{n_epochs}  loss={loss:.4f}  "
                  f"test_MAE={mae:.4f}  ({time.time()-t0:.0f}s)"
                  f"  [train≈{loss*norm.std:.4f}]")

    return evaluate(ema_model, test_data, norm, device)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Kronecker Hamiltonian Coulomb experiment')
    parser.add_argument(
        '--models', nargs='+', default=ALL_MODELS,
        metavar='MODEL',
        help=(f'models to run (default: all). choices: {ALL_MODELS}'),
    )
    parser.add_argument('--epochs', type=int, default=100, help='training epochs (default: 100)')
    parser.add_argument('--lr',     type=float, default=3e-3, help='learning rate (default: 3e-3)')
    args = parser.parse_args()

    # validate
    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}. choices: {ALL_MODELS}")

    torch.manual_seed(42)
    device = 'cpu'

    # ── dataset ─────────────────────────────────────────────
    N_A, N_B     = 5, 5
    SEPARATION   = 8.0
    CLUSTER_STD  = 0.5
    CUTOFF       = 4.0
    CHARGE_SCALE = 1.0
    N_EPOCHS     = args.epochs
    LR           = args.lr

    print("=" * 58)
    print(" Kronecker Hamiltonian — Long-Range Coulomb Experiment")
    print("=" * 58)
    print(f"\nGeometry: n_A={N_A}, n_B={N_B}, sep={SEPARATION} Å, cutoff={CUTOFF} Å")
    print(f"Models:   {args.models}  |  epochs={N_EPOCHS}  lr={LR}")
    print("Generating dataset …")

    train_data = generate_dataset(1000, N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=0)
    test_data  = generate_dataset(200,  N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=1000)

    e_totals = [s['E_total'].item() for s in train_data]
    e_abs    = [s['E_AB'].item()    for s in train_data]
    norm     = Normaliser(e_totals)

    print(f"\nDataset statistics (train, N={len(train_data)}):")
    print(f"  E_total : std = {np.std(e_totals):.3f}")
    print(f"  E_AB    : std = {np.std(e_abs):.3f}  ← LocalGNN irreducible error floor")

    # ── shared configs ───────────────────────────────────────
    scalar_cfg   = dict(hidden=64, n_rbf=20, cutoff=CUTOFF, n_layers=3)
    kron_cfg     = dict(basis_dim=4, K=4, k_states=8)
    e3nn_base    = dict(hidden=64, edge_sh_lmax=2, n_layers=3, cutoff=CUTOFF, **kron_cfg)

    # ── model table ─────────────────────────────────────────
    # short_name → (display_label, factory)
    model_table = {
        'local': (
            'LocalGNN (no e3nn, no Kronecker)',
            lambda: LocalGNN(**scalar_cfg).to(device),
        ),
        'scalar': (
            'KronHamModel (no e3nn, +Kronecker)',
            lambda: KronHamModel(**scalar_cfg, **kron_cfg).to(device),
        ),
        'e3nn-none': (
            'KronHamModelE3NN (16x0e, uvw, linear)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='16x0e',
                                     self_interaction='none', tp_mode='uvw').to(device),
        ),
        'e3nn-nequip': (
            'KronHamModelE3NN (16x0e, uvw, NequIP)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='16x0e',
                                     self_interaction='nequip', tp_mode='uvw').to(device),
        ),
        'e3nn-scalarmix': (
            'KronHamModelE3NN (16x0e, uvw, ScalarMix)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='16x0e',
                                     self_interaction='scalar_mix', tp_mode='uvw').to(device),
        ),
        'e3nn-64': (
            'KronHamModelE3NN (64x0e, uvu, ScalarMix)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='64x0e',
                                     self_interaction='scalar_mix', tp_mode='uvu').to(device),
        ),
        'e3nn-scalarmpnn': (
            'KronHamModelE3NN (64x0e, uvu, ScalarMPNN-style: sigmoid gate + MLP update)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='64x0e',
                                     self_interaction='scalar_mpnn', tp_mode='uvu',
                                     use_sigmoid_gate=True).to(device),
        ),
        'e3nn-sigmoid-only': (
            'KronHamModelE3NN (64x0e, uvu, sigmoid gate only, Linear update)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='64x0e',
                                     self_interaction='scalar_mix', tp_mode='uvu',
                                     use_sigmoid_gate=True).to(device),
        ),
        'e3nn-mlpupdate-only': (
            'KronHamModelE3NN (64x0e, uvu, MLP update only, no sigmoid)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='64x0e',
                                     self_interaction='scalar_mpnn', tp_mode='uvu',
                                     use_sigmoid_gate=False).to(device),
        ),
        'e3nn-normsage': (
            'KronHamModelE3NN (64x0e, uvu, NormSAGE — L=0 only, equiv to ScalarMPNN)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='64x0e',
                                     self_interaction='norm_sage', tp_mode='uvu').to(device),
        ),
        'e3nn-normsage-mixed': (
            'KronHamModelE3NN (16x0e+4x1o+2x2e, uvw, NormSAGE — L>0 norm gating)',
            lambda: KronHamModelE3NN(**e3nn_base, node_irreps='16x0e + 4x1o + 2x2e',
                                     self_interaction='norm_sage', tp_mode='uvw').to(device),
        ),
    }

    # ── run selected models ──────────────────────────────────
    results = {}
    for key in args.models:
        label, factory = model_table[key]
        if key.startswith('e3nn') and not HAS_E3NN_COULOMB:
            print(f"\n[skip] {label} — e3nn not installed")
            continue
        m = factory()
        results[label] = run(m, train_data, test_data, norm, N_EPOCHS, LR, device, label=label)

    # ── summary ──────────────────────────────────────────────
    if not results:
        return

    std_e_ab  = float(np.std([s['E_AB'].item() for s in test_data]))
    mae_local = results.get('LocalGNN (no e3nn, no Kronecker)')
    mae_kron  = results.get('KronHamModel (no e3nn, +Kronecker)')

    print(f"\n{'='*58}")
    print(" Results Summary")
    print(f"{'='*58}")
    print(f"  std(E_AB) test = {std_e_ab:.4f}  ← LocalGNN cannot beat this\n")
    print(f"  {'Model':<50} {'MAE':>7}  {'vs scalar':>10}")
    print(f"  {'-'*70}")
    for name, mae in results.items():
        if mae_kron:
            vs = f"{mae_kron/mae:.2f}x" if mae > mae_kron * 0.5 else f"{mae_kron/mae:.1f}x better"
        elif mae_local:
            vs = f"{mae_local/mae:.1f}x better" if mae < mae_local else "—"
        else:
            vs = "—"
        print(f"  {name:<50} {mae:>7.4f}  {vs:>10}")

    if mae_kron and mae_local and mae_kron < mae_local * 0.80:
        print("\n✓ Kronecker spectral coupling captures long-range E_AB")


if __name__ == '__main__':
    main()
