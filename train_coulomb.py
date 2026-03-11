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
            # Huber loss (δ=0.1 in normalised space ≈ 0.21 eV original):
            #   quadratic for |err| < δ  → smooth gradient near zero, no plateau
            #   linear  for |err| > δ  → robust to outliers, asymptotically MAE
            loss   = loss + torch.nn.functional.huber_loss(pred, target, delta=0.1)

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
        weight_decay=1e-4, warmup_epochs=10):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*58}")
    print(f"  {label}  ({n_params:,} params)")
    print(f"{'─'*58}")

    opt = make_optimizer(model, lr, weight_decay)

    # Linear warmup (0.1×lr → lr) then cosine decay (lr → lr/50)
    # Warmup gives L>0 channels time to align before LR starts falling —
    # eliminates the non-monotone test_MAE artifact seen in L=0+1 / L=0+1+2 models.
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs - warmup_epochs, eta_min=lr / 50)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        loss = train_epoch(model, train_data, opt, norm, device)
        sched.step()
        if ep % (n_epochs // 5) == 0 or ep == 1:
            mae = evaluate(model, test_data, norm, device)
            # loss is Huber in normalised space; ×std gives rough train-MAE proxy
            print(f"  epoch {ep:3d}/{n_epochs}  loss={loss:.4f}  "
                  f"test_MAE={mae:.4f}  ({time.time()-t0:.0f}s)"
                  f"  [train≈{loss*norm.std:.4f}]")

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

    # ── e3nn sanity check: three self-interaction variants ───
    # All use L=0 only (pure scalars, identical information to KronHamModel).
    # Goal: find which self-interaction style closes the 3× gap vs ScalarMPNN.
    #
    #  'none'        current baseline — linear only, known 3× gap
    #  'nequip'      SiLU(Linear(h_L0_combined)) — NequIP gated nonlinearity
    #  'scalar_mix'  MLP(cat(h_self_L0, h_aggr_L0)) — explicit self+msg mixing
    #
    # L=0 scalars are SO(3)-invariant → all three variants are equivariant.
    if HAS_E3NN_COULOMB:
        e3nn_base_cfg = dict(
            hidden=64, node_irreps='16x0e',
            edge_sh_lmax=2, n_layers=3,
            cutoff=CUTOFF, **kron_cfg,
        )
        for si_mode in ('none', 'nequip', 'scalar_mix'):
            label = {
                'none':        'KronHamModelE3NN (L=0, linear only)',
                'nequip':      'KronHamModelE3NN (L=0, NequIP SiLU)',
                'scalar_mix':  'KronHamModelE3NN (L=0, ScalarMix MLP)',
            }[si_mode]
            m = KronHamModelE3NN(**e3nn_base_cfg, self_interaction=si_mode).to(device)
            results[label] = run(
                m, train_data, test_data, norm, N_EPOCHS, LR, device, label=label,
            )
    else:
        print("\n[skip] e3nn not installed — KronHamModelE3NN not tested")

    # ── summary ──────────────────────────────────────────────
    std_e_ab  = float(np.std([s['E_AB'].item() for s in test_data]))
    mae_local = results['LocalGNN (no e3nn, no Kronecker)']
    mae_kron  = results.get('KronHamModel (no e3nn, +Kronecker)')

    print(f"\n{'='*58}")
    print(" Results Summary")
    print(f"{'='*58}")
    print(f"  std(E_AB) test = {std_e_ab:.4f}  ← LocalGNN cannot beat this\n")
    print(f"  {'Model':<46} {'MAE':>7}  {'vs LocalGNN':>11}")
    print(f"  {'-'*66}")
    for name, mae in results.items():
        vs = f"{mae_local/mae:.1f}x better" if mae < mae_local else "—"
        print(f"  {name:<46} {mae:>7.4f}  {vs:>11}")

    print()
    if mae_kron and mae_kron < mae_local * 0.80:
        print("✓ Kronecker spectral coupling captures long-range E_AB")

    # Self-interaction comparison
    mae_none  = results.get('KronHamModelE3NN (L=0, linear only)')
    mae_neq   = results.get('KronHamModelE3NN (L=0, NequIP SiLU)')
    mae_smix  = results.get('KronHamModelE3NN (L=0, ScalarMix MLP)')
    if mae_kron and mae_none:
        print(f"\n── Self-interaction ablation (target: scalar={mae_kron:.4f}) ──")
        for name, mae in [('none (linear)', mae_none),
                          ('NequIP SiLU',   mae_neq),
                          ('ScalarMix MLP', mae_smix)]:
            if mae is not None:
                ratio = mae_kron / mae
                flag = "✓ matched" if 0.7 < ratio < 1.3 else ("↑ improved" if ratio > 0.8 else "⚠ gap")
                print(f"  {name:<16} MAE={mae:.4f}  ratio={ratio:.2f}x  {flag}")


if __name__ == '__main__':
    main()
