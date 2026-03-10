"""
train_coulomb.py
================
Train and compare three models on the Coulomb long-range dataset:

  1. LocalGNN        — pure PyTorch MPNN, no e3nn, per-atom sum
  2. KronHamModel    — pure PyTorch MPNN + Kronecker core, no e3nn
  3. KronHamModelV2  — e3nn equivariant backbone + Kronecker core (model_v2.py)

Key question: does using e3nn's equivariance help for this scalar task?

Physics setup:
  Two clusters A and B, separation=8 Å >> cutoff=4 Å.
  E_total = E_AA + E_BB + E_AB (Coulomb, 1/r).
  LocalGNN provably cannot learn E_AB (zero A-B edges).
  Kronecker models implicitly capture E_AB via spectral cross-products.

Expected outcome:
  LocalGNN:      MAE ≈ std(E_AB)       [cannot fit long-range term]
  KronHamModel:  MAE << std(E_AB)      [Kronecker spectral coupling helps]
  KronHamV2:     similar to KronHamModel, possibly slightly better
                 (e3nn gives orientation features, but Coulomb is scalar → limited gain)
"""

import time
import torch
import torch.nn as nn
import numpy as np

from model_coulomb import generate_dataset, LocalGNN, KronHamModel

# ── optional e3nn model ──────────────────────────────────────
try:
    from model_v2 import KroneckerHamiltonianModelV2
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False
    print("[warn] e3nn not available — skipping KronHamModelV2")


# ══════════════════════════════════════════════════════════════
# Dataset normalisation helper
# ══════════════════════════════════════════════════════════════

class Normaliser:
    """Shift-scale normalisation fitted on training targets."""

    def __init__(self, values: list):
        arr = np.array(values, dtype=np.float32)
        self.mean = float(arr.mean())
        self.std  = float(arr.std()) or 1.0

    def normalise(self, x: float) -> float:
        return (x - self.mean) / self.std

    def denormalise(self, x: float) -> float:
        return x * self.std + self.mean

    def denormalise_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════

def train_epoch(model, dataset, optimiser, norm: Normaliser, device, use_e3nn=False):
    """One epoch: iterate every sample, accumulate gradients in mini-batches."""
    model.train()
    indices = torch.randperm(len(dataset)).tolist()
    total_loss = 0.0

    BATCH = 32
    for start in range(0, len(dataset), BATCH):
        batch = [dataset[i] for i in indices[start:start + BATCH]]
        optimiser.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for sample in batch:
            charges = sample['charges'].to(device)
            pos     = sample['pos'].to(device)
            sids    = sample['subsystem_ids'].to(device)
            target  = torch.tensor(
                norm.normalise(sample['E_total'].item()),
                dtype=torch.float32, device=device
            )

            if use_e3nn:
                # KroneckerHamiltonianModelV2 needs atom_types (use subsystem_ids as proxy)
                out = model(pos, sids.int(), sids)
                pred = out['energy']
            else:
                out  = model(charges, pos, sids)
                pred = out['energy']

            batch_loss = batch_loss + (pred - target) ** 2

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += batch_loss.item()

    return total_loss / max(1, len(dataset) // BATCH)


@torch.no_grad()
def evaluate(model, dataset, norm: Normaliser, device, use_e3nn=False):
    """Returns MAE on E_total (original scale) and array of E_AB truths."""
    model.eval()
    errs_total = []
    e_ab_true  = []

    for sample in dataset:
        charges = sample['charges'].to(device)
        pos     = sample['pos'].to(device)
        sids    = sample['subsystem_ids'].to(device)

        if use_e3nn:
            out = model(pos, sids.int(), sids)
        else:
            out = model(charges, pos, sids)

        pred_norm = out['energy'].item()
        pred      = norm.denormalise(pred_norm)
        true      = sample['E_total'].item()

        errs_total.append(abs(pred - true))
        e_ab_true.append(sample['E_AB'].item())

    return float(np.mean(errs_total)), np.array(e_ab_true)


def train_model(
    model,
    train_data, test_data,
    norm: Normaliser,
    n_epochs:   int   = 100,
    lr:         float = 3e-3,
    device:     str   = 'cpu',
    label:      str   = 'Model',
    use_e3nn:   bool  = False,
):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*55}")
    print(f"  {label}  ({n_params:,} params)")
    print(f"{'─'*55}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr / 20
    )

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, train_data, optimiser, norm, device, use_e3nn)
        scheduler.step()

        if epoch % (n_epochs // 5) == 0 or epoch == 1:
            mae, _ = evaluate(model, test_data, norm, device, use_e3nn)
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{n_epochs}  loss={loss:.4f}  "
                  f"test_MAE={mae:.4f}  ({elapsed:.0f}s)")

    final_mae, e_ab = evaluate(model, test_data, norm, device, use_e3nn)
    return final_mae, e_ab


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    device = 'cpu'

    # ── Dataset config ───────────────────────────────────────
    N_A, N_B      = 5, 5
    SEPARATION    = 8.0    # Å — far beyond GNN cutoff
    CLUSTER_STD   = 0.5   # Å — tight clusters, all intra-pairs within cutoff
    CUTOFF        = 4.0   # Å — no A-B edges when separation >> cutoff
    CHARGE_SCALE  = 1.0

    print("=" * 55)
    print(" Kronecker Hamiltonian — Long-Range Coulomb Demo")
    print("=" * 55)
    print(f"\nDataset: n_A={N_A}, n_B={N_B}, sep={SEPARATION}Å, cutoff={CUTOFF}Å")
    print("Generating dataset …")

    train_data = generate_dataset(1000, N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=0)
    test_data  = generate_dataset(200,  N_A, N_B, SEPARATION, CLUSTER_STD, CHARGE_SCALE, seed=1000)

    # Statistics
    e_totals = [s['E_total'].item() for s in train_data]
    e_locals = [s['E_local'].item() for s in train_data]
    e_abs    = [s['E_AB'].item()    for s in train_data]

    norm = Normaliser(e_totals)   # fit normaliser on training set

    print(f"\nTraining set statistics:")
    print(f"  E_total : mean={np.mean(e_totals):+.3f}  std={np.std(e_totals):.3f}")
    print(f"  E_local : mean={np.mean(e_locals):+.3f}  std={np.std(e_locals):.3f}")
    print(f"  E_AB    : mean={np.mean(e_abs):+.3f}  std={np.std(e_abs):.3f}")
    print(f"\n  LocalGNN lower-bound MAE ≈ std(E_AB) = {np.std(e_abs):.3f}")
    print(f"  (because LocalGNN cannot predict E_AB at all — zero A-B edges)")

    # ── Shared hyper-params ──────────────────────────────────
    N_EPOCHS = 100
    LR       = 3e-3
    common   = dict(hidden=64, n_rbf=20, cutoff=CUTOFF, n_layers=3)

    results = {}

    # ── 1. LocalGNN ──────────────────────────────────────────
    model_local = LocalGNN(**common).to(device)
    mae_local, e_ab_test = train_model(
        model_local, train_data, test_data, norm,
        N_EPOCHS, LR, device, label='LocalGNN (no e3nn, no Kronecker)',
    )
    results['LocalGNN'] = mae_local

    # ── 2. KronHamModel (pure PyTorch) ───────────────────────
    model_kron = KronHamModel(**common, basis_dim=4, K=4, k_states=8).to(device)
    mae_kron, _ = train_model(
        model_kron, train_data, test_data, norm,
        N_EPOCHS, LR, device, label='KronHamModel (no e3nn, +Kronecker)',
    )
    results['KronHamModel'] = mae_kron

    # ── 3. KronHamModelV2 with e3nn backbone ─────────────────
    if HAS_E3NN:
        e3nn_config = {
            'atom_types':     2,       # we use subsystem_id (0 or 1) as atom type
            'atom_embed_dim': 32,
            'node_irreps':    '4x0e + 4x1o + 4x2e',  # equal muls for uvu mode
            'edge_sh_lmax':   2,
            'n_mp_layers':    3,
            'K':              4,
            'vector_irreps':  '1x0e + 1x1o',           # basis_dim = 4
            'basis_dim':      4,
            'k_keep':         32,
            'k_states':       8,
            'cutoff':         CUTOFF,
        }
        model_e3nn = KroneckerHamiltonianModelV2(e3nn_config).to(device)
        mae_e3nn, _ = train_model(
            model_e3nn, train_data, test_data, norm,
            N_EPOCHS, LR, device,
            label='KronHamModelV2 (e3nn backbone, +Kronecker)',
            use_e3nn=True,
        )
        results['KronHamV2 (e3nn)'] = mae_e3nn

    # ── Summary ──────────────────────────────────────────────
    std_e_ab = np.std([s['E_AB'].item() for s in test_data])

    print(f"\n{'='*55}")
    print(" Results Summary")
    print(f"{'='*55}")
    print(f"  std(E_AB) on test = {std_e_ab:.4f}  ← LocalGNN floor")
    print()
    print(f"  {'Model':<35} {'MAE':>8}  {'vs LocalGNN':>12}")
    print(f"  {'-'*56}")
    mae_local_v = results['LocalGNN']
    for name, mae in results.items():
        ratio = f"{mae_local_v / mae:.1f}x better" if mae < mae_local_v else "—"
        print(f"  {name:<35} {mae:>8.4f}  {ratio:>12}")

    print()
    print("Interpretation:")
    if results.get('KronHamModel', 1e9) < mae_local_v * 0.8:
        print("  ✓ KronHamModel significantly outperforms LocalGNN")
        print("  ✓ Kronecker spectral coupling captures long-range E_AB")
    else:
        print("  → Increase n_epochs or check if E_AB signal is large enough")

    if HAS_E3NN and 'KronHamV2 (e3nn)' in results:
        diff = abs(results['KronHamV2 (e3nn)'] - results['KronHamModel'])
        if diff < 0.01 * std_e_ab * 10:
            print("  ≈ e3nn backbone gives similar MAE to plain PyTorch backbone")
            print("    (expected: Coulomb energy is scalar — orientational equivariance")
            print("     doesn't help beyond what distance features already capture)")
        elif results['KronHamV2 (e3nn)'] < results['KronHamModel']:
            print("  ✓ e3nn backbone improves over plain PyTorch backbone")
        else:
            print("  ✗ e3nn backbone did not help — scalar task, extra symmetry unused")


if __name__ == '__main__':
    main()
