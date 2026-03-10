"""
test_kronecker.py
=================
独立测试文件，只依赖 torch。
测试 Kronecker 对角化的：
  1. 数学正确性（特征值验证）
  2. 加速效果（计时对比）
  3. 梯度可传播性
  4. vec trick 正确性

运行:
    python test_kronecker.py
"""

import torch
import time


# ══════════════════════════════════════════════════════════════
# Kronecker 对角化核心（无 e3nn 依赖）
# ══════════════════════════════════════════════════════════════

def make_sym(A: torch.Tensor) -> torch.Tensor:
    """确保矩阵实对称"""
    return (A + A.T) / 2


def kronecker_diag(H_A: torch.Tensor, H_B: torch.Tensor, perturb: float = 1e-6):
    """
    H = H_A ⊗ I_B + I_A ⊗ H_B

    利用 Kronecker 结构：
      ε_{ij} = ε_i^A + ε_j^B
      Φ_{ij} = φ_i^A ⊗ ψ_j^B

    复杂度: O(n_A³ + n_B³)
    """
    if perturb > 0:
        H_A = H_A + perturb * torch.eye(H_A.shape[0], dtype=H_A.dtype)
        H_B = H_B + perturb * torch.eye(H_B.shape[0], dtype=H_B.dtype)

    evals_A, evecs_A = torch.linalg.eigh(H_A)   # O(n_A³)
    evals_B, evecs_B = torch.linalg.eigh(H_B)   # O(n_B³)

    # 广播求和: [n_A, 1] + [1, n_B] → [n_A, n_B]
    evals_full = (evals_A.unsqueeze(1) + evals_B.unsqueeze(0)).reshape(-1)
    evals_sorted, sort_idx = torch.sort(evals_full)

    return evals_sorted, evecs_A, evecs_B, sort_idx


def apply_global_evecs_T(evecs_A, evecs_B, v):
    """
    计算 (evecs_A ⊗ evecs_B)^T @ v，不显式构建 Kronecker 矩阵。

    vec trick: (A ⊗ B)^T vec(V) = vec(A^T V B)
    其中 V = reshape(v, [n_A, n_B])
    """
    n_A, n_B = evecs_A.shape[0], evecs_B.shape[0]
    V = v.reshape(n_A, n_B)
    return (evecs_A.T @ V @ evecs_B).reshape(-1)


def brute_force_diag(H_A: torch.Tensor, H_B: torch.Tensor):
    """
    暴力对角化：显式构建 H_full = H_A⊗I + I⊗H_B，然后 eigh。
    复杂度: O((n_A * n_B)³)
    """
    n_A, n_B = H_A.shape[0], H_B.shape[0]
    I_A = torch.eye(n_A, dtype=H_A.dtype)
    I_B = torch.eye(n_B, dtype=H_B.dtype)
    H_full = torch.kron(H_A, I_B) + torch.kron(I_A, H_B)
    return torch.linalg.eigh(H_full)


# ══════════════════════════════════════════════════════════════
# 简化版完整模型（无 e3nn，用随机线性层模拟等变层）
# ══════════════════════════════════════════════════════════════

class MockGNN(torch.nn.Module):
    """
    用 MLP 代替等变 GNN，只为测试后续 Kronecker pipeline。
    真实模型换成 e3nn EquivariantMessagePassing。
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class KroneckerModelMock(torch.nn.Module):
    """
    完整 pipeline 的 mock 版本：
      原子特征 → MLP → 低秩向量 + onsite → H_A, H_B → Kronecker 对角化 → 能量
    """
    def __init__(self, atom_feat_dim: int, node_dim: int, basis_dim: int, K: int, k_states: int = 8):
        super().__init__()
        self.basis_dim = basis_dim
        self.K = K
        self.k_states = k_states

        # 节点特征提取
        self.gnn = MockGNN(atom_feat_dim, 64, node_dim)

        # 低秩向量头（A 和 B 各一套）
        self.vec_A = torch.nn.ModuleList([
            torch.nn.Linear(node_dim, basis_dim) for _ in range(K)
        ])
        self.vec_B = torch.nn.ModuleList([
            torch.nn.Linear(node_dim, basis_dim) for _ in range(K)
        ])

        # Onsite 头
        self.onsite = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, basis_dim),
        )

        # 能量头
        feat_dim = 3 * k_states + 4
        self.energy_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, atom_feats: torch.Tensor, subsystem_ids: torch.Tensor):
        # ── 节点特征 ──
        node_feat = self.gnn(atom_feats)             # [N, node_dim]

        mask_A = subsystem_ids == 0
        mask_B = subsystem_ids == 1
        feat_A = node_feat[mask_A]                   # [n_A, node_dim]
        feat_B = node_feat[mask_B]                   # [n_B, node_dim]
        n_A, n_B = feat_A.shape[0], feat_B.shape[0]
        dim_A, dim_B = n_A * self.basis_dim, n_B * self.basis_dim

        # ── 低秩矩阵 M_A, M_B ──
        M_A = torch.zeros(dim_A, dim_A)
        M_B = torch.zeros(dim_B, dim_B)
        for k in range(self.K):
            a_k = self.vec_A[k](feat_A).reshape(-1)   # [dim_A]
            b_k = self.vec_B[k](feat_B).reshape(-1)   # [dim_B]
            M_A = M_A + torch.outer(a_k, a_k)
            M_B = M_B + torch.outer(b_k, b_k)

        # ── Onsite ──
        d_A = self.onsite(feat_A).reshape(-1)          # [dim_A]
        d_B = self.onsite(feat_B).reshape(-1)          # [dim_B]

        # ── 子系统 Hamiltonian ──
        H_A = make_sym(torch.diag(d_A) + M_A)
        H_B = make_sym(torch.diag(d_B) + M_B)

        # ── Kronecker 解析对角化 ──
        evals, evecs_A, evecs_B, _ = kronecker_diag(H_A, H_B)

        # ── 特征提取 ──
        evals_A = torch.linalg.eigvalsh(H_A)
        evals_B = torch.linalg.eigvalsh(H_B)

        k = self.k_states
        feat = torch.cat([
            evals[:k].reshape(k) if evals.shape[0] >= k
                else torch.nn.functional.pad(evals, (0, k - evals.shape[0])),
            evals_A[:k] if evals_A.shape[0] >= k
                else torch.nn.functional.pad(evals_A, (0, k - evals_A.shape[0])),
            evals_B[:k] if evals_B.shape[0] >= k
                else torch.nn.functional.pad(evals_B, (0, k - evals_B.shape[0])),
            torch.stack([
                (evals_A[:k].unsqueeze(1) - evals_B[:k].unsqueeze(0)).min(),
                (evals_A[:k].unsqueeze(1) - evals_B[:k].unsqueeze(0)).max(),
                (evals_A[:k].unsqueeze(1) - evals_B[:k].unsqueeze(0)).abs().mean(),
                (evals_A[:k].unsqueeze(1) - evals_B[:k].unsqueeze(0)).std(),
            ]),
        ])

        energy = self.energy_head(feat).squeeze()

        return {
            'energy':  energy,
            'evals':   evals,
            'evecs_A': evecs_A,
            'evecs_B': evecs_B,
            'H_A':     H_A,
            'H_B':     H_B,
            'evals_A': evals_A,
            'evals_B': evals_B,
        }


# ══════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════

def sep(title=""):
    w = 60
    if title:
        print(f"\n{'─'*((w-len(title)-2)//2)} {title} {'─'*((w-len(title)-2)//2)}")
    else:
        print("─" * w)


def ok(passed: bool) -> str:
    return "✓" if passed else "✗"


def test_correctness() -> bool:
    sep("1. 数学正确性")
    torch.manual_seed(0)
    passed = True

    for n_A, n_B in [(4, 4), (6, 5), (10, 8)]:
        H_A = make_sym(torch.randn(n_A, n_A))
        H_B = make_sym(torch.randn(n_B, n_B))

        evals_kron, _, _, _ = kronecker_diag(H_A, H_B, perturb=0)
        evals_brute, _ = brute_force_diag(H_A, H_B)

        err = (evals_kron - evals_brute).abs().max().item()
        case_ok = err < 1e-4
        passed = passed and case_ok
        print(f"  n_A={n_A}, n_B={n_B}  →  全局dim={n_A*n_B}  "
              f"最大特征值误差: {err:.2e}  {ok(case_ok)}")

    return passed


def test_eigenvec_orthogonality() -> bool:
    sep("2. 特征向量正交性")
    torch.manual_seed(1)

    n_A, n_B = 6, 5
    H_A = make_sym(torch.randn(n_A, n_A))
    H_B = make_sym(torch.randn(n_B, n_B))
    _, evecs_A, evecs_B, _ = kronecker_diag(H_A, H_B)

    err_A = (evecs_A.T @ evecs_A - torch.eye(n_A)).abs().max().item()
    err_B = (evecs_B.T @ evecs_B - torch.eye(n_B)).abs().max().item()
    a_ok = err_A < 1e-5
    b_ok = err_B < 1e-5
    print(f"  evecs_A 正交误差: {err_A:.2e}  {ok(a_ok)}")
    print(f"  evecs_B 正交误差: {err_B:.2e}  {ok(b_ok)}")
    return a_ok and b_ok


def test_vec_trick() -> bool:
    sep("3. Vec trick 正确性")
    torch.manual_seed(2)

    n_A, n_B = 5, 4
    H_A = make_sym(torch.randn(n_A, n_A))
    H_B = make_sym(torch.randn(n_B, n_B))
    _, evecs_A, evecs_B, _ = kronecker_diag(H_A, H_B)

    v = torch.randn(n_A * n_B)
    result_trick = apply_global_evecs_T(evecs_A, evecs_B, v)
    result_brute = torch.kron(evecs_A, evecs_B).T @ v

    err = (result_trick - result_brute).abs().max().item()
    passed = err < 1e-5
    print(f"  最大误差: {err:.2e}  {ok(passed)}")
    return passed


def test_gradient() -> bool:
    sep("4. 梯度传播")
    torch.manual_seed(3)

    n_A, n_B = 5, 5
    # Keep leaf tensors separate — make_sym produces a non-leaf, so
    # .grad would never be populated if we checked the output of make_sym.
    raw_A = torch.randn(n_A, n_A, requires_grad=True)
    raw_B = torch.randn(n_B, n_B, requires_grad=True)
    H_A = make_sym(raw_A)
    H_B = make_sym(raw_B)

    evals_A, _ = torch.linalg.eigh(H_A + 1e-6 * torch.eye(n_A))
    evals_B, _ = torch.linalg.eigh(H_B + 1e-6 * torch.eye(n_B))
    evals_full = (evals_A.unsqueeze(1) + evals_B.unsqueeze(0)).reshape(-1)

    loss = evals_full[:4].sum()
    loss.backward()

    grad_A_ok = raw_A.grad is not None and not raw_A.grad.isnan().any()
    grad_B_ok = raw_B.grad is not None and not raw_B.grad.isnan().any()
    print(f"  raw_A 梯度: {ok(grad_A_ok)}"
          + (f"  norm={raw_A.grad.norm().item():.4f}" if grad_A_ok else ""))
    print(f"  raw_B 梯度: {ok(grad_B_ok)}"
          + (f"  norm={raw_B.grad.norm().item():.4f}" if grad_B_ok else ""))
    return grad_A_ok and grad_B_ok


def test_full_model() -> bool:
    sep("5. 完整模型前向 + 梯度")
    torch.manual_seed(4)

    model = KroneckerModelMock(
        atom_feat_dim=16,
        node_dim=32,
        basis_dim=4,
        K=4,
        k_states=8,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")

    N_A, N_B = 5, 5
    atom_feats = torch.randn(N_A + N_B, 16)
    subsystem_ids = torch.tensor([0]*N_A + [1]*N_B)

    out = model(atom_feats, subsystem_ids)
    print(f"  H_A: {out['H_A'].shape}  H_B: {out['H_B'].shape}")
    print(f"  全局特征值数: {out['evals'].shape[0]}")
    print(f"  预测能量: {out['energy'].item():.4f}")

    out['energy'].backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    no_nan = all(not p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
    passed = has_grad and no_nan
    print(f"  端到端梯度: {ok(passed)}")
    return passed


def test_timing() -> bool:
    sep("6. 计时对比：Kronecker vs 暴力")
    torch.manual_seed(5)
    REPEAT = 5

    print(f"  {'n_A=n_B':<8} {'全局dim':<10} "
          f"{'暴力(ms)':<12} {'Kronecker(ms)':<15} {'加速'}")
    print(f"  {'─'*55}")

    for n in [4, 8, 16, 24, 32]:
        H_A = make_sym(torch.randn(n, n))
        H_B = make_sym(torch.randn(n, n))

        times_b = []
        for _ in range(REPEAT):
            t0 = time.perf_counter(); brute_force_diag(H_A, H_B)
            times_b.append(time.perf_counter() - t0)

        times_k = []
        for _ in range(REPEAT):
            t0 = time.perf_counter(); kronecker_diag(H_A, H_B, perturb=0)
            times_k.append(time.perf_counter() - t0)

        t_brute = min(times_b) * 1000
        t_kron  = min(times_k) * 1000
        speedup = t_brute / t_kron if t_kron > 0 else float('inf')
        print(f"  {n:<8} {n*n:<10} {t_brute:<12.2f} {t_kron:<15.2f} {speedup:.1f}x")

    return True  # timing is informational — always passes


def test_eigenvalue_structure() -> bool:
    sep("7. 特征值结构验证（ε_i^A + ε_j^B）")
    torch.manual_seed(6)

    n_A, n_B = 3, 3
    H_A = make_sym(torch.randn(n_A, n_A))
    H_B = make_sym(torch.randn(n_B, n_B))

    evals_kron, _, _, _ = kronecker_diag(H_A, H_B, perturb=0)
    evals_A = torch.linalg.eigvalsh(H_A)
    evals_B = torch.linalg.eigvalsh(H_B)

    expected = sorted([
        (evals_A[i] + evals_B[j]).item()
        for i in range(n_A) for j in range(n_B)
    ])

    print(f"  ε_A: {[f'{v:.3f}' for v in evals_A.tolist()]}")
    print(f"  ε_B: {[f'{v:.3f}' for v in evals_B.tolist()]}")
    print(f"  期望: {[f'{v:.3f}' for v in expected]}")
    print(f"  实际: {[f'{v:.3f}' for v in sorted(evals_kron.tolist())]}")

    err = max(abs(a - b) for a, b in zip(expected, sorted(evals_kron.tolist())))
    passed = err < 1e-4
    print(f"  最大误差: {err:.2e}  {ok(passed)}")
    return passed


def test_model_v2() -> bool:
    sep("8. KroneckerHamiltonianModelV2 (e3nn)")
    torch.manual_seed(7)

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from model_v2 import KroneckerHamiltonianModelV2
    except ImportError as e:
        print(f"  跳过：导入失败 ({e})")
        return True  # skip, not a failure

    config = {
        'atom_types':     10,
        'atom_embed_dim': 16,
        # Equal multiplicities required by e3nn 'uvu' TensorProduct mode
        'node_irreps':    "4x0e + 4x1o + 4x2e",
        'edge_sh_lmax':   2,
        'n_mp_layers':    1,
        'K':              2,
        'vector_irreps':  "1x0e + 1x1o",
        'basis_dim':      4,
        'k_keep':         8,
        'k_states':       4,
        'cutoff':         5.0,
    }

    model = KroneckerHamiltonianModelV2(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    N_A, N_B = 3, 3
    pos           = torch.randn(N_A + N_B, 3) * 2.0
    atom_types    = torch.randint(0, 10, (N_A + N_B,))
    subsystem_ids = torch.tensor([0] * N_A + [1] * N_B)

    out = model(pos, atom_types, subsystem_ids)
    print(f"  H_A: {out['H_A'].shape}  H_B: {out['H_B'].shape}")
    print(f"  全局特征值数: {out['evals'].shape[0]}")
    print(f"  预测能量: {out['energy'].item():.4f}")

    # 特征值结构：evals[0] ≈ ε_A[0] + ε_B[0]
    expected_min = (out['evals_A'][0] + out['evals_B'][0]).item()
    actual_min   = out['evals'][0].item()
    eval_err     = abs(expected_min - actual_min)
    eval_ok      = eval_err < 1e-4
    print(f"  ε_A[0]+ε_B[0]={expected_min:.4f}  evals[0]={actual_min:.4f}  "
          f"误差={eval_err:.2e}  {ok(eval_ok)}")

    # 端到端梯度
    out['energy'].backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    no_nan   = all(not p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
    grad_ok  = has_grad and no_nan
    print(f"  端到端梯度: {ok(grad_ok)}")

    return eval_ok and grad_ok


# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("   Kronecker Hamiltonian 测试套件")
    print("=" * 60)

    tests = [
        test_correctness,
        test_eigenvec_orthogonality,
        test_vec_trick,
        test_gradient,
        test_full_model,
        test_timing,
        test_eigenvalue_structure,
        test_model_v2,
    ]

    results = []
    for fn in tests:
        results.append((fn.__name__, fn()))

    sep()
    n_pass = sum(1 for _, p in results if p)
    n_fail = len(results) - n_pass
    print(f"结果: {n_pass}/{len(results)} 通过"
          + (f"  ({n_fail} 失败: {', '.join(n for n, p in results if not p)})" if n_fail else ""))
    print("全部测试完成。" if n_fail == 0 else "部分测试失败。")
