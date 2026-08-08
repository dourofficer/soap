"""Fast CPU invariants for the successor-side top-w variants (exp-soap).

    # from repo root
    pytest exp-soap/test_succ.py -v

Toy keeper/weighting helpers follow the pattern of tests/test_invariants.py.
"""
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parent))                    # repo root -> import src.*
sys.path.insert(0, str(EXP))

from succ import (SuccWCache, backprop_succ_loop, mask_columns_nearest,
                  mask_columns_strongest, MASKS)
from src.rescore.strategies import backprop_vec
from src.rescore.weights import build_W


class _Keeper:
    """Minimal keeper: 3 trajectories (copied pattern from tests/test_invariants.py)."""
    def __init__(self):
        from src.stores import StepIndex
        self.index = []
        self.traj_ranges = []
        self.traj_meta = {}
        row = 0
        specs = [
            (1, [(0, "A", False), (1, "B", True), (2, "A", False)], "B"),
            (2, [(0, "X", False), (1, "Y", False), (2, "Z", True), (3, "X", False)], "Z"),
            (3, [(0, "Orchestrator (thought)", True), (1, "W", False)], "Orchestrator"),
        ]
        for tid, steps, agent in specs:
            start = row
            for s, role, mis in steps:
                self.index.append(StepIndex(row, tid, s, role, mis))
                row += 1
            self.traj_ranges.append((start, row))
            self.traj_meta[tid] = {"mistake_agent": agent}
        self.lookup = {}
        self.device = torch.device("cpu")


def _toy_weighting(keeper):
    """Each step attends to all predecessors with random (distinct, tie-free) mass."""
    torch.manual_seed(3)
    weighting = {}
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        tid = str(entries[0].traj_idx)
        weighting[tid] = {}
        for i, e in enumerate(entries):
            preds = [entries[j].step_idx for j in range(i)]
            if not preds:
                continue
            wv = torch.rand(len(preds)) + 0.1
            weighting[tid][e.step_idx] = {
                "ctx_indices": torch.tensor(preds),
                "weights": wv / wv.sum(),
            }
    return weighting


def _succ_mats(keeper, weighting, w, variant):
    return [MASKS[variant](Wj, w) for Wj in build_W(keeper, weighting, "all")]


def test_gamma0_identity_both_variants():
    keeper = _Keeper()
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    for variant in ("strongest", "nearest"):
        for w in (1, 3, "all"):
            out = backprop_vec(s, keeper, _succ_mats(keeper, _toy_weighting(keeper), w, variant),
                               [0.0, 0.5])
            assert torch.allclose(out[:, 0], s), (variant, w)


def test_w_all_parity_with_backprop():
    """At w='all' both variants must equal the existing backprop on the full W exactly."""
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    W_all = build_W(keeper, weighting, "all")
    ref = backprop_vec(s, keeper, W_all, [0.0, 0.3, 1.0])
    for variant in ("strongest", "nearest"):
        out = backprop_vec(s, keeper, _succ_mats(keeper, weighting, "all", variant),
                           [0.0, 0.3, 1.0])
        assert torch.equal(out, ref), variant


def test_vec_equals_loop_both_variants():
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    # tie-heavy scores to stress ordering-independence of the arithmetic
    s = (torch.randn(N) * 2).round()
    for variant in ("strongest", "nearest"):
        for w in (1, 2, 3, "all"):
            mats = _succ_mats(keeper, weighting, w, variant)
            for gamma in (0.1, 0.7, 1.0):
                vec = backprop_vec(s, keeper, mats, [gamma])[:, 0]
                loop = backprop_succ_loop(s, keeper, weighting, gamma, w, variant)
                assert torch.allclose(vec, loop, atol=1e-6), (variant, w, gamma)


def test_sink_pass_through():
    """A step no successor attends to keeps its score (true sink)."""
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    last_positions = [end - 1 for _, end in keeper.traj_ranges]   # final steps: no successors
    for variant in ("strongest", "nearest"):
        for w in (1, "all"):
            out = backprop_vec(s, keeper, _succ_mats(keeper, weighting, w, variant), [1.0])[:, 0]
            for p in last_positions:
                assert out[p] == s[p], (variant, w, p)
    # mid-trajectory sink: in traj 2 (steps 0..3), step 2's successors skip it entirely
    skip = {tid: dict(steps) for tid, steps in weighting.items()}
    skip["2"][3] = {"ctx_indices": torch.tensor([0, 1]),
                    "weights": torch.tensor([0.4, 0.6])}          # step 3 ignores step 2
    start = keeper.traj_ranges[1][0]
    for variant in ("strongest", "nearest"):
        out = backprop_vec(s, keeper, _succ_mats(keeper, skip, "all", variant), [1.0])[:, 0]
        assert out[start + 2] == s[start + 2], variant


def test_strong_vs_near_differ():
    """Crafted weights where step 1's strongest successor is not its nearest one."""
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    # traj 2, steps 0..3. Column i=1: w_{1,2}=0.1 (near), w_{1,3}=0.8 (strong).
    weighting["2"] = {
        1: {"ctx_indices": torch.tensor([0]), "weights": torch.tensor([1.0])},
        2: {"ctx_indices": torch.tensor([0, 1]), "weights": torch.tensor([0.9, 0.1])},
        3: {"ctx_indices": torch.tensor([0, 1, 2]), "weights": torch.tensor([0.1, 0.8, 0.1])},
    }
    start = keeper.traj_ranges[1][0]
    N = keeper.traj_ranges[-1][1]
    s = torch.zeros(N)
    s[start + 2], s[start + 3] = 1.0, 5.0                          # distinct successor scores
    gamma = 1.0
    strong = backprop_vec(s, keeper, _succ_mats(keeper, weighting, 1, "strongest"), [gamma])[:, 0]
    near = backprop_vec(s, keeper, _succ_mats(keeper, weighting, 1, "nearest"), [gamma])[:, 0]
    # hub-normalized single-successor correction = that successor's score
    assert torch.isclose(strong[start + 1], torch.tensor(5.0))     # collected from step 3
    assert torch.isclose(near[start + 1], torch.tensor(1.0))       # collected from step 2
    # loops agree with the crafted expectation too
    assert torch.allclose(backprop_succ_loop(s, keeper, weighting, gamma, 1, "strongest"), strong)
    assert torch.allclose(backprop_succ_loop(s, keeper, weighting, gamma, 1, "nearest"), near)


def test_mask_correctness():
    torch.manual_seed(7)
    T = 9
    W = torch.tril(torch.rand(T, T), diagonal=-1)
    W[W < 0.25] = 0.0                                              # some structural zeros
    for w in (1, 2, 4):
        for mask in (mask_columns_strongest, mask_columns_nearest):
            M = mask(W, w)
            for i in range(T):
                nz_orig = W[:, i].nonzero(as_tuple=True)[0]
                nz_kept = M[:, i].nonzero(as_tuple=True)[0]
                assert nz_kept.numel() == min(w, nz_orig.numel()), (mask.__name__, w, i)
                assert set(nz_kept.tolist()) <= set(nz_orig.tolist())
                assert torch.equal(M[nz_kept, i], W[nz_kept, i])   # values untouched
        # per-column semantics: strongest keeps the max entry, nearest the first one
        Ms, Mn = mask_columns_strongest(W, 1), mask_columns_nearest(W, 1)
        for i in range(T):
            nz = W[:, i].nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                continue
            assert Ms[:, i].max() == W[nz, i].max()
            assert Mn[nz[0], i] == W[nz[0], i]
    assert torch.equal(mask_columns_strongest(W, "all"), W)
    assert torch.equal(mask_columns_nearest(W, "all"), W)


def test_gt_sentinel_transparent():
    """A GT_STEP=-1 context bucket (with-GT extractions) is dropped inside build_W, so
    the masked matrices match the plain ones exactly."""
    keeper = _Keeper()
    plain = _toy_weighting(keeper)
    gt_w = {tid: {s: {"ctx_indices": torch.cat([torch.tensor([-1]), d["ctx_indices"]]),
                      "weights": torch.cat([torch.tensor([0.3]), d["weights"] * 0.7])}
                  for s, d in steps.items()}
            for tid, steps in plain.items()}
    for variant in ("strongest", "nearest"):
        for w in (1, 2, "all"):
            for a, b in zip(_succ_mats(keeper, plain, w, variant),
                            _succ_mats(keeper, gt_w, w, variant)):
                assert torch.allclose(a, b, atol=1e-6), (variant, w)


def test_succ_wcache_matches_direct_masking():
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    ws = [1, 2, "all"]
    cache = SuccWCache([weighting, weighting], keeper, ws)
    for variant in ("strongest", "nearest"):
        for r_idx in (0, 1):
            for w in ws:
                for a, b in zip(cache.mats(r_idx, w)[variant],
                                _succ_mats(keeper, weighting, w, variant)):
                    assert torch.equal(a, b), (variant, r_idx, w)


def test_strategies_succ_dispatch():
    """The STRATEGIES-compatible closures pick their own variant's mats out of the
    SuccWCache dict and reproduce plain backprop_vec on them."""
    from succ import STRATEGIES_SUCC, STRATEGY_VARIANTS
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    cache = SuccWCache([weighting], keeper, [1, "all"])
    for name, variant in STRATEGY_VARIANTS.items():
        for w in (1, "all"):
            out = STRATEGIES_SUCC[name](s, keeper, cache.mats(0, w), [0.0, 0.6])
            ref = backprop_vec(s, keeper, _succ_mats(keeper, weighting, w, variant),
                               [0.0, 0.6])
            assert torch.equal(out, ref), (name, w)
