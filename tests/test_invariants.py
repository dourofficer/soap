"""Fast CPU invariant tests (no GPU/data needed): scorer identities + batched metrics.

    # from v2/
    pytest tests/test_invariants.py -v

Rescore invariants cover the three strategies (backprop / succ-strong / succ-near):
gamma=0 identity, orient none identity, the backprop transpose by hand, vec == the
successor-side reference loop, w="all" coincidence of all strategies, sink
pass-through, strongest-vs-nearest divergence, and column-mask correctness. The
GT-context block at the bottom covers the with-GT extraction mode (pinned
[question, answer] prefix + GT_STEP sentinel).
"""
import numpy as np
import pytest
import torch

from src.score.scorers import proj, resid, angres
from src.metrics import compute_metrics, compute_metrics_batch


torch.manual_seed(0)
D, T, K = 32, 40, 20


def _fit(n=200):
    G = torch.randn(n, D)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    return Vh[:K].T.contiguous(), S.contiguous(), G.mean(0)


def test_resid_identity():
    V, S, ref = _fit()
    R = torch.randn(T, D)
    # resid(0,c) == ||R||^2 - c * proj(0,c)   (proj is band-MEAN; resid uses band-SUM)
    for c in (1, 5, 20):
        r = resid(R, V, 0, c)
        p = proj(R, V, 0, c)                       # mean over band
        expect = R.square().sum(1) - c * p
        assert torch.allclose(r, expect, atol=1e-4), c


def test_angres_identity():
    V, S, ref = _fit()
    R = torch.randn(T, D)
    for c in (2, 10):
        a = angres(R, V, 0, c)
        r = resid(R, V, 0, c)
        assert torch.allclose(a, r / (R.square().sum(1) + 1e-12), atol=1e-5)
        assert (a >= -1e-6).all() and (a <= 1 + 1e-6).all()


class _Keeper:
    """Minimal keeper: 3 trajectories with known mistake steps/roles."""
    def __init__(self):
        from src.stores import StepIndex
        self.index = []
        self.traj_ranges = []
        self.traj_meta = {}
        row = 0
        specs = [  # (traj_idx, [(step, role, is_mistake)], mistake_agent)
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


def test_batch_equals_loop_with_ties():
    keeper = _Keeper()
    N = keeper.traj_ranges[-1][1]
    torch.manual_seed(1)
    # include a tie-heavy vector (rounded) to exercise earliest-step tie-break
    S = torch.stack([torch.randn(N), (torch.randn(N) * 3).round(), torch.zeros(N)])
    for direction in ("asc", "desc"):
        batch = compute_metrics_batch(S, keeper, [1, 3], direction)
        for i in range(S.shape[0]):
            loop = compute_metrics(S[i], keeper, [1, 3], direction)
            for k in (1, 3):
                for kind in ("step", "agent"):
                    key = f"{kind}@{k}_{direction}"
                    assert abs(batch[key][i] - loop[key]) < 1e-12, (direction, i, key)


# ── rescore strategy invariants ─────────────────────────────────────────────
def _toy_weighting(keeper):
    """A weighting dict: each step attends to all predecessors with random mass."""
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


def test_orient_none_identity():
    from src.rescore.strategies import orient
    x = torch.randn(10)
    assert torch.equal(orient(x, "none"), x)


def test_gamma0_identity_all_strategies():
    from src.rescore.weights import strategy_mats
    from src.rescore.strategies import STRATEGIES
    keeper = _Keeper()
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    for w in (1, 3, "all"):
        mats = strategy_mats(keeper, _toy_weighting(keeper), w)
        for name, fn in STRATEGIES.items():
            out = fn(s, keeper, mats, [0.0, 0.5])
            assert torch.allclose(out[:, 0], s), (name, w)


def test_vec_equals_loop_succ():
    from src.rescore.weights import strategy_mats
    from src.rescore.strategies import STRATEGIES, backprop_succ_loop
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    # tie-heavy scores to stress ordering-independence of the arithmetic
    s = (torch.randn(N) * 2).round()
    for name, variant in (("succ-strong", "strongest"), ("succ-near", "nearest")):
        for w in (1, 2, 3, "all"):
            mats = strategy_mats(keeper, weighting, w)
            for gamma in (0.1, 0.7, 1.0):
                vec = STRATEGIES[name](s, keeper, mats, [gamma])[:, 0]
                loop = backprop_succ_loop(s, keeper, weighting, gamma, w, variant)
                assert torch.allclose(vec, loop, atol=1e-6), (name, w, gamma)


def test_w_all_all_strategies_coincide():
    """At w='all' the row trim and both column masks are the identity, so every
    strategy must equal plain backprop on the full W exactly."""
    from src.rescore.weights import build_W, strategy_mats
    from src.rescore.strategies import STRATEGIES, backprop_vec
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    ref = backprop_vec(s, keeper, build_W(keeper, weighting, "all"), [0.0, 0.3, 1.0])
    mats = strategy_mats(keeper, weighting, "all")
    for name, fn in STRATEGIES.items():
        assert torch.equal(fn(s, keeper, mats, [0.0, 0.3, 1.0]), ref), name


def test_succ_sink_pass_through():
    """A step no successor attends to keeps its score under the succ variants."""
    from src.rescore.weights import strategy_mats
    from src.rescore.strategies import STRATEGIES
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    last_positions = [end - 1 for _, end in keeper.traj_ranges]  # final steps: no successors
    for w in (1, "all"):
        mats = strategy_mats(keeper, weighting, w)
        for name in ("succ-strong", "succ-near"):
            out = STRATEGIES[name](s, keeper, mats, [1.0])[:, 0]
            for p in last_positions:
                assert out[p] == s[p], (name, w, p)
    # mid-trajectory sink: in traj 2 (steps 0..3), step 3 skips step 2 entirely
    skip = {tid: dict(steps) for tid, steps in weighting.items()}
    skip["2"][3] = {"ctx_indices": torch.tensor([0, 1]),
                    "weights": torch.tensor([0.4, 0.6])}
    start = keeper.traj_ranges[1][0]
    mats = strategy_mats(keeper, skip, "all")
    for name in ("succ-strong", "succ-near"):
        out = STRATEGIES[name](s, keeper, mats, [1.0])[:, 0]
        assert out[start + 2] == s[start + 2], name


def test_succ_strong_vs_near_differ():
    """Crafted weights where step 1's strongest successor is not its nearest one."""
    from src.rescore.weights import strategy_mats
    from src.rescore.strategies import STRATEGIES, backprop_succ_loop
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
    s[start + 2], s[start + 3] = 1.0, 5.0             # distinct successor scores
    mats = strategy_mats(keeper, weighting, 1)
    strong = STRATEGIES["succ-strong"](s, keeper, mats, [1.0])[:, 0]
    near = STRATEGIES["succ-near"](s, keeper, mats, [1.0])[:, 0]
    # hub-normalized single-successor correction = that successor's score
    assert torch.isclose(strong[start + 1], torch.tensor(5.0))   # collected from step 3
    assert torch.isclose(near[start + 1], torch.tensor(1.0))     # collected from step 2
    assert torch.allclose(backprop_succ_loop(s, keeper, weighting, 1.0, 1, "strongest"), strong)
    assert torch.allclose(backprop_succ_loop(s, keeper, weighting, 1.0, 1, "nearest"), near)


def test_column_mask_correctness():
    from src.rescore.weights import mask_columns_strongest, mask_columns_nearest
    torch.manual_seed(7)
    T = 9
    W = torch.tril(torch.rand(T, T), diagonal=-1)
    W[W < 0.25] = 0.0                                 # some structural zeros
    for w in (1, 2, 4):
        for mask in (mask_columns_strongest, mask_columns_nearest):
            M = mask(W, w)
            for i in range(T):
                nz_orig = W[:, i].nonzero(as_tuple=True)[0]
                nz_kept = M[:, i].nonzero(as_tuple=True)[0]
                assert nz_kept.numel() == min(w, nz_orig.numel()), (mask.__name__, w, i)
                assert set(nz_kept.tolist()) <= set(nz_orig.tolist())
                assert torch.equal(M[nz_kept, i], W[nz_kept, i])  # values untouched
    # per-column semantics at w=1: strongest keeps the max entry, nearest the first
    Ms, Mn = mask_columns_strongest(W, 1), mask_columns_nearest(W, 1)
    for i in range(T):
        nz = W[:, i].nonzero(as_tuple=True)[0]
        if nz.numel() == 0:
            continue
        assert Ms[:, i].max() == W[nz, i].max()
        assert Mn[nz[0], i] == W[nz[0], i]
    assert torch.equal(mask_columns_strongest(W, "all"), W)
    assert torch.equal(mask_columns_nearest(W, "all"), W)


def test_backprop_transpose_by_hand():
    from src.rescore.weights import build_W
    from src.rescore.strategies import backprop_vec
    keeper = _Keeper()
    weighting = _toy_weighting(keeper)
    W = build_W(keeper, weighting, "all")
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    gamma = 0.5
    out = backprop_vec(s, keeper, W, [gamma])[:, 0]
    # hand compute per trajectory: bp_i = sum_t w[t,i] s_t / sum_t w[t,i]
    expect = s.clone()
    for (start, end), Wj in zip(keeper.traj_ranges, W):
        sj = s[start:end]
        num = Wj.T @ sj
        den = Wj.T.sum(1)
        expect[start:end] = sj + gamma * num / (den + 1e-12)
    assert torch.allclose(out, expect, atol=1e-6)


# ── with-GT context mode (pinned [question, answer] block) ──────────────────
class _StubTok:
    """Char-level tokenizer: 1 token per char, so spans are exact and decodable."""
    name_or_path = "stub"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<O>" + msgs[0]["content"] + "<C>"     # sentinel round-trips through content

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _decode(ids):
    return "".join(map(chr, ids))


def _toy_traj(n_turns=5, gt="42"):
    from src.data.trajectory import Trajectory
    hist = [{"role": f"agent{i}", "content": f"step {i} says " + "x" * (4 + 3 * i)}
            for i in range(n_turns)]
    return Trajectory(filename="7.json", question_id="q7", history=hist,
                      mistake_agent="agent1", mistake_step=1, level=1, subset="toy",
                      question="What is six times seven?", system=None, ground_truth=gt)


def _gt_text(traj):
    return (f"The problem is: {traj.question}\n"
            f"The Answer for the problem is: {traj.ground_truth}")


def test_gt_off_layout_unchanged():
    """with_gt=False (and the default) produce exactly the documented layout:
    open ++ chunk_0 ++ sep ++ ... ++ close ++ content, spans covering every chunk."""
    from src.data.context import _build, _serialize_turns
    tok, traj = _StubTok(), _toy_traj()
    b = _build(traj, 3, tok, None, None)
    assert b == _build(traj, 3, tok, None, None, with_gt=False)
    chunks = [_serialize_turns(traj.history, [i]) for i in range(3)]
    expect = "<O>" + "\n\n".join(chunks) + "<C>" + _serialize_turns(traj.history, [3])
    assert _decode(b["input_ids"]) == expect
    assert b["ctx_len"] == len(expect) - len(_serialize_turns(traj.history, [3]))
    assert set(b["step_tokens"]) == {0, 1, 2, 3}
    for i, chunk in enumerate(chunks):
        assert _decode([b["input_ids"][p] for p in b["step_tokens"][i]]) == chunk
    assert not b["hard_truncated"]


def test_gt_block_prepended():
    from src.data.context import _build, GT_STEP
    tok, traj = _StubTok(), _toy_traj()
    b0 = _build(traj, 3, tok, None, None)
    bg = _build(traj, 3, tok, None, None, with_gt=True)
    gt = _gt_text(traj)
    shift = len(gt) + 2                                # gt block + one "\n\n" sep
    assert bg["step_tokens"][GT_STEP] == list(range(3, 3 + len(gt)))   # right after "<O>"
    assert _decode([bg["input_ids"][p] for p in bg["step_tokens"][GT_STEP]]) == gt
    assert bg["ctx_len"] == b0["ctx_len"] + shift
    for i in (0, 1, 2):                                # real turns: same ids, shifted
        assert bg["step_tokens"][i] == [p + shift for p in b0["step_tokens"][i]]
    assert bg["input_ids"][bg["ctx_len"]:] == b0["input_ids"][b0["ctx_len"]:]
    assert bg["step_tokens"][3] == list(range(bg["ctx_len"], len(bg["input_ids"])))


def test_gt_pinned_under_truncation():
    from src.data.context import _build, GT_STEP
    tok, traj = _StubTok(), _toy_traj()
    full = _build(traj, 4, tok, None, None, with_gt=True)
    budget = len(full["input_ids"]) - 1                # forces at least one drop
    b = _build(traj, 4, tok, budget, None, with_gt=True)
    assert not b["hard_truncated"] and len(b["input_ids"]) <= budget
    assert _decode([b["input_ids"][p] for p in b["step_tokens"][GT_STEP]]) == _gt_text(traj)
    kept = sorted(k for k in b["step_tokens"] if 0 <= k < 4)
    assert kept and kept == list(range(4 - len(kept), 4))   # oldest real turns go first
    assert len(kept) < 4
    # degenerate budget: tail-keep, scored step only, no phantom GT bucket
    d = _build(traj, 4, tok, 5, None, with_gt=True)
    assert d["hard_truncated"] and len(d["input_ids"]) == 5
    assert set(d["step_tokens"]) == {4}


def test_gt_empty_raises():
    from src.data.context import _build
    tok, traj = _StubTok(), _toy_traj(gt="  ")
    with pytest.raises(AssertionError):
        _build(traj, 2, tok, None, None, with_gt=True)
    _build(traj, 2, tok, None, None)                   # without GT: fine


def _gt_weighting(plain, gt_mass):
    """plain weighting + a GT_STEP=-1 bucket carrying gt_mass, rest rescaled."""
    return {tid: {s: {"ctx_indices": torch.cat([torch.tensor([-1]), d["ctx_indices"]]),
                      "weights": torch.cat([torch.tensor([gt_mass]),
                                            d["weights"] * (1 - gt_mass)])}
                  for s, d in steps.items()}
            for tid, steps in plain.items()}


def test_build_W_drops_gt_sentinel():
    """A GT_STEP=-1 context bucket must behave exactly like WW's unscored turn 0:
    dropped by build_W with the surviving real predecessors renormalized."""
    from src.rescore.weights import build_W, strategy_mats
    from src.rescore.strategies import STRATEGIES, backprop_succ_loop
    keeper = _Keeper()
    assert all(e.step_idx >= 0 for e in keeper.index)  # sentinel can never be a keeper row
    plain = _toy_weighting(keeper)
    gt_w = _gt_weighting(plain, gt_mass=0.3)
    for a, b in zip(build_W(keeper, plain, "all"), build_W(keeper, gt_w, "all")):
        assert torch.allclose(a, b, atol=1e-6)         # drop + renorm recovers plain W
    # w=1 with the GT bucket strongest: it claims the only top-w slot, then drops ->
    # all-zero rows, scores pass through (the documented turn-0 slot-consumption path;
    # backprop-only — the succ variants mask the ALREADY-FILTERED full W, so the GT
    # bucket can never consume one of their slots)
    for Wj in build_W(keeper, _gt_weighting(plain, gt_mass=0.9), 1):
        assert torch.equal(Wj, torch.zeros_like(Wj))
    # sentinel is transparent to the succ variants: masked matrices match plain ones,
    # and vectorized == reference loop with the sentinel present, across w regimes
    N = keeper.traj_ranges[-1][1]
    s = (torch.randn(N) * 2).round()
    for w in (1, 2, "all"):
        mats_p, mats_g = strategy_mats(keeper, plain, w), strategy_mats(keeper, gt_w, w)
        for name, variant in (("succ-strong", "strongest"), ("succ-near", "nearest")):
            for a, b in zip(mats_p[name], mats_g[name]):
                assert torch.allclose(a, b, atol=1e-6), (name, w)
            for gamma in (0.1, 1.0):
                vec = STRATEGIES[name](s, keeper, mats_g, [gamma])[:, 0]
                loop = backprop_succ_loop(s, keeper, gt_w, gamma, w, variant)
                assert torch.allclose(vec, loop, atol=1e-6), (name, w, gamma)
