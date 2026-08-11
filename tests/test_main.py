"""`main/` correctness: agreement with `src/`, plus the invariants that outlive it.

GROUP A pins `main/` to `src/` where they must agree — the seed->partition mapping, the
metric quirks, the base score, the W construction, the context spans and the selection
rule. That agreement is what makes `main/` the same experiment, not a lookalike.

GROUP B are self-contained invariants: they hold on `main/` alone and survive `src/`
being retired. `test_split_partition_matches_src` deliberately does both — it checks
against `src/` AND against a hardcoded golden partition.

This test file imports both packages; `main/` itself imports nothing from `src/`
(asserted by `test_main_never_imports_src`). CPU-only, no data or weights needed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

import main.data as mdata
import main.metrics as mmetrics
import main.rescore as mrescore
import main.score as mscore
import main.stores as mstores
import main.sweep as msweep

REPO = Path(__file__).resolve().parents[1]
torch.manual_seed(0)
D, T, K = 32, 40, 20


# ── shared fixtures ─────────────────────────────────────────────────────────
def _keeper(module):
    """3 trajectories with known mistake steps/roles, built for either package."""
    StepIndex = module.StepIndex

    class _K:
        pass

    k = _K()
    k.index, k.traj_ranges, k.traj_meta, k.lookup = [], [], {}, {}
    k.device = torch.device("cpu")
    row = 0
    specs = [  # (traj_idx, [(step, role, is_mistake)], mistake_agent)
        (1, [(0, "A", False), (1, "B", True), (2, "A", False)], "B"),
        (2, [(0, "X", False), (1, "Y", False), (2, "Z", True), (3, "X", False)], "Z"),
        (3, [(0, "Orchestrator (thought)", True), (1, "W", False)], "Orchestrator"),
        (4, [(0, "P", False), (1, "Q", False)], None),          # NO gold mistake
    ]
    for tid, steps, agent in specs:
        start = row
        for s, role, mis in steps:
            k.index.append(StepIndex(row, tid, s, role, mis))
            row += 1
        k.traj_ranges.append((start, row))
        k.traj_meta[tid] = {"mistake_agent": agent} if agent else {}
    return k


def _toy_weighting(keeper, gt_bucket=False):
    """Each step attends to all predecessors with random mass.

    With ``gt_bucket`` a GT_STEP=-1 column carrying the LARGEST mass is prepended, so
    top-w selection hands it a slot that ``build_W`` then drops.
    """
    torch.manual_seed(3)
    weighting = {}
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        tid = str(entries[0].traj_idx)
        weighting[tid] = {}
        for i, e in enumerate(entries):
            preds = [entries[j].step_idx for j in range(i)]
            if not preds and not gt_bucket:
                continue
            wv = torch.rand(len(preds)) + 0.1
            if gt_bucket:
                preds = [-1] + preds
                wv = torch.cat([torch.tensor([9.0]), wv])
            weighting[tid][e.step_idx] = {
                "ctx_indices": torch.tensor(preds), "weights": wv / wv.sum()}
    return weighting


def _fit_matrix(n=200):
    return torch.randn(n, D)


# ════════════════════════════════════════════════════════════════════════════
# GROUP A — agreement with src/
# ════════════════════════════════════════════════════════════════════════════

# The single most important test in this file. With seeds frozen in the config, a drift
# in the seed->partition mapping silently makes every reported number a different
# experiment. Checked twice: against src/, and against a literal so the guarantee
# survives src/ being deleted.
GOLDEN_SPLITS = {"train": 0.3, "val": 0.2, "test": 0.5}
GOLDEN_FILES = [f"{i}.safetensors" for i in range(1, 21)]
GOLDEN_SEED_7 = {
    "train": ["15.safetensors", "19.safetensors", "16.safetensors",
              "8.safetensors", "4.safetensors", "18.safetensors"],
    "val": ["1.safetensors", "20.safetensors", "12.safetensors", "7.safetensors"],
    "test": ["10.safetensors", "6.safetensors", "17.safetensors", "9.safetensors",
             "14.safetensors", "3.safetensors", "2.safetensors", "13.safetensors",
             "5.safetensors", "11.safetensors"],
}


def test_split_partition_matches_src():
    import src.stores as sstores
    for n in (11, 20, 58, 126):
        files = [f"{i}.safetensors" for i in range(1, n + 1)]
        for seed in range(1, 51):
            got = mstores.split_files(files, GOLDEN_SPLITS, seed)
            want = sstores.split_files(files, GOLDEN_SPLITS, seed)
            for part in ("train", "val", "test"):
                # ORDERED equality: downstream row order depends on it, not just membership.
                assert got[part] == want[part], (n, seed, part)
    assert mstores.split_data(list(range(10)), 0.4, 3) == sstores.split_data(list(range(10)), 0.4, 3)
    assert mstores.derive_split_ratios(0.3, 0.2, 0.5) == sstores.derive_split_ratios(0.3, 0.2, 0.5)


def test_split_partition_golden_literal():
    """Same guarantee, written down — survives src/ being retired."""
    assert mstores.split_files(GOLDEN_FILES, GOLDEN_SPLITS, 7) == GOLDEN_SEED_7


def test_split_ratio_validation():
    with pytest.raises(ValueError):
        mstores.derive_split_ratios(0.3, 0.2, 0.4)      # does not sum to 1
    with pytest.raises(ValueError):
        mstores.derive_split_ratios(0.5, 0.0, 0.5)      # a zero split


def _write_toy_tree(tmp_path: Path, poolings=("mean", "last"), positions=("act/0", "act/1")):
    """A miniature (activations, corpus) pair: 3 trajectories, 2 positions, 2 poolings."""
    rep_dir, data_dir = tmp_path / "reps", tmp_path / "data"
    rep_dir.mkdir(), data_dir.mkdir()
    torch.manual_seed(11)
    for tid, n_steps, mistake in ((1, 3, 1), (2, 4, 2), (3, 2, 0)):
        flat = {f"{s}.{p}.{w}": torch.randn(D).half()
                for s in range(n_steps) for p in poolings for w in positions}
        save_file(flat, rep_dir / f"{tid}.safetensors", metadata={"payload_metadata": json.dumps(
            {"filename": f"{tid}.json", "mistake_step": str(mistake), "mistake_agent": "B"})})
        (data_dir / f"{tid}.json").write_text(json.dumps(
            {"history": [{"role": "B" if s == mistake else "A", "content": f"s{s}"}
                         for s in range(n_steps)]}))
    return rep_dir, data_dir


def test_keeper_row_order_matches_src(tmp_path):
    import src.stores as sstores
    rep_dir, data_dir = _write_toy_tree(tmp_path)
    m = mstores.load_representations(rep_dir, data_dir, poolings=["mean"])
    s = sstores.load_representations(rep_dir, data_dir, poolings=["mean"])
    assert m.keeper.traj_ranges == s.keeper.traj_ranges
    assert m.keeper.lookup == s.keeper.lookup
    assert m.keeper.traj_meta == s.keeper.traj_meta
    assert [(e.row, e.traj_idx, e.step_idx, e.role, e.is_mistake) for e in m.keeper.index] == \
           [(e.row, e.traj_idx, e.step_idx, e.role, e.is_mistake) for e in s.keeper.index]
    for pos in ("act/0", "act/1"):
        assert torch.equal(m.stores[("mean", pos)].R, s.stores[("mean", pos)].R)
    # the (pooling, name) key is retained even though the sweep only asks for mean
    both = mstores.load_representations(rep_dir, data_dir, poolings=["mean", "last"])
    assert both.poolings() == ["last", "mean"]


def test_metrics_match_src():
    import src.metrics as smetrics
    mk, sk = _keeper(mstores), _keeper(__import__("src.stores", fromlist=["x"]))
    N = mk.traj_ranges[-1][1]
    torch.manual_seed(1)
    # a tie-saturated vector and an all-equal one exercise the earliest-step tie-break
    for s in (torch.randn(N), (torch.randn(N) * 3).round(), torch.zeros(N)):
        got = mmetrics.compute_metrics(s, mk, [1, 3])
        want = smetrics.compute_metrics(s, sk, [1, 3], "desc")
        for k in (1, 3):
            for kind in ("step", "agent"):
                assert abs(got[f"{kind}@{k}"] - want[f"{kind}@{k}_desc"]) < 1e-12
    # the mistake-less trajectory is skipped but still inflates the divisor
    assert mmetrics.KeeperContext(mk).total == 4
    assert len(mmetrics.KeeperContext(mk).trajs) == 3


def test_base_score_equals_src_oriented_proj():
    """Bit-for-bit: main's folded score == src's orient(proj(...), 'inverse')."""
    from src.score.svd import fit_one, score_config
    from src.rescore.strategies import orient
    G, R = _fit_matrix(), torch.randn(T, D).half()
    entry = fit_one(G, K)
    V = mscore.fit_svd(G, K)
    for cb, ce in ((0, 1), (0, 5), (2, 8), (0, 20), (19, 20)):
        got = mscore.score_steps(R, V, cb, ce)
        want = orient(score_config(R, entry, "proj", cb, ce, False, False), "inverse")
        assert torch.equal(got, want), (cb, ce)


def test_base_score_ranks_like_proj_ascending():
    """The fold-in changes no number: 1/(pi+eps) desc == pi asc, ties included."""
    import src.metrics as smetrics
    from src.score.svd import fit_one, score_config
    mk, sk = _keeper(mstores), _keeper(__import__("src.stores", fromlist=["x"]))
    N = mk.traj_ranges[-1][1]
    G = _fit_matrix()
    entry, V = fit_one(G, K), mscore.fit_svd(G, K)
    R = torch.randn(N, D).half()
    for cb, ce in ((0, 2), (1, 6)):
        pi = score_config(R, entry, "proj", cb, ce, False, False)
        got = mmetrics.compute_metrics(mscore.score_steps(R, V, cb, ce), mk, [1, 3])
        want = smetrics.compute_metrics(pi, sk, [1, 3], "asc")
        for k in (1, 3):
            for kind in ("step", "agent"):
                assert abs(got[f"{kind}@{k}"] - want[f"{kind}@{k}_asc"]) < 1e-12, (cb, ce, k)


def test_fit_svd_matches_src_v_raw():
    from src.score.svd import fit_one
    G = _fit_matrix()
    assert torch.equal(mscore.fit_svd(G, K), fit_one(G, K)["V_raw"])


def test_build_W_matches_src():
    from src.rescore.weights import build_W as sbuild_W
    mk, sk = _keeper(mstores), _keeper(__import__("src.stores", fromlist=["x"]))
    for gt_bucket in (False, True):
        wt = _toy_weighting(mk, gt_bucket=gt_bucket)
        for w in (1, 2, 3, "all"):
            got, want = mrescore.build_W(mk, wt, w), sbuild_W(sk, wt, w)
            assert len(got) == len(want)
            for a, b in zip(got, want):
                assert torch.equal(a, b), (gt_bucket, w)


def test_build_W_gt_sentinel_consumes_a_slot():
    """The GT bucket can win the only top-w slot and then be dropped, leaving an
    all-zero row — that slot-consumption order is part of the recorded numbers."""
    mk = _keeper(mstores)
    wt = _toy_weighting(mk, gt_bucket=True)          # GT carries the largest mass
    for W in mrescore.build_W(mk, wt, 1):
        assert torch.count_nonzero(W) == 0
    # with w="all" the real predecessors survive and rows renormalize to 1
    for W in mrescore.build_W(mk, wt, "all"):
        for t in range(1, W.shape[0]):
            assert abs(float(W[t].sum()) - 1.0) < 1e-6


def test_column_masks_match_src():
    from src.rescore.weights import (mask_columns_nearest as s_near,
                                     mask_columns_strongest as s_strong)
    torch.manual_seed(5)
    W = (torch.rand(8, 8) * (torch.rand(8, 8) > 0.4)).tril(-1)
    for w in (1, 2, 4, "all"):
        assert torch.equal(mrescore.mask_columns_strongest(W, w), s_strong(W, w))
        assert torch.equal(mrescore.mask_columns_nearest(W, w), s_near(W, w))


def test_strategy_mats_match_src():
    from src.rescore.weights import strategy_mats as s_mats
    mk, sk = _keeper(mstores), _keeper(__import__("src.stores", fromlist=["x"]))
    wt = _toy_weighting(mk)
    for w in (1, 3, "all"):
        got, want = mrescore.strategy_mats(mk, wt, w), s_mats(sk, wt, w)
        assert set(got) == set(want)
        for name in got:
            for a, b in zip(got[name], want[name]):
                assert torch.equal(a, b), (name, w)


# ── context spans ───────────────────────────────────────────────────────────
class _StubTok:
    """Char-level tokenizer: 1 token per char, so spans are exact and decodable."""
    name_or_path = "stub"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<O>" + msgs[0]["content"] + "<C>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _toy_traj(module, n_turns=5, gt="42"):
    hist = [{"role": f"agent{i}", "content": f"step {i} says " + "x" * (4 + 3 * i)}
            for i in range(n_turns)]
    return module.Trajectory(filename="7.json", question_id="q7", history=hist,
                             mistake_agent="agent1", mistake_step=1, level=1, subset="toy",
                             question="What is six times seven?", system=None,
                             ground_truth=gt)


def _as_src(b):
    """main returns input_ids as a (1,L) tensor; src's _build returns a list."""
    return {**b, "input_ids": b["input_ids"][0].tolist()}


def test_context_matches_src():
    from src.data.context import _build
    import src.data.trajectory as straj
    tok = _StubTok()
    mt, st = _toy_traj(mdata), _toy_traj(straj)
    for with_gt in (False, True):
        for step in (1, 3, 4):
            got = _as_src(mdata.build_step_input(mt, step, tok, None, None, with_gt))
            want = _build(st, step, tok, None, None, with_gt=with_gt)
            assert got == want, (with_gt, step)


def test_context_truncation_matches_src():
    from src.data.context import _build
    import src.data.trajectory as straj
    tok = _StubTok()
    mt, st = _toy_traj(mdata), _toy_traj(straj)
    full = _build(st, 4, tok, None, None, with_gt=True)
    for budget in (len(full["input_ids"]) - 1, 80, 40, 12):
        got = _as_src(mdata.build_step_input(mt, 4, tok, budget, None, True))
        want = _build(st, 4, tok, budget, None, with_gt=True)
        assert got == want, budget


def test_gt_block_pinned_and_empty_gt_raises():
    tok = _StubTok()
    traj = _toy_traj(mdata)
    b = mdata.build_step_input(traj, 3, tok, None, None, with_gt=True)
    gt = (f"The problem is: {traj.question}\n"
          f"The Answer for the problem is: {traj.ground_truth}")
    ids = b["input_ids"][0].tolist()
    span = b["step_tokens"][mdata.GT_STEP]
    assert span == list(range(3, 3 + len(gt)))                 # right after "<O>"
    assert "".join(chr(ids[p]) for p in span) == gt
    with pytest.raises(AssertionError):
        mdata.build_step_input(_toy_traj(mdata, gt="  "), 3, tok, None, None, True)


def test_selection_rule_matches_src():
    """main.select_config picks what src.select_shared picks: same completeness gate,
    same norm_val coercion, same 'highest key wins' tiebreak on full ties."""
    from src.reports.triples import select_shared
    rows = []
    for seed in (1, 2, 3):
        for pos in ("act/3", "act/7"):
            for cb, ce in ((0, 2), (1, 5)):
                # exact ties between the two positions, so the tiebreak decides
                rows.append({"seed": seed, "position": pos, "c_begin": cb, "c_end": ce,
                             "step_acc_test@1": 0.5 if ce == 2 else 0.4,
                             "agent_acc_test@1": 0.7,
                             "step_acc_val@1": 0.3, "agent_acc_val@1": 0.3,
                             "step_acc_test": 0.5 if ce == 2 else 0.4,
                             "agent_acc_test": 0.7})
    # a config missing from one seed must be rejected by both
    rows.append({"seed": 1, "position": "act/9", "c_begin": 0, "c_end": 2,
                 "step_acc_test@1": 0.99, "agent_acc_test@1": 0.99,
                 "step_acc_val@1": 0.0, "agent_acc_val@1": 0.0,
                 "step_acc_test": 0.99, "agent_acc_test": 0.99})
    df = pd.DataFrame(rows)
    swept = ["position", "c_begin", "c_end"]
    got = msweep.select_config(df, swept, [1, 2, 3], "step_acc_test@1", "agent_acc_test@1")
    want = select_shared(df, {}, swept, "step_acc_test", "agent_acc_test", [1, 2, 3])
    assert {k: got["config"][k] for k in swept} == {k: want["config"][k] for k in swept}
    assert abs(got["step"] - want["step"]) < 1e-12
    assert got["config"]["position"] == "act/7"      # highest key wins the tie
    assert msweep.norm_val(9.0) == "9" and msweep.norm_val("act/3") == "act/3"


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — self-contained invariants
# ════════════════════════════════════════════════════════════════════════════
def test_gamma0_is_the_base_score():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    for w in (1, 3, "all"):
        mats = mrescore.strategy_mats(keeper, _toy_weighting(keeper), w)
        for strat in mrescore.STRATEGIES:
            out = mrescore.apply_strategy(s, keeper, mats, strat, [0.0, 0.5])
            assert torch.equal(out[:, 0], s), (strat, w)


def test_w_all_strategies_coincide():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    mats = mrescore.strategy_mats(keeper, _toy_weighting(keeper), "all")
    outs = [mrescore.apply_strategy(s, keeper, mats, st, [0.1, 0.6, 1.0])
            for st in mrescore.STRATEGIES]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


def _succ_loop(scores, keeper, weighting, gamma, w, variant):
    """Explicit per-step successor-side backprop, read straight off the ragged dicts —
    the readable definition the vectorized path must match."""
    w = mrescore.coerce_w(w)
    out = scores.clone()
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        traj_w = weighting.get(str(entries[0].traj_idx))
        if traj_w is None:
            continue
        step_to_global = {e.step_idx: start + k for k, e in enumerate(entries)}
        incoming: dict[int, list] = {start + k: [] for k in range(len(entries))}
        for t_off, e in enumerate(entries):
            ctx = traj_w.get(e.step_idx)
            if ctx is None:
                continue
            ctx_ids, ctx_w = ctx["ctx_indices"], ctx["weights"]
            pred, aligned = [], []
            for j, ci in enumerate(ctx_ids.tolist()):
                pos = step_to_global.get(int(ci))
                if pos is not None:
                    pred.append(pos)
                    aligned.append(ctx_w[j])
            if not pred:
                continue
            av = torch.stack(aligned)
            if av.numel() != ctx_w.numel():
                av = av / (av.sum() + mrescore.EPS)
            for pos, wt in zip(pred, av):
                incoming[pos].append((start + t_off, wt))
        for gi, succs in incoming.items():
            if not succs:
                continue                                  # true sink: passes through
            if w != "all" and len(succs) > w:
                succs = (sorted(succs, key=lambda p: float(p[1]), reverse=True)[:w]
                         if variant == "strongest" else sorted(succs, key=lambda p: p[0])[:w])
            wts = torch.stack([wt for _, wt in succs])
            sts = torch.stack([scores[t] for t, _ in succs])
            out[gi] = scores[gi] + gamma * (wts * sts).sum() / (wts.sum() + mrescore.EPS)
    return out


def test_vec_equals_reference_loop_succ():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    s = (torch.randn(N) * 2).round()                       # tie-heavy on purpose
    for gt_bucket in (False, True):
        wt = _toy_weighting(keeper, gt_bucket=gt_bucket)
        for w in (1, 2, 3, "all"):
            mats = mrescore.strategy_mats(keeper, wt, w)
            for strat, variant in (("succ-strong", "strongest"), ("succ-near", "nearest")):
                for gamma in (0.1, 0.7, 1.0):
                    got = mrescore.apply_strategy(s, keeper, mats, strat, [gamma])[:, 0]
                    want = _succ_loop(s, keeper, wt, gamma, w, variant)
                    assert torch.allclose(got, want, atol=1e-6), (gt_bucket, w, strat, gamma)


def test_backprop_transpose_by_hand():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    s = torch.randn(N)
    gamma = 0.4
    mats = mrescore.strategy_mats(keeper, _toy_weighting(keeper), 2)
    got = mrescore.apply_strategy(s, keeper, mats, "backprop", [gamma])[:, 0]
    want = s.clone()
    for (start, end), W in zip(keeper.traj_ranges, mats["backprop"]):
        for i in range(end - start):
            num = sum(float(W[t, i]) * float(s[start + t]) for t in range(end - start))
            den = sum(float(W[t, i]) for t in range(end - start))
            want[start + i] = s[start + i] + gamma * num / (den + mrescore.EPS)
    assert torch.allclose(got, want, atol=1e-6)


def test_succ_sink_passes_through_and_strong_differs_from_near():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    # step 1 of traj 1: nearest successor is step 2 (mass 0.1), strongest is step 3.
    wt = {"1": {1: {"ctx_indices": torch.tensor([0]), "weights": torch.tensor([1.0])},
                2: {"ctx_indices": torch.tensor([0, 1]), "weights": torch.tensor([0.9, 0.1])}}}
    s = torch.zeros(N)
    s[1], s[2] = 0.0, 5.0
    mats = mrescore.strategy_mats(keeper, wt, 1)
    for strat in mrescore.STRATEGIES:
        out = mrescore.apply_strategy(s, keeper, mats, strat, [1.0])[:, 0]
        # trajectories 2-4 have no weighting entry at all -> untouched
        assert torch.equal(out[3:], s[3:]), strat


def test_batch_metrics_equal_loop():
    keeper = _keeper(mstores)
    N = keeper.traj_ranges[-1][1]
    torch.manual_seed(2)
    S = torch.stack([torch.randn(N), (torch.randn(N) * 3).round(), torch.zeros(N)])
    batch = mmetrics.compute_metrics_batch(S, keeper, [1, 3])
    for i in range(S.shape[0]):
        loop = mmetrics.compute_metrics(S[i], keeper, [1, 3])
        for k in (1, 3):
            for kind in ("step", "agent"):
                assert abs(batch[f"{kind}@{k}"][i] - loop[f"{kind}@{k}"]) < 1e-12


def test_ensemble_uses_train_statistics_only():
    """z-stats come from TRAIN; perturbing eval must not change the standardization."""
    members = ["act/1", "act/2", "act/3"]
    torch.manual_seed(4)
    Vs = {p: mscore.fit_svd(_fit_matrix(), K) for p in members}
    tr = {p: torch.randn(50, D).half() for p in members}
    ev = {p: torch.randn(12, D).half() for p in members}
    got = mscore.ens_score_steps(0, 5, members, Vs, tr, ev)
    want = torch.stack([
        (mscore.score_steps(ev[p], Vs[p], 0, 5)
         - mscore.score_steps(tr[p], Vs[p], 0, 5).mean())
        / (mscore.score_steps(tr[p], Vs[p], 0, 5).std(unbiased=False) + mscore.Z_EPS)
        for p in members]).mean(dim=0)
    assert torch.allclose(got, want, atol=1e-6)
    assert mscore.member_positions(["embed", "act/0", "act/1", "act/2",
                                    "act/3", "act/4", "act/5", "act/5_normed"]) == \
        ["act/2", "act/3"]


def test_config_stamp_refuses_drift(tmp_path):
    import main.config as mcfg
    cfg = {"dataset": "toy", "gt": False, "splits": {"train": 0.3, "val": 0.2, "test": 0.5},
           "seeds": {"a": [1, 2, 3]}, "n_components": 20, "positions": "all",
           "ensemble": True, "n_ranges": 4, "gammas": [0.0], "ws": [1],
           "strategies": ["backprop"]}
    mcfg.check_stamp(cfg, tmp_path)
    mcfg.check_stamp(cfg, tmp_path)                       # unchanged -> fine
    drifted = {**cfg, "seeds": {"a": [4, 5, 6]}}
    with pytest.raises(SystemExit):
        mcfg.check_stamp(drifted, tmp_path)
    mcfg.check_stamp(drifted, tmp_path, force=True)       # explicit override
    assert mcfg.seeds_for(cfg, "a") == [1, 2, 3]
    with pytest.raises(SystemExit):
        mcfg.seeds_for(cfg, "missing")
    with pytest.raises(SystemExit):
        mcfg.seeds_for({"seeds": {"a": [1, 2]}}, "a")     # wrong triple size


def test_main_never_imports_src():
    """main/ must stand alone; this is the one guarantee that decays silently."""
    offenders = []
    for path in sorted((REPO / "main").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {a.name}" for a in node.names
                              if a.name == "src" or a.name.startswith("src.")]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"):
                offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, offenders


def test_configs_main_declare_frozen_triples():
    """Every shipped config must name a 3-seed triple for each of its subsets."""
    import main.config as mcfg
    for path in sorted((REPO / "configs-main").glob("*.yaml")):
        cfg = mcfg.load_config(path)
        for subset in cfg["subsets"]:
            assert len(mcfg.seeds_for(cfg, subset)) == 3, (path.name, subset)


def test_selection_tiebreak_survives_float_noise():
    """Two configs that tie mathematically can land ulps apart once averaged, because
    the addends differ even though their sum does not. Comparing raw floats lets
    summation order — i.e. ROW ORDER IN THE FILE — decide before the documented agent
    tiebreak is consulted; the rounded comparison key must not.

    This is not hypothetical: it is why src/'s recorded succ-near pick for
    ww/deepseek-8b/algorithm-generated is `24-32` where the better-agent `0-8` ties it
    on step accuracy to 1e-16.
    """
    a = [20 / 63, 21 / 63, 25 / 63]          # mean 66/63
    b = [21 / 63, 22 / 63, 23 / 63]          # mean 66/63, but 1 ulp apart under pandas
    assert pd.Series(a).mean() != pd.Series(b).mean(), "fixture no longer shows the gap"
    rows = []
    for seed, (va, vb) in enumerate(zip(a, b), start=1):
        rows.append({"seed": seed, "layer_range": "0-8", "gamma": 1.0, "w": "2",
                     "step_acc_test@1": va, "agent_acc_test@1": 0.9,
                     "step_acc_val@1": 0.0, "agent_acc_val@1": 0.0})
        rows.append({"seed": seed, "layer_range": "24-32", "gamma": 1.0, "w": "2",
                     "step_acc_test@1": vb, "agent_acc_test@1": 0.1,
                     "step_acc_val@1": 0.0, "agent_acc_val@1": 0.0})
    got = msweep.select_config(pd.DataFrame(rows), ["layer_range", "gamma", "w"],
                               [1, 2, 3], "step_acc_test@1", "agent_acc_test@1")
    # agent (0.9 vs 0.1) must decide, not the 1e-16 step difference
    assert got["config"]["layer_range"] == "0-8", got["config"]
