"""The sweep and the selection — one table, one pass.

TWO GRIDS, STAGED:

  base     position x c_begin x c_end          (dense; ~210 bands per position)
  rescore  layer_range x gamma x w x strategy  (504 configs, for the CHOSEN base only)

The rescore grid is expanded for the SELECTED base config, not for all of them. The full
cross product would be ~3.7M rows per seed for a 35-position model; staged, the whole
table is ~24k rows. That staging — not any approximation — is why one table suffices.

NO SEPARATE BASE TABLE. At gamma=0, ``S~ = s + 0*(M^T s)/(M^T 1) = s`` EXACTLY for every
matrix (``M^T 1 > 0`` always, and zero rows give B=0, so there is no NaN path). So the
gamma=0 rows of the rescore grid ARE the base score, and "SOAP w/o rescoring" needs no
distinct artifact. ``run_sweep`` asserts that identity against the base rows on every
run — a free self-check that the fold-in and the plumbing agree.

Base rows are marked by the ``strategy = "base"`` sentinel and carry ``layer_range="-"``,
``w="-"``, ``gamma=0.0``.

    python -m main sweep  --config configs-main/ww.yaml
    python -m main select --config configs-main/ww.yaml
"""
from __future__ import annotations

import pandas as pd
import torch
from tqdm.auto import tqdm

from . import config as C
from .metrics import KeeperContext, compute_metrics_batch
from .rescore import WCache, aggregate_attn, apply_strategy
from .score import (ENSEMBLE_POSITION, band_bounds, base_positions, ens_score_steps,
                    fit_svd, member_positions, score_steps)
from .stores import list_rep_files, load_representations, split_files

BASE_STRATEGY = "base"
POOLING = "mean"          # frozen: the sweep reads one pooling, the extractor writes both

CONFIG_COLS = ["position", "c_begin", "c_end", "strategy", "layer_range", "gamma", "w"]
BASE_SWEPT = ["position", "c_begin", "c_end"]
RESCORE_SWEPT = ["layer_range", "gamma", "w"]


def metric_cols(ks) -> list[str]:
    return [f"{kind}_acc_{split}@{k}"
            for split in ("val", "test") for k in ks for kind in ("step", "agent")]


def sweep_cols(ks) -> list[str]:
    return ["model", "subset", "seed"] + CONFIG_COLS + metric_cols(ks)


# ── selection ───────────────────────────────────────────────────────────────
def norm_val(v) -> str:
    """String form that survives pandas' int->float round-trip ('9.0' -> '9')."""
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


# Accuracies are rationals with small denominators (hits / n_trajectories), so two
# configs that tie mathematically can land 1-2 ulps apart once averaged over seeds —
# the addends differ even though the sums do not (26+24+27 vs 25+24+28, over 63).
# Comparing raw floats therefore lets summation order, i.e. ROW ORDER IN THE FILE,
# decide a tie before the documented agent tiebreak is ever consulted. Rounding the
# comparison key restores the intended rule: equal-on-step configs are separated by
# agent accuracy, and only then by the highest config key. Reported values keep full
# precision. 12 decimals is far below any real accuracy difference (1/n_traj >= 1/2226)
# and far above float noise.
_TIE_DP = 12


def select_config(df: pd.DataFrame, swept: list[str], seeds: list[int],
                  step_col: str, agent_col: str) -> dict | None:
    """Argmax of the mean test metric over ``seeds``; the config must appear in EVERY
    seed. Tiebreak on the mean agent metric, then the HIGHEST config key (the ``>=``
    over a sorted groupby makes full ties resolve deterministically)."""
    df = df[df["seed"].isin(seeds)]
    best = best_key = None
    for _, g in df.groupby(swept, sort=True):
        if len(g) < len(seeds):
            continue
        cand = {"config": {ax: g.iloc[0][ax] for ax in swept},
                "step": float(g[step_col].mean()),
                "agent": float(g[agent_col].mean()),
                "step_val": float(g[step_col.replace("test", "val")].mean()),
                "agent_val": float(g[agent_col.replace("test", "val")].mean())}
        key = (round(cand["step"], _TIE_DP), round(cand["agent"], _TIE_DP))
        if best is None or key >= best_key:
            best, best_key = cand, key
    return best


# ── the two passes ──────────────────────────────────────────────────────────
def _load_split(rep_dir, data_dir, files, device):
    return load_representations(rep_dir, data_dir, poolings=[POOLING],
                                files=files, device=device)


def _metrics_row(vm, tm, ks, i) -> dict:
    out = {}
    for split, m in (("val", vm), ("test", tm)):
        for k in ks:
            out[f"step_acc_{split}@{k}"] = float(m[f"step@{k}"][i])
            out[f"agent_acc_{split}@{k}"] = float(m[f"agent@{k}"][i])
    return out


def _base_pass(cfg, model, subset, seeds, rep_dir, data_dir, device) -> list[dict]:
    """Dense base grid over (position, band) for each frozen seed."""
    ks = cfg["ks"]
    n_comp = cfg["n_components"]
    bands = band_bounds(n_comp)
    files = list_rep_files(rep_dir)
    rows: list[dict] = []

    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        train = _load_split(rep_dir, data_dir, parts["train"], device)
        val = _load_split(rep_dir, data_dir, parts["val"], device)
        test = _load_split(rep_dir, data_dir, parts["test"], device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        available = train.positions()
        positions = base_positions(available, cfg.get("positions", "all"),
                                   cfg.get("ensemble", True))
        members = member_positions(available)
        fits: dict[str, torch.Tensor] = {}

        for position in tqdm(positions, desc=f"base {model}/{subset} s{seed}", leave=False):
            if position == ENSEMBLE_POSITION:
                for p in members:
                    fits.setdefault(p, fit_svd(train.stores[(POOLING, p)].R, n_comp))
                tr = {p: train.stores[(POOLING, p)].R for p in members}
                vR = {p: val.stores[(POOLING, p)].R for p in members}
                tR = {p: test.stores[(POOLING, p)].R for p in members}
                vs = [ens_score_steps(cb, ce, members, fits, tr, vR) for cb, ce in bands]
                ts = [ens_score_steps(cb, ce, members, fits, tr, tR) for cb, ce in bands]
            else:
                V = fits.setdefault(position, fit_svd(train.stores[(POOLING, position)].R, n_comp))
                Rv = val.stores[(POOLING, position)].R
                Rt = test.stores[(POOLING, position)].R
                vs = [score_steps(Rv, V, cb, ce) for cb, ce in bands]
                ts = [score_steps(Rt, V, cb, ce) for cb, ce in bands]
            vm = compute_metrics_batch(torch.stack(vs), None, ks, ctx=val_ctx)
            tm = compute_metrics_batch(torch.stack(ts), None, ks, ctx=test_ctx)
            for i, (cb, ce) in enumerate(bands):
                rows.append({"model": model, "subset": subset, "seed": seed,
                             "position": position, "c_begin": cb, "c_end": ce,
                             "strategy": BASE_STRATEGY, "layer_range": "-",
                             "gamma": 0.0, "w": "-", **_metrics_row(vm, tm, ks, i)})
        del train, val, test
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def _rescore_pass(cfg, model, subset, seeds, base_cfg, rep_dir, data_dir,
                  weightings, range_labels, device) -> list[dict]:
    """The rescore grid, conditional on the chosen base config."""
    ks = cfg["ks"]
    n_comp = cfg["n_components"]
    gammas = list(cfg["gammas"])
    ws = list(cfg["ws"])
    strategies = list(cfg["strategies"])
    position = base_cfg["position"]
    cb, ce = int(base_cfg["c_begin"]), int(base_cfg["c_end"])
    files = list_rep_files(rep_dir)
    rows: list[dict] = []

    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        train = _load_split(rep_dir, data_dir, parts["train"], device)
        val = _load_split(rep_dir, data_dir, parts["val"], device)
        test = _load_split(rep_dir, data_dir, parts["test"], device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)

        if position == ENSEMBLE_POSITION:
            members = member_positions(train.positions())
            fits = {p: fit_svd(train.stores[(POOLING, p)].R, n_comp) for p in members}
            tr = {p: train.stores[(POOLING, p)].R for p in members}
            s_val = ens_score_steps(cb, ce, members, fits,
                                    tr, {p: val.stores[(POOLING, p)].R for p in members})
            s_test = ens_score_steps(cb, ce, members, fits,
                                     tr, {p: test.stores[(POOLING, p)].R for p in members})
        else:
            V = fit_svd(train.stores[(POOLING, position)].R, n_comp)
            s_val = score_steps(val.stores[(POOLING, position)].R, V, cb, ce)
            s_test = score_steps(test.stores[(POOLING, position)].R, V, cb, ce)

        val_WC = WCache(weightings, val.keeper, ws, device=device)
        test_WC = WCache(weightings, test.keeper, ws, device=device)
        for r_idx, label in enumerate(range_labels):
            for w in ws:
                vmats, tmats = val_WC.mats(r_idx, w), test_WC.mats(r_idx, w)
                for strat in strategies:
                    # (N, G) -> (G, N): gammas are the metric batch dimension.
                    Sv = apply_strategy(s_val, val.keeper, vmats, strat, gammas).T.contiguous()
                    St = apply_strategy(s_test, test.keeper, tmats, strat, gammas).T.contiguous()
                    vm = compute_metrics_batch(Sv, None, ks, ctx=val_ctx)
                    tm = compute_metrics_batch(St, None, ks, ctx=test_ctx)
                    for gi, gamma in enumerate(gammas):
                        rows.append({"model": model, "subset": subset, "seed": seed,
                                     "position": position, "c_begin": cb, "c_end": ce,
                                     "strategy": strat, "layer_range": label,
                                     "gamma": gamma, "w": w,
                                     **_metrics_row(vm, tm, ks, gi)})
        del train, val, test, val_WC, test_WC
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def _assert_gamma0_is_base(df: pd.DataFrame, ks) -> None:
    """Every gamma=0 rescore row must equal its base row exactly."""
    base = df[df.strategy == BASE_STRATEGY]
    zero = df[(df.strategy != BASE_STRATEGY) & (df.gamma == 0.0)]
    if zero.empty:
        return
    key = ["seed", "position", "c_begin", "c_end"]
    cols = metric_cols(ks)
    ref = base.set_index(key)[cols]
    for _, r in zero.iterrows():
        want = ref.loc[(r.seed, r.position, r.c_begin, r.c_end)]
        bad = [c for c in cols if abs(float(r[c]) - float(want[c])) > 1e-12]
        assert not bad, (f"gamma=0 row disagrees with its base row on {bad} "
                         f"(seed={r.seed} {r.strategy} {r.layer_range} w={r.w})")


# ── drivers ─────────────────────────────────────────────────────────────────
def run_sweep(cfg: dict) -> None:
    device = cfg.get("device", "cuda")
    force = cfg.get("force", False)
    ks = cfg["ks"]
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            seeds = C.seeds_for(cfg, subset)
            out_dir = C.sweep_dir(cfg, model, subset)
            out = out_dir / "sweep.tsv"
            if out.exists() and not force:
                print(f"[skip] {out}")
                continue
            if cfg.get("dry_run"):
                print(f"[dry] sweep {model}/{subset} seeds={seeds}")
                continue
            C.check_stamp(cfg, out_dir, force=force)

            rep_dir = C.reps_root(cfg) / model / subset
            data_dir = C.data_root(cfg) / subset
            # Attention is independent of the base score and of the seed split.
            weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                                n_ranges=cfg["n_ranges"], device=device)
            range_labels = [f"{lo}-{hi}" for lo, hi in bounds]

            base_rows = _base_pass(cfg, model, subset, seeds, rep_dir, data_dir, device)
            base_df = pd.DataFrame(base_rows)
            best = select_config(base_df, BASE_SWEPT, seeds,
                                 f"step_acc_test@{ks[0]}", f"agent_acc_test@{ks[0]}")
            assert best is not None, f"no base config present in all seeds for {model}/{subset}"
            print(f"[base] {model}/{subset}: {best['config']} "
                  f"step@{ks[0]}={best['step']:.4f}")

            resc_rows = _rescore_pass(cfg, model, subset, seeds, best["config"],
                                      rep_dir, data_dir, weightings, range_labels, device)
            df = pd.DataFrame(base_rows + resc_rows)[sweep_cols(ks)]
            _assert_gamma0_is_base(df, ks)
            df = df.sort_values(f"step_acc_test@{ks[0]}", ascending=False, kind="mergesort")
            df.to_csv(out, sep="\t", index=False)
            print(f"  wrote {out}  ({len(df)} rows)")


def run_select(cfg: dict) -> None:
    """Pure read-side reduction of the sweep tables — no recomputation."""
    ks = cfg["ks"]
    step_col, agent_col = f"step_acc_test@{ks[0]}", f"agent_acc_test@{ks[0]}"
    out_dir = C.select_dir(cfg)
    rows = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            seeds = C.seeds_for(cfg, subset)
            path = C.sweep_dir(cfg, model, subset) / "sweep.tsv"
            if not path.exists():
                print(f"[skip] no sweep for {model}/{subset}")
                continue
            df = pd.read_csv(path, sep="\t")
            seeds_str = ",".join(map(str, seeds))

            base = select_config(df[df.strategy == BASE_STRATEGY], BASE_SWEPT, seeds,
                                 step_col, agent_col)
            if base is None:
                print(f"[skip] no complete base config for {model}/{subset}")
                continue
            rows.append({"model": model, "subset": subset, "row": "svd", "seeds": seeds_str,
                         **base["config"], "strategy": "", "layer_range": "", "gamma": 0.0,
                         "w": "", "step_acc_test": base["step"], "agent_acc_test": base["agent"],
                         "step_acc_val": base["step_val"], "agent_acc_val": base["agent_val"]})

            for strat in cfg["strategies"]:
                sw = df[df.strategy == strat]
                for ax in BASE_SWEPT:
                    sw = sw[sw[ax].astype(str).map(norm_val)
                            == norm_val(base["config"][ax])]
                sel = select_config(sw, RESCORE_SWEPT, seeds, step_col, agent_col)
                if sel is None:
                    continue
                rows.append({"model": model, "subset": subset, "row": strat,
                             "seeds": seeds_str, **base["config"], "strategy": strat,
                             **sel["config"],
                             "step_acc_test": sel["step"], "agent_acc_test": sel["agent"],
                             "step_acc_val": sel["step_val"], "agent_acc_val": sel["agent_val"]})
    if not rows:
        print("[select] nothing to write")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    C.check_stamp(cfg, out_dir, force=True)
    out = out_dir / "selection.tsv"
    cols = (["model", "subset", "row", "seeds"] + CONFIG_COLS
            + ["step_acc_test", "agent_acc_test", "step_acc_val", "agent_acc_val"])
    pd.DataFrame(rows)[cols].to_csv(out, sep="\t", index=False)
    print(f"[select] wrote {out}  ({len(rows)} rows)")
