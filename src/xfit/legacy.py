"""Retired main-pipeline machinery that ONLY the xfit strand still uses.

When the triples protocol became the main selection path, the reduce conventions
(`base_*`/`crr_*`/`backprop_*` reduced tables, `_test`/`_val` argmax selection), the
legacy pooling/seed picks, and the discount (CRR) strategy were removed from the core
pipeline. xfit's cross-dataset generalization tables are defined AGAINST those
conventions (its verify gate compares byte-for-byte with the archived reduced trees),
so the pieces it needs live on here, copied from ``src_v2/reports/{reduce,
main_table}.py`` and ``src_v2/rescore/{weights,strategies}.py`` unchanged in behavior.
Nothing outside ``src/xfit`` (and the optional legacy-parity test) may import this.

    from src.xfit.legacy import reduce_base, reduce_crr, LEGACY_STRATEGIES, WCacheLegacy
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import pandas as pd
import torch

from ..common import paths
from ..reports.baselines import MODEL_DISPLAY, load_scores
from ..rescore.weights import build_W, coerce_w
from ..rescore.strategies import backprop_vec

DEFAULT_HEADLINE = ["proj", "resid", "angres"]
_EPS = 1e-12


# ── reduction helpers (src_v2/reports/reduce.py) ─────────────────────────────
def _seed_keys(cfg) -> list[str]:
    """Reduction grouping: ['seed'] (pooling competes) or ['pooling','seed'] (pooling fixed)."""
    return ["pooling", "seed"] if cfg.get("pooling_mode") == "separate" else ["seed"]


def _fname(prefix: str, cfg, conv: str) -> str:
    v = cfg.get("variant")
    return f"{prefix}_{v}_{conv}.tsv" if v else f"{prefix}_{conv}.tsv"


def sweep_name(cfg) -> str:
    """Rescore sweep filename for this variant (writer and reader must agree)."""
    v = cfg.get("variant")
    return f"sweep_{v}.tsv" if v else "sweep.tsv"


def best_per_group(df: pd.DataFrame, metrics: list[str], group_keys: list[str],
                   top_k: int = 1) -> pd.DataFrame:
    """Top-``top_k`` rows per group by ``metrics`` (lexicographic, stable). Adds ``rank``.

    ``group_keys`` defines what is held FIXED while everything else competes.
    ``mergesort`` is required: it is the only stable sort in pandas, so ties resolve by
    the incoming row order, making reductions deterministic across runs. Selecting on
    a ``*_test`` column is optimistic by construction; ``*_val`` is the leak-free
    convention — the legacy pipeline emitted both so the gap stayed visible."""
    d = df.sort_values(metrics, ascending=False, kind="mergesort").copy()
    d["rank"] = d.groupby(group_keys, sort=False).cumcount() + 1
    return d[d["rank"] <= top_k].reset_index(drop=True)


def _restrict_seeds(cfg, df: pd.DataFrame) -> pd.DataFrame:
    seeds = cfg.get("seeds")
    return df[df["seed"].isin(seeds)] if seeds else df


def reduce_base(cfg) -> list:
    """Recorded scores -> base[_<variant>]_{test,val}.tsv (+ by-method diagnostics)."""
    headline = cfg.get("headline_methods", DEFAULT_HEADLINE)
    top_k = cfg.get("base_top_k", 1)
    gk = _seed_keys(cfg)
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            df = load_scores(cfg, model, subset)
            if df is None:
                print(f"[skip] no scores for {model}/{subset}")
                continue
            head = df[df["method"].isin(headline)]
            excl = cfg.get("exclude_positions", [])
            if excl:
                head = head[~head["position"].isin(excl)]
            out_dir = paths.reduced_root(cfg) / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            tables = {
                _fname("base", cfg, "test"): best_per_group(head, ["step_acc_test", "agent_acc_test"], gk, top_k),
                _fname("base", cfg, "val"):  best_per_group(head, ["step_acc_val", "agent_acc_val"], gk, top_k),
            }
            if not cfg.get("variant"):
                tables["base_by_method_test.tsv"] = best_per_group(
                    df, ["step_acc_test", "agent_acc_test"], ["seed", "method"], 1)
                tables["base_by_method_val.tsv"] = best_per_group(
                    df, ["step_acc_val", "agent_acc_val"], ["seed", "method"], 1)
            for name, t in tables.items():
                p = out_dir / name
                t.to_csv(p, sep="\t", index=False)
                written.append(p)
            print(f"[base] {model}/{subset}: {len(next(iter(tables.values())))} rows "
                  f"(pooling_mode={cfg.get('pooling_mode','joint')}, variant={cfg.get('variant') or '-'})")
    return written


def reduce_crr(cfg) -> list:
    """Rescore sweep -> crr[_<variant>]_{test,val}.tsv (discount rows) +
    backprop[_<variant>]_{test,val}.tsv (backprop rows), diff columns added."""
    gk = _seed_keys(cfg)
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            sweep = paths.rescore_root(cfg) / model / subset / sweep_name(cfg)
            if not sweep.exists():
                print(f"[skip] no sweep for {model}/{subset} ({sweep.name})")
                continue
            df = _restrict_seeds(cfg, pd.read_csv(sweep, sep="\t"))
            df["diff_step_acc_test"] = df["disc_step_acc_test"] - df["undisc_step_acc_test"]
            df["diff_agent_acc_test"] = df["disc_agent_acc_test"] - df["undisc_agent_acc_test"]
            out_dir = paths.reduced_root(cfg) / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            for strat, prefix in (("discount", "crr"), ("backprop", "backprop")):
                sub = df[df["strategy"] == strat]
                if sub.empty:
                    continue
                for conv, metrics in (("test", ["disc_step_acc_test", "disc_agent_acc_test"]),
                                      ("val", ["disc_step_acc_val", "disc_agent_acc_val"])):
                    best = best_per_group(sub, metrics, gk, 1)
                    p = out_dir / _fname(prefix, cfg, conv)
                    best.to_csv(p, sep="\t", index=False)
                    written.append(p)
            print(f"[crr] {model}/{subset}: reduced {len(df)} sweep rows")
    return written


# ── legacy table/selection helpers (src_v2/reports/main_table.py) ────────────
def _read_tsv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_pooling_per_seed(rows: list[dict]) -> dict[int, dict]:
    """seed -> chosen (pooling) row: max (undisc_step_acc_test, disc_step_acc_test, pooling)."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    return {s: max(c, key=lambda r: (float(r["undisc_step_acc_test"]),
                                     float(r["disc_step_acc_test"]), r["pooling"]))
            for s, c in by_seed.items()}


def _by_seed(rows: list[dict]) -> dict[int, dict]:
    return {int(r["seed"]): r for r in rows}


def _mean_over(cell: dict, seeds, col):
    vals = [float(cell[s][col]) for s in seeds if s in cell and cell[s].get(col) not in (None, "")]
    return mean(vals) if vals else None


def _selection_cells(cfg, subsets, crr_file):
    """selected[(model,subset)] = {seed: chosen row} from a reduced CRR table."""
    reduced = paths.reduced_root(cfg)
    selected, have = {}, False
    for _, mk in MODEL_DISPLAY:
        for _, sk in subsets:
            rows = _read_tsv(reduced / mk / sk / crr_file)
            selected[(mk, sk)] = pick_pooling_per_seed(rows) if rows else {}
            have |= bool(rows)
    return selected, have


def _choose_seeds(selected, n_select) -> list[int]:
    """Top-n seeds by mean disc_step_acc_test across cells, tiebreak mean diff (legacy)."""
    seeds = sorted({s for cell in selected.values() for s in cell})

    def means(seed):
        disc = [float(c[seed]["disc_step_acc_test"]) for c in selected.values() if seed in c]
        diff = [float(c[seed].get("diff_step_acc_test", 0.0)) for c in selected.values() if seed in c]
        return (mean(disc) if disc else -1e9, mean(diff) if diff else -1e9)

    ranked = sorted(seeds, key=means, reverse=True)
    return ranked[:n_select]


# ── discount (CRR) strategy + list-valued WCache (src_v2/rescore) ────────────
def discount_loop(scores, keeper, weighting, gamma, w):
    """Single-pass discount, written out per step: S~_t = s_t - gamma * sum_i w_{i,t} s_i.
    The readable reference ``discount_vec`` must match. Scores must be 'higher = error'."""
    w = coerce_w(w)
    out = scores.clone()
    device = scores.device
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        if not entries:
            continue
        traj_w = weighting.get(str(entries[0].traj_idx))
        if traj_w is None:
            continue
        step_to_global = {e.step_idx: start + i for i, e in enumerate(entries)}
        for offset, e in enumerate(entries):
            ctx = traj_w.get(e.step_idx)
            if ctx is None:
                continue
            ctx_ids, ctx_w = ctx["ctx_indices"], ctx["weights"].to(device)
            n_ctx = ctx_w.shape[0]
            if n_ctx == 0:
                continue
            if w == "all" or (isinstance(w, int) and w >= n_ctx):
                kept_w, kept_ids = ctx_w, ctx_ids
            else:
                vals, idx = torch.topk(ctx_w, int(w))
                kept_w = vals / (vals.sum() + _EPS)
                kept_ids = ctx_ids[idx]
            pred, aligned = [], []
            for j, ci in enumerate(kept_ids.tolist()):
                pos = step_to_global.get(int(ci))
                if pos is not None:
                    pred.append(scores[pos])
                    aligned.append(kept_w[j])
            if not pred:
                continue
            pred = torch.stack(pred)
            aligned = torch.stack(aligned)
            if aligned.numel() != kept_w.numel():
                aligned = aligned / (aligned.sum() + _EPS)
            out[start + offset] = scores[start + offset] - gamma * (aligned * pred).sum()
    return out


def _discount_column(s, keeper, Wmats) -> torch.Tensor:
    """D = W s assembled over trajectories (per-step discount sum). (N,)."""
    D = torch.zeros_like(s)
    for (start, end), W in zip(keeper.traj_ranges, Wmats):
        D[start:end] = W.to(s) @ s[start:end]
    return D


def discount_vec(s, keeper, Wmats, gammas) -> torch.Tensor:
    """S~ for every gamma: (N, G). S~ = s - gamma * (W s)."""
    D = _discount_column(s, keeper, Wmats)
    g = torch.as_tensor(gammas, dtype=s.dtype, device=s.device)
    return s[:, None] - g[None, :] * D[:, None]


# The legacy strategy fns take a LIST of per-trajectory matrices (not the new
# per-strategy dict), and the legacy cache stores exactly that.
LEGACY_STRATEGIES = {"discount": discount_vec, "backprop": backprop_vec}


class WCacheLegacy:
    """Per-(model, subset, split) cache of row-trimmed W matrices, keyed (range_idx, w)."""

    def __init__(self, weightings: list[dict], keeper, ws, device="cpu"):
        self.keeper = keeper
        self._mats: dict[tuple[int, str], list[torch.Tensor]] = {}
        for r_idx, weighting in enumerate(weightings):
            for w in ws:
                self._mats[(r_idx, str(w))] = build_W(keeper, weighting, w, device)

    def mats(self, r_idx: int, w) -> list[torch.Tensor]:
        return self._mats[(r_idx, str(w))]
