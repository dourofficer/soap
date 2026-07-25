"""Reduce sweep tables to best-config-per-seed, in BOTH selection conventions.

Two stages, selected by ``stage`` (base | crr | both):

* base — reduce the score TSVs (all seeds) to the best base config per seed.
* crr  — reduce the rescore sweep to CRR + backprop winners.

POOLING MODE (``pooling_mode``) — a first-class toggle, because how pooling is treated
is a real methodological choice, not a detail:

* ``joint`` (default): pooling is an ordinary hyperparameter and competes inside the
  per-seed reduction (group by ``seed``) — one winner per seed across pooling / position
  / method / band. This is the native behaviour.
* ``separate``: pooling is held FIXED in the reduction (group by ``pooling, seed``) —
  one winner per (pooling, seed) — and which pooling to report is decided LATER, at
  table-build time. This reproduces the legacy protocol where pooling was chosen as a
  separate stage after everything else, and it is what the faithful main-table path uses.

The two modes can give different headline numbers; both are supported so a faithful
reproduction and the native result can sit side by side.

VARIANT (``variant``) — an output-filename suffix so a specialised pass (e.g. a
proj-only, pooling=separate reduction feeding the faithful table) coexists with the
default all-methods reduction under the same split tag: ``base_<variant>_test.tsv`` etc.

Writes under ``reduced/<tag>/<model>/<subset>/``:
    base[_<variant>]_{test,val}.tsv    best per group by (step_acc_*, agent_acc_*)
    base_by_method_{test,val}.tsv      best per (seed, method)   [diagnostics; joint only]
    crr[_<variant>]_{test,val}.tsv     discount winners
    backprop[_<variant>]_{test,val}.tsv

    # from repo root
    python -m src.reports.reduce --config configs/reduce/correct-full.yaml --set stage=base
    # faithful proj/separate variant:
    python -m src.reports.reduce --config configs/reduce/ww.yaml --set stage=base \
        --set headline_methods=[proj] --set pooling_mode=separate --set variant=proj
"""
from __future__ import annotations

import pandas as pd

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer

DEFAULT_HEADLINE = ["proj", "resid", "angres"]


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


_sweep_name = sweep_name


def best_per_group(df: pd.DataFrame, metrics: list[str], group_keys: list[str],
                   top_k: int = 1) -> pd.DataFrame:
    """Top-``top_k`` rows per group by ``metrics`` (lexicographic, stable). Adds ``rank``.

    ``group_keys`` defines what is held FIXED while everything else competes, and that
    choice is the whole statistical content of a reduction. Grouping by ``seed`` alone
    (what the base reduction does) means pooling, position, method, band and centering
    all compete against each other within a seed — one winner per seed, then seeds are
    averaged. Adding a key (e.g. ``method``) instead reports the best each family can do,
    which is a diagnostic, not a headline.

    ``mergesort`` is required: it is the only stable sort in pandas, so ties resolve by
    the incoming row order rather than arbitrarily, making reductions deterministic
    across runs.

    Selection metric matters as much as the grouping. Selecting on a ``*_test`` column
    means the reported number is the maximum over the sweep, which is optimistic by
    construction; selecting on ``*_val`` and reporting test is the leak-free protocol.
    Both are emitted so the gap between them is always visible.
    """
    d = df.sort_values(metrics, ascending=False, kind="mergesort").copy()
    d["rank"] = d.groupby(group_keys, sort=False).cumcount() + 1
    return d[d["rank"] <= top_k].reset_index(drop=True)


def _restrict_seeds(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the manifest's ``seeds``. Score/sweep files may hold more seeds than the
    experiment uses (e.g. extra seeds scored later); the reduction must reflect the
    configured seed set, or seed selection downstream diverges from the intended run."""
    seeds = cfg.get("seeds")
    return df[df["seed"].isin(seeds)] if seeds else df


def _load_scores(cfg, model, subset) -> pd.DataFrame | None:
    d = paths.scores_root(cfg) / model / subset
    files = sorted(d.glob("seed-*.tsv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    df = _restrict_seeds(cfg, df)
    return df[df["k"] == 1].reset_index(drop=True)


def reduce_base(cfg) -> list:
    headline = cfg.get("headline_methods", DEFAULT_HEADLINE)
    top_k = cfg.get("base_top_k", 1)
    gk = _seed_keys(cfg)
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            df = _load_scores(cfg, model, subset)
            if df is None:
                print(f"[skip] no scores for {model}/{subset}")
                continue
            head = df[df["method"].isin(headline)]
            # exclude pseudo-positions (e.g. the ens-mid3 ensemble) — used by the faithful
            # reproduction, which must not include v2-only features the legacy run lacked.
            excl = cfg.get("exclude_positions", [])
            if excl:
                head = head[~head["position"].isin(excl)]
            out_dir = paths.reduced_root(cfg) / model / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            tables = {
                _fname("base", cfg, "test"): best_per_group(head, ["step_acc_test", "agent_acc_test"], gk, top_k),
                _fname("base", cfg, "val"):  best_per_group(head, ["step_acc_val", "agent_acc_val"], gk, top_k),
            }
            # by-method diagnostics only for the default (joint, unsuffixed) reduction.
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
    """Reduce the rescore sweep to CRR + backprop tables, both conventions.

    Best per group (``pooling_mode``: joint -> per seed; separate -> per pooling×seed,
    matching v1's reduced ``svd.tsv`` shape so the main-table builder can pick pooling
    itself). ``discount`` rows -> crr[_<variant>]_{test,val}.tsv; ``backprop`` rows ->
    backprop[_<variant>]_{test,val}.tsv (kept apart so the ablation never contaminates
    the headline CRR cell). Adds diff = disc - undisc. Reads the matching (variant)
    sweep file.
    """
    gk = _seed_keys(cfg)
    written = []
    for model in cfg["models"]:
        for subset in cfg["subsets"]:
            sweep = paths.rescore_root(cfg) / model / subset / _sweep_name(cfg)
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
            print(f"[crr] {model}/{subset}: reduced {len(df)} sweep rows "
                  f"(pooling_mode={cfg.get('pooling_mode','joint')}, variant={cfg.get('variant') or '-'})")
    return written


def run(cfg) -> None:
    stage = cfg.get("stage", "both")
    with RunTimer(cfg, "reduced") as rec:
        rec.note(reduce_stage=stage)
        if stage in ("base", "both"):
            for p in reduce_base(cfg):
                rec.add_output(p)
        if stage in ("crr", "both"):
            for p in reduce_crr(cfg):
                rec.add_output(p)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
