"""Choose the best seed triple per subset — swappably, in seconds.

Reads ONLY `results-sweep/selections_all.tsv` (~8.4k rows), so changing the protocol
never touches the 10.6 GB of sweep tables. Adding a rule is one function plus one dict
entry.

    python scripts/main/pick_triple.py                              # sum, test, backprop
    python scripts/main/pick_triple.py --rule sum-diff
    python scripts/main/pick_triple.py --rule val --strategy succ-near
    python scripts/main/pick_triple.py --restrict-to 1-20            # cross-check vs src/

SCOPE. `ww` and `traceelephant` pick per SUBSET. `correct-error` picks per DATASET: its
manuscript column is the macro-average over the 7 subsets, and `configs-main/correct-error.yaml`
already declares that all 7 share one triple — picking per subset would produce a config
that disagrees with the aggregation the number is reported under. `--scope` overrides.

TIES. All comparison keys are rounded to 12 dp. Accuracies are rationals over small
denominators, so triples that tie mathematically can land ulps apart once summed; without
rounding, float noise decides before the documented tiebreak is consulted. Same reasoning
and same constant as `main/sweep.py:_TIE_DP`.

Note the column is `with_gt`, not `gt` — `df.gt` is pandas' greater-than method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "results-sweep"
TIE_DP = 12
PER_DATASET_SCOPE = {"correct-error"}       # see SCOPE in the docstring


def _seeds(triple: int) -> list[int]:
    return [triple, triple + 1, triple + 2]


def _wide(df: pd.DataFrame, strategy: str, metric: str) -> pd.DataFrame:
    """One row per (with_gt, dataset, subset, triple): per-model strategy and svd values."""
    step, agent = f"step_acc_{metric}", f"agent_acc_{metric}"
    keep = df[df["row"].isin([strategy, "svd"])]
    idx = ["with_gt", "dataset", "subset", "triple", "model"]
    p = keep.pivot_table(index=idx, columns="row", values=[step, agent], aggfunc="first")
    p.columns = [f"{a}__{b}" for a, b in p.columns]
    return p.reset_index()


def _aggregate(w: pd.DataFrame, strategy: str, metric: str, scope: str) -> pd.DataFrame:
    """Collapse to one row per (with_gt, key, triple) with the summed-across-backbones
    objective. For dataset scope, macro-average the subsets FIRST (matching how the
    manuscript's CE column is formed), then sum across backbones."""
    step, agent = f"step_acc_{metric}__{strategy}", f"agent_acc_{metric}__{strategy}"
    svd = f"step_acc_{metric}__svd"
    w = w.copy()
    w["key"] = w["subset"] if scope == "subset" else w["dataset"]
    per_model = (w.groupby(["with_gt", "dataset", "key", "triple", "model"], as_index=False)
                   [[step, agent, svd]].mean())          # no-op for subset scope
    per_model["beats_svd"] = per_model[step] > per_model[svd] + 1e-12
    per_model["diff"] = per_model[step] - per_model[svd]
    g = per_model.groupby(["with_gt", "dataset", "key", "triple"], as_index=False).agg(
        score=(step, "sum"), agent_score=(agent, "sum"),
        diff_sum=("diff", "sum"), n_beat=("beats_svd", "sum"), n_models=(step, "size"))
    for c in ("score", "agent_score", "diff_sum"):
        g[c] = g[c].round(TIE_DP)
    return g


# ── rules: each returns the frame sorted best-first ─────────────────────────
def rule_sum(g: pd.DataFrame) -> pd.DataFrame:
    """Current protocol: argmax of the strategy's step accuracy summed across backbones;
    tiebreak the agent sum, then the earliest triple."""
    return g.sort_values(["score", "agent_score", "triple"],
                         ascending=[False, False, True], kind="mergesort")


def rule_sum_diff(g: pd.DataFrame) -> pd.DataFrame:
    """A triple where the rescoring beats SVD (proj) on EVERY backbone always outranks one
    where it does not. Within each group, `rule_sum` decides.

    The first key is a BOOLEAN, not the count of backbones beaten. A count would make a
    triple that lifts one backbone outrank a higher-scoring triple that lifts neither,
    which is not what "beats the base scorer" means. Binary keeps the promise exact:
    lifted-everywhere first, then plain accuracy inside each group.

    No filtering, so the rule degrades gracefully. If no triple lifts every backbone, the
    first key is False everywhere and this reduces to `rule_sum`."""
    g = g.assign(beats_all=g["n_beat"] == g["n_models"])
    return g.sort_values(["beats_all", "score", "agent_score", "triple"],
                         ascending=[False, False, False, True], kind="mergesort")


RULES = {"sum": rule_sum, "sum-diff": rule_sum_diff, "val": rule_sum}
RULE_METRIC = {"val": "val"}          # `val` is rule_sum on the validation columns


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selections", default=str(SWEEP / "selections_all.tsv"))
    p.add_argument("--rule", default="sum", choices=sorted(RULES))
    p.add_argument("--metric", default=None, choices=["test", "val"],
                   help="default: val for --rule val, else test")
    p.add_argument("--strategy", default="backprop",
                   choices=["backprop", "succ-strong", "succ-near"])
    p.add_argument("--scope", default="auto", choices=["auto", "subset", "dataset"])
    p.add_argument("--restrict-to", default=None,
                   help="limit triples to a seed range, e.g. 1-20 (for cross-checks)")
    p.add_argument("--top", type=int, default=3, help="runners-up to print")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    metric = args.metric or RULE_METRIC.get(args.rule, "test")
    df = pd.read_csv(args.selections, sep="\t")
    if args.restrict_to:
        lo, hi = (int(x) for x in args.restrict_to.split("-"))
        df = df[(df["triple"] >= lo) & (df["triple"] + 2 <= hi)]
    n_triples = df["triple"].nunique()

    wide = _wide(df, args.strategy, metric)
    rows, blocks = [], {}
    for dataset in sorted(wide["dataset"].unique()):
        scope = args.scope if args.scope != "auto" else (
            "dataset" if dataset in PER_DATASET_SCOPE else "subset")
        g = _aggregate(wide[wide["dataset"] == dataset], args.strategy, metric, scope)
        for with_gt in sorted(g["with_gt"].unique()):
            for key in sorted(g[g["with_gt"] == with_gt]["key"].unique()):
                cell = g[(g["with_gt"] == with_gt) & (g["key"] == key)]
                ranked = RULES[args.rule](cell)
                best = ranked.iloc[0]
                runner = ranked.iloc[1] if len(ranked) > 1 else None
                rows.append({
                    "with_gt": with_gt, "dataset": dataset, "scope": scope, "key": key,
                    "rule": args.rule, "metric": metric, "strategy": args.strategy,
                    "triple": int(best["triple"]),
                    "seeds": ",".join(map(str, _seeds(int(best["triple"])))),
                    "score": best["score"], "agent_score": best["agent_score"],
                    "diff_sum": best["diff_sum"], "n_beat": int(best["n_beat"]),
                    "n_models": int(best["n_models"]), "n_triples": len(ranked),
                    "n_beats_all": int((cell["n_beat"] == cell["n_models"]).sum()),
                    "margin": (best["score"] - runner["score"]) if runner is not None else None,
                    "runners_up": ",".join(str(int(t)) for t in
                                           ranked["triple"].iloc[1:1 + args.top]),
                })
                blocks.setdefault((dataset, with_gt), []).append(
                    (key, scope, _seeds(int(best["triple"]))))

    out = pd.DataFrame(rows)
    dest = Path(args.out or SWEEP / f"best_triples_{args.rule}_{metric}_{args.strategy}.tsv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, sep="\t", index=False)

    print(f"rule={args.rule} metric={metric} strategy={args.strategy} "
          f"triples={n_triples}\n")
    show = ["with_gt", "dataset", "key", "triple", "seeds", "score", "diff_sum",
            "n_beat", "margin", "runners_up"]
    print(out[show].to_string(index=False))
    print(f"\nwrote {dest}")

    print("\n" + "=" * 72)
    print("ready-to-paste seeds: blocks")
    import yaml
    for (dataset, with_gt), items in sorted(blocks.items()):
        cfg = yaml.safe_load((REPO / "configs-main" / f"{dataset}.yaml").read_text())
        print(f"\n# configs-main/{dataset}.yaml   "
              f"({'with-GT' if with_gt else 'without-GT'}, rule={args.rule}, "
              f"metric={metric}, strategy={args.strategy})")
        print("seeds:")
        by_key = {k: s for k, _, s in items}
        for sub in cfg["subsets"]:
            seeds = by_key.get(sub) or by_key.get(dataset)
            print(f"  {sub + ':':22}{seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
