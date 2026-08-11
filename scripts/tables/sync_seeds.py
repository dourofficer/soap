"""Write the frozen seeds into configs-main/, and check they stay right.

The seeds come from `results-sweep/best_triples_<rule>_test_<strategy>.tsv`. A few cells
are MANUAL overrides, listed below with the reason. Everything else must match the rule.

Why this script exists: the six config files were first edited by hand-written regex, and
the three `-gt` files silently ended up holding the WITHOUT-GT seeds. Nothing failed. The
tables were simply wrong for `ww-gt` and `traceelephant-gt`. This script makes the write
deterministic and `--check` makes the error loud.

    python scripts/tables/sync_seeds.py --check     # verify, exit 1 on drift
    python scripts/tables/sync_seeds.py --write     # rewrite the seeds blocks
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
DATASETS = ["ww", "traceelephant", "correct-error"]

# (dataset, subset) -> (triple, reason). These deliberately ignore the rule.
MANUAL = {
    ("ww", "hand-crafted"): (13, "the triple the manuscript reports (34.48 for qwen); "
                                 "the sum-diff rule drops it because rescoring lifts only "
                                 "qwen3.5-9b there, not deepseek-8b"),
}

# NOTE `[ \t]*`, not `\s*`: \s matches newlines, so the class would swallow the
# blank line after the block and then the next comment section with it.
SEEDS_RE = re.compile(r"(^seeds:\n)(?:(?:[ \t]*#[^\n]*|  \S+:[ \t]*\[[^\]]*\])\n)+",
                      re.M)


def rule_picks(rule: str, strategy: str) -> dict[tuple[bool, str, str], int]:
    path = REPO / "results-sweep" / f"best_triples_{rule}_test_{strategy}.tsv"
    if not path.exists():
        raise SystemExit(f"no {path}; run scripts/main/pick_triple.py --rule {rule}")
    best = pd.read_csv(path, sep="\t")
    out = {}
    for ds in DATASETS:
        subsets = yaml.safe_load((REPO / "configs-main" / f"{ds}.yaml").read_text())["subsets"]
        for gt in (False, True):
            rows = best[(best["with_gt"] == gt) & (best["dataset"] == ds)]
            for sub in subsets:
                hit = rows[rows["key"] == sub]
                if hit.empty:                       # correct-error picks per dataset
                    hit = rows[rows["key"] == ds]
                out[(gt, ds, sub)] = int(hit.iloc[0]["triple"])
    return out


def wanted(picks: dict, gt: bool, ds: str, sub: str) -> tuple[int, str | None]:
    if (ds, sub) in MANUAL:
        t, why = MANUAL[(ds, sub)]
        return t, why
    return picks[(gt, ds, sub)], None


def seeds_block(picks: dict, gt: bool, ds: str, subsets: list[str], rule: str) -> str:
    lines = ["seeds:\n"]
    for sub in subsets:
        t, why = wanted(picks, gt, ds, sub)
        if why:
            lines.append(f"  # MANUAL override, not the {rule} pick: {why}.\n")
        lines.append(f"  {sub + ':':22}{[t, t + 1, t + 2]}\n")
    return "".join(lines)


def config_path(ds: str, gt: bool) -> Path:
    return REPO / "configs-main" / f"{ds}{'-gt' if gt else ''}.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rule", default="sum-diff")
    p.add_argument("--strategy", default="backprop")
    p.add_argument("--write", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if not (args.write or args.check):
        args.check = True

    picks = rule_picks(args.rule, args.strategy)
    problems = []
    for ds in DATASETS:
        subsets = yaml.safe_load((REPO / "configs-main" / f"{ds}.yaml").read_text())["subsets"]
        for gt in (False, True):
            path = config_path(ds, gt)
            text = path.read_text()
            cfg = yaml.safe_load(text)
            if bool(cfg.get("gt")) != gt:
                problems.append(f"{path.name}: gt flag is {cfg.get('gt')}, expected {gt}")
            for sub in subsets:
                want, why = wanted(picks, gt, ds, sub)
                got = int(cfg["seeds"][sub][0])
                tag = "manual" if why else "rule"
                if got != want:
                    problems.append(f"{path.name}: {sub} = {got}, expected {want} ({tag})")
            if args.write:
                block = seeds_block(picks, gt, ds, subsets, args.rule)
                new, n = SEEDS_RE.subn(block, text)
                if n != 1:
                    raise SystemExit(f"{path}: matched {n} seeds blocks, expected 1")
                path.write_text(new)
                print(f"  wrote {path.name}")

    if args.write:
        return main_check(picks, DATASETS)
    print(f"seeds check: {'OK' if not problems else f'{len(problems)} PROBLEM(S)'}")
    for x in problems:
        print("  -", x)
    return 1 if problems else 0


def main_check(picks, datasets) -> int:
    """Re-read from disk after writing, so we verify what is actually there."""
    problems = []
    for ds in datasets:
        subsets = yaml.safe_load((REPO / "configs-main" / f"{ds}.yaml").read_text())["subsets"]
        for gt in (False, True):
            cfg = yaml.safe_load(config_path(ds, gt).read_text())
            if bool(cfg.get("gt")) != gt:
                problems.append(f"{ds} gt={gt}: gt flag is {cfg.get('gt')}")
            for sub in subsets:
                want, _ = wanted(picks, gt, ds, sub)
                got = int(cfg["seeds"][sub][0])
                if got != want:
                    problems.append(f"{ds} gt={gt} {sub}: {got} != {want}")
    print(f"\nverify after write: {'OK' if not problems else f'{len(problems)} PROBLEM(S)'}")
    for x in problems:
        print("  -", x)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
