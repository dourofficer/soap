"""Check the prompting evaluation before you trust its numbers.

Four checks. Each one guards a mistake that would produce wrong numbers in silence.

  V1  The test split is SOAP's split.
  V2  We read the right GT folder. This is the easiest thing to get backwards, because
      `../attrib-prompting/outputs/` is the WITH-GT run while soap's `outputs/` is the
      without-GT one.
  V3  Every cell holds the whole corpus, and every split holds half of it.
  V4  How far apart the two agent rules are. We report the exact rule. This prints the
      substring rule beside it once, so the size of that choice is known.

    python scripts/prompting/verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "prompting"))

from evaluate import (COLUMNS, DATASETS, JUDGES, METHODS, SETTINGS, SPLIT_MODEL,  # noqa: E402
                      SRC, read_cell, test_ids)
from main.config import load_config, seeds_for                                    # noqa: E402
from main.stores import list_rep_files, split_files                               # noqa: E402

CORPUS = {("ww", "algorithm-generated"): 126, ("ww", "hand-crafted"): 58,
          ("traceelephant", "captain"): 85, ("traceelephant", "magentic"): 91,
          ("correct-error", "arc"): 304, ("correct-error", "gaia"): 50,
          ("correct-error", "hotpot"): 578, ("correct-error", "math500"): 157,
          ("correct-error", "mmlu_pro"): 92, ("correct-error", "musique"): 312,
          ("correct-error", "wikimqa"): 733}


def v1_split_matches_soap() -> list[str]:
    """The ids we score must be the ids main/ evaluates, in the same order."""
    bad = []
    for _root, _gt, tree, suffix in SETTINGS:
        for ds in DATASETS:
            cfg = load_config(REPO / "configs-main" / f"{ds}{suffix}.yaml")
            for subset in cfg["subsets"]:
                for seed in seeds_for(cfg, subset):
                    ours = test_ids(ds, subset, tree, cfg["splits"], seed)
                    rep_dir = REPO / tree / ds / "activations" / SPLIT_MODEL / subset
                    theirs = [Path(f).stem for f in
                              split_files(list_rep_files(rep_dir), cfg["splits"], seed)["test"]]
                    if ours != theirs:
                        bad.append(f"{tree}/{ds}/{subset} seed {seed}")
    return bad


def v2_gt_mapping() -> tuple[list[str], list[str]]:
    """Two independent checks that we did not swap the GT folders."""
    flag_bad, same_bad = [], []
    for ds in DATASETS:
        cfg = load_config(REPO / "configs-main" / f"{ds}.yaml")
        for subset in cfg["subsets"]:
            for judge in JUDGES:
                for method in METHODS:
                    cells = {}
                    for root, with_gt, _tree, _sfx in SETTINGS:
                        preds = read_cell(root, ds, subset, judge, method)
                        cells[with_gt] = preds
                        # where the flag exists it must agree with the folder
                        for row in preds.values():
                            if "gt_in_prompt" in row and row["gt_in_prompt"] != with_gt:
                                flag_bad.append(f"{root}/{ds}/{subset}/{judge}/{method}")
                                break
                    a, b = cells.get(False, {}), cells.get(True, {})
                    if not a or not b:
                        continue
                    diff = sum(1 for i in a if i in b
                               and a[i].get("predicted_step") != b[i].get("predicted_step"))
                    if diff == 0:
                        same_bad.append(f"{ds}/{subset}/{judge}/{method}")
    return flag_bad, same_bad


def v3_counts() -> list[str]:
    bad = []
    for root, _gt, tree, suffix in SETTINGS:
        for ds in DATASETS:
            cfg = load_config(REPO / "configs-main" / f"{ds}{suffix}.yaml")
            for subset in cfg["subsets"]:
                want = CORPUS[(ds, subset)]
                for judge in JUDGES:
                    for method in METHODS:
                        n = len(read_cell(root, ds, subset, judge, method))
                        if n and n != want:
                            bad.append(f"{root}/{ds}/{subset}/{judge}/{method}: {n} != {want}")
                seed = seeds_for(cfg, subset)[0]
                n_test = len(test_ids(ds, subset, tree, cfg["splits"], seed))
                # split_data cuts at int(len*ratio), so for an odd corpus the TEST half
                # takes the extra trajectory. Do not write want // 2 here.
                sp = cfg["splits"]
                expect = want - int(want * (sp["train"] + sp["val"]))
                if n_test != expect:
                    bad.append(f"{tree}/{ds}/{subset}: test {n_test} != {expect}")
    return bad


def v4_agent_rules(cell_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cell_path, sep="\t")
    df["gap"] = df["agent_acc_substring"] - df["agent_acc"]
    return df.nlargest(6, "gap")[["with_gt", "judge", "method", "dataset", "subset",
                                  "agent_acc", "agent_acc_substring", "gap"]]


def main() -> int:
    fails = 0

    bad = v1_split_matches_soap()
    print(f"V1  test split == SOAP's split ....... {'OK' if not bad else f'{len(bad)} BAD'}")
    for x in bad[:5]:
        print("      -", x)
    fails += len(bad)

    flag_bad, same_bad = v2_gt_mapping()
    print(f"V2a gt_in_prompt agrees with folder .. {'OK' if not flag_bad else f'{len(flag_bad)} BAD'}")
    for x in flag_bad[:5]:
        print("      -", x)
    print(f"V2b the two settings disagree ........ {'OK' if not same_bad else f'{len(same_bad)} identical'}")
    for x in same_bad[:5]:
        print("      -", x)
    fails += len(flag_bad) + len(same_bad)

    bad = v3_counts()
    print(f"V3  corpus and split sizes ........... {'OK' if not bad else f'{len(bad)} BAD'}")
    for x in bad[:5]:
        print("      -", x)
    fails += len(bad)

    cell = REPO / "results-prompting" / "by_cell.tsv"
    if cell.exists():
        print("\nV4  exact vs substring agent rule, the 6 widest gaps:")
        print(v4_agent_rules(cell).to_string(index=False))
    else:
        print("\nV4  skipped: run evaluate.py first")

    print(f"\n{'ALL CHECKS PASS' if not fails else f'{fails} PROBLEM(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
