"""Build ``results_synthfit_paper.tsv`` — the paper-protocol synthetic-reference table.

Mirrors the manuscript's tab:synth: per backbone and scorer, three reference-corpus
rows — Real (train split), Synthetic (Qwen3.5-9B), Synthetic (Qwen3.5-35B-A3B) — with,
per subset, the base scorer **S** and the rescored **+RECAP** (RECAP = the backprop
strategy; the discount/CRR reductions exist on disk but are not a table row). Cells are
``paper.convention``-selected (file suffix) and always report the TEST metric, meaned
over ``paper_seeds`` — so under the defaults (test convention, main-top3 seeds) the
Real row equals the main ``results_extended.tsv`` cells, which ``src.xfit.verify``
checks mechanically at per-seed granularity.

Unlike the legacy ceiling table, this protocol is leakage-controlled: the fit corpus
carries the train split's own questions (plus a seeded same-pool fill that excludes
every question of the target subset), so no val/test question ever reaches the fit.

    python -m src.xfit.table_paper --set setting=paper [--dataset ww]
"""
from __future__ import annotations

import csv

from ..common import paths
from ..reports.baselines import MODEL_DISPLAY, SUBSET_DISPLAY, fmt as _fmt
from .legacy import _read_tsv, _by_seed, _mean_over
from . import prov
from .common import (load_config, source_tag, target_cfg, setting, paper_cfg,
                     paper_targets, paper_seeds, REAL_SOURCE)

CORPORA = [(REAL_SOURCE, "Real (train split)"),
           ("q9", "Synthetic (Qwen3.5-9B)"),
           ("q35", "Synthetic (Qwen3.5-35B-A3B)")]

CAVEAT = [
    ["# SYNTHETIC REFERENCE FIT (paper protocol) — results_synthfit_paper.tsv"],
    ["# Standard 325 partition: real val (20%) selects, real test (50%) reports — "
     "byte-identical to the main tables. The real train split is discarded; the fit "
     "corpus is a per-seed synthetic subsample matched to the train split's SIZE and "
     "QUESTIONS (missing questions filled by a seeded same-pool draw that excludes "
     "every question of the target subset — leakage-controlled, unlike the legacy "
     "ceiling table)."],
    ["# Real (train split) re-runs the real train fit through the same code path and "
     "must equal the main results_extended cells (checked by src.xfit.verify)."],
]


def _corpus_tag(cfg, harness, corpus) -> str | None:
    if corpus == REAL_SOURCE:
        return source_tag(REAL_SOURCE, cfg)
    source = cfg["sources"].get(harness, {}).get(corpus)
    return source_tag(source, cfg) if source else None


def build_dataset(cfg, dataset: str):
    pc = paper_cfg(cfg)
    conv = pc["convention"]
    seeds = paper_seeds(cfg, dataset)
    ds_targets = [(h, sub) for h, d, sub in paper_targets(cfg) if d == dataset]
    subsets = [(disp, sk) for disp, sk in SUBSET_DISPLAY[dataset]
               if any(sub == sk for _, sub in ds_targets)]
    harness_of = {sub: h for h, sub in ds_targets}

    header_a = ["Backbone", "Scorer", "Reference corpus"]
    header_b = ["", "", ""]
    for disp, _ in subsets:
        header_a += [disp, "", "", ""]
        header_b += ["S step", "S agent", "+RECAP step", "+RECAP agent"]
    lines = list(CAVEAT)
    lines += [[f"# convention={conv} seeds={seeds} scorers={pc['table_scorers']}"],
              header_a, header_b]

    for disp_model, mk in MODEL_DISPLAY:
        first_model = True
        for scorer in pc["table_scorers"]:
            for corpus, corpus_disp in CORPORA:
                cells = []
                for _, sk in subsets:
                    tag = _corpus_tag(cfg, harness_of[sk], corpus)
                    root = (paths.reduced_root(target_cfg(dataset, sk, cfg, split_tag=tag))
                            if tag else None)
                    if root is None:
                        cells += ["", "", "", ""]
                        continue
                    base = _by_seed(_read_tsv(root / mk / sk / f"base_ext_{scorer}_{conv}.tsv") or [])
                    bp = _by_seed(_read_tsv(root / mk / sk / f"backprop_ext_{scorer}_{conv}.tsv") or [])
                    cells += [_fmt(_mean_over(base, seeds, "step_acc_test")),
                              _fmt(_mean_over(base, seeds, "agent_acc_test")),
                              _fmt(_mean_over(bp, seeds, "disc_step_acc_test")),
                              _fmt(_mean_over(bp, seeds, "disc_agent_acc_test"))]
                lines.append([disp_model if first_model else "",
                              scorer if corpus == REAL_SOURCE else "",
                              corpus_disp] + cells)
                first_model = False

    out = paths.tables_root(target_cfg(dataset, subsets[0][1], cfg)) / "results_synthfit_paper.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(lines)
    print(f"[synthfit-paper] {dataset} conv={conv} seeds={seeds} -> {out}")
    return out


def run(cfg, only_dataset=None) -> None:
    if setting(cfg) != "paper":
        raise SystemExit("table_paper is a paper-setting stage (--set setting=paper)")
    datasets = list(dict.fromkeys(d for _, d, _s in paper_targets(cfg)))
    outs = []
    for ds in datasets:
        if only_dataset in (None, ds):
            outs.append(build_dataset(cfg, ds))
    for ds, out in zip([d for d in datasets if only_dataset in (None, d)], outs):
        subset = next(s for _, d, s in paper_targets(cfg) if d == ds)
        prov.record(cfg, target_cfg(ds, subset, cfg), "table", [out])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    run(load_config(args.overrides), only_dataset=args.dataset)


if __name__ == "__main__":
    main()
