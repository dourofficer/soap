"""E4 — does the content of the reference set matter?

SOAP fits its singular vectors on failed trajectories from the target benchmark. A
reviewer asked whether that reference captures failure structure or only the backbone's
generic geometry. This runner keeps E2's protocol (standard val/test partitions of the
frozen triple, dependency weights from the target's own attention) and swaps the fit
set:

  real            the seed's train split — Table 1; must reproduce selection.tsv
  syn-qwen9b      E2's synthetic corpora, successes and failures mixed   (WW only)
  syn-gpt4o
  syn-*-fail      the same corpora restricted to `outcome == "fail"`     (WW only)
  wikitext        120 pseudo-trajectories of WikiText-103 paragraphs
                  (scripts/ablations/e4_stage_wikitext.py)
  random          no reference at all: a random orthonormal 20-d basis per position

Each reference is reported twice. `anchor`: Table 1's configuration, frozen — what the
swap costs with nothing re-tuned. `reselect`: the full configuration re-selected for
that reference by the standard rule (dense base grid, then the backprop rescore grid)
— the best that reference can do, the same optimism the real row enjoys.

The random reference draws one basis per position, shared by the triple. Its ensemble
position needs train-split statistics to z-score members; it takes them from the whole
target corpus, labels unread.

    python scripts/ablations/e4_reference_controls.py [--models qwen3.5-9b] [--refs ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BACKBONES, POOLING, REPO, RESULTS_DIR, anchor_rows,  # noqa: E402
                    assert_close, cell_paths, iter_cells, load_selection,
                    select_config)
from e2_synthfit import (METRIC_COLS, Reference, base_grid, config_metrics,  # noqa: E402
                         rescore_grid)
from main import config as C                                          # noqa: E402
from main.rescore import aggregate_attn                               # noqa: E402
from main.stores import list_rep_files                                # noqa: E402
from main.sweep import BASE_SWEPT, RESCORE_SWEPT, rule_cols           # noqa: E402

OUT = RESULTS_DIR / "e4_reference_controls.tsv"
SHORT = {"algorithm-generated": "WW-AG", "hand-crafted": "WW-HC",
         "captain": "TE-Cap", "magentic": "TE-Mag"}
SYN_SUBSETS = {"WW-AG": {"syn-qwen9b": "ag-qwen9b", "syn-gpt4o": "ag-gpt4o"},
               "WW-HC": {"syn-qwen9b": "hc-qwen9b", "syn-gpt4o": "hc-gpt4o"}}
REF_ORDER = ["real", "syn-qwen9b", "syn-qwen9b-fail", "syn-gpt4o", "syn-gpt4o-fail",
             "wikitext", "random"]


class RandomReference(Reference):
    """A reference that fits nothing: every position gets a random orthonormal basis.

    ``draw`` picks the basis: draw 0 is the ``random`` row of the main run, draws 1..k
    the extra bases of the multi-draw study (``--random-draws``).
    """
    draw = 0

    def train(self, seed_files=None):
        store = super().train()
        for (_, position), st in store.stores.items():
            if position not in self.fits:
                g = torch.Generator(device="cpu").manual_seed(
                    zlib.crc32(position.encode()) + self.draw)
                Q, _ = torch.linalg.qr(torch.randn(st.R.shape[1], self.n_comp, generator=g))
                self.fits[position] = Q.to(self.device)
        return store


def failed_files(rep_dir: Path, data_dir: Path) -> list[str]:
    return [f for f in list_rep_files(rep_dir)
            if json.loads((data_dir / (Path(f).stem + ".json")).read_text())["outcome"] == "fail"]


def build_refs(cell, model, syn_cfg, device, wanted) -> dict[str, Reference]:
    refs = {}
    if "real" in wanted:
        refs["real"] = Reference("real", cell["rep_dir"], cell["data_dir"], device,
                                 static=False, files=cell["files"])
    for name, sub in SYN_SUBSETS.get(cell["name"], {}).items():
        rep, data = C.reps_root(syn_cfg) / model / sub, C.data_root(syn_cfg) / sub
        if name in wanted:
            refs[name] = Reference(name, rep, data, device, static=True)
        if f"{name}-fail" in wanted:
            refs[f"{name}-fail"] = Reference(f"{name}-fail", rep, data, device, static=True,
                                             files=failed_files(rep, data))
    if "wikitext" in wanted:
        refs["wikitext"] = Reference("wikitext", C.reps_root(syn_cfg) / model / "wikitext",
                                     C.data_root(syn_cfg) / "wikitext", device, static=True)
    for name in wanted:
        if name == "random" or name.startswith("random-d"):
            ref = RandomReference(name, cell["rep_dir"], cell["data_dir"], device,
                                  static=True, files=cell["files"])
            ref.n_comp = cell["cfg"]["n_components"]
            ref.draw = 0 if name == "random" else int(name[len("random-d"):])
            refs[name] = ref
    return refs


def main(build=build_refs, order=REF_ORDER, out=OUT, extra_args=None) -> int:
    """E5 and E6 reuse this loop with their own references: ``build`` makes them,
    ``order`` names them (``real`` first, so the self-check runs before any control),
    and ``extra_args`` adds the runner's own flags, handed to ``build`` as ``args``."""
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="+", default=BACKBONES)
    p.add_argument("--refs", nargs="+", default=order)
    p.add_argument("--select-rule", default="test", choices=["test", "val"])
    p.add_argument("--out", default=str(out))
    p.add_argument("--targets", nargs="+", default=None, help="e.g. WW-AG WW-HC")
    p.add_argument("--random-draws", type=int, default=0,
                   help="also run random-d1..dK: K more random bases per cell")
    if extra_args:
        extra_args(p)
    args = p.parse_args()
    device = args.device

    syn_cfg = C.load_config(REPO / "configs-main/synthetic.yaml")
    extra = [f"random-d{k}" for k in range(1, args.random_draws + 1)]
    args.refs = list(args.refs) + extra
    order = list(order) + extra
    rows = []
    for cfg, model, subset in iter_cells(overrides=[f"select_rule={args.select_rule}"],
                                         models=args.models):
        if args.targets and SHORT[subset] not in args.targets:
            continue
        step_col, agent_col = rule_cols(cfg)
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        cell = {"cfg": cfg, "subset": subset, "name": SHORT[subset], "rep_dir": rep_dir,
                "data_dir": data_dir, "files": files, "seeds": C.seeds_for(cfg, subset)}
        n = len(cell["seeds"])
        weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                            n_ranges=cfg["n_ranges"], device=device)
        labels = [f"{lo}-{hi}" for lo, hi in bounds]
        anchor_base = {"position": svd_row["position"], "c_begin": int(svd_row["c_begin"]),
                       "c_end": int(svd_row["c_end"])}
        anchor_gamma = float(bp_row["gamma"])
        anchor_resc = ({"layer_range": bp_row["layer_range"], "gamma": anchor_gamma,
                        "w": str(bp_row["w"]).removesuffix(".0")} if anchor_gamma > 0 else None)
        print(f"[{model}] {cell['name']}  anchor base={anchor_base} rescoring={anchor_resc}")

        refs = (build(cell, model, syn_cfg, device, args.refs, args) if extra_args
                else build(cell, model, syn_cfg, device, args.refs))
        for ref_name in [r for r in order if r in refs]:
            ref = refs[ref_name]
            base_df = base_grid(cell, ref, device)
            sel_base = select_config(base_df, BASE_SWEPT, list(range(n)),
                                     step_col, agent_col)["config"]
            grids = {}

            def resc(base_cfg):
                key = tuple(base_cfg[a] for a in BASE_SWEPT)
                if key not in grids:
                    grids[key] = rescore_grid(cell, ref, base_cfg, weightings, labels, device)
                return grids[key]

            out = {}
            out[("anchor", "base")] = (anchor_base, None,
                                       config_metrics(base_df, anchor_base, n))
            out[("anchor", "soap")] = (
                (anchor_base, anchor_resc, config_metrics(resc(anchor_base), anchor_resc, n))
                if anchor_resc else (anchor_base, None, out[("anchor", "base")][2]))
            sel_resc = select_config(resc(sel_base), RESCORE_SWEPT, list(range(n)),
                                     step_col, agent_col)["config"]
            out[("reselect", "base")] = (sel_base, None, config_metrics(base_df, sel_base, n))
            out[("reselect", "soap")] = (sel_base, sel_resc,
                                         config_metrics(resc(sel_base), sel_resc, n))

            if ref_name == "real":
                for mode in ("anchor", "reselect"):
                    assert_close(out[(mode, "base")][2]["step_acc_test@1"],
                                 float(svd_row["step_acc_test"]), f"{model} {subset} real base")
                    assert_close(out[(mode, "soap")][2]["step_acc_test@1"],
                                 float(bp_row["step_acc_test"]), f"{model} {subset} real soap")
                print("  real reference verified against the selection table")

            for (mode, row), (bcfg, rcfg, m) in out.items():
                rcfg = rcfg or {}
                rows.append({"model": model, "target": cell["name"], "reference": ref_name,
                             "config": mode, "row": row, "corpus_n": len(ref.files)
                             if ref.static else len(files) * 3 // 10,
                             "position": bcfg["position"], "c_begin": int(bcfg["c_begin"]),
                             "c_end": int(bcfg["c_end"]),
                             "layer_range": rcfg.get("layer_range", ""),
                             "gamma": float(rcfg.get("gamma", 0.0)), "w": rcfg.get("w", ""),
                             "seeds": ",".join(map(str, cell["seeds"])),
                             **{c.replace("@1", ""): m[c] for c in METRIC_COLS},
                             **getattr(ref, "extra", {})})
            a, r = out[("anchor", "soap")][2], out[("reselect", "soap")][2]
            print(f"  [{ref_name:16s}] anchor soap {a['step_acc_test@1']:.4f}   "
                  f"reselect soap {r['step_acc_test@1']:.4f}  base={sel_base}")
            pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)   # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")
    for (model, mode, row), g in df.groupby(["model", "config", "row"]):
        pivot = g.pivot_table(index="reference", columns="target",
                              values="step_acc_test") * 100
        print(f"\n=== {model}, {mode}, {row} (step acc %) ===")
        print(pivot.reindex([r for r in order if r in pivot.index]).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
