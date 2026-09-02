"""E2 / tab:synth — synthetic reference trajectories, main-experiment protocol.

The experiment is IDENTICAL to the main one — same frozen triples, same standard
val/test partitions, same test-selection rule — except the fit set: R is fit on a
synthetic corpus instead of the seed's train split, which goes unused. One corpus
per (target, generator), shared across the triple's seeds, so per-seed variance
comes from the val/test partitions alone.

References per target (staged by e2_stage_data.py, extracted through
configs-main/synthetic.yaml into results-nogt/synthetic/activations/):

    real        the seed's own train split — exactly Table 1's selection problem,
                so its base and SOAP rows must reproduce selection.tsv (asserted)
    syn-qwen9b  data/synthetic/{ag,hc}-qwen9b   (generator Qwen3.5-9B)
    syn-gpt4o   data/synthetic/{ag,hc}-gpt4o    (generator GPT-4o)

For every reference the full config is RE-SELECTED by the standard rule — dense
base grid (position x band), then the backprop rescore grid on the winning base
config, argmax of mean TEST step accuracy over the triple, tiebreak agent
accuracy — so each row is "the best that reference corpus can do", matching the
optimistic protocol of the real-corpus row. Val metrics ride along in the TSV for
the later val-selection conversion, as do the question-overlap counts between the
fit corpus and each seed's val/test partition (the overlap is by design: the
synthetic runs cover WW's own question pool).

Dependency weights always come from the target trajectories' own attention; the
synthetic corpora need activations only.

    python scripts/ablations/e2_synthfit.py [--models qwen3.5-9b] [--targets WW-AG]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, REPO, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    cell_paths, iter_cells, load_selection, select_config)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W      # noqa: E402
from main.score import (ENSEMBLE_POSITION, band_bounds, base_positions,  # noqa: E402
                        ens_score_steps, fit_svd, member_positions, score_steps)
from main.stores import (list_rep_files, load_representations,        # noqa: E402
                         split_files)
from main.sweep import BASE_SWEPT, RESCORE_SWEPT                      # noqa: E402

OUT = RESULTS_DIR / "e2_synthfit.tsv"
SHORT = {"algorithm-generated": "WW-AG", "hand-crafted": "WW-HC"}
SYN_SUBSETS = {"WW-AG": {"syn-gpt4o": "ag-gpt4o", "syn-qwen9b": "ag-qwen9b"},
               "WW-HC": {"syn-gpt4o": "hc-gpt4o", "syn-qwen9b": "hc-qwen9b"}}
REF_ORDER = ["real", "syn-qwen9b", "syn-gpt4o"]
METRIC_COLS = ["step_acc_val@1", "agent_acc_val@1", "step_acc_test@1", "agent_acc_test@1"]


def questions_of(data_dir: Path, json_names) -> set[str]:
    return {json.loads((Path(data_dir) / n).read_text())["question"].strip()
            for n in json_names}


class Reference:
    """One fit set: the real per-seed train split, or a static synthetic corpus.

    A static reference loads its store and fits its SVDs ONCE and reuses them for
    every seed — the corpus is shared across the triple by design.
    """

    def __init__(self, name, rep_dir, data_dir, device, static, files=None):
        self.name = name
        self.rep_dir, self.data_dir = Path(rep_dir), Path(data_dir)
        self.device, self.static = device, static
        self.files = files if files is not None else list_rep_files(rep_dir)
        self.n_traj = len(self.files)
        self.questions = questions_of(
            self.data_dir, [Path(f).with_suffix("").name + ".json" for f in self.files])
        self._store = None
        self.fits: dict[str, torch.Tensor] = {}   # position -> V, static refs only

    def train(self, seed_files=None):
        """The fit-set store; ``seed_files`` only applies to the non-static real ref."""
        if self.static:
            if self._store is None:
                self._store = load_representations(
                    self.rep_dir, self.data_dir, poolings=[POOLING],
                    weight_names="all", files=self.files, device=self.device)
            return self._store
        return load_representations(self.rep_dir, self.data_dir, poolings=[POOLING],
                                    weight_names="all", files=seed_files,
                                    device=self.device)


def _metrics_row(vm, tm, j) -> dict:
    return {"step_acc_val@1": float(vm["step@1"][j]), "agent_acc_val@1": float(vm["agent@1"][j]),
            "step_acc_test@1": float(tm["step@1"][j]), "agent_acc_test@1": float(tm["agent@1"][j])}


def _band_scores(cfg, position, bands, fits, train, split, members):
    """Base scores of one split for every band of one position, sweep-identical."""
    n_comp = cfg["n_components"]
    if position == ENSEMBLE_POSITION:
        for p in members:
            fits.setdefault(p, fit_svd(train.stores[(POOLING, p)].R, n_comp))
        tr = {p: train.stores[(POOLING, p)].R for p in members}
        ev = {p: split.stores[(POOLING, p)].R for p in members}
        return [ens_score_steps(cb, ce, members, fits, tr, ev) for cb, ce in bands]
    V = fits.setdefault(position, fit_svd(train.stores[(POOLING, position)].R, n_comp))
    return [score_steps(split.stores[(POOLING, position)].R, V, cb, ce)
            for cb, ce in bands]


def load_target_split(cell, files, device):
    return load_representations(cell["rep_dir"], cell["data_dir"], poolings=[POOLING],
                                weight_names="all", files=files, device=device)


def base_grid(cell, ref: Reference, device) -> pd.DataFrame:
    """Dense (position x band) grid: R fit on ``ref``, scored on the standard
    val/test partitions of the frozen triple. ``seed`` is the triple index 0..2."""
    cfg = cell["cfg"]
    bands = band_bounds(cfg["n_components"])
    rows = []
    for i, seed in enumerate(cell["seeds"]):
        parts = split_files(cell["files"], cfg["splits"], seed)
        train = ref.train(parts["train"])
        val = load_target_split(cell, parts["val"], device)
        test = load_target_split(cell, parts["test"], device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        available = train.positions()
        assert available == val.positions(), "reference/target position sets differ"
        positions = base_positions(available, cfg.get("positions", "all"),
                                   cfg.get("ensemble", True))
        members = member_positions(available)
        fits = ref.fits if ref.static else {}
        for position in tqdm(positions, desc=f"{cell['name']} {ref.name} i{i}",
                             leave=False):
            vs = _band_scores(cfg, position, bands, fits, train, val, members)
            ts = _band_scores(cfg, position, bands, fits, train, test, members)
            vm = compute_metrics_batch(torch.stack(vs), None, [1], ctx=val_ctx)
            tm = compute_metrics_batch(torch.stack(ts), None, [1], ctx=test_ctx)
            for j, (cb, ce) in enumerate(bands):
                rows.append({"seed": i, "position": position, "c_begin": cb, "c_end": ce,
                             **_metrics_row(vm, tm, j)})
        del val, test
        if not ref.static:
            del train
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def rescore_grid(cell, ref: Reference, base_cfg, weightings, labels, device) -> pd.DataFrame:
    """Backprop rescore grid (layer_range x gamma x w) on one base config."""
    cfg = cell["cfg"]
    bands = [(int(base_cfg["c_begin"]), int(base_cfg["c_end"]))]
    position = base_cfg["position"]
    gammas, ws = list(cfg["gammas"]), list(cfg["ws"])
    rows = []
    for i, seed in enumerate(cell["seeds"]):
        parts = split_files(cell["files"], cfg["splits"], seed)
        train = ref.train(parts["train"])
        val = load_target_split(cell, parts["val"], device)
        test = load_target_split(cell, parts["test"], device)
        val_ctx, test_ctx = KeeperContext(val.keeper), KeeperContext(test.keeper)
        members = member_positions(train.positions())
        fits = ref.fits if ref.static else {}
        s_val = _band_scores(cfg, position, bands, fits, train, val, members)[0]
        s_test = _band_scores(cfg, position, bands, fits, train, test, members)[0]
        for r_idx, label in enumerate(labels):
            for w in ws:
                vmats = {"backprop": build_W(val.keeper, weightings[r_idx], w, device)}
                tmats = {"backprop": build_W(test.keeper, weightings[r_idx], w, device)}
                Sv = apply_strategy(s_val, val.keeper, vmats, "backprop", gammas).T.contiguous()
                St = apply_strategy(s_test, test.keeper, tmats, "backprop", gammas).T.contiguous()
                vm = compute_metrics_batch(Sv, None, [1], ctx=val_ctx)
                tm = compute_metrics_batch(St, None, [1], ctx=test_ctx)
                for gi, gamma in enumerate(gammas):
                    rows.append({"seed": i, "layer_range": label, "gamma": gamma,
                                 "w": str(w), **_metrics_row(vm, tm, gi)})
        del val, test
        if not ref.static:
            del train
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def config_metrics(df: pd.DataFrame, config: dict, n_seeds: int) -> dict:
    g = df
    for ax, v in config.items():
        g = g[g[ax].astype(str) == str(v)]
    assert len(g) == n_seeds, f"config {config} has {len(g)} rows, expected {n_seeds}"
    return {c: float(g[c].mean()) for c in METRIC_COLS}


def _json_names(files) -> list[str]:
    return [Path(f).with_suffix("").name + ".json" for f in files]


def overlap_counts(cell, ref: Reference) -> dict:
    """Mean count (over the triple) of val/test trajectories whose question also
    appears in the fit set — recorded because the overlap is by design. For the
    real reference the fit set is the seed's train split (WW questions are unique
    per trajectory, so these counts are 0 — asserted implicitly by the numbers)."""
    out = {"val": [], "test": []}
    sizes = {}
    for seed in cell["seeds"]:
        parts = split_files(cell["files"], cell["cfg"]["splits"], seed)
        fit_qs = (ref.questions if ref.static
                  else questions_of(cell["data_dir"], _json_names(parts["train"])))
        for split in ("val", "test"):
            qs = questions_of(cell["data_dir"], _json_names(parts[split]))
            out[split].append(len(qs & fit_qs))
            sizes[split] = len(parts[split])
    return {"q_overlap_val": sum(out["val"]) / len(out["val"]),
            "q_overlap_test": sum(out["test"]) / len(out["test"]),
            "n_val": sizes["val"], "n_test": sizes["test"]}


def out_row(model, cell, ref, row, base_cfg, resc_cfg, m) -> dict:
    resc_cfg = resc_cfg or {}
    return {"model": model, "target": cell["name"], "reference": ref.name,
            "row": row, "corpus_n": ref.n_traj,
            "position": base_cfg["position"], "c_begin": int(base_cfg["c_begin"]),
            "c_end": int(base_cfg["c_end"]),
            "layer_range": resc_cfg.get("layer_range", ""),
            "gamma": float(resc_cfg.get("gamma", 0.0)),
            "w": resc_cfg.get("w", ""),
            "seeds": ",".join(map(str, cell["seeds"])),
            "step_acc_test": m["step_acc_test@1"], "agent_acc_test": m["agent_acc_test@1"],
            "step_acc_val": m["step_acc_val@1"], "agent_acc_val": m["agent_acc_val@1"],
            **overlap_counts(cell, ref)}


def run_cell(cell, model, syn_cfg, device, ref_names=REF_ORDER) -> list[dict]:
    n_seeds = len(cell["seeds"])
    weightings, bounds = aggregate_attn(C.attn_root(cell["cfg"]), model, cell["subset"],
                                        n_ranges=cell["cfg"]["n_ranges"], device=device)
    labels = [f"{lo}-{hi}" for lo, hi in bounds]
    refs = {}
    if "real" in ref_names:
        refs["real"] = Reference("real", cell["rep_dir"], cell["data_dir"], device,
                                 static=False, files=cell["files"])
        # The real ref's n_traj is the per-seed train-split size, not the corpus.
        refs["real"].n_traj = len(split_files(cell["files"], cell["cfg"]["splits"],
                                              cell["seeds"][0])["train"])
    for ref_name, syn_subset in SYN_SUBSETS[cell["name"]].items():
        if ref_name in ref_names:
            refs[ref_name] = Reference(
                ref_name, C.reps_root(syn_cfg) / model / syn_subset,
                C.data_root(syn_cfg) / syn_subset, device, static=True)

    rows = []
    for ref_name in [r for r in REF_ORDER if r in refs]:
        ref = refs[ref_name]
        base_df = base_grid(cell, ref, device)
        base_sel = select_config(base_df, BASE_SWEPT, list(range(n_seeds)),
                                 "step_acc_test@1", "agent_acc_test@1")
        bcfg = base_sel["config"]
        resc_df = rescore_grid(cell, ref, bcfg, weightings, labels, device)
        soap_sel = select_config(resc_df, RESCORE_SWEPT, list(range(n_seeds)),
                                 "step_acc_test@1", "agent_acc_test@1")
        bm = config_metrics(base_df, bcfg, n_seeds)
        sm = config_metrics(resc_df, soap_sel["config"], n_seeds)
        rows.append(out_row(model, cell, ref, "base", bcfg, None, bm))
        rows.append(out_row(model, cell, ref, "soap", bcfg, soap_sel["config"], sm))
        print(f"  [{ref_name}] base={bcfg} {bm['step_acc_test@1']:.4f} "
              f"soap={soap_sel['config']} {sm['step_acc_test@1']:.4f}")

        # The real reference repeats Table 1's selection problem exactly.
        if ref_name == "real":
            assert_close(bm["step_acc_test@1"], float(cell["svd"]["step_acc_test"]),
                         f"{model} {cell['name']} real base vs Table 1")
            assert_close(sm["step_acc_test@1"], float(cell["bp"]["step_acc_test"]),
                         f"{model} {cell['name']} real soap vs Table 1")
            print(f"  real reference verified against the selection table")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--targets", nargs="+", default=None, help="WW-AG WW-HC; default both")
    p.add_argument("--refs", nargs="+", default=list(REF_ORDER),
                   help="real syn-qwen9b syn-gpt4o; default all three")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    syn_cfg = C.load_config(REPO / "configs-main/synthetic.yaml")
    rows = []
    for cfg, model, subset in iter_cells(["configs-main/ww.yaml"]):
        if args.models and model not in args.models:
            continue
        name = SHORT[subset]
        if args.targets and name not in args.targets:
            continue
        svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        cell = {"cfg": cfg, "subset": subset, "name": name, "svd": svd_row,
                "bp": bp_row, "rep_dir": rep_dir, "data_dir": data_dir,
                "files": files, "seeds": C.seeds_for(cfg, subset)}
        print(f"[{model}] {name}")
        rows.extend(run_cell(cell, model, syn_cfg, args.device, args.refs))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    for (model, row), g in df.groupby(["model", "row"]):
        pivot = g.pivot_table(index="reference", columns="target",
                              values="step_acc_test") * 100
        pivot = pivot.reindex([r for r in REF_ORDER if r in pivot.index])
        print(f"\n=== {model}, {row} (step acc %) ===")
        print(pivot.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
