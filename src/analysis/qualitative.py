"""Qualitative examples: where the base scorer picks a downstream consequence and SOAP
recovers the source.

Finds trajectories in which the base (spectral) scorer's argmax lands STRICTLY AFTER
the gold decisive step -- the downstream-contamination failure mode -- while the
rescored score's argmax is exactly the gold step. For each (model, subset) cell it
sweeps every seed in the manifest, emits EVERY such "flip", and writes each one's raw
trajectory record, its per-step signal at every pipeline stage, and a before/after
score figure, so the set can be browsed and a manuscript example chosen from it.

The frozen configs are NOT hardcoded: they are read from the manuscript's own
bookkeeping file (the seed-window "triples" selection produced by exp-august), so the
configuration is always the one behind the main-table number. Both accuracies are
re-derived over the manuscript seed window and asserted against the recorded ones
before anything is drawn -- a mismatch means the wrong config/window, not noise, and
aborts the run. Seeds outside that window re-partition train/val/test; their flips are
kept (the config is identical) but flagged ``manuscript_window=False``, and the
headline figure always prefers a window seed.

Because SOAP shares its base config with the SVD (proj) row in that protocol, one
reproduction per (cell, seed) yields BOTH the "w/o rescoring" and the rescored
predictions, so the comparison is apples-to-apples within a single scoring pass.

    # from the repo root
    python -m src.analysis.qualitative --config configs/datasets/ww.yaml --model qwen3.5-9b

    # pin the headline (combined.pdf) example, restrict to the manuscript seeds,
    # or restyle without touching code:
    python -m src.analysis.qualitative --config configs/datasets/ww.yaml --model qwen3.5-9b \
        --set pick.hand-crafted=[13,42] --set seeds_scope=manuscript \
        --set colors.soap=#d98032
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from ..common import paths
from ..common.cli import base_parser, load_and_narrow
from ..common.provenance import RunTimer
from ..metrics import compute_metrics, get_mistake_meta, standardize_role
from ..reproduce.core import ReproContext, reproduce_row
from ..rescore.weights import coerce_w

DEFAULT_SELECTION_TSV = "exp-august/outputs/manuscript-tables/table1_main_selection.tsv"
DEFAULT_OUT_ROOT = "artifacts/qualitative_examples"

# Series colours, overridable per run: --set colors.base=#807EAF --set colors.soap=...
# These are the manuscript's own plot inks lightened: purpleinplot #7676a4 and
# orangeinplot #e29c7a from main.tex.
DEFAULT_COLORS = {"base": "#807EAF", "soap": "#F1A484"}

# The hyperparameter keys reproduce_row reads off a row (see src/reproduce/core.py).
HP_KEYS = ("pooling", "position", "method", "c_begin", "c_end", "centered", "weighted",
           "direction", "strategy", "orient", "score_norm", "layer_range", "gamma", "w")

# The selection TSV stores accuracies rounded to 4 decimals, so the reproduction gate
# cannot use the 1e-9 tolerance src/reproduce/run.py applies to the reduced tables.
ACC_TOL = 5e-5

# Plot-legibility bucket for the auto-pick: enough steps that "downstream" is visible,
# few enough that the markers do not crowd at column width.
READABLE_STEPS = (8, 30)


# ── the frozen configs (manuscript bookkeeping) ─────────────────────────────
def _load_cells(cfg: dict, tsv: Path) -> list[dict]:
    """Manuscript cells for this dataset, narrowed to the requested models/subsets."""
    if not tsv.exists():
        raise SystemExit(f"selection TSV not found: {tsv}")
    ds = paths._ds(cfg)                      # 'dataset' on a stage config, 'name' on a manifest
    df = pd.read_csv(tsv, sep="\t")          # parses centered/weighted to real bools
    df = df[(df["role"] == "cell") & (df["dataset"] == ds)
            & (df["model"].isin(cfg["models"])) & (df["subset"].isin(cfg["subsets"]))]
    if df.empty:
        raise SystemExit(f"no cells in {tsv} for dataset={ds!r} "
                         f"models={cfg['models']} subsets={cfg['subsets']}")
    cells = []
    for _, row in df.iterrows():
        # The TSV mixes SOAP rows with baseline rows that leave the hyperparameter
        # columns empty, so pandas widens c_begin/c_end to float and w to object.
        # Narrow them back, or the frozen config logged into meta.json reads as
        # "c_end: 5.0" / "w: '2'" where the manuscript says 5 and 2.
        hp = {k: row[k] for k in HP_KEYS}
        hp["c_begin"], hp["c_end"] = int(hp["c_begin"]), int(hp["c_end"])
        hp["gamma"] = float(hp["gamma"])
        hp["w"] = coerce_w(hp["w"])
        cells.append({
            "model": row["model"], "subset": row["subset"], "column": row["column"],
            "seeds": [int(s) for s in str(row["seeds"]).split(",")],
            "recorded_svd": float(row["svd_step"]),
            "recorded_soap": float(row["soap_step"]),
            "hp": hp,
        })
    return cells


# ── the base side (native "lower = error" convention) ───────────────────────
def _base_view(r) -> tuple[pd.DataFrame, list[int]]:
    """Per-trajectory base predictions + a per-step base rank, in keeper order.

    ``base`` is in the scorer's native convention: for ``proj`` LOWER is more
    error-like, so the base prediction is the ARGMIN, with ties broken toward the
    EARLIEST step -- the convention ``compute_metrics`` uses (its stable ascending
    sort leaves tied entries in step order). ``_tabulate`` in src/reproduce/core.py
    must not be reused here: its ``(value, -i)`` sort key breaks ascending ties toward
    the LATER step, which would disagree with the metric on exact ties.
    """
    ascending = r.direction == "asc"
    vals = r.base.detach().float().cpu()
    ranks: list[int] = [0] * len(vals)
    rows = []
    m_steps, m_roles = get_mistake_meta(r.keeper)

    for (start, end), true_step, true_role in zip(r.keeper.traj_ranges, m_steps, m_roles):
        entries = r.keeper.index[start:end]
        seg = vals[start:end].tolist()
        order = sorted(range(len(seg)), key=lambda i: (seg[i], i), reverse=not ascending)
        for rank, i in enumerate(order):
            ranks[start + i] = rank + 1
        best = order[0]
        gold_i = next((i for i, e in enumerate(entries) if e.step_idx == true_step), None)
        rows.append({
            "traj_idx": entries[0].traj_idx, "n_steps": len(entries),
            "true_step": true_step, "true_agent": true_role,
            "base_pred_step": entries[best].step_idx,
            "base_pred_agent": standardize_role(entries[best].role),
            "gold_base_rank": ranks[start + gold_i] if gold_i is not None else None,
            "base_step_correct": true_step is not None and entries[best].step_idx == true_step,
        })
    return pd.DataFrame(rows), ranks


def _soap_margin(r, start: int, end: int, gold_i: int) -> float:
    """final[gold] - max(final[every other step]); positive iff SOAP's argmax is gold."""
    seg = r.final.detach().float().cpu()[start:end]
    others = torch.cat([seg[:gold_i], seg[gold_i + 1:]])
    return float(seg[gold_i] - others.max()) if others.numel() else float("inf")


# ── the correctness gate ────────────────────────────────────────────────────
def _verify_cell(cell: dict, repros: dict) -> dict:
    """Re-derive both main-table numbers from the reproductions; abort on mismatch."""
    per_seed = {}
    for seed, r in repros.items():
        base_acc = compute_metrics(r.base, r.keeper, [1], r.direction)[f"step@1_{r.direction}"]
        soap_acc = r.metrics["step@1_desc"]
        bp, _ = _base_view(r)
        explicit = float(bp["base_step_correct"].sum()) / len(r.keeper.traj_ranges)
        assert abs(explicit - base_acc) < 1e-12, (
            f"{cell['column']} seed {seed}: explicit base argmin accuracy {explicit} != "
            f"metric {base_acc} -- tie-break convention drift")
        per_seed[seed] = {"base_step_acc": base_acc, "soap_step_acc": soap_acc,
                          "n_trajectories": len(r.keeper.traj_ranges)}

    mean_base = sum(v["base_step_acc"] for v in per_seed.values()) / len(per_seed)
    mean_soap = sum(v["soap_step_acc"] for v in per_seed.values()) / len(per_seed)
    ok_base = abs(mean_base - cell["recorded_svd"]) <= ACC_TOL
    ok_soap = abs(mean_soap - cell["recorded_soap"]) <= ACC_TOL
    if not (ok_base and ok_soap):
        raise AssertionError(
            f"{cell['model']}/{cell['column']}: reproduction does not match the "
            f"manuscript. base {mean_base:.6f} vs {cell['recorded_svd']} "
            f"(ok={ok_base}), soap {mean_soap:.6f} vs {cell['recorded_soap']} "
            f"(ok={ok_soap}). Wrong config or seed window.")
    return {"seeds": cell["seeds"], "per_seed": per_seed, "tolerance": ACC_TOL,
            "reproduced_base_step_acc": mean_base, "recorded_svd_step": cell["recorded_svd"],
            "reproduced_soap_step_acc": mean_soap, "recorded_soap_step": cell["recorded_soap"],
            "passed": True}


# ── candidates ──────────────────────────────────────────────────────────────
def _find_flips(repros: dict, split: str, in_window: bool,
                relaxed: bool = False) -> pd.DataFrame:
    """Trajectories where base predicts a downstream step and SOAP recovers the gold one.

    Strict criterion: ``base_pred > gold`` and ``soap_pred == gold``. The relaxed
    criterion only requires SOAP to lift the gold step into the top 2; it changes the
    claim from "recovers" to "nearly recovers" and is flagged in the output.
    """
    out = []
    for seed, r in repros.items():
        base_df, _ = _base_view(r)
        merged = base_df.merge(
            r.predictions[["traj_idx", "pred_step", "pred_agent", "true_step_rank"]],
            on="traj_idx", how="inner")
        ranges = {r.keeper.index[s].traj_idx: (s, e) for s, e in r.keeper.traj_ranges}
        for _, row in merged.iterrows():
            gold = row["true_step"]
            if gold is None or pd.isna(gold):
                continue
            if not row["base_pred_step"] > gold:
                continue
            recovered = row["pred_step"] == gold
            near = (not recovered) and row["true_step_rank"] is not None \
                and pd.notna(row["true_step_rank"]) and int(row["true_step_rank"]) <= 2
            if not (recovered or (relaxed and near)):
                continue
            start, end = ranges[row["traj_idx"]]
            gold_i = next(i for i, e in enumerate(r.keeper.index[start:end])
                          if e.step_idx == gold)
            out.append({
                "seed": seed, "split": split, "manuscript_window": in_window,
                "relaxed": not recovered,
                "traj_idx": int(row["traj_idx"]), "n_steps": int(row["n_steps"]),
                "true_step": int(gold), "true_agent": row["true_agent"],
                "base_pred_step": int(row["base_pred_step"]),
                "base_pred_agent": row["base_pred_agent"],
                "soap_pred_step": int(row["pred_step"]), "soap_pred_agent": row["pred_agent"],
                "offset": int(row["base_pred_step"] - gold),
                "gold_base_rank": int(row["gold_base_rank"]),
                "gold_final_rank": int(row["true_step_rank"]),
                "soap_margin": _soap_margin(r, start, end, gold_i),
            })
    return pd.DataFrame(out)


def _rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic preference order: legible plot, then the strongest rescue story.

    ``_prefix`` prefers a gold step with steps on BOTH sides: a decisive error at step 0
    makes the contamination story invisible, since every step is downstream of it and
    the figure has no clean prefix to contrast against. Manuscript-window seeds sort
    first so the headline figure stays drawn from the seeds behind the reported number.
    """
    if df.empty:
        return df
    lo, hi = READABLE_STEPS
    df = df.copy()
    df["_window"] = (~df["manuscript_window"]).astype(int)           # 0 = in window
    df["_readable"] = (~df["n_steps"].between(lo, hi)).astype(int)   # 0 = in range
    df["_prefix"] = df["true_step"].clip(upper=2).rsub(2)            # gold >=2 -> 0, 1 -> 1, 0 -> 2
    df = df.sort_values(
        by=["relaxed", "_window", "_readable", "_prefix", "offset", "gold_base_rank",
            "soap_margin", "seed", "traj_idx"],
        ascending=[True, True, True, True, False, False, False, True, True],
    ).drop(columns=["_window", "_readable", "_prefix"]).reset_index(drop=True)
    df.insert(0, "pick_rank", df.index + 1)
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best-ranked seed per trajectory; record where else it flipped.

    The same trajectory flipping under several seeds is the same qualitative story told
    from different train/test partitions, so drawing each one would pad the gallery
    without adding an example. Every occurrence stays listed in ``candidates.tsv``.
    """
    if df.empty:
        return df
    seeds = (df.groupby("traj_idx")["seed"]
               .apply(lambda s: ",".join(str(v) for v in sorted(set(s)))))
    df = df.copy()
    df["flip_seeds"] = df["traj_idx"].map(seeds)
    df["n_flip_seeds"] = df["traj_idx"].map(df.groupby("traj_idx")["seed"].nunique())
    return df.drop_duplicates(subset="traj_idx", keep="first").reset_index(drop=True)


# ── outputs ─────────────────────────────────────────────────────────────────
def _steps_frame(r, base_ranks: list[int], traj_idx: int, gold, base_pred, soap_pred):
    """Per-step signal at every pipeline stage for one trajectory, plot-ready."""
    df = r.per_step[r.per_step["traj_idx"] == traj_idx].copy()
    start = next(s for s, e in r.keeper.traj_ranges if r.keeper.index[s].traj_idx == traj_idx)
    df["base_rank"] = [base_ranks[start + i] for i in range(len(df))]
    df = df.rename(columns={"rank": "final_rank"})
    df["is_gold"] = df["step_idx"] == gold
    df["is_base_pred"] = df["step_idx"] == base_pred
    df["is_soap_pred"] = df["step_idx"] == soap_pred
    cols = ["model", "subset", "seed", "split", "traj_idx", "n_steps", "step_idx", "role",
            "base", "oriented", "normalized", "final", "base_rank", "final_rank",
            "is_gold", "is_base_pred", "is_soap_pred"]
    return df[cols]


def _write_example(cfg, cell, r, base_ranks, cand: dict, verification: dict,
                   out_dir: Path, tsv: Path) -> dict:
    """Raw trajectory record + per-step scores + full provenance for one example."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_idx = cand["traj_idx"]

    src_json = paths.data_root(cfg) / cell["subset"] / f"{traj_idx}.json"
    shutil.copyfile(src_json, out_dir / "trajectory.json")        # verbatim, byte-identical
    raw = json.loads(src_json.read_text())

    steps = _steps_frame(r, base_ranks, traj_idx, cand["true_step"],
                         cand["base_pred_step"], cand["soap_pred_step"])
    steps.to_csv(out_dir / "steps.tsv", sep="\t", index=False)

    meta = {
        "cell": {"dataset": paths._ds(cfg), "model": cell["model"], "subset": cell["subset"],
                 "column": cell["column"], "manuscript_seeds": cell["seeds"]},
        "frozen_config": r.config,
        "selection": {k: cand[k] for k in (
            "seed", "split", "manuscript_window", "relaxed", "pick_rank", "traj_idx",
            "n_steps", "offset", "gold_base_rank", "gold_final_rank", "soap_margin",
            "flip_seeds", "n_flip_seeds") if k in cand},
        "gold": {"step": cand["true_step"], "agent": cand["true_agent"],
                 "mistake_step_raw": raw.get("mistake_step"),
                 "mistake_agent_raw": raw.get("mistake_agent"),
                 "mistake_reason": raw.get("mistake_reason"),
                 "question_ID": raw.get("question_ID"),
                 "n_history_turns": len(raw.get("history", []))},
        "predictions": {
            "base": {"step": cand["base_pred_step"], "agent": cand["base_pred_agent"]},
            "soap": {"step": cand["soap_pred_step"], "agent": cand["soap_pred_agent"]}},
        "scored_step_indices": steps["step_idx"].tolist(),
        "verification": verification,
        "sources": {
            "selection_tsv": str(tsv),
            "trajectory_json": str(src_json),
            "activations": str(paths.reps_root(cfg) / cell["model"] / cell["subset"]
                               / f"{traj_idx}.safetensors"),
            "attention": str(paths.attn_root(cfg) / cell["model"] / cell["subset"]
                             / f"{traj_idx}.safetensors"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


# ── driver ──────────────────────────────────────────────────────────────────
def _reproduce(ctx, cell: dict, seeds, split: str) -> dict:
    return {int(s): reproduce_row(ctx, {**cell["hp"], "seed": int(s)}, split=split)
            for s in seeds}


def _collect(ctx, cfg, cell, split, allow_relaxed, seeds_scope):
    """Every flip the frozen config produces, over the requested seeds.

    The manuscript window is reproduced first and gated against the recorded
    accuracies. Remaining seeds re-partition train/val/test, so their flips are
    legitimate illustrations of the SAME frozen config but are marked
    ``manuscript_window=False`` -- they are not the seeds behind the reported number.
    """
    window = [int(s) for s in cell["seeds"]]
    repros = _reproduce(ctx, cell, window, split)
    verification = _verify_cell(cell, repros)
    print(f"[qual] {cell['model']}/{cell['column']}: base "
          f"{verification['reproduced_base_step_acc']:.4f} (rec {cell['recorded_svd']}) "
          f"soap {verification['reproduced_soap_step_acc']:.4f} "
          f"(rec {cell['recorded_soap']}) -- verified")

    pools = {(split, s): r for s, r in repros.items()}
    frames = [_find_flips(repros, split, in_window=True)]

    if seeds_scope == "all":
        others = [int(s) for s in cfg.get("seeds", []) if int(s) not in window]
        for s in others:
            rr = _reproduce(ctx, cell, [s], split)
            frames.append(_find_flips(rr, split, in_window=False))
            pools.update({(split, k): v for k, v in rr.items()})
            # Drop the seed's representation matrices; the Reproduction keeps only the
            # per-step vectors and its keeper, so the pool stays small across 20 seeds.
            ctx._seed_cache.pop((s, split), None)
        print(f"[qual] {cell['column']}: swept {len(window) + len(others)} seeds")

    cands = pd.concat(frames, ignore_index=True)
    if cands.empty and allow_relaxed:
        print(f"[qual] {cell['column']}: no strict flip anywhere -- relaxed criterion")
        cands = _find_flips(repros, split, in_window=True, relaxed=True)
    return cands, pools, verification


def run(cfg) -> None:
    tsv = Path(cfg.get("selection_tsv", DEFAULT_SELECTION_TSV))
    out_root = Path(cfg.get("out_root", DEFAULT_OUT_ROOT))
    split = cfg.get("split", "test")
    picks = cfg.get("pick", {}) or {}
    allow_relaxed = bool(cfg.get("allow_relaxed", False))
    make_plots = bool(cfg.get("plots", True))
    plot_scale = cfg.get("plot_scale", "auto")
    seeds_scope = cfg.get("seeds_scope", "all")        # all | manuscript
    dedupe = bool(cfg.get("dedupe_by_traj", True))
    colors = {**DEFAULT_COLORS, **(cfg.get("colors") or {})}

    cells = _load_cells(cfg, tsv)
    with RunTimer(cfg, "analysis") as rec:
        rec.note(selection_tsv=str(tsv), out_root=str(out_root), split=split,
                 seeds_scope=seeds_scope, dedupe_by_traj=dedupe, colors=colors,
                 cells=[c["column"] for c in cells])
        headline, drawn = [], []
        for cell in cells:
            sub_cfg = dict(cfg, poolings=[cell["hp"]["pooling"]])   # skip unused pooling
            ctx = ReproContext(sub_cfg, cell["model"], cell["subset"],
                               n_ranges=int(cfg.get("n_ranges", 4)))
            cands, pools, verification = _collect(ctx, cfg, cell, split,
                                                  allow_relaxed, seeds_scope)
            cell_dir = out_root / cell["model"] / cell["subset"]
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "verification.json").write_text(json.dumps(verification, indent=2))
            rec.add_output(cell_dir / "verification.json")

            if cands.empty:
                print(f"[qual] {cell['column']}: no candidates "
                      f"(re-run with --set allow_relaxed=true to widen)")
                continue

            cands = _rank_candidates(cands)
            examples = _dedupe(cands) if dedupe else cands.copy()
            cands.to_csv(cell_dir / "candidates.tsv", sep="\t", index=False)
            rec.add_output(cell_dir / "candidates.tsv")
            print(f"[qual] {cell['column']}: {len(cands)} flip(s) over "
                  f"{cands.traj_idx.nunique()} trajectories -> drawing {len(examples)}"
                  f"{' (deduped by trajectory)' if dedupe else ''}")

            cell_examples = []
            for _, row in examples.iterrows():
                cand = row.to_dict()
                r = pools[(cand["split"], cand["seed"])]
                _, base_ranks = _base_view(r)
                ex_dir = (cell_dir / "examples"
                          / f"seed-{cand['seed']}_traj-{cand['traj_idx']}")
                meta = _write_example(sub_cfg, cell, r, base_ranks, cand, verification,
                                      ex_dir, tsv)
                for name in ("trajectory.json", "steps.tsv", "meta.json"):
                    rec.add_output(ex_dir / name)
                cell_examples.append({
                    "cell": cell, "cand": cand, "meta": meta, "dir": ex_dir,
                    "steps": pd.read_csv(ex_dir / "steps.tsv", sep="\t")})

            want = picks.get(cell["subset"])
            if want is not None:
                seed, traj = int(want[0]), int(want[1])
                hit = [e for e in cell_examples
                       if e["cand"]["seed"] == seed and e["cand"]["traj_idx"] == traj]
                if not hit:
                    raise SystemExit(
                        f"--set pick.{cell['subset']}=[{seed},{traj}] is not a drawn "
                        f"example. Available (seed, traj_idx): "
                        f"{[(e['cand']['seed'], e['cand']['traj_idx']) for e in cell_examples]}")
                headline.append(hit[0])
            else:
                headline.append(cell_examples[0])
            drawn.append((cell, cell_dir, cell_examples))

        if make_plots and drawn:
            from .qualitative_plot import plot_examples, plot_gallery, plot_combined
            for cell, cell_dir, cell_examples in drawn:
                for out in plot_examples(cell_examples, colors, plot_scale):
                    rec.add_output(out)
                for out in plot_gallery(cell["column"], cell_examples, cell_dir,
                                        colors, plot_scale):
                    rec.add_output(out)
            for out in plot_combined(headline, out_root, colors, plot_scale):
                rec.add_output(out)

        _write_manifest(out_root, tsv, drawn, headline, seeds_scope, dedupe)
        rec.add_output(out_root / "MANIFEST.md")


def _write_manifest(out_root: Path, tsv: Path, drawn: list, headline: list,
                    seeds_scope: str, dedupe: bool) -> None:
    import subprocess
    from datetime import datetime, timezone
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "unknown"
    top = {(e["cell"]["column"], e["cand"]["seed"], e["cand"]["traj_idx"])
           for e in headline}
    lines = [
        "# Qualitative examples: base picks a downstream consequence, SOAP recovers the source",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"at commit `{sha}` by `python -m src.analysis.qualitative`.",
        "",
        f"Frozen configs and recorded accuracies come from `{tsv}` (the manuscript's",
        "without-GT seed-window selection). Before any example is selected, the",
        "reproduced base/SOAP step accuracies over the manuscript seed window are",
        "asserted equal to the recorded main-table numbers (see `verification.json`).",
        "",
        f"Seed scope: **{seeds_scope}**. Rows marked `window = no` come from a seed",
        "OUTSIDE the manuscript triple: same frozen config, different train/val/test",
        "partition. They are valid illustrations but are not the seeds behind the",
        "reported number, so prefer a `window = yes` row for the paper.",
        ""
        + ("Figures are deduplicated by trajectory (best-ranked seed kept); every "
           "occurrence is still listed in `candidates.tsv`." if dedupe else ""),
        "",
        "`combined.pdf` is the manuscript-ready figure (marked ★ below).",
        "Each cell has a paginated `gallery` contact sheet of all its examples.",
        "",
        "| | cell | seed | traj | steps | gold | base pred | SOAP pred | offset | window |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, _, examples in drawn:
        for e in examples:
            k = e["cand"]
            star = "★" if (cell["column"], k["seed"], k["traj_idx"]) in top else ""
            lines.append(
                f"| {star} | {cell['column']} | {k['seed']} | {k['traj_idx']} | "
                f"{k['n_steps']} | {k['true_step']} | {k['base_pred_step']} | "
                f"{k['soap_pred_step']} | {k['offset']} | "
                f"{'yes' if k['manuscript_window'] else 'no'} |")
    lines += [
        "",
        "Per example, `examples/seed-<s>_traj-<t>/` holds the verbatim trajectory JSON,",
        "`steps.tsv` (the signal at every pipeline stage: `base -> oriented -> normalized",
        "-> final`, with both rankings), `meta.json` (frozen config, gold/predictions,",
        "source paths) and the figure.",
        "",
        "Figures plot the ORIENTED base score (higher = more error-like, the quantity the",
        "rescoring adds to) against the final SOAP score; the lower panel is the",
        r"promotion `delta = final - oriented` on its own scale. Steps where the two",
        "curves coincide exactly are steps that are nobody's top-`w` predecessor and so",
        "receive no correction at all -- a property of the weight construction, not a",
        "plotting artifact.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    run(load_and_narrow(args))


if __name__ == "__main__":
    main()
