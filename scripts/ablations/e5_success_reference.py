"""E5 — does a reference of FAILED runs score better than one of SUCCESSFUL runs?

The benchmarks ship no successes, so the contrast comes from MCP-Atlas, the corpus OAT
trains on: one agent, one task pool, both outcomes
(`scripts/ablations/e5_stage_mcp_atlas.py`). Both arms are fit out of distribution and
scored on the real val/test splits with the target's own attention, as in E4.

  real        the seed's train split — Table 1; must reproduce selection.tsv
  mcp-succ    50 successful MCP-Atlas runs
  mcp-fail    50 failed MCP-Atlas runs

Each arm is one draw without replacement under `--draw-seed`. The arms should hold
about the same number of steps; the runner prints both counts and records them.

The loop, the two modes (`anchor`, `reselect`) and the self-check are E4's. Run once
per selection rule:

    python scripts/ablations/e5_success_reference.py [--select-rule val --out ...]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR                                        # noqa: E402
from e2_synthfit import Reference                                     # noqa: E402
from e4_reference_controls import main as e4_main                     # noqa: E402
from main import config as C                                          # noqa: E402
from main.stores import list_rep_files                                # noqa: E402

OUT = RESULTS_DIR / "e5_success_reference.tsv"
REF_ORDER = ["real", "mcp-succ", "mcp-fail"]
OUTCOME = {"mcp-succ": "success", "mcp-fail": "fail"}
SUBSET = "mcp-atlas"


def sample_arm(rep_dir: Path, data_dir: Path, outcome: str, n: int, seed: int):
    """``n`` runs of one outcome, and the number of steps they hold."""
    pool = []
    for f in list_rep_files(rep_dir):
        rec = json.loads((data_dir / (Path(f).stem + ".json")).read_text())
        if rec["outcome"] == outcome:
            pool.append((f, len(rec["history"])))
    assert len(pool) >= n, f"only {len(pool)} {outcome} runs extracted"
    picked = random.Random(seed).sample(sorted(pool), n)
    return [f for f, _ in picked], sum(k for _, k in picked)


def build_refs(cell, model, syn_cfg, device, wanted, args) -> dict[str, Reference]:
    refs = {}
    if "real" in wanted:
        refs["real"] = Reference("real", cell["rep_dir"], cell["data_dir"], device,
                                 static=False, files=cell["files"])
    rep, data = C.reps_root(syn_cfg) / model / SUBSET, C.data_root(syn_cfg) / SUBSET
    for name, outcome in OUTCOME.items():
        if name in wanted:
            files, n_steps = sample_arm(rep, data, outcome, args.n_runs, args.draw_seed)
            refs[name] = Reference(name, rep, data, device, static=True, files=files)
            refs[name].extra = {"n_steps": n_steps, "draw_seed": args.draw_seed}
            print(f"  {name}: {len(files)} runs, {n_steps} steps (draw seed {args.draw_seed})")
    return refs


def extra_args(p) -> None:
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--draw-seed", type=int, default=0)


if __name__ == "__main__":
    sys.exit(e4_main(build=build_refs, order=REF_ORDER, out=OUT, extra_args=extra_args))
