"""E6 — a reference built from the steps BEFORE the decisive error.

No benchmark in our question pools ships successful trajectories. By the paper's own
account, though, the steps before the decisive step t* are ordinary: nothing has gone
wrong yet. This runner fits R on those steps alone and compares it with the full train
split. It reads step labels on the reference split, so it is a diagnostic, never a
variant of the method.

  real      every step of the seed's train split — Table 1; must reproduce
  prefix    the train split's steps with step_idx < mistake_step
  random    steps drawn uniformly without replacement from the same train split, as
            many as `prefix` holds for that seed (one fixed draw per split) — separates
            "fewer rows" from "earlier, cleaner rows"

Only the fit set changes: validation and test splits, scoring and dependency weights
are E4's. The loop, the two modes (`anchor`, `reselect`) and the self-check are E4's
too. Run once per selection rule:

    python scripts/ablations/e6_prefix_reference.py [--select-rule val --out ...]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR                                        # noqa: E402
from e2_synthfit import Reference                                     # noqa: E402
from e4_reference_controls import main as e4_main                     # noqa: E402
from main.stores import RepresentationStore, RepresentationStores     # noqa: E402

OUT = RESULTS_DIR / "e6_prefix_reference.tsv"
REF_ORDER = ["real", "prefix", "random"]


def prefix_rows(keeper, data_dir: Path) -> list[int]:
    """Rows of the steps before each trajectory's decisive step. Checks the store's
    label against the JSON's `mistake_step` first, so the mask cuts where we think."""
    rows = []
    for start, end in keeper.traj_ranges:
        entries = keeper.index[start:end]
        traj = entries[0].traj_idx
        t_star = int(json.loads((data_dir / f"{traj}.json").read_text())["mistake_step"])
        marked = [e.step_idx for e in entries if e.is_mistake]
        assert marked in ([t_star], []), f"traj {traj}: store marks {marked}, JSON {t_star}"
        rows += [e.row for e in entries if e.step_idx < t_star]
    return rows


class MaskedReference(Reference):
    """The real per-seed train split, cut down to a subset of its rows."""

    def __init__(self, name, rep_dir, data_dir, device, files):
        super().__init__(name, rep_dir, data_dir, device, static=False, files=files)
        self.n_rows: dict[tuple, tuple[int, int]] = {}
        self.extra = {}

    def train(self, seed_files=None):
        full = super().train(seed_files)
        keep = prefix_rows(full.keeper, self.data_dir)
        n_full = len(full.keeper.index)
        if self.name == "random":
            # One fixed draw per split, seeded by the split's own file list.
            g = torch.Generator().manual_seed(zlib.crc32("|".join(seed_files).encode()))
            keep = torch.randperm(n_full, generator=g)[:len(keep)].sort().values.tolist()
        self.n_rows[tuple(seed_files)] = (len(keep), n_full)
        kept = [k for k, _ in self.n_rows.values()]
        self.extra = {"n_rows": sum(kept) / len(kept), "n_rows_min": min(kept),
                      "n_rows_full": sum(f for _, f in self.n_rows.values()) / len(kept)}
        idx = torch.tensor(keep, device=self.device)
        stores = {k: RepresentationStore(R=st.R[idx], pooling=st.pooling, name=st.name)
                  for k, st in full.stores.items()}
        return RepresentationStores(stores=stores, keeper=full.keeper)


def build_refs(cell, model, syn_cfg, device, wanted) -> dict[str, Reference]:
    refs = {}
    if "real" in wanted:
        refs["real"] = Reference("real", cell["rep_dir"], cell["data_dir"], device,
                                 static=False, files=cell["files"])
    for name in ("prefix", "random"):
        if name in wanted:
            refs[name] = MaskedReference(name, cell["rep_dir"], cell["data_dir"], device,
                                         cell["files"])
    return refs


if __name__ == "__main__":
    sys.exit(e4_main(build=build_refs, order=REF_ORDER, out=OUT))
