"""Per-step token counts, tokenizer only — no forward pass.

A1 recorded each step's token count as a by-product of its NLL pass, but only for the
four Who&When / TraceElephant cells. This script produces the same count for any
config (CORRECT-Error in particular) with the extractor's exact context construction:
the step's own tokens are what follows `ctx_len` in the encoded input. Output:
`results-ablations/ntokens/<ds>-<subset>-<model>.tsv` (traj_idx, step_idx, n_tokens),
which `common.load_ntokens` reads when A1's file is absent.

    python scripts/ablations/ntokens.py --configs configs-main/correct-error.yaml
    python scripts/ablations/ntokens.py --configs configs-main/ww.yaml --check
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import pandas as pd

sys.modules["torchvision"] = None      # built against another torch; nothing here needs them
sys.modules["torchaudio"] = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BACKBONES, REPO, RESULTS_DIR, iter_cells     # noqa: E402
from main import config as C                                    # noqa: E402

OUT_DIR = RESULTS_DIR / "ntokens"


def count(cfg, model, subset) -> pd.DataFrame:
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from src.data import build_context, iter_scoreable_steps, load_dataset
    from src.models import get_adapter

    path = str((REPO / cfg["model_paths"][model]).resolve())
    tokenizer = AutoTokenizer.from_pretrained(path)
    context_fn = functools.partial(build_context, with_gt=False,
                                   template_kwargs=get_adapter(path).template_kwargs())
    rows = []
    for traj in tqdm(load_dataset(C.data_root(cfg), subset=subset), desc=f"{model}/{subset}"):
        traj_idx = int(traj.filename.replace(".json", ""))
        for step_idx in iter_scoreable_steps(traj):
            enc = context_fn(traj, step_idx, tokenizer, max_tokens=cfg["max_tokens"])
            n = enc["input_ids"].shape[1] - max(enc["ctx_len"], 1)   # A1's convention
            if n > 0:
                rows.append({"traj_idx": traj_idx, "step_idx": step_idx, "n_tokens": n})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--check", action="store_true",
                   help="compare against A1's counts instead of writing")
    args = p.parse_args()
    for cfg, model, subset in iter_cells(args.configs, models=BACKBONES):
        name = f"{cfg['dataset']}-{subset}-{model}.tsv"
        if args.check:
            a1 = pd.read_csv(RESULTS_DIR / "a1_scorefn" / "nll" / name, sep="\t")
            m = a1.merge(count(cfg, model, subset), on=["traj_idx", "step_idx"],
                         how="outer", suffixes=("_a1", ""))
            bad = m[m["n_tokens_a1"] != m["n_tokens"]]
            print(f"{name}: {len(m)} steps, {len(bad)} mismatches")
            continue
        out = OUT_DIR / name
        if out.exists():
            print(f"[skip] {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        count(cfg, model, subset).to_csv(out, sep="\t", index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
