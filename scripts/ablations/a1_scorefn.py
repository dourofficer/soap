"""A1 / tab:scorefn — is the spectral band the right base score?

Eight scoring functions on the anchor position, no rescoring anywhere. Orientations
are FIXED, not selected: perplexity reads "higher = more error"; every projection
row (random/top/tail/full/ours) and both norm rows read "lower = more error",
implemented as the same 1/(x+eps) fold the pipeline uses everywhere.

  perplexity    mean NLL of the step's tokens under the proxy, in context
  random        projection onto a random orthonormal basis of dimension |C|,
                redrawn per seed
  top           spectral band [0, |C|)
  tail          the trailing |C| of the 20 computed components
  full          all 20 computed components
  norm-l1/l2    the step vector's L1 / L2 norm
  ours          the anchor band [c_begin, c_end) — must reproduce Table 1's base row

|C| = c_end - c_begin of the anchor band, so every subspace row is dimension-matched
to ours. Two stages:

  --stage nll    GPU: one forward pass per step (the extractor's exact context
                 construction) writing per-step mean NLL TSVs. Resumable per cell.
  --stage score  CPU: everything else + the table. Requires the nll TSVs.

    python scripts/ablations/a1_scorefn.py --stage nll --models qwen3.5-9b
    python scripts/ablations/a1_scorefn.py --stage score
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POOLING, REPO, RESULTS_DIR, anchor_rows, assert_close,  # noqa: E402
                    cell_paths, iter_cells, load_selection, position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch         # noqa: E402
from main.score import EPS, fit_svd, score_steps                      # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

OUT_DIR = RESULTS_DIR / "a1_scorefn"
OUT = OUT_DIR / "scorefn.tsv"
N_COMP = 20


def nll_path(cfg, model, subset) -> Path:
    return OUT_DIR / "nll" / f"{cfg['dataset']}-{subset}-{model}.tsv"


# ── stage nll ───────────────────────────────────────────────────────────────
def run_nll(args) -> None:
    """Per-step mean NLL with the extractor's exact context construction."""
    # The venv's torchvision/torchaudio were built against a different torch and
    # crash transformers' lazy imports; nothing here needs them, so block both.
    sys.modules["torchvision"] = None
    sys.modules["torchaudio"] = None

    from tqdm import tqdm

    from src.data import build_context, iter_scoreable_steps, load_dataset
    from src.models import get_adapter
    import functools

    for cfg, model, subset in iter_cells(args.configs):
        if args.models and model not in args.models:
            continue
        out = nll_path(cfg, model, subset)
        if out.exists():
            print(f"[skip] {out}")
            continue
        model_path = (REPO / cfg["model_paths"][model]).resolve()
        adapter = get_adapter(str(model_path))
        lm, tokenizer = adapter.load(str(model_path), torch.bfloat16, {"": args.device})
        lm.eval()
        context_fn = functools.partial(build_context,
                                       template_kwargs=adapter.template_kwargs(),
                                       with_gt=False)
        trajs = load_dataset(C.data_root(cfg), subset=subset)
        rows = []
        for traj in tqdm(trajs, desc=f"nll {model}/{subset}"):
            traj_idx = int(traj.filename.replace(".json", ""))
            for step_idx in iter_scoreable_steps(traj):
                enc = context_fn(traj, step_idx, tokenizer,
                                 max_tokens=cfg["max_tokens"])
                input_ids = enc["input_ids"].to(args.device)
                ctx_len = enc["ctx_len"]
                if input_ids.shape[1] <= ctx_len:
                    continue
                with torch.no_grad():
                    logits = lm(input_ids, use_cache=False).logits[0]
                # Hard-truncated steps can leave ctx_len = 0; the first token then
                # has no predecessor, so its NLL is undefined and it is skipped.
                start = max(ctx_len, 1)
                targets = input_ids[0, start:]
                pred = logits[start - 1:-1]
                # fp32 in chunks: a full fp32 copy of (tokens, vocab) is GBs.
                total, n = 0.0, targets.numel()
                for lo in range(0, n, 1024):
                    total += float(F.cross_entropy(pred[lo:lo + 1024].float(),
                                                   targets[lo:lo + 1024],
                                                   reduction="sum"))
                rows.append({"traj_idx": traj_idx, "step_idx": step_idx,
                             "n_tokens": n, "mean_nll": total / n})
            torch.cuda.empty_cache()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(f"wrote {out}  ({len(rows)} rows)")
        del lm
        torch.cuda.empty_cache()


# ── stage score ─────────────────────────────────────────────────────────────
def inv(x: torch.Tensor) -> torch.Tensor:
    """The pipeline's orientation fold: lower raw value = more error."""
    return 1.0 / (x + EPS)


def random_basis(d: int, k: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return Q.to(device)


def nll_scores(nll: pd.DataFrame, keeper) -> torch.Tensor:
    """Per-step mean NLL aligned to keeper row order (higher = more error)."""
    table = {(int(r.traj_idx), int(r.step_idx)): float(r.mean_nll)
             for r in nll.itertuples()}
    vals = []
    for e in keeper.index:
        key = (e.traj_idx, e.step_idx)
        assert key in table, f"missing NLL for traj {key[0]} step {key[1]}"
        vals.append(table[key])
    return torch.tensor(vals)


def run_score(args) -> None:
    rows_out = []
    for cfg, model, subset in iter_cells(args.configs):
        seeds = C.seeds_for(cfg, subset)
        svd_row, _ = anchor_rows(load_selection(cfg), model, subset)
        position = svd_row["position"]
        cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
        width = ce - cb
        rep_dir, data_dir, files = cell_paths(cfg, model, subset)
        members, names = position_load_names(rep_dir, files, position)
        assert members is None, "ensemble anchors are not supported here"
        nll = pd.read_csv(nll_path(cfg, model, subset), sep="\t")
        print(f"[{cfg['dataset']}] {model}/{subset} anchor={position} [{cb},{ce}) "
              f"|C|={width}")

        rows = ["perplexity", "random", "top", "tail", "full",
                "norm-l1", "norm-l2", "ours"]
        acc = {r: {"step_t": 0.0, "agent_t": 0.0, "step_v": 0.0, "agent_v": 0.0}
               for r in rows}
        for seed in seeds:
            parts = split_files(files, cfg["splits"], seed)
            loads = {sp: load_representations(rep_dir, data_dir, poolings=[POOLING],
                                              weight_names=names, files=parts[sp],
                                              device=args.device)
                     for sp in ("train", "val", "test")}
            R_train = loads["train"].stores[(POOLING, position)].R
            V = fit_svd(R_train, N_COMP)
            Q = random_basis(R_train.shape[1], width, seed, args.device)
            for sp, kt, ka in (("test", "step_t", "agent_t"),
                               ("val", "step_v", "agent_v")):
                split = loads[sp]
                R = split.stores[(POOLING, position)].R
                Rf = R.float()
                scores = {
                    "perplexity": nll_scores(nll, split.keeper).to(args.device),
                    "random": inv((Rf @ Q).square().mean(dim=1)),
                    "top": score_steps(R, V, 0, width),
                    "tail": score_steps(R, V, N_COMP - width, N_COMP),
                    "full": score_steps(R, V, 0, N_COMP),
                    "norm-l1": inv(Rf.norm(p=1, dim=1)),
                    "norm-l2": inv(Rf.norm(p=2, dim=1)),
                    "ours": score_steps(R, V, cb, ce),
                }
                S = torch.stack([scores[r].double() for r in rows])
                m = compute_metrics_batch(S, None, [1], ctx=KeeperContext(split.keeper))
                for i, r in enumerate(rows):
                    acc[r][kt] += float(m["step@1"][i]) / len(seeds)
                    acc[r][ka] += float(m["agent@1"][i]) / len(seeds)
            del loads
            if args.device == "cuda":
                torch.cuda.empty_cache()

        assert_close(acc["ours"]["step_t"], float(svd_row["step_acc_test"]),
                     f"{model}/{subset} ours vs Table 1 base")
        for r in rows:
            rows_out.append({"dataset": cfg["dataset"], "model": model,
                             "subset": subset, "seeds": ",".join(map(str, seeds)),
                             "position": position, "c_begin": cb, "c_end": ce,
                             "width": width, "row": r,
                             "step_acc_test": acc[r]["step_t"],
                             "agent_acc_test": acc[r]["agent_t"],
                             "step_acc_val": acc[r]["step_v"],
                             "agent_acc_val": acc[r]["agent_v"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT}  ({len(df)} rows)")

    df["cell"] = df["dataset"] + "/" + df["subset"]
    order = ["perplexity", "random", "top", "tail", "full",
             "norm-l1", "norm-l2", "ours"]
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index="row", columns="cell", values="step_acc_test") * 100
        print(f"\n=== {model} (step acc %, test) ===")
        print(pivot.reindex(order).round(2).to_string())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["nll", "score"], required=True)
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.configs is None:
        from common import CONFIGS_NOGT
        args.configs = CONFIGS_NOGT
    if args.stage == "nll":
        run_nll(args)
    else:
        run_score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
