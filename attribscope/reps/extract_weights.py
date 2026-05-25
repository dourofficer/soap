"""Extract raw step-to-step similarity scores from hidden state activations.

Per scored step, saves three tensors per trajectory:
    "{step_idx}.raw_cosine"  : (L+1, n_ctx)  float32
    "{step_idx}.raw_dot"     : (L+1, n_ctx)  float32
    "{step_idx}.ctx_indices" : (n_ctx,)      int64

Raw scores only — no temperature, no sqrt(d) scaling, no softmax. All
downstream reshaping (temperature, scaling, softmax, layer aggregation) is
left to consumers.

python -m attribscope.reps.extract_weights \
    --model      llama-3.1-8b \
    --subset     hand-crafted \
    --input      data/ww \
    --output-root  outputs/weighting \
    --max_tokens 8192 \
    --context    dependency \
    --device     cuda \
    --dtype      bfloat16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizer,
)
from safetensors.torch import save_file

from ..data.trajectory import Trajectory, load_dataset
from ..data.context    import iter_scoreable_steps, preprocess_context

MODELS = {
    "llama-3.1-8b": "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b": "/data/hoang/resources/models/Qwen/Qwen3-8B"
}
# ────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction
# ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_trajectory_scores(
    traj:       Trajectory,
    model:      PreTrainedModel,
    tokenizer:  PreTrainedTokenizer,
    max_tokens: int,
    device:     str,
    context:    str,
) -> dict[str, Tensor]:
    flat: dict[str, Tensor] = {}

    for step_idx in iter_scoreable_steps(traj):
        encoded     = preprocess_context(
            traj, step_idx, tokenizer,
            max_tokens=max_tokens, strategy=context,
        )
        input_ids   = encoded["input_ids"].to(device)
        step_tokens = encoded["step_tokens"]

        ctx_step_ids = sorted(m for m in step_tokens if m != step_idx)
        if not ctx_step_ids:
            continue

        # Forward pass → hidden states (L+1, seq_len, d)
        out    = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        del out

        # Mean-pool per step per layer
        def pool(idxs: list[int]) -> Tensor:
            sel = torch.tensor(idxs, device=device, dtype=torch.long)
            return hidden[:, sel, :].mean(dim=1).float()         # (L+1, d)

        h_i   = pool(step_tokens[step_idx])                       # (L+1, d)
        h_ctx = torch.stack(
            [pool(step_tokens[m]) for m in ctx_step_ids], dim=0,
        )                                                          # (n_ctx, L+1, d)
        del hidden

        # Raw similarities — no scaling, no softmax
        w_dot = torch.einsum("ld,nld->ln", h_i, h_ctx)             # (L+1, n_ctx)
        w_cos = torch.einsum(
            "ld,nld->ln",
            F.normalize(h_i,   dim=-1),
            F.normalize(h_ctx, dim=-1),
        )                                                          # (L+1, n_ctx)

        flat[f"{step_idx}.raw_cosine"]  = w_cos.cpu().contiguous()
        flat[f"{step_idx}.raw_dot"]     = w_dot.cpu().contiguous()
        flat[f"{step_idx}.ctx_indices"] = torch.tensor(ctx_step_ids, dtype=torch.long)

    return flat


def _extract_metadata(traj: Trajectory) -> dict:
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
    }


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract raw step-to-step similarity scores from activations."
    )
    p.add_argument("--model",      required=True, help="HF model name or local path.")
    p.add_argument("--subset",     default=None)
    p.add_argument("--input",      required=True, help="Dataset directory.")
    p.add_argument("--output-root", required=True, help="Output directory for .safetensors files.")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--start_idx",  type=int, default=0)
    p.add_argument("--end_idx",    type=int, default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument(
        "--context", choices=["dependency", "all"], default="dependency",
        help="Context selection strategy for hand-crafted trajectories.",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }[args.dtype]

    model_path = MODELS[args.model]
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model → {device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map={"": device},
    )
    model.eval()

    trajectories = load_dataset(args.input, subset=args.subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"Processing {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    out_root = Path(args.output_root) / args.model / args.subset 
    out_root.mkdir(parents=True, exist_ok=True)

    for traj in tqdm(trajectories, desc="trajectories"):
        out_path = out_root / f"{traj.filename}.safetensors"
        if out_path.exists():
            continue

        flat = extract_trajectory_scores(
            traj, model, tokenizer,
            max_tokens=args.max_tokens, device=device, context=args.context,
        )
        if not flat:
            continue

        save_file(
            flat, out_path,
            metadata={"payload_metadata": json.dumps(_extract_metadata(traj))},
        )


if __name__ == "__main__":
    main()