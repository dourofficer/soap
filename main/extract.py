"""Stage 1+2: pooled hidden states and attention mass. A faithful port of ``src/extract``.

This module is deliberately NOT simplified relative to ``src/``. Extraction is the
expensive, once-only stage and its artifacts are shared with the frozen ``outputs/``
tree, so the on-disk schema must stay byte-compatible:

    activations   "{step}.{pool}.{shorthand}"      fp16, BOTH poolings written
    attention     "{step}.raw_attn"                (L, n_ctx)   head-averaged
                  "{step}.raw_attn_per_head"       (L, H, n_ctx)
                  "{step}.attn_residual_mass"      (L,)
                  "{step}.ctx_indices"             (n_ctx,)

Only the READING side of the pipeline pins pooling to ``mean``; the extractor keeps
writing ``last`` as well, so switching the sweep back is a config change, not a
re-extraction. Existing outputs are skipped, so extraction is resumable.

LAYER SHORTHAND (and the off-by-one that bites)
    tuple idx 0     -> "embed"              embedding output, before any block
    tuple idx k+1   -> "act/k"              residual stream AFTER block k
    (derived)       -> "act/{N-1}_normed"   final residual through the model's final norm
``act/k`` is tuple index ``k+1``, NOT ``k``. The attention stage indexes attention
BLOCKS instead, so ``L`` there is a different (often shorter) list — for Qwen3.5 only
the full-attention layers of its 3:1 DeltaNet:Attention stack ever dispatch.

WHY THE SDPA OVERRIDE
    ``output_attentions=True`` materialises (1,H,N,N) per layer — tens of GB at 8k
    tokens, so it OOMs. Reconstructing Q/K by hand is fragile across architectures.
    Instead the registered ``sdpa`` function is temporarily swapped for a wrapper that
    (1) delegates to the genuine implementation so the model's forward is bit-for-bit
    unchanged, and (2) separately slices Q to the scored step's rows and scores those
    against full K. Peak attention memory is O(H * |T_t| * N), not O(H * N^2).

    python -m main extract --config configs-main/ww.yaml [--stage activations]
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from safetensors.torch import save_file
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from . import config as C
from .data import load_dataset, iter_scoreable_steps, build_step_input, extract_metadata

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ── model adapters ──────────────────────────────────────────────────────────
def find_decoder(model) -> nn.Module:
    """Return the submodule holding the decoder ``.layers`` (and ``.norm``)."""
    inner = getattr(model, "model", None)
    for m in (inner,
              getattr(inner, "language_model", None),
              getattr(inner, "text_model", None),
              getattr(model, "language_model", None)):
        if m is not None and hasattr(m, "layers"):
            return m
    raise RuntimeError("Could not locate the decoder block list; extend find_decoder().")


def num_hidden_layers(cfg) -> int:
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    text = getattr(cfg, "text_config", None)
    if text is not None and hasattr(text, "num_hidden_layers"):
        return text.num_hidden_layers
    raise AttributeError("config has no num_hidden_layers (checked text_config too).")


class ModelAdapter:
    """Default = Llama-3.1 / Qwen3 (AutoModelForCausalLM + AutoTokenizer)."""
    _loader = AutoModelForCausalLM

    def load(self, path, torch_dtype, device_map):
        tok = AutoTokenizer.from_pretrained(path)
        model = self._loader.from_pretrained(path, torch_dtype=torch_dtype,
                                             device_map=device_map)
        model.eval()
        return model, tok

    def template_kwargs(self) -> dict:
        return {}

    def final_norm(self, model) -> nn.Module:
        return find_decoder(model).norm

    def num_layers(self, model) -> int:
        return num_hidden_layers(model.config)

    def extract_block_indices(self, model) -> list[int]:
        """Decoder block indices to extract from. Default: every block."""
        return list(range(self.num_layers(model)))


class Qwen35Adapter(ModelAdapter):
    """Hybrid Qwen3.5: load as image-text-to-text with a text tokenizer; extract only
    the full-attention layers (3:1 DeltaNet:Attention stack)."""
    _loader = AutoModelForImageTextToText

    def template_kwargs(self) -> dict:
        return {"enable_thinking": False}

    def extract_block_indices(self, model) -> list[int]:
        cfg = getattr(model.config, "text_config", model.config)
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is None:
            raise AttributeError("Qwen3.5 config exposes no layer_types.")
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]


_BY_MODEL_TYPE = {"qwen3_5": Qwen35Adapter}


def get_adapter(model_path: str) -> ModelAdapter:
    return _BY_MODEL_TYPE.get(AutoConfig.from_pretrained(model_path).model_type,
                              ModelAdapter)()


# ── layer shorthand ─────────────────────────────────────────────────────────
def shorthand_to_layer(shorthand: str) -> int:
    base = shorthand.removesuffix("_normed")
    if base == "embed":
        return 0
    if base.startswith("act/"):
        return int(base[4:]) + 1
    raise ValueError(f"unknown shorthand {shorthand!r}")


def all_shorthands(n_layers: int) -> list[str]:
    return ["embed"] + [f"act/{i}" for i in range(n_layers)] + [f"act/{n_layers - 1}_normed"]


# ── pooling ─────────────────────────────────────────────────────────────────
def pool_last(h: Tensor, ctx_len: int) -> Tensor:
    """Hidden state of the step's LAST token. Single samples are never padded."""
    assert h.shape[0] > ctx_len
    return h[-1].half().cpu()


def pool_mean(h: Tensor, ctx_len: int) -> Tensor:
    """Mean over the step's OWN tokens. Averaged in fp32 before the fp16 cast —
    summing hundreds of fp16 values loses precision fast."""
    assert h.shape[0] > ctx_len
    return h[ctx_len:].float().mean(dim=0).half().cpu()


def _pooled(h: Tensor, ctx_len: int) -> dict[str, Tensor]:
    return {"last": pool_last(h, ctx_len), "mean": pool_mean(h, ctx_len)}


# ── activations ─────────────────────────────────────────────────────────────
def extract_hidden(model, input_ids, ctx_len, layers, final_norm) -> dict[str, Tensor]:
    """One forward pass; return ``{"{pool}.{shorthand}": vector}``.

    ``use_cache=False`` (nothing is generated, so a KV cache is pure overhead).
    ``act/{N-1}_normed`` is handled apart from the raw residuals because it is not a
    distinct tuple entry: it is the LAST residual pushed through the final norm — the
    representation the LM head actually sees.
    """
    n_layers = num_hidden_layers(model.config)
    valid = set(all_shorthands(n_layers))
    normed_sh = f"act/{n_layers - 1}_normed"
    wanted = valid if layers == "all" else set(layers)
    bad = wanted - valid
    if bad:
        raise ValueError(f"unknown shorthands {bad}")

    with torch.no_grad():
        out = model(input_ids, attention_mask=None, use_cache=False,
                    output_hidden_states=True)
    hs = out.hidden_states
    result: dict[str, Tensor] = {}
    for sh in wanted - {normed_sh}:
        h = hs[shorthand_to_layer(sh)][0].float()
        for stat, vec in _pooled(h, ctx_len).items():
            result[f"{stat}.{sh}"] = vec
    if normed_sh in wanted:
        with torch.no_grad():
            hn = final_norm(hs[shorthand_to_layer(normed_sh)][0].unsqueeze(0))[0].float()
        for stat, vec in _pooled(hn, ctx_len).items():
            result[f"{stat}.{normed_sh}"] = vec
    return result


def activations_for(traj, model, tokenizer, max_tokens, layers, final_norm, tk, with_gt):
    device = next(model.parameters()).device
    hidden = {}
    for step_idx in iter_scoreable_steps(traj):
        enc = build_step_input(traj, step_idx, tokenizer, max_tokens=max_tokens,
                               template_kwargs=tk, with_gt=with_gt)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= enc["ctx_len"]:
            continue
        hidden[step_idx] = extract_hidden(model, input_ids, enc["ctx_len"], layers, final_norm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return hidden


# ── attention: the streaming SDPA override ──────────────────────────────────
def build_key_mask(ctx_step_ids, step_tokens, seq_len, device) -> Tensor:
    """(n_ctx, N) one-hot mask: row j marks every token position belonging to step j.

    Bucketing token attention into per-step mass is then one matmul instead of a Python
    loop over steps. Rows are disjoint by construction (``main.data`` guarantees every
    token belongs to exactly one step), so nothing is double counted; positions in no
    row (scaffolding, the scored step itself) fall out into the residual mass.
    """
    M = torch.zeros(len(ctx_step_ids), seq_len, device=device, dtype=torch.float32)
    for j, i in enumerate(ctx_step_ids):
        idx = torch.tensor(step_tokens[i], device=device, dtype=torch.long)
        M[j].index_fill_(0, idx, 1.0)
    return M


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped-query KV heads so each query head has its own K."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class _StreamState:
    armed = False
    query_idx = None
    key_mask = None
    query_pool = "mean"
    out_per_head: dict[int, Tensor] = {}


_STREAM = _StreamState()


@torch.no_grad()
def _reduce(module, query, key, scaling):
    """Bucket ONE layer's step-t attention into per-predecessor mass.

    ``query``/``key`` are (1,H,N,d) post-projection and post-RoPE — exactly what the
    model is about to attend with, which is why no architecture-specific reconstruction
    is needed. ``scores`` is (H, |T_t|, N), never (H, N, N).

    Causality is re-applied by hand because SDPA's internal mask was bypassed; without
    it the softmax would normalise over future tokens. Softmax runs in fp32 (the model
    may be bf16, and normalising a distribution in bf16 loses real precision).
    """
    n_q, n_kv = query.shape[1], key.shape[1]
    if n_q != n_kv:
        key = _repeat_kv(key, n_q // n_kv)
    dev = query.device
    qi = _STREAM.query_idx.to(dev, non_blocking=True)
    km = _STREAM.key_mask.to(dev, non_blocking=True)
    q_t = query[0].index_select(1, qi)                              # (H, |T_t|, d)
    k_full = key[0]                                                 # (H, N, d)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(q_t, k_full.transpose(-1, -2)) * scale    # (H, |T_t|, N)
    key_pos = torch.arange(k_full.shape[1], device=dev)
    not_causal = key_pos[None, :] > qi[:, None]                     # (|T_t|, N)
    scores = scores.masked_fill(not_causal[None], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1)
    A_t = probs.mean(dim=1) if _STREAM.query_pool == "mean" else probs[:, -1, :]
    _STREAM.out_per_head[module.layer_idx] = (A_t @ km.T).cpu()     # (H, n_ctx)


def _streaming_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """Drop-in for ``sdpa_attention_forward``: real output first, our reduction second."""
    out = sdpa_attention_forward(module, query, key, value, attention_mask,
                                 scaling=scaling, dropout=dropout, **kw)
    if _STREAM.armed:
        _reduce(module, query, key, scaling)
    return out


@contextmanager
def _override_sdpa():
    """Install the wrapper globally for one forward pass, restore in ``finally``."""
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _streaming_sdpa
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig


def _force_sdpa(model) -> None:
    """Pin the attention implementation to 'sdpa' so the override is actually hit.

    A checkpoint may default to flash-attention or eager; either bypasses the registry
    entry we replace and the extraction would silently capture nothing.
    """
    cfgs = {id(model.config): model.config}
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        cfgs[id(tc)] = tc
    for cfg in cfgs.values():
        cfg._attn_implementation = "sdpa"


@torch.no_grad()
def attention_for(traj, model, tokenizer, max_tokens, tk, query_pool="mean", with_gt=False):
    flat: dict[str, Tensor] = {}
    device = next(model.parameters()).device
    for step_idx in iter_scoreable_steps(traj):
        enc = build_step_input(traj, step_idx, tokenizer, max_tokens=max_tokens,
                               template_kwargs=tk, with_gt=with_gt)
        step_tokens = enc["step_tokens"]
        ctx_step_ids = sorted(m for m in step_tokens if m != step_idx)
        if not any(m >= 0 for m in ctx_step_ids):
            # No real predecessors. In with-GT mode the GT bucket (GT_STEP < 0) alone
            # does not count: recording it would give first steps attention entries that
            # do not exist in the without-GT tree, breaking step-entry parity.
            continue
        input_ids = enc["input_ids"].to(device)
        _STREAM.out_per_head = {}
        _STREAM.query_idx = torch.tensor(step_tokens[step_idx], device=device, dtype=torch.long)
        _STREAM.key_mask = build_key_mask(ctx_step_ids, step_tokens, input_ids.shape[1], device)
        _STREAM.query_pool = query_pool
        _STREAM.armed = True
        try:
            with _override_sdpa():
                model(input_ids, use_cache=False)
        finally:
            _STREAM.armed = False
        oph = _STREAM.out_per_head
        if not oph:
            raise RuntimeError(f"no attention captured at step {step_idx} of {traj.filename}")
        # Stack captured layers in decoder order -> (L, H, n_ctx). For hybrid models only
        # full-attention blocks were ever called, so L is exactly those.
        per_head = torch.stack([oph[i] for i in sorted(oph)], dim=0).float().contiguous()
        flat[f"{step_idx}.raw_attn"] = per_head.mean(dim=1).contiguous()
        flat[f"{step_idx}.raw_attn_per_head"] = per_head
        # Each head's row sums to <= 1 over predecessors; the shortfall went to
        # scaffolding / the step's own tokens / sinks. Averaged over heads.
        flat[f"{step_idx}.attn_residual_mass"] = (
            1.0 - per_head.sum(dim=-1).mean(dim=-1)).contiguous()
        flat[f"{step_idx}.ctx_indices"] = torch.tensor(ctx_step_ids, dtype=torch.long)
    return flat


# ── driver ──────────────────────────────────────────────────────────────────
def _check_gt(cfg: dict, trajs, out_path: Path) -> None:
    """GT artifacts must never mix into a without-GT tree, and an empty answer would
    silently degrade the run to without-GT — fail loudly on both."""
    if not cfg["gt"]:
        return
    assert any(p.endswith("-gt") for p in out_path.parts), \
        f"gt runs must write under a *-gt tree, got {out_path}"
    missing = [t.filename for t in trajs if not (t.ground_truth or "").strip()]
    assert not missing, f"gt but empty ground_truth in {len(missing)} files, e.g. {missing[:5]}"


def run_extract(cfg: dict, stages=("activations", "attention")) -> None:
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[cfg.get("dtype", "bfloat16")]
    device_map = "auto" if device == "auto" else {"": device}
    max_tokens = cfg["max_tokens"]
    dry = cfg.get("dry_run", False)
    start, end = cfg.get("start_idx", 0), cfg.get("end_idx")

    for model_name in cfg["models"]:
        model_path = cfg["model_paths"][model_name]
        model = tokenizer = adapter = None
        for subset in cfg["subsets"]:
            trajs_all = load_dataset(C.data_root(cfg), subset)
            trajs = trajs_all[start:end if end is not None else len(trajs_all)]
            for stage in stages:
                out_dir = (C.reps_root(cfg) if stage == "activations" else C.attn_root(cfg))
                out_dir = out_dir / model_name / subset
                _check_gt(cfg, trajs, out_dir)
                todo = [t for t in trajs
                        if not (out_dir / f"{Path(t.filename).stem}.safetensors").exists()]
                print(f"[{stage}] {model_name}/{subset}: {len(todo)}/{len(trajs)} to do "
                      f"-> {out_dir}")
                if dry or not todo:
                    continue
                if model is None:
                    adapter = get_adapter(model_path)
                    model, tokenizer = adapter.load(model_path, dtype, device_map)
                    _force_sdpa(model)
                tk = adapter.template_kwargs()
                out_dir.mkdir(parents=True, exist_ok=True)

                if stage == "activations":
                    n_layers = adapter.num_layers(model)
                    blocks = adapter.extract_block_indices(model)
                    layers = ["embed"] + [f"act/{i}" for i in blocks] + \
                             [f"act/{n_layers - 1}_normed"]
                    (out_dir / "config.json").write_text(json.dumps(
                        {"model": model_name, "layers": layers, "pool": "all",
                         "max_tokens": max_tokens, "dtype": cfg.get("dtype", "bfloat16"),
                         "subset": subset, "gt": bool(cfg["gt"])}, indent=2))
                else:
                    (out_dir / "config.json").write_text(json.dumps(
                        {"model": model_name, "subset": subset, "max_tokens": max_tokens,
                         "query_pool": "mean", "dtype": cfg.get("dtype", "bfloat16"),
                         "impl": "sdpa_streaming",
                         "attn_block_indices": adapter.extract_block_indices(model),
                         "gt": bool(cfg["gt"])}, indent=2))

                for traj in tqdm(todo, desc=f"{stage} {model_name}/{subset}"):
                    out_path = out_dir / f"{Path(traj.filename).stem}.safetensors"
                    meta = {"payload_metadata": json.dumps(extract_metadata(traj))}
                    if stage == "activations":
                        hidden = activations_for(traj, model, tokenizer, max_tokens,
                                                 layers, adapter.final_norm(model), tk,
                                                 cfg["gt"])
                        flat = {f"{s}.{k}": t.contiguous()
                                for s, d in hidden.items() for k, t in d.items()}
                        assert flat, f"no hidden states for {traj.filename}"
                        save_file(flat, out_path, metadata=meta)
                    else:
                        flat = attention_for(traj, model, tokenizer, max_tokens, tk,
                                             with_gt=cfg["gt"])
                        if flat:
                            save_file(flat, out_path, metadata=meta)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
