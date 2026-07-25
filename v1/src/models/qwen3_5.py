"""Adapter for Qwen3.5 (config ``model_type == "qwen3_5"``).

Qwen3.5 is a multimodal checkpoint, so the *model* is loaded with
``AutoModelForImageTextToText``, but extraction is **text-only**: we pair it with
a plain ``AutoTokenizer`` rather than the ``AutoProcessor``. The processor's chat
template expects each message's ``content`` to be a list of typed parts
(``[{"type": "text", ...}]``) and raises on the plain-string content the context
builder produces; the text tokenizer accepts strings exactly like the
Llama-3.1 / Qwen3 path.

Hybrid stack: 3 Gated DeltaNet (linear-attention) layers per 1 Gated Attention
(full softmax) layer. Only the full-attention layers expose a softmax attention
matrix, so we extract from those only — for *both* hidden states and attention
mass — reading their positions off ``config.layer_types`` (the per-layer
"linear_attention" / "full_attention" list), so nothing is hardcoded.

The decoder stack is at ``model.model.layers``; ``find_decoder`` (in base)
resolves it, so decoder/norm location is inherited unchanged.
"""
from __future__ import annotations

from transformers import AutoModelForImageTextToText, AutoTokenizer

from .base import ModelAdapter


def _text_config(config):
    """layer_types may live on the top config or under text_config."""
    return getattr(config, "text_config", config)


class Qwen35Adapter(ModelAdapter):

    def load(self, path, torch_dtype, device_map, eager: bool = False):
        # Text tokenizer, NOT the multimodal processor: its chat template
        # accepts plain-string message content (the processor wants typed parts).
        tok = AutoTokenizer.from_pretrained(path)
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if eager:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
        model.eval()
        return model, tok

    def template_kwargs(self) -> dict:
        # Suppress the default <think> block. Drop this if a checkpoint's
        # template doesn't accept enable_thinking.
        return {"enable_thinking": False}

    def extract_block_indices(self, model) -> list[int]:
        layer_types = getattr(_text_config(model.config), "layer_types", None)
        if layer_types is None:
            raise AttributeError(
                "Qwen3.5 config exposes no layer_types; cannot locate the "
                "full-attention layers."
            )
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]