"""Model adapters — the few architecture-specific bits of extraction, in one file.

Only extraction stages need this;
scoring/rescoring never load a model. Dispatch by ``config.model_type``:

    from src.models import get_adapter
    adapter = get_adapter("../hub/Qwen/Qwen3.5-9B")
    model, tok = adapter.load(path, torch.bfloat16, {"": "cuda"})

DefaultAdapter = Llama-3.1 / Qwen3 (AutoModelForCausalLM + AutoTokenizer).
Qwen35Adapter = hybrid Qwen3.5 (image-text-to-text load, text tokenizer, extract
only full-attention blocks read off config.layer_types).
"""
from __future__ import annotations

from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


# ── module-tree helpers (robust to multimodal wrappers) ─────────────────────
def find_decoder(model) -> nn.Module:
    """Return the submodule holding the decoder ``.layers`` (and ``.norm``)."""
    inner = getattr(model, "model", None)
    candidates = [
        inner,                                    # Llama / Qwen3
        getattr(inner, "language_model", None),   # *ForImageTextToText
        getattr(inner, "text_model", None),
        getattr(model, "language_model", None),
    ]
    for m in candidates:
        if m is not None and hasattr(m, "layers"):
            return m
    raise RuntimeError("Could not locate the decoder block list; extend find_decoder().")


def num_hidden_layers(config) -> int:
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    text = getattr(config, "text_config", None)
    if text is not None and hasattr(text, "num_hidden_layers"):
        return text.num_hidden_layers
    raise AttributeError("config has no num_hidden_layers (checked text_config too).")


# ── adapters ────────────────────────────────────────────────────────────────
class ModelAdapter:
    """Default = Llama-3.1 / Qwen3 (AutoModelForCausalLM + AutoTokenizer)."""

    def load(self, path, torch_dtype, device_map, eager: bool = False):
        tok = AutoTokenizer.from_pretrained(path)
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if eager:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.eval()
        return model, tok

    def template_kwargs(self) -> dict:
        return {}

    def decoder_layers(self, model) -> nn.Module:
        return find_decoder(model).layers

    def final_norm(self, model) -> nn.Module:
        return find_decoder(model).norm

    def num_layers(self, model) -> int:
        return num_hidden_layers(model.config)

    def extract_block_indices(self, model) -> list[int]:
        """Decoder block indices to extract from. Default: every block."""
        return list(range(self.num_layers(model)))


class DefaultAdapter(ModelAdapter):
    """Explicit name for the Llama-3.1 / Qwen3 default behaviour."""
    pass


def _text_config(config):
    return getattr(config, "text_config", config)


class Qwen35Adapter(ModelAdapter):
    """Hybrid Qwen3.5: load as image-text-to-text with a text tokenizer; extract only
    the full-attention layers (3:1 DeltaNet:Attention stack)."""

    def load(self, path, torch_dtype, device_map, eager: bool = False):
        tok = AutoTokenizer.from_pretrained(path)
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if eager:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
        model.eval()
        return model, tok

    def template_kwargs(self) -> dict:
        return {"enable_thinking": False}

    def extract_block_indices(self, model) -> list[int]:
        layer_types = getattr(_text_config(model.config), "layer_types", None)
        if layer_types is None:
            raise AttributeError("Qwen3.5 config exposes no layer_types.")
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]


_BY_MODEL_TYPE: dict[str, type[ModelAdapter]] = {"qwen3_5": Qwen35Adapter}


def get_adapter(model_path: str) -> ModelAdapter:
    model_type = AutoConfig.from_pretrained(model_path).model_type
    return _BY_MODEL_TYPE.get(model_type, DefaultAdapter)()
