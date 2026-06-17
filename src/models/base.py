"""Model adapters — isolate the architecture-specific bits of extraction.

Only the handful of points that actually vary across architectures live here:

  * which ``Auto*`` classes load the model and its tokenizer / processor,
  * extra kwargs the chat template needs (e.g. disabling a default think block),
  * where the decoder block list and final norm live in the module tree,
  * which decoder blocks we extract from (all of them, or a subset).

``DefaultAdapter`` reproduces the original Llama-3.1 / Qwen3 behaviour, so those
models are unaffected. New architectures subclass and override the few methods
that differ. Dispatch is by ``config.model_type`` in ``models.get_adapter``.
"""
from __future__ import annotations

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Module-tree helpers (robust to multimodal wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def find_decoder(model) -> nn.Module:
    """Return the submodule that holds the decoder ``.layers`` ModuleList.

    Tries the common locations across plain CausalLMs and multimodal wrappers
    (``*ForImageTextToText`` nests the text stack under ``model.language_model``).
    The returned module also exposes ``.norm`` (the final RMSNorm) on every
    architecture we target.
    """
    inner = getattr(model, "model", None)
    candidates = [
        inner,                                          # Llama / Qwen3
        getattr(inner, "language_model", None),         # *ForImageTextToText
        getattr(inner, "text_model", None),
        getattr(model, "language_model", None),
    ]
    for m in candidates:
        if m is not None and hasattr(m, "layers"):
            return m
    raise RuntimeError(
        "Could not locate the decoder block list. Inspect the module tree and "
        "extend find_decoder() (or override the adapter's decoder_layers)."
    )


def num_hidden_layers(config) -> int:
    """``num_hidden_layers``, looking under ``text_config`` for wrappers."""
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    text = getattr(config, "text_config", None)
    if text is not None and hasattr(text, "num_hidden_layers"):
        return text.num_hidden_layers
    raise AttributeError("config has no num_hidden_layers (checked text_config too).")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter base / default
# ─────────────────────────────────────────────────────────────────────────────

class ModelAdapter:
    """Default = Llama-3.1 / Qwen3 (AutoModelForCausalLM + AutoTokenizer)."""

    def load(self, path, torch_dtype, device_map, eager: bool = False):
        """Return ``(model, tokenizer_or_processor)``, eval mode.

        ``eager=True`` forces ``attn_implementation="eager"`` (needed by the
        eager attention extractor, which reads post-softmax probabilities).
        """
        tok = AutoTokenizer.from_pretrained(path)
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if eager:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.eval()
        return model, tok

    def template_kwargs(self) -> dict:
        """Extra kwargs splatted into ``apply_chat_template``. Empty by default."""
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