"""Model-adapter registry.

``get_adapter(model_path)`` returns the right :class:`ModelAdapter` by reading
``config.model_type``. Unknown types fall back to :class:`DefaultAdapter`
(Llama-3.1 / Qwen3 behaviour), so existing models keep working unchanged.
"""
from __future__ import annotations

from transformers import AutoConfig

from .base import DefaultAdapter, ModelAdapter
from .qwen3_5 import Qwen35Adapter

_BY_MODEL_TYPE: dict[str, type[ModelAdapter]] = {
    "qwen3_5": Qwen35Adapter,
}


def get_adapter(model_path: str) -> ModelAdapter:
    model_type = AutoConfig.from_pretrained(model_path).model_type
    return _BY_MODEL_TYPE.get(model_type, DefaultAdapter)()


__all__ = ["ModelAdapter", "DefaultAdapter", "Qwen35Adapter", "get_adapter"]