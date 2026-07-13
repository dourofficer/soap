"""Derive every stage's output root from a resolved config.

The on-disk layout is preserved **byte-for-byte** — these functions reproduce the
exact directory strings the hand-written configs used, so existing
activations / attention / projections stay valid. What changes is that the
split-tag (e.g. ``325``) and the ``outputs-<dataset>`` prefix are now derived
from the dataset manifest in one place instead of being threaded through every
config by hand.

Layout (base = ``outputs-<dataset>`` unless the manifest overrides ``outputs_base``)::

    {base}/activations                         reps_root         (not split-tagged)
    {base}/attention                           attn_root         (not split-tagged)
    {base}/weighted-projections/{tag}          svd_root
    {base}/undiscounted-splits/{tag}           undisc_root
    {base}/discounted-splits/sweep/{tag}       rescore_sweep_root
    {base}/discounted-splits/reduced/{tag}     disc_root
    {base}/reproductions/{tag}/{split}         reproductions_root   (new)
"""
from __future__ import annotations

from pathlib import Path

from experiments._common.config import split_tag


def outputs_base(cfg: dict) -> Path:
    base = cfg.get("outputs_base") or f"outputs-{cfg['dataset']}"
    return Path(base)


def _tag(cfg: dict) -> str:
    return split_tag(cfg["splits"], cfg.get("split_tag"))


def reps_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "activations"


def attn_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "attention"


def svd_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "weighted-projections" / _tag(cfg)


def undisc_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "undiscounted-splits" / _tag(cfg)


def rescore_sweep_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "discounted-splits" / "sweep" / _tag(cfg)


def disc_root(cfg: dict) -> Path:
    return outputs_base(cfg) / "discounted-splits" / "reduced" / _tag(cfg)


def reproductions_root(cfg: dict, split: str) -> Path:
    return outputs_base(cfg) / "reproductions" / _tag(cfg) / split
