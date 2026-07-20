"""Reproduce a single frozen config end-to-end and expose its PER-STEP scores.

The sweep stages answer "which config is best?" by collapsing every run to one
accuracy number. This package answers the complementary question — "what did that
config actually do to each step of each trajectory?" — which is what you need to
plot score curves, eyeball failures, and compare methods case by case.
"""
from .core import Reproduction, reproduce_row, ReproContext

__all__ = ["Reproduction", "reproduce_row", "ReproContext"]
