"""Answer checkers, dispatched by a pool's `answer_type`.

Each checker returns `(is_correct, reason)`, or `(None, reason)` when it cannot
decide and the caller should escalate to the LLM judge.

The harness drivers ship their own in-run judges, but those are permissive
substring matches ("18" matches "1183") and are never trusted; these run
post-hoc over `summary.json` and produce the authoritative `verdict.json`.
"""
from __future__ import annotations

import re
import string
from collections import Counter

# ── normalization ─────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace (SQuAD-style)."""
    text = (text or "").lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _strip_math_wrappers(text: str) -> str:
    """Remove LaTeX/markdown decoration around a numeric answer."""
    t = (text or "").strip()
    t = re.sub(r"\\boxed\s*{(.+?)}", r"\1", t, flags=re.DOTALL)
    t = t.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"\\text\s*{(.*?)}", r"\1", t)
    t = t.replace(",", "")            # thousands separators
    t = t.rstrip(".")
    return t.strip()


def _to_number(text: str):
    """Parse a scalar from free text, or None. Handles %, fractions, LaTeX frac."""
    t = _strip_math_wrappers(text)
    m = re.fullmatch(r"\\d?frac\s*{(-?[\d.]+)}\s*{(-?[\d.]+)}", t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", t):
        a, b = t.split("/")
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    t = t.rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def _last_number(text: str):
    """Fall back to the final number in a verbose answer."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", _strip_math_wrappers(text))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


# ── checkers ──────────────────────────────────────────────────────────────────

def check_numeric(extracted: str, gold: str) -> tuple[bool | None, str]:
    if not extracted:
        return False, "no answer extracted"
    g = _to_number(gold)
    e = _to_number(extracted)
    if g is None:
        # Non-numeric gold (e.g. an algebraic expression): fall back to string.
        return check_exact(extracted, gold)
    if e is None:
        e = _last_number(extracted)
        if e is None:
            return False, f"no number in answer {extracted[:60]!r}"
    close = abs(e - g) <= 1e-6 * max(1.0, abs(g))
    return close, f"numeric {e} vs gold {g}"


def check_mcq(extracted: str, gold: str) -> tuple[bool | None, str]:
    if not extracted:
        return False, "no answer extracted"
    gold_letter = (gold or "").strip().upper()[:1]
    t = _strip_math_wrappers(extracted).strip()
    # Leading option letter: "B", "B.", "(B)", "B) foo".
    # The letter must be the whole answer or be followed by option-style
    # punctuation — plain whitespace would make "I think ..." parse as "I".
    m = re.match(r"^[\(\[]?([A-Z])(?:[\)\].:,]\s*|\s*$)", t.upper())
    if m:
        return m.group(1) == gold_letter, f"letter {m.group(1)} vs gold {gold_letter}"
    # "Answer: B", "The correct option is C", "The answer is A) 42".
    # Upper-case both sides so a single pattern covers any casing; the lazy gap
    # skips filler words, and \b…\b keeps it to a standalone letter (so the "I"
    # in "IS" cannot match).
    m = re.search(r"\b(?:ANSWER|OPTION)\b.{0,20}?\b([A-Z])\b", t.upper(), re.DOTALL)
    if m:
        return m.group(1) == gold_letter, f"letter {m.group(1)} vs gold {gold_letter}"
    return None, f"no option letter in {extracted[:60]!r}"


def _f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def check_exact(extracted: str, gold: str, f1_threshold: float = 0.6) -> tuple[bool | None, str]:
    """Short-answer EM, then token-F1; ambiguous middle escalates to the judge."""
    if not extracted:
        return False, "no answer extracted"
    ne, ng = normalize(extracted), normalize(gold)
    if ne == ng:
        return True, "exact match"
    # A verbose reply containing the gold span as a whole phrase.
    if ng and re.search(rf"\b{re.escape(ng)}\b", ne):
        return True, "gold span contained in answer"
    f1 = _f1(extracted, gold)
    if f1 >= f1_threshold:
        return None, f"borderline f1={f1:.2f} — escalate"
    return False, f"f1={f1:.2f} below threshold"


CHECKERS = {
    "numeric": check_numeric,
    "mcq": check_mcq,
    "exact": check_exact,
    # `open` has no programmatic checker — always LLM-judged.
}
