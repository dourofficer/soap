"""E4 staging — an unrelated-text reference corpus for the SVD fit.

Writes `data/synthetic/wikitext/<n>.json`: 120 pseudo-trajectories of 8 "steps", each
step one WikiText-103 paragraph (400+ characters, headings skipped), drawn without
replacement under a fixed seed. The files carry the minimal schema `main.data`
requires, with `mistake_step = -1` as in the other fit-only corpora. The extractor
serializes every turn as an agent-log line, so this control keeps the log format and
changes only the content.

    python scripts/ablations/e4_stage_wikitext.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "synthetic" / "wikitext"
N_FILES, N_STEPS, MIN_CHARS, SEED = 120, 8, 400, 0


def main() -> None:
    text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    paras = [t.strip() for t in text
             if len(t.strip()) >= MIN_CHARS and not t.strip().startswith("=")]
    random.Random(SEED).shuffle(paras)
    assert len(paras) >= N_FILES * N_STEPS, f"only {len(paras)} paragraphs"
    OUT.mkdir(parents=True, exist_ok=True)
    for n in range(N_FILES):
        chunk = paras[n * N_STEPS:(n + 1) * N_STEPS]
        record = {"question_ID": f"wikitext-{n}", "question": "", "ground_truth": "",
                  "history": [{"role": "assistant", "content": c} for c in chunk],
                  "mistake_agent": "", "mistake_step": -1, "mistake_reason": "",
                  "level": -1, "subset": "wikitext"}
        (OUT / f"{n}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"wrote {N_FILES} files to {OUT}")


if __name__ == "__main__":
    main()
