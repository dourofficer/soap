#!/usr/bin/env python3
"""Generate a patched tokenizer directory for checkpoints whose declared
tokenizer class mis-decodes under the installed transformers.

Why this exists
---------------
`DeepSeek-R1-Distill-Llama-8B` ships `tokenizer_config.json` declaring
`tokenizer_class: LlamaTokenizerFast` alongside sentencepiece-era fields
(`legacy`, `sp_model_kwargs`), but its `tokenizer.json` is a ByteLevel BPE.
Under transformers 5.x `AutoTokenizer` resolves this to a class that does not
apply the ByteLevel decoder, so text round-trips corrupted:

    "Hello world\\n\\nsecond line"  ->  "Helloworldsecondline"

Served through vLLM that surfaces as raw byte tokens in every completion
(`Ċ` for newline, `Ġ` for space), which would poison every collected
trajectory. Loading the same `tokenizer.json` directly is correct, so the fix
is to copy the tokenizer files and rewrite the declared class.

The original checkpoint is never modified — vLLM is pointed at the patched
directory with `--tokenizer` (see `tokenizer:` in configs/serve.yaml).

    python datagen/serve/fix_tokenizer.py --model deepseek-8b
    python datagen/serve/fix_tokenizer.py --check-all     # audit, no writes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

OUT_ROOT = common.DATAGEN_DIR / "serve" / "tokenizers"
# Fields that steer transformers toward the sentencepiece code path.
DROP_KEYS = ["legacy", "sp_model_kwargs", "add_bos_token", "add_eos_token"]
PROBE = "Hello world\n\nsecond line\ttabbed"


def roundtrips(path: str | Path) -> tuple[bool, str]:
    """Does `AutoTokenizer.from_pretrained(path)` survive an encode/decode?"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    ids = tok.encode(PROBE, add_special_tokens=False)
    got = tok.decode(ids)
    return got == PROBE, f"{type(tok).__name__}: {got!r}"


def patch(src: Path, dst: Path) -> None:
    """Copy tokenizer files, declaring the generic fast class."""
    dst.mkdir(parents=True, exist_ok=True)
    tok_json = src / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(
            f"{src} has no tokenizer.json — this patch only handles checkpoints "
            f"that carry a self-contained fast tokenizer.")
    shutil.copy2(tok_json, dst / "tokenizer.json")

    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    for key in DROP_KEYS:
        cfg.pop(key, None)
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    # Keep any generation-side config the server may want alongside it.
    for extra in ["special_tokens_map.json", "generation_config.json"]:
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", dest="models",
                    help="model key from serve.yaml (repeatable)")
    ap.add_argument("--check-all", action="store_true",
                    help="report round-trip status for every model; write nothing")
    args = ap.parse_args()

    cfg = common.load_cfg("serve")
    repo_root = common.REPO_ROOT

    if args.check_all:
        rc = 0
        for name, spec in cfg["models"].items():
            src = (repo_root / spec["path"]).resolve()
            ok, detail = roundtrips(src)
            print(f"  {'OK  ' if ok else 'BAD '} {name:14s} {detail}")
            rc |= (0 if ok else 1)
        return rc

    if not args.models:
        ap.error("pass --model <key> (repeatable) or --check-all")

    for name in args.models:
        spec = cfg["models"][name]
        src = (repo_root / spec["path"]).resolve()
        dst = OUT_ROOT / name

        ok, detail = roundtrips(src)
        print(f"[{name}] source tokenizer: {'OK' if ok else 'BROKEN'} — {detail}")
        if ok:
            print("  nothing to patch; drop the `tokenizer:` key from serve.yaml")
            continue

        patch(src, dst)
        ok, detail = roundtrips(dst)
        print(f"  patched -> {dst}")
        print(f"  {'OK' if ok else 'STILL BROKEN'} — {detail}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
