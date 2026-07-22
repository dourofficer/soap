#!/usr/bin/env python3
"""Render an autogen-0.2 `OAI_CONFIG_LIST` from the serve.yaml registry.

Captain-Agent is built on a vendored autogen 0.2 fork, which discovers models
through `config_list_from_json` reading an OAI_CONFIG_LIST file rather than
through env vars. Generating it from serve.yaml keeps a single source of truth
for endpoints instead of a second hand-maintained copy that drifts.

    python datagen/serve/gen_oai_config.py            # write configs/oai_config_list.json
    python datagen/serve/gen_oai_config.py --print    # stdout only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen import common  # noqa: E402

OUT_PATH = common.CONFIGS_DIR / "oai_config_list.json"


def build(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or common.load_cfg("serve")
    entries = []
    for name in cfg["models"]:
        ep = common.resolve_endpoint(name, cfg)
        entries.append({
            "model": ep["model"],
            "base_url": ep["base_url"],
            "api_key": ep["api_key"],
            # Any OpenAI-compatible server (vLLM here) uses api_type "openai".
            "api_type": "openai",
            "tags": [name],
            # No "price" key: autogen 0.2.20 forwards unknown entry fields
            # straight into Completions.create(), which rejects them.
        })
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    args = ap.parse_args()

    entries = build()
    text = json.dumps(entries, indent=2)
    if args.to_stdout:
        print(text)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"wrote {out} ({len(entries)} models: {[e['model'] for e in entries]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
