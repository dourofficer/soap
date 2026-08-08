"""Assert outputs-gt/<ds> extraction mirrors outputs/<ds> exactly (read-only).

Why this matters: ``src.stores.split_files`` derives the seed->partition mapping from
the FILE SET of a reps directory, and the baseline cells' splits are derived the same
way. If the with-GT extraction produced a different set of trajectory files — or a
different set of step entries inside one — every split silently shifts and GT vs non-GT
numbers stop being comparable. Run this after GT extraction, before scoring.

Checked per (dataset, model, subset), for activations and attention:
  * the sorted .safetensors stems match between outputs/ and outputs-gt/;
  * per file, the set of step indices in the keys matches
    (activations: identical key sets; attention: identical step-entry sets — GT files
    gain one extra ctx column per step, never a step entry).

    python scripts/check_gt_parity.py --dataset ww
    python scripts/check_gt_parity.py --dataset traceelephant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]


def _steps(path: Path) -> set[str]:
    with safe_open(path, framework="pt") as f:
        return {k.split(".", 1)[0] for k in f.keys()}


def _keys(path: Path) -> set[str]:
    with safe_open(path, framework="pt") as f:
        return set(f.keys())


def check_dir(plain: Path, gt: Path, per_file) -> list[str]:
    errs = []
    if not plain.is_dir() or not gt.is_dir():
        return [f"missing directory: {plain if not plain.is_dir() else gt}"]
    p_stems = sorted(p.stem for p in plain.glob("*.safetensors"))
    g_stems = sorted(p.stem for p in gt.glob("*.safetensors"))
    if p_stems != g_stems:
        only_p = sorted(set(p_stems) - set(g_stems))[:5]
        only_g = sorted(set(g_stems) - set(p_stems))[:5]
        errs.append(f"file-set mismatch under {gt}: only-plain={only_p} only-gt={only_g} "
                    f"({len(p_stems)} vs {len(g_stems)} files)")
        return errs
    for stem in p_stems:
        a, b = per_file(plain / f"{stem}.safetensors"), per_file(gt / f"{stem}.safetensors")
        if a != b:
            errs.append(f"{gt / stem}.safetensors: step/key sets differ "
                        f"(only-plain={sorted(a - b)[:5]} only-gt={sorted(b - a)[:5]})")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--subsets", nargs="*", default=None)
    args = ap.parse_args()

    manifest = yaml.safe_load((REPO / "configs" / "datasets" / f"{args.dataset}.yaml").read_text())
    models = args.models or manifest["models"]
    subsets = args.subsets or manifest["subsets"]

    errs = []
    for model in models:
        for subset in subsets:
            for stage, per_file in (("activations", _keys), ("attention", _steps)):
                plain = REPO / "outputs" / args.dataset / stage / model / subset
                gt = REPO / "outputs-gt" / args.dataset / stage / model / subset
                found = check_dir(plain, gt, per_file)
                tag = f"{args.dataset}/{stage}/{model}/{subset}"
                if found:
                    errs += [f"[{tag}] {e}" for e in found]
                else:
                    print(f"[ok] {tag}")
    if errs:
        print("\n".join(errs), file=sys.stderr)
        sys.exit(1)
    print(f"[parity] {args.dataset}: outputs-gt mirrors outputs exactly")


if __name__ == "__main__":
    main()
