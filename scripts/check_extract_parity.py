"""Prove main/'s extractor is bit-identical to the one that produced outputs/.

main/extract.py is a port, and the bulk tensors are COPIED into the results trees rather
than recomputed — so its equivalence has to be demonstrated directly, not inferred. This
re-extracts a handful of trajectories into a scratch tree and compares, against the files
already on disk:

  * the safetensors KEY SET  (so "{step}.{pool}.{shorthand}" naming, both poolings, and
    the full attention key quartet all match)
  * torch.equal on EVERY tensor (bitwise — extraction is deterministic under no_grad
    with a fixed dtype)
  * the payload_metadata header and config.json

Run it BEFORE seeding the results trees. The --gt pass additionally exercises the pinned
[question, answer] block, the GT_STEP=-1 context column, and the parity guard that skips
steps whose only predecessor is the GT bucket.

    python scripts/check_extract_parity.py                      # both models, both settings
    python scripts/check_extract_parity.py --model qwen3.5-9b --n 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")


def _read(path: Path) -> tuple[dict, dict]:
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


def compare(got_dir: Path, ref_dir: Path, stems: list[str], label: str) -> tuple[list, list]:
    """Return (fatal, notes).

    FATAL is anything that changes a number downstream: a differing key set, a differing
    tensor, a missing file. NOTES are provenance-only drift (config.json fields,
    payload_metadata keys). The reference `outputs/` tree was extracted by an older
    revision than `outputs-gt/`, so it predates the `gt` config field, the `ground_truth`
    metadata key, and the switch from passing the model PATH to the model shorthand as
    `--model`. None of those are read by the scoring pipeline — `main.stores` uses only
    `mistake_step` and `mistake_agent` — so they are reported and not failed on.
    """
    fatal, notes = [], []
    gc, rc = got_dir / "config.json", ref_dir / "config.json"
    if gc.exists() and rc.exists():
        g, r = json.loads(gc.read_text()), json.loads(rc.read_text())
        for k in sorted(set(g) | set(r)):
            if g.get(k) != r.get(k):
                notes.append(f"{label}: config.json[{k}] got {g.get(k)!r}, ref {r.get(k)!r}")
    for stem in stems:
        gp, rp = got_dir / f"{stem}.safetensors", ref_dir / f"{stem}.safetensors"
        if not rp.exists():
            fatal.append(f"{label}/{stem}: no reference file at {rp}")
            continue
        if not gp.exists():
            fatal.append(f"{label}/{stem}: extractor produced nothing")
            continue
        gt_, gm = _read(gp)
        rt_, rm = _read(rp)
        if set(gt_) != set(rt_):
            only_g, only_r = sorted(set(gt_) - set(rt_)), sorted(set(rt_) - set(gt_))
            fatal.append(f"{label}/{stem}: key sets differ "
                         f"(+{only_g[:4]} -{only_r[:4]}; {len(gt_)} vs {len(rt_)})")
            continue
        bad = [k for k in gt_ if not torch.equal(gt_[k], rt_[k])]
        if bad:
            k = bad[0]
            fatal.append(f"{label}/{stem}: {len(bad)}/{len(gt_)} tensors differ, "
                         f"e.g. {k} max|d|={float((gt_[k].float()-rt_[k].float()).abs().max()):.3e}")
        gj = json.loads(gm.get("payload_metadata", "{}"))
        rj = json.loads(rm.get("payload_metadata", "{}"))
        diff = {k: (gj.get(k), rj.get(k)) for k in set(gj) | set(rj) if gj.get(k) != rj.get(k)}
        if diff:
            notes.append(f"{label}/{stem}: payload_metadata {sorted(diff)}")
    return fatal, notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ww")
    p.add_argument("--subset", default="hand-crafted", help="smallest cell by default")
    p.add_argument("--model", action="append", dest="models",
                   help="repeatable; default both backbones")
    p.add_argument("--n", type=int, default=3, help="trajectories to re-extract")
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep", action="store_true", help="keep the scratch tree")
    args = p.parse_args()

    cfg_path = REPO / "configs-main" / f"{args.dataset}.yaml"
    models = args.models or json.loads(
        subprocess.run([PY, "-c",
                        f"import yaml,json;print(json.dumps(yaml.safe_load(open({str(cfg_path)!r}))['models']))"],
                       capture_output=True, text=True, check=True).stdout)

    fatal: list[str] = []
    notes: list[str] = []
    for gt in (False, True):
        ref_root = REPO / ("outputs-gt" if gt else "outputs") / args.dataset
        if not (ref_root / "activations").exists():
            print(f"[skip] no reference tree at {ref_root}")
            continue
        scratch = REPO / (".parity-scratch-gt" if gt else ".parity-scratch")
        shutil.rmtree(scratch, ignore_errors=True)
        for model in models:
            print(f"\n=== {'with-GT' if gt else 'no-GT'} / {model} / {args.subset} "
                  f"({args.n} trajectories) ===")
            cmd = [PY, "-m", "main", "extract", "--config", str(cfg_path),
                   "--model", model, "--subset", args.subset,
                   "--device", args.device, "--start-idx", "0", "--end-idx", str(args.n),
                   "--set", f"results_base={scratch.name}"]
            if gt:
                cmd += ["--set", "gt=true"]
            r = subprocess.run(cmd, cwd=REPO)
            if r.returncode != 0:
                fatal.append(f"{'gt' if gt else 'nogt'}/{model}: extract exited "
                             f"{r.returncode}")
                continue
            ref_reps = ref_root / "activations" / model / args.subset
            stems = [p.stem for p in sorted(ref_reps.glob("*.safetensors"),
                                            key=lambda x: int(x.stem))][:args.n]
            tag = f"{'gt' if gt else 'nogt'}/{model}"
            for kind, got_dir, ref_dir in (
                    ("activations", scratch / args.dataset / "activations" / model / args.subset,
                     ref_reps),
                    ("attention", scratch / args.dataset / "attention" / model / args.subset,
                     ref_root / "attention" / model / args.subset)):
                f_, n_ = compare(got_dir, ref_dir, stems, f"{tag}/{kind}")
                fatal += f_
                notes += n_
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)

    print("\n" + "=" * 70)
    if notes:
        print(f"provenance-only drift vs the reference tree ({len(notes)}, not failures):")
        for x in notes:
            print("  ~", x)
        print("  (outputs/ predates the `gt` config field, the `ground_truth` metadata")
        print("   key, and the model-shorthand `--model`; none are read when scoring.)")
        print()
    if fatal:
        print(f"EXTRACTION PARITY FAILED — {len(fatal)} problem(s):")
        for x in fatal:
            print("  -", x)
        return 1
    print("EXTRACTION PARITY OK: every key set and every tensor is bit-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
