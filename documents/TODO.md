# TODO — rewire `baselines/` and `datagen/` imports to the v2 layout

## Context

`baselines/` and `datagen/` were moved from `v1/` to the repo root but still import against
the **v1** code layout (`src.utils.*`, `src.models` as a package, `experiments._common`).
At the root, `src/` is now **v2**, whose module structure differs, so a few imports no
longer resolve. This is the checklist to point them at v2 equivalents so both packages run
from the root via `python -m baselines.…` / `python -m datagen.…`.

Scope: **only 5 import sites** (verified by grep — there are no other `src.`/`experiments.`
references). Both packages already have `__init__.py`, so they run as packages from the
root once the imports resolve. **Do not change anything under `src/`** — v2 is the source of
truth; only edit files in `baselines/` and `datagen/`.

## Import map (v1 → v2)

| v1 import | v2 equivalent | same signature? |
|---|---|---|
| `src.utils.utils.split_data` | `src.stores.split_data` | yes (identical body) |
| `src.utils.utils.standardize_role` | `src.metrics.standardize_role` | yes (identical) |
| `experiments._common.config.load_yaml` | `src.common.config.load_yaml` | yes (identical) |
| `src.models.get_adapter` | `src.models.get_adapter` | **no change** — v2 provides it |
| `src.utils.common._get_sorted_json_files` | `src.data.trajectory._sorted_json_files` | yes, but **private + renamed** |
| `src.utils.common._load_json_data` | `src.data.trajectory._load_json` | **behaviour differs** (see gotcha) |

## Part A — the import edits (makes them import-clean)

1. **`baselines/prompting/report.py:41`**
   ```python
   # from src.utils.utils import split_data, standardize_role
   from src.stores  import split_data
   from src.metrics import standardize_role
   ```

2. **`baselines/prompting/engine.py:24`** — **no change needed.** v2 exposes
   `src.models.get_adapter` (single-file module), and its `ModelAdapter.template_kwargs()`
   is what `engine.py` calls (`get_adapter(model_path).template_kwargs()`), so the import
   and usage are already compatible. Just re-verify after the other edits (see Verification).

3. **`datagen/common.py:20`**
   ```python
   # from experiments._common.config import load_yaml
   from src.common.config import load_yaml
   ```
   Note the `# noqa: E402` — this import sits after a `sys.path` bootstrap in `common.py`.
   Confirm that bootstrap inserts the **repo root** (now the v2 root) and not a hardcoded
   `v1/` path; if it hardcodes a path, update it to the repo root so `src.common` resolves.

4. **`baselines/prompting/predict.py:29`** and **`baselines/correct/schemagen.py:35`** —
   both do `from src.utils.common import _get_sorted_json_files, _load_json_data`. See the
   gotcha below; pick one option and apply it to both files.

## Gotcha — the two JSON helpers (`predict.py`, `schemagen.py`)

v2's equivalents live in `src/data/trajectory.py` but are **private, renamed, and one
behaves differently**:
- `_get_sorted_json_files(dir)` ↔ `_sorted_json_files(dir)` — same behaviour (numeric-sorted
  `*.json` filenames).
- `_load_json_data(path)` **returns `None` on a malformed/missing file** (v1 wraps in
  try/except); v2's `_load_json(path)` **raises**. `predict.py` uses the result as
  `data = _load_json_data(Path(directory)/fn)` then consumes `data`, so a swallowed `None`
  vs an exception is a real behavioural change if any input file is bad.

Two options — **prefer Option B** (stable, preserves behaviour, no dependency on v2 private
names that may churn):

- **Option A (import v2 privates):**
  ```python
  from src.data.trajectory import _sorted_json_files as _get_sorted_json_files, \
                                  _load_json as _load_json_data
  ```
  Accepts v2's raise-on-bad-file semantics; brittle (imports underscore-privates).

- **Option B (vendor a 6-line local helper) — recommended:** add
  `baselines/_io.py` and import from it in both files:
  ```python
  # baselines/_io.py
  import json, os
  from pathlib import Path
  def sorted_json_files(directory) -> list[str]:
      files = [f for f in os.listdir(directory) if f.endswith(".json")]
      return sorted(files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
  def load_json_data(path):
      try:
          with open(path, encoding="utf-8") as f:
              return json.load(f)
      except Exception:
          return None          # preserves v1's None-on-error contract
  ```
  then in `predict.py` / `schemagen.py`:
  `from baselines._io import sorted_json_files as _get_sorted_json_files, load_json_data as _load_json_data`.

## Part B — to actually RUN against v2 outputs (do after Part A)

Import-clean ≠ runnable end-to-end. The baseline **report** configs still point at the v1
output layout and must be repointed at v2's (`src/common/paths.py`) tree:

- `outputs-<ds>/activations` → `outputs/<ds>/activations`
- `outputs-<ds>/{prompting,chief,correct}` (predictions) → `outputs/<ds>/baselines/<b>/…`
- `outputs-<ds>/discounted-splits/reduced/325` → `outputs/<ds>/reduced/325`
- `out_root: outputs-<ds>/…-reports` → wherever you want v2 report outputs

Affected: `baselines/*/configs/*.yaml` (grep `outputs-`, `reps_root`, `pred_root`,
`crr_reduced_root`). This is config-only; no code. (If you only need the baseline *numbers*
in the main table, note that `src/reports/main_table.py` already scores the prediction
JSONLs directly from `outputs/<ds>/baselines/…` — you may not need the standalone baseline
report at all.)

`datagen` has no output-path config coupling to v2; it reads models from `../hub` and writes
`data/synthetic/` — both already correct from the root. Its only code dependency is the
single `load_yaml` import in Part A.

## Verification (after each part)

Per edited file, import-resolve without running the model:
```bash
python -c "import baselines.prompting.report, baselines.prompting.predict, \
                  baselines.prompting.engine, baselines.correct.schemagen; print('baselines import OK')"
python -c "import datagen.common; print('datagen import OK')"
```
Behavioural spot-checks:
- `report.py`: `split_data([... ], 0.5, 1)` and `standardize_role('Orchestrator (thought)')`
  return the same as before (v2 `split_data`/`standardize_role` are byte-identical ports).
- `engine.py`: `get_adapter('<a model path>').template_kwargs()` returns a dict (e.g.
  `{'enable_thinking': False}` for Qwen3.5).
- Part B: run one baseline report (e.g. `python -m baselines.prompting.report --config
  baselines/prompting/configs/report_ww.yaml --check-only`) and confirm it finds the reps
  and prediction files at the v2 paths.

## Do NOT

- Change anything under `src/` (v2). Only `baselines/` and `datagen/` files.
- Re-run baseline LLM inference — the prediction JSONLs under `outputs/<ds>/baselines/` are
  reused as-is; only scoring/reporting is re-run.
