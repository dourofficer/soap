## What this is
 
Failure attribution in LLM multi-agent systems using a **proxy model's internal representations**. Given a failed trajectory (e.g., from Who&When, CORRECT-Error, and TraceElephant bechmarks), predict which step was the *decisive error* — the earliest step that irrecoverably derailed the run — and hence which agent is responsible.
 
## The method: Causal Residual Rescoring (CRR)
 
Prediction is the argmax over rescored per-step scores: `t̂ = argmax_t S̃(s_t)`.
 
CRR corrects an off-the-shelf per-step base score `S(s_t)` by discounting the part it inherits from causally upstream steps:
 
> `S̃(s_t) = S(s_t) − γ · Σ_{i<t} w_{i,t} · S(s_i)`,  with `S̃(s_1) = S(s_1)`.
 
- `S(s_t)` — **base score** (higher = more error-like). Primary form is the SVD projection score: `S(s_t) = −(1/|C|) Σ_{c∈C} ⟨ṽ_t, V_{:,c}⟩²`, the negated mean-squared projection of the step embedding `v_t` onto a mid-spectrum band `C` of right singular vectors of a reference matrix (erroneous steps project *smaller*, hence the negation). `ṽ_t` is optionally mean-centered. A norm baseline swaps in `‖v_t‖₂`.
- `w_{i,t}` — **attention-mass dependency weight**: the fraction of step-`t` query attention that lands in predecessor step `i`, `m_{i,t} = mean over T_t of Σ_{q∈T_i} A_{p,q}`, head-averaged over a layer band, then renormalized over predecessors so `Σ_{i<t} w_{i,t} = 1` (softmax-sharpening variant also supported).
- `γ ∈ [0,1]` — **discount strength**. `γ=0` recovers the base scorer; `γ=1` subtracts a full attention-weighted predecessor mean. **Single-pass**: the discount reads the *original* `S(s_i)`, not `S̃`, so corrections don't cascade (a cascading variant is only an ablation).
Base scores rank *within a trajectory* — never across trajectories. Evaluation reports step@k / agent@k.
 
## The pipeline (and where each stage lives)
 
1. **Per-step representations** — one forward pass of the proxy model over each step in context; pool its hidden states to one vector `v_t`. → `src/activations/`
2. **Base score `S(s_t)`** — fit SVD on a reference split, score each step by its ranged projection (or norm). → `src/svd/`
3. **Attention weights `w_{i,t}`** — extract per-step attention mass into predecessors. → `src/attention/`
4. **CRR rescoring** — aggregate attention over a layer range, orient the base scores to "higher = error", apply the discount. → `src/rescore/`
5. **Evaluate / tabulate** — per-trajectory ranking, best-config selection, result tables. → `experiments/reports/`
6. **Reproduce / apply** — re-run the full SVD→orient→discount pipeline for a chosen (optimal) config, either to *validate* the reduced-table metrics or to *apply* the frozen config and score all trajectories. → `src/rescore/reproduce.py`, driven by `experiments/reproduce/`.

The whole flow is driven end-to-end by the `scripts/` wrappers (`gen_embeddings.sh`, `extract_attention.sh`, `run_analysis.sh`, `reproduce.sh`) — see `scripts/README.md`.
## Repository layout: the `src/` ↔ `experiments/` split
 
The repo is deliberately **two-layered**, and this is the main thing to internalize before editing:
 
- **`src/` = core logic.** Pure, config-free functions and CLIs. Each stage is a package with the algorithm and a thin `main()` you can invoke directly for one model/subset.
- **`experiments/` = orchestration.** For (almost) every `src/<stage>` there is an `experiments/<stage>/` holding `configs/*.yaml` (the sweep space) and a sweep driver (e.g., `sweep.py`). The driver loads a YAML, expands the grid, and shells out to a runner (which imports from `src/`). Nothing in `experiments/` reimplements algorithm logic — it only chooses arguments and manages runs.
So: **change behavior in `src/`, change what gets run in `experiments/`.** When you add a `src/` capability, add the matching config axis + sweep wiring in `experiments/`.
 
Sweep drivers share an interface: `python -m experiments.<stage>.<sweep_name> --config <yaml> [--set key.subkey=value ...] [--dry-run]`. `--set` does dot-path overrides; `--dry-run` prints the commands without running.
 
Three pieces make the orchestration DRY:
- **`experiments/_common/`** — shared orchestration: `config.py` (the one canonical `load_cfg` + dataset-manifest merge), `paths.py` (derives every stage's output root from `(dataset, split-ratio)`, reproducing the `outputs-<ds>/…/<tag>/` layout), `sweep.py` (the `format_command`/`run`/`run_grid` engine). Nothing else re-implements config loading or path building.
- **`experiments/datasets/<ds>.yaml`** — the **single source of truth** per dataset (`models`, `model_paths`, `subsets`, `data_root`, `max_tokens`, `splits`, `seeds`). Stage configs are now *thin*: `dataset: <ds>` plus only that stage's sweep axes; the manifest fills in the rest and output roots are **derived** (never hand-written). Precedence: manifest < stage config < `--set`.
- **`experiments/legacy/` and `experiments/reports/legacy/`** — archived superseded scripts/configs (v1 table builders, summary builders, one-offs, ww numbered/`default`/`seeding` configs, `run_all_positions.py`). Not on any active path; kept for reference.
 
`experiments/` also has: `reports/` (the two v2 table builders + `build_main_tables.py`; the figure illustrator `draw_trajectories.py` now lives in `reports/legacy/`), `reproduce/` (the apply/validate-a-frozen-config stage), and `notebooks/` (you MUST NOT read as they are just gibberish code)
 
## Datasets, configs & running
 
The mental model in one place: **manifest → thin stage config → derived paths → wrapper script.**
- One manifest per dataset (`experiments/datasets/<ds>.yaml`) holds everything shared across stages. Thin stage configs add only their sweep axes and reference it via `dataset: <ds>`.
- Output roots are computed by `experiments/_common/paths.py` from `(dataset, split-tag)` — you never write `outputs-<ds>/…/<tag>/` paths by hand.
- Run via the `scripts/*.sh` wrappers, controlled by env knobs `DATASET`, `MODELS`, `GPU`, `DRY_RUN`, `EXTRA_SET` (the last forwards `--set` overrides). See `scripts/README.md` for examples and `scripts/_common.sh`'s header for onboarding a new dataset (write one manifest + six thin configs).
 
## Core modules (in `src/`)
 
- **`src/data/`** dictates how a trajectory is represented, how a dataset of trajectories is loaded, how the the context for a step is handled (e.g., context length, step selection, etc).
- **`src/activations/`** handles the representation extraction logic, which runs forward passes and write one `.safetensors` per trajectory with flat keys `"{step}.{pool}.{shorthand}"`.
- **`src/attention/`** — two extractors with **identical output schemas**, differing only in how they get the attention mass. `streaming.py` (the **live** extractor) re-runs `q_proj`/`k_proj` + RoPE in a pre-hook and never forms `(N,N)` — memory drops from `O(H·N²)` to `O(H·|T_t|·N)`, needed for Qwen3-14B and multi-GPU. `legacy/eager.py` (the **validation reference**) uses HF eager attention + `output_attentions=True` and reduces in a forward hook — exact and simple but materializes `(N,N)`; use it only to check `streaming` agrees to ~1e-4 on a short trajectory. Shared helpers live in `_common.py` (`build_key_mask`) and `src/utils/metadata.py`; both extractors take a resolved `--model-path` (from the manifest `model_paths`) — there is no hardcoded model dict. Both are wired for Llama-3.1 and Qwen3 RoPE (and Qwen3 per-head q/k norm) only; a new architecture needs the RoPE/norm helpers swapped, then re-validate.
- **`src/svd/`** — `core.py` has the scoring primitive `ranged_projection_svd` (other variants can be safely ignored); `computation.py` fits raw+centered SVD and scores across the config grid (`fit_all`, `score_all`, `precompute_svd`); `score.py` is the per-`(model, subset, pooling, seed)` **base-score runner** the svd sweep shells out to (was `experiments/svd/run_all_positions.py`); `reproduce.py` regenerates the **base score** w.r.t a specific config for spot-checking (SVD only — the full-CRR reproducer is in `src/rescore/`).
- **`src/rescore/`** — `weights.py` loads the attention `.safetensors` and aggregates per layer-range into `process_weighting`-compatible dicts (`aggregate_attn`, `layer_ranges`); `discount.py` is the CRR math: `orient_svd_scores()` flips SVD's "lower = error" to "higher = error" (`negate` / `inverse` / `sigmoid`) and `apply_discount()` runs the single-pass discount (assumes scores are already "higher = error" — callers must orient first). `run.py` is the per-`(model, subset)` **CRR sweep runner** the rescore sweep shells out to. `reproduce.py` is the **full-CRR reproducer** (`reproduce_crr`): it composes `svd/reproduce` + `weights` + `discount` to apply/validate a frozen optimal config — distinct from svd's base-score reproducer.
- **`src/utils/`** contains helpers for storing trajectories, evaluation metrics (`compute_metrics`, `get_mistake_meta`), the shared `.safetensors` trajectory-metadata header (`metadata.py`), and others.
## Navigating the data
 
- Layout: `data/ww/<subset>/<id>.json`, one trajectory per file. Subsets present: `algorithm-generated` (126), `hand-crafted` (58). The current scope only concerns `algorithm-generated` and `hand-crafted`, however, as the project expands new datasets will be added.
- Each JSON has: `history` (the ordered turns; `step t == history[t]`, **0-indexed**), `question` / `question_ID`, `ground_truth` (final answer), `mistake_agent`, `mistake_step` (string index into `history`), `mistake_reason`, `level`. See any file for the exact shape.
- Turn roles are agent/system names (e.g. `WebSurfer`, `Orchestrator (thought)`, `Orchestrator (-> WebSurfer)`); `standardize_role` normalizes them, and `mistake_agent` matches a turn's `role`. Not every turn is scoreable — `iter_scoreable_steps()` filters (e.g. skips the initial human turn / planning scaffolding).
- Trajectory lengths vary widely: algorithm-generated ≤10 steps; hand-crafted ~50 average, up to ~130. This is why memory-efficient attention extraction matters and why ranking is per-trajectory.
## Conventions that bite
 
- **Layer-index off-by-one between stages.** Activations store `L+1` rows: index 0 = `embed`, `act/k` = tuple index `k+1`, plus a final `act/{N-1}_normed`. Attention stores `L` rows: index 0 = `layers[0].self_attn`, `act/k` = index `k`. Mixing an activation weight source and an attention weight source in one sweep is where this bites.
- **Two sign conventions.** SVD projection scores are "lower = error"; CRR's `apply_discount` wants "higher = error". Always route SVD scores through `orient_svd_scores` first. Check which convention any score is in before comparing across modules.
- **Split ratio is a single source of truth now.** It lives once in the dataset manifest (`splits:`) and is *derived* into the split-tag (`0.3/0.2/0.5 → 325`) that names every split-tagged output root, so svd and rescore stay in sync automatically. When you reuse an existing output, seed + ratio must still match the `<tag>` on that path.
- **Model paths come from the manifest `model_paths`.** One registry per dataset — no per-module `MODELS`/`HUB` dicts. The activations/attention runners take a resolved `--model-path`; the sweep drivers resolve the shorthand.
- **Run from repo root as `src.…` / `experiments.…`.** The code is in `src/` (and `experiments/`); imports are `from src.…` / `from experiments.…`, so **invoke via `python -m` from the repo root** (`pyproject.toml`'s `packages` is set to `["src", "experiments"]`).