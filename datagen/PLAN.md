# Synthetic trajectory-generation pipeline (`datagen/`)

## Context

The SVD+CRR pipeline currently splits each benchmark (ww, traceelephant,
correct-full) 3/2/5 train/val/test. Two bottlenecks: the train split is
unlabeled reference data "wasted" from scarce benchmarks, and val is too small
for reliable hyperparameter selection. Fix: a synthetic generation pipeline in
two phases —

- **Phase 1 — trajectory generation**: run Magentic-One and Captain-Agent
  (the vendored clone at `datagen/TraceElephant/`; paper at
  `datagen/TraceElephant/traceelephant.pdf`) over comprehensive question
  pools with deepseek-8b / qwen3.5-9b agent backbones served via vLLM, judge
  success/failure, convert failures into an **unlabeled corpus** for SVD
  fitting. This alone is sufficient for the experimental setup where each
  benchmark is split val/test = 50/50 (SVD fit = synthetic, hyperparameter
  selection = benchmark val, report = benchmark test).
- **Phase 2 — labeling (later)**: CORRECT-style **error injection** into
  successful Phase-1 trajectories to produce a labeled synthetic validation
  set, plus AgenTracer-style **counterfactual replay** as a label-quality
  audit. How many trajectories get injected / replay-audited is a config
  knob, not hardcoded.

All new code lives inside `datagen/` (the harness clone only receives two thin
driver scripts). Nothing in `src/` or `experiments/` changes in Phase 1 except
adding dataset manifests + thin stage configs for the new data.

### User decisions (2026-07-19)

1. Labeled val = hybrid: injection for the bulk + a replay-verified audit
   subset (both Phase 2; sizes config-driven).
2. Question pools = comprehensive (GAIA, AssistantBench, HotpotQA, 2WikiMQA,
   Musique, ARC, MMLU-Pro, Math500, GSM8K); what feeds SVD/CRR is filtered
   downstream, not at collection time.
3. Both harnesses (Magentic-One + Captain-Agent) from day 1.
4. Injector/judge model = qwen3.5-9b / deepseek-8b endpoints for now; every
   model role reads a configurable endpoint so larger local models swap in
   later.

### What the papers established (read this session)

- **CORRECT** (papers/correct.pdf, §4.1 "Bootstrap Error Synthesis", App. A.9
  pp.15–16): builds CORRECT-Error by **error injection into successful
  Magentic-One(AutoGen-variant) trajectories** — an LLM picks a step and
  rewrites it into a realistic error via a two-prompt flow (injection-strategy
  prompt → message-modification prompt); the injected step is a free
  ground-truth label (`mistake_step`/`agent`/`reason` by construction); a
  verification checklist enforces realism incl. "leads to incorrect final
  answer?"; special rule: for consecutive CodeExecutor steps modify only the
  final step. Text-level edit, no re-execution. Human seeds are used only for
  schema matching (BGE-M3 similarity). Blind human study: synthetic errors
  ≈ indistinguishable from real. No Captain-Agent anywhere in the paper.
- **AgenTracer** (papers/agentracer.pdf, pp.4–5, App. B pp.16–18): two
  branches, both gated by re-running the outcome judge Ω — (1) counterfactual
  replay on natural failures: an analyzer proposes a minimally-invasive
  corrected action at step t (without leaking the solution), the system
  re-simulates downstream, and the earliest step whose correction flips
  failure→success is the label; (2) fault injection on successes, kept only
  if the outcome actually flips to failure. Decisive error = earliest
  counterfactually outcome-flipping step (same definition in both papers).
  Neither Magentic-One nor Captain-Agent generates its data.
- **TraceElephant repo** (`datagen/TraceElephant/`): runnable Magentic-One
  (`code/agent_system/Magentic-One/`, autogen-agentchat 0.4+, drivers
  `run_gaia_benchmark.py` / `run_assistant_bench.py`, in-process per-step LLM
  logging via monkeypatched `OpenAIChatCompletionClient.create`, `.env`
  config `OPENAI_API_BASE`/`OPENAI_API_KEY`/`M1_MODEL`) and Captain-Agent
  (`code/agent_system/Captain-Agent/`, vendored AutoGen-0.2 fork, drivers
  `scripts/run_gaia.py` / `run_assistantbench.py`, `OAI_CONFIG_LIST` json
  with `api_type: openai` + `base_url` — default `--model` is already
  qwen3_235b, so non-OpenAI endpoints are proven; Serper search key). Both
  → a vLLM OpenAI-compatible endpoint plugs in directly. Per-run success
  signals exist (`judge.json` / LLM judge). **No injection or
  auto-annotation pipeline** (labels were human); the LLM-judge attribution
  prompts in `code/trace_locate/lib/utils.py` are reusable.

## Directory layout

```
datagen/
├── TraceElephant/                    # vendored clone (only run_pool.py drivers added inside)
│   └── code/agent_system/
│       ├── Magentic-One/run_pool.py           # NEW (own venv)
│       └── Captain-Agent/scripts/run_pool.py  # NEW (own venv)
├── common.py                 # sys.path bootstrap + config loading (reuse experiments/_common/config.load_yaml),
│                             #   pattern: exp-synthetic-correct/common.py:14-17
├── configs/
│   ├── serve.yaml            # vLLM endpoint registry (single source of truth: base_url/model/port)
│   ├── collect.yaml          # harness × backbone × pool matrix, concurrency, timeouts, judge endpoint
│   ├── inject.yaml           # Phase 2: injector/verifier endpoints, taxonomy weights, cascade mode, n_inject
│   ├── replay.yaml           # Phase 2: oracle endpoint, n_audit, per-harness replay mode
│   └── oai_config_list.json  # generated: Captain-Agent OAI_CONFIG_LIST for the vLLM endpoints
├── serve/launch_vllm.sh
├── pools/prepare.py          # → datagen/pools/data/<pool>.jsonl
├── collect/run_batch.py      # orchestrator: matrix expansion, subprocess per task, resumable
├── judge/{rejudge.py, checkers.py}
├── inject/{taxonomy.py, strategy.py, modify.py, verify.py, run_inject.py}   # Phase 2
├── replay/{replay_captain.py, replay_magentic.py, audit.py}                # Phase 2
├── convert.py                # raw runs → data/synthetic*, pattern: data/convert_traceelephant.py
└── runs/<harness>/<backbone>/<pool>/<task_id>-r<rep>/   # summary.json, llm_steps/, verdict.json
```

Venv reality (verified): Magentic-One needs autogen-agentchat/autogen-ext
0.4+/openai≥1.59; Captain-Agent imports the vendored `autogen` 0.2 fork from
its repo root; attribscope's venv has transformers/vllm. Three environments —
the orchestrator invokes each harness driver with that harness's own
`.venv/bin/python` (paths in `collect.yaml`).

═══════════════════════════════════════════════════════════════════
# PHASE 1 — Trajectory generation
═══════════════════════════════════════════════════════════════════

## A. Serving (vLLM)

`datagen/serve/launch_vllm.sh` (attribscope venv already has vllm≥0.19):

```bash
# GPU 6 — qwen3.5-9b
vllm serve ../hub/Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b --port 8001 \
  --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser hermes
  # VERIFY at smoke: parser choice for Qwen3.5; add --reasoning-parser if hybrid-thinking
# GPU 7 — deepseek-8b
vllm serve ../hub/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-8b --port 8002 --max-model-len 32768 \
  --reasoning-parser deepseek_r1     # NO tool parser: R1-distill isn't tool-call trained;
                                     # the reasoning parser keeps <think> out of logged content
```

GPUs 0–5 stay free for activations/attention extraction. Scale = second
replica per model + round-robin in `collect.yaml` if throughput-bound.

Integration facts (verified in the harness code):
- Magentic-One's `OpenAIChatCompletionClient` is built with no `model_info`
  (`run_assistant_bench.py:411-416`); autogen-ext 0.4+ raises for
  non-registry model names → `run_pool.py` must pass
  `model_info={vision: False, function_calling: True (False for deepseek-8b),
  json_output: True, family: "unknown", structured_output: False,
  multiple_system_messages: True}`. The MagenticOne orchestrator uses
  JSON-prompted ledgers (not function calls) — verify lenient parsing on
  deepseek-8b at the M1 smoke.
- **MultimodalWebSurfer must be excluded** (sends screenshots, checks
  `model_info["vision"]`, needs the Patchright browser block,
  `run_assistant_bench.py:348-374`). Team is hardcoded at
  `run_assistant_bench.py:495-499` → `run_pool.py` adds `--agents`:
  `fs,coder,executor` for qwen3.5-9b; `coder,executor` for deepseek-8b
  (FileSurfer needs function calling). Consequence: web-dependent questions
  mostly fail — acceptable; failures are exactly what the unlabeled corpus
  wants, and injection candidates come from math/MCQ/QA pools.
- Captain-Agent: `_build_llm_configs` (`scripts/run_assistantbench.py:55-91`)
  loads `OAI_CONFIG_LIST` via `config_list_from_json` with
  `filter_dict={"model": [args.model]}`. vLLM entry:
  `{"model": "qwen3.5-9b", "base_url": "http://127.0.0.1:8001/v1",
  "api_key": "EMPTY", "api_type": "openai", "tags": ["qwen3.5-9b"]}`.
  Captain agents use markdown code blocks + a retrieved tool library — no
  native function calling → the friendlier harness for deepseek-8b. Needs
  `SERPER_API_KEY` for web pools; downloads all-MiniLM-L6-v2 for tool
  retrieval.

## B. Question pools

Uniform jsonl at `datagen/pools/data/<pool>.jsonl`:

```json
{"id": "gsm8k-dev-0017", "question": "...", "answer": "72", "pool": "gsm8k",
 "level": null, "answer_type": "numeric", "file_path": null}
```

`answer_type ∈ {numeric, mcq, exact, open}` drives judging. `prepare.py
--pool <name>` for: gsm8k, math500, arc (Challenge), mmlu-pro, hotpotqa,
2wikimqa, musique (HF `datasets`, already a dep); gaia (miromind gaia-val.zip
per Magentic-One README; keep `file_path`, set `GAIA_DATA_DIR`);
assistantbench (HF dev split). Neither harness ships the pool jsonls —
prep scripts are mandatory first-milestone work.

## C. Generic drivers (the only edits inside the clone)

- `Magentic-One/run_pool.py` — copy of `run_assistant_bench.py` with:
  uniform task fields; `--agents` flag; **`--task-index` single-task mode**
  (the logging monkeypatch uses process-globals `llm_call_logs` /
  `current_steps_dir` — one task per process kills cross-task bleed and makes
  timeouts clean); `--output-dir` from the orchestrator; `model_info`
  override; keep `format_history` untouched (its `summary.json` —
  `history:[{role, content}]` with roles `human` / `Orchestrator (thought)` /
  `Orchestrator (-> X)` — is already ~the attribscope schema); generic prompt
  template = the existing `## ANSWER`/`## REASON` contract + GAIA file-path
  prefix when `file_path` set.
- `Captain-Agent/scripts/run_pool.py` — copy of `scripts/run_assistantbench.py`
  (already single-task via `--task-index`) with uniform fields,
  `--output-dir`, and a `--build-state` reuse knob (default: one autobuild
  per (pool, backbone) to amortize cost; `--fresh-build` for diversity).

## D. Collection orchestration

`datagen/collect/run_batch.py` + `configs/collect.yaml`:

```yaml
endpoints:
  qwen3.5-9b:  {base_url: "http://127.0.0.1:8001/v1", model: qwen3.5-9b}
  deepseek-8b: {base_url: "http://127.0.0.1:8002/v1", model: deepseek-8b}
harnesses:
  magentic: {python: datagen/TraceElephant/code/agent_system/Magentic-One/.venv/bin/python,
             driver: run_pool.py, cwd: .../Magentic-One,
             agents: {qwen3.5-9b: "fs,coder,executor", deepseek-8b: "coder,executor"}}
  captain:  {python: .../Captain-Agent/.venv/bin/python, driver: scripts/run_pool.py,
             cwd: .../Captain-Agent, extra_env: {SERPER_API_KEY: "..."}}
matrix:
  - {harness: [magentic, captain], backbone: [qwen3.5-9b, deepseek-8b],
     pool: [gsm8k, math500, arc, mmlu-pro, hotpotqa, 2wikimqa, musique], limit: 500, reps: 1}
  - {harness: [magentic, captain], backbone: [qwen3.5-9b, deepseek-8b],
     pool: [gaia, assistantbench], reps: 2}
judge: {base_url: "http://127.0.0.1:8001/v1", model: qwen3.5-9b}   # config knob
concurrency: 24
timeout_s: {default: 900, gaia: 1800, assistantbench: 1800}
max_round: 20
out_root: datagen/runs
```

Mechanics: expand matrix → task list; ThreadPoolExecutor of `concurrency`
launches one subprocess per (harness, backbone, pool, task_id, rep) with env
`OPENAI_API_BASE` / `OPENAI_API_KEY=EMPTY` / `M1_MODEL` (dotenv in the
drivers does not override pre-set env, so the clone's `.env` files stay
untouched). Resumable: done iff `<run_dir>/summary.json` exists; timeout →
SIGKILL + marker + 1 retry. `manifest.jsonl` appended per completed run
(harness, backbone, pool, task_id, wall time, verdict) is the corpus ledger.
Cost controls: `max_round`, per-task timeout, vLLM `--max-model-len`.

## E. Success/failure judging

Do NOT trust in-run judges (Magentic-One's is permissive substring matching,
`run_gaia_benchmark.py:411-420`). Post-hoc `datagen/judge/rejudge.py` reads
`summary.json`, re-extracts `## ANSWER`, dispatches on `answer_type`:
- numeric (gsm8k, math500): normalize + math_verify/sympy equivalence.
- mcq (arc, mmlu-pro): option/letter match after normalization.
- exact/open QA (hotpotqa, 2wikimqa, musique): EM + token-F1 threshold;
  LLM judge on near-misses.
- open (gaia, assistantbench): LLM judge — reuse the strict-JSON prompt from
  Captain's `_llm_judge_answer` (`scripts/run_assistantbench.py:369-391`),
  plain `openai` client against the configurable `judge:` endpoint.

Writes `<run_dir>/verdict.json` `{is_correct, method, extracted, reason}` —
this, not `judge.json`, routes downstream: failure → unlabeled corpus;
success → Phase-2 injection candidate (kept on disk either way).

## F. Trajectory schemas (raw run dir → converted dataset)

### F.1 Raw run directory (written by the harness drivers)

`datagen/runs/<harness>/<backbone>/<pool>/<task_id>-r<rep>/`

| File | Written by | Content |
|---|---|---|
| `summary.json` | driver | the conversation trace (**load-bearing**; schemas below) |
| `judge.json` | driver | the harness's own correctness guess — **untrusted**, superseded by `verdict.json` |
| `verdict.json` | `judge/rejudge.py` | `{is_correct, method, extracted, reason}` — the routing signal |
| `llm_steps/step_{N}.json` | Magentic-One only | one file per real LLM call: `{request, response, timestamp}` (monkeypatch at `run_assistant_bench.py:322-334`), plus `llm_steps/images/`. Raw provenance; not consumed downstream |
| `timeout.marker` | orchestrator | present iff the run was killed on timeout |

**Magentic-One `summary.json`** (from `format_history`, `run_assistant_bench.py:684-692`):

```json
{
  "history": [
    {"role": "human",                            "content": "..."},
    {"role": "Orchestrator (thought)",           "content": "..."},
    {"role": "Orchestrator (-> WebSurfer)",      "content": "..."},
    {"role": "WebSurfer",                        "content": "..."},
    {"role": "Orchestrator (termination condition)", "content": "..."}
  ],
  "question": "<task text>", "ground_truth": "<answer>",
  "question_ID": "<task id>", "is_corrected": false
}
```

Roles are *synthesized* by `format_history`: `human` (source `user`);
`Orchestrator (thought)` when an orchestrator message matches ledger keywords
(GIVEN OR VERIFIED FACTS / Here is the plan / Updated Ledger / fact sheet /
Next speaker); `Orchestrator (-> <NextAgent>)` for instructions, inferred by
peeking at the *next* message's source; the bare agent name for agent replies;
`Orchestrator (termination condition)` for the end-of-run object. These are
exactly the ww role conventions — `standardize_role` and
`iter_scoreable_steps` already handle them.

**Captain-Agent `summary.json`** (`scripts/run_assistantbench.py:443-456`):

```json
{
  "is_correct": false, "question": "...", "question_ID": "...",
  "difficulty": "...", "ground_truth": "...", "gold_url": "", "explanation": "",
  "history": [{"content": "...", "name": "Expert_Agent"}],
  "agents": [ ... built-team info ... ]
}
```

Note the difference that drives converter branching: Captain turns are
`{content, name}` (the `role` field is deliberately stripped at
`run_assistantbench.py:155-190`, agent lives under `name`, empty-content
messages already dropped); Magentic-One turns are `{role, content}`.

### F.2 Converted trajectory (`datagen/convert.py` → `data/synthetic*/<N>.json`)

The schema `src.data.trajectory.load_dataset` consumes (verified against a
real `data/traceelephant/magentic/0.json`):

```json
{
  "question_ID":   "gsm8k/dev-0017-r0",
  "question":      "...",
  "ground_truth":  "72",
  "history":       [{"role": "...", "content": "..."}],
  "mistake_agent": "",
  "mistake_step":  -1,
  "mistake_reason": "",
  "level":         -1,
  "system":        "<agent system intro>",
  "subset":        "magentic-qwen",
  "pool":          "gsm8k"
}
```

- Magentic-One's `summary.json` is already ~the target schema (same
  `{role, content}` history, `question`, `ground_truth`, `question_ID`) —
  conversion mainly adds mistake/subset/level fields. Captain needs
  `name → role` remapping.
- `mistake_step` is **0-indexed into `history`**; `-1` + empty
  `mistake_agent` marks unlabeled (Phase 1 corpus). Phase-2 injected
  trajectories carry the real triple, with the invariant
  `history[mistake_step]["role"] == mistake_agent` (enforced exactly as in
  `data/convert_traceelephant.py:166-169`).
- `pool` is an extra key; the loader ignores unknown keys.

## G. Conversion + onboarding

`datagen/convert.py` (pattern + validation from `data/convert_traceelephant.py`):
- magentic runs: `summary.json.history` already `[{role, content}]` in ww
  role conventions; reuse tool-call serialization/truncation where needed.
- captain runs: history entries are `{content, name}`
  (`scripts/run_assistantbench.py:154-190`) → map name→role, drop empties.
- Fields: `question_ID` = `<pool>/<task_id>-r<rep>`, `question`,
  `ground_truth`, `level` (pool difficulty or -1), `system`, `subset`;
  unlabeled → `mistake_step=-1`, `mistake_agent=""`. Keep a `pool` extra key
  (loader ignores unknown keys) + `filename_map.csv` provenance (precedent:
  `merge_correct_full.py`).
- Filters: `--outcome {fail,success,all}` (default fail — matches the
  all-failure reference distribution of the benchmarks), max-steps cap,
  dedup by (pool id, rep), `--exclude-pools gaia` knob (GAIA questions
  underlie benchmark test trajectories — distributional overlap; report SVD
  fits with/without).
- **Never name a subset `algorithm-generated`** — special-cased at
  `src/data/trajectory.py:97-105` (reads `system_prompt`).

**Dataset naming — two datasets, subsets encode harness-backbone** (the
generator backbone is orthogonal to the manifest `models:` field, which lists
the PROXY extraction models):
- `data/synthetic/{magentic-qwen, magentic-dsr1, captain-qwen, captain-dsr1,
  mixed}/<N>.json` — unlabeled failure corpus; `mixed` = converter-
  materialized merge (renumbered + filename_map.csv) so a combined SVD fit
  needs no loader changes.
- `data/synthetic-err/{...same subsets...}/<N>.json` — Phase-2 injected
  labeled set (naming precedent: correct-error).

Manifests `experiments/datasets/{synthetic,synthetic-err}.yaml`:
`models: [deepseek-8b, qwen3.5-9b]`, `model_paths` as in
`traceelephant.yaml:22-26`, `max_tokens: 8192`, `splits: 3/2/5` (tag
derivation only), `seeds: [1,2,3]` + the six thin stage configs per the
`scripts/_common.sh` onboarding header. Then
`DATASET=synthetic ./scripts/gen_embeddings.sh` etc.

**Verified safety/cost facts:**
- Extraction never reads mistake fields; `mistake_step="-1"` round-trips
  through the safetensors header (`src/utils/metadata.py:16-25`).
- `compute_metrics` skips `mistake_step=None` trajectories in hits but keeps
  them in the `total_trajs` denominator (`src/utils/utils.py:290-299`) →
  the unlabeled corpus is **fit-only**; never put unlabeled trajectories in a
  metric split.
- Fit-only data needs **activations but NOT attention** (attention is
  consumed only by rescoring on val/test) — only `synthetic-err` ever needs
  `extract_attention.sh`. Large compute saving.

## Phase-1 milestones

| M | Deliverable | Verification | Scale |
|---|---|---|---|
| 1.0 | vLLM launch + endpoint registry + pool prep (gsm8k, hotpotqa, gaia first) | `curl :8001/v1/models`; tool-call + JSON-mode round-trips on both endpoints; `wc -l` pool jsonls | — |
| 1.1 | Magentic-One venv + `run_pool.py` | 5 GSM8K runs on qwen3.5-9b → run dirs with summary.json + llm_steps/; `convert.py` → JSONs that `src.data.trajectory.load_dataset` loads; 2 runs on deepseek-8b | 5–10 runs |
| 1.2 | Captain venv + `run_pool.py` + generated OAI_CONFIG_LIST | 5 GSM8K runs end-to-end; converter handles `{content,name}` histories | 5–10 runs |
| 1.3 | Orchestrator + rejudge + pilot corpus + manifests + activations | 2 harnesses × 2 backbones × {gsm8k:100, hotpotqa:100} ≈ 800 runs (~1 day at concurrency 24) → expect ~250–450 failures → `data/synthetic`; `DATASET=synthetic ./scripts/gen_embeddings.sh` runs clean on `mixed` | ~800 runs |
| 1.4 | Full collection | all 9 pools × 4 combos ≈ 15k runs (~2–3 days); corpus ledger stats (per-pool failure rates, lengths) | ~5–8k unlabeled |

Phase 1 done ⇒ the benchmark val/test=50/50 setup is unblocked (SVD fit on
`data/synthetic`, selection on benchmark val, report on benchmark test).

═══════════════════════════════════════════════════════════════════
# PHASE 2 — Labeling: error injection + counterfactual replay (later)
═══════════════════════════════════════════════════════════════════

## H. Injection (labeled synthetic val)

`datagen/inject/` implements CORRECT App. A.9's two-prompt flow. How many
successes get injected = `inject.yaml: n_inject` (or `all`), never hardcoded.

**Replacing the human-seed schema matching** (no human seed set exists here):
default = a generic error-pattern menu from the MAST taxonomy
(`taxonomy.py`, ~14 patterns: disobey task/role spec, step repetition, lost
context, premature termination, no/incorrect verification, fabricated tool
output, faulty calculation, reasoning–action mismatch, misread instruction,
wrong-entity retrieval, unit/format error, off-by-one…), weighted sampling,
instantiated into prompt 1's "ERROR INFORMATION" slot. Leak-proof by
construction. Optional `--exemplars` mode (condition on benchmark
`mistake_reason` texts) for ablation only — config hard-refuses any exemplar
source that appears as a test target in the active protocol.

Per successful trajectory (from `verdict.json`):
1. `strategy.py` — injection-plan prompt (keep the consecutive-CodeExecutor
   rule + cascade-analysis requirement); parse `<injection_step/agent/
   error_pattern/injection_strategy/expected_impact>`; reject if step not
   scoreable (`iter_scoreable_steps` — step 0 is the human turn) or agent ≠
   that step's role.
2. `modify.py` — message-modification prompt → `<MODIFIED CONTENT>`.
3. Cascade handling `--cascade {none, rewrite}`, default **rewrite**:
   sequentially regenerate each step after the injection point conditioned on
   the modified prefix (CORRECT's downstream steps are consistent with the
   injected error; a pure text edit leaves an incoherent suffix a scorer
   could exploit). `none` kept as the cheap ablation.
4. `verify.py` — second-pass verifier judge (separate call): the A.9
   checklist as explicit yes/no + "is the final answer now incorrect w.r.t.
   ground truth?" + programmatic prefix-untouched check. Reject-and-resample
   up to `max_attempts` (config).

Output: labeled trajectory + `{mistake_step (0-indexed), mistake_agent
(= step role, satisfying the converter validation), mistake_reason (pattern +
strategy text)}` + provenance sidecar (source run, pattern, injector model,
attempts). All model calls via `inject.yaml` `injector:`/`verifier:` endpoint
blocks — the swap-in-a-bigger-model knob. Converted via `convert.py` into
`data/synthetic-err/` (needs activations AND attention).

## I. Counterfactual replay (audit tool, not bulk labeler)

`datagen/replay/` — sample size = `replay.yaml: n_audit` (config). Two uses:
(a) step-sweep natural failures → replay-verified labels; (b) audit injected
labels in reverse: replay from the injected step with the original step
restored — outcome should flip back to success.

Per-harness feasibility (honest):
- **Captain-Agent** (truer): the 0.2 fork has no resume API, but
  `GroupChat.messages` / agents' `chat_messages` are plain lists —
  `replay_captain.py` rebuilds the team from saved `build_state.json`,
  pre-seeds the history prefix with step t replaced by the oracle-corrected
  action, then drives the speaker-selection loop forward. ~100 lines against
  internal APIs; brittle across agent sets — audit-only scope.
- **Magentic-One** (compromise): `MagenticOneGroupChat` re-plans from its
  ledger and can't be faithfully mid-run-seeded without `save_state` blobs we
  never captured. Use prefix-conditioned continuation ("transcript of work
  completed so far: … continue from this point") — measures "does correcting
  step t let the system recover?" rather than exact-replay counterfactuals.
  Stated as a fidelity limit wherever results are reported.

Oracle correction = configurable endpoint rewriting step t given the ground
truth (without leaking it verbatim — AgenTracer's analyzer constraint).
Earliest outcome-flipping step via **binary search** (~log₂N re-runs per
trajectory). `audit.py` reports injected-vs-replay agreement (exact / ±1 /
same-agent).

## Phase-2 milestones

| M | Deliverable | Verification |
|---|---|---|
| 2.1 | Injection pipeline + `synthetic-err` onboarding | Inject on pilot successes (n from config) → verified-label yield ≥60%; manual spot-check ~15; activations+attention extract clean |
| 2.2 | Replay audit | n_audit trajectories (config), binary-search sweep; agreement report (exact/±1/agent) |

## Risks and mitigations

| Risk | Assessment | Mitigation |
|---|---|---|
| Injection quality at 9B (unrealistic edits, wrong-step drift, incoherent cascades) | Headline risk | Verifier second pass + resample; cascade=rewrite; injector endpoint knob → larger model later; replay audit quantifies label noise |
| deepseek-8b tool calling | Near-certain broken | No tool parser; coder+executor-only team on Magentic-One; prefer Captain (code-block protocol) |
| WebSurfer needs vision + headful browser | Certain blocker for 8–9B backbones | Exclude via `--agents`; web-task failures are corpus signal, not loss |
| autogen-ext model_info / ledger-JSON quirks | Certain (model_info) / likely-fine (ledger) | M1.1 smoke is exactly this; overrides specified in §A |
| `<think>` leakage into logged content | Likely without care | vLLM `--reasoning-parser deepseek_r1`; converter regex-strip fallback |
| Qwen3.5 tool-parser support in vLLM 0.19 | Verify | M1.0 tool-call round-trip test |
| GAIA-question overlap with benchmark test trajectories | Distributional leakage into fit corpus | `--exclude-pools gaia` converter knob; report both |
| Mixed labeled/unlabeled metric deflation | Silent (`total_trajs` denominator) | Convention: unlabeled subsets are fit-only; assert labels ≥0 on any metric split |
| Replay fidelity (Magentic-One) | Fundamental | Framed as audit only; Captain gets the truer seeded-groupchat variant; report agreement bands, not labels |
