# datagen — synthetic trajectory generation

Runs Magentic-One and Captain-Agent over question pools with locally-served
backbones, then converts the failures into an unlabeled corpus for SVD fitting.

**To run any of it, see [SCRIPTS.md](SCRIPTS.md).** This file is the reference
for *what the pieces are* and the non-obvious behaviour of the environment.

## Installation

Four environments, deliberately separate: the main repo venv serves the models
and runs the pipeline; each harness has its own venv (Magentic-One needs autogen
0.7, Captain-Agent imports a vendored autogen 0.2 fork, and neither can share
the other's or the repo's); Chromium runs in a Docker image. All commands are
run from the repo root. `uv` is assumed (`pip install uv` otherwise).

### 1. Main repo venv (serving + pipeline)

Already provisioned as `.venv` (vllm 0.19.1, transformers 5.13, datasets 4.8).
It serves the vLLM endpoints and runs `pools/`, `collect/`, `judge/`,
`convert.py`. If recreating:

```bash
uv sync         # from pyproject.toml at the repo root
```

Model checkpoints are expected under `../hub/` (a sibling of the repo):
`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-35B-A3B`,
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. Paths live in `configs/serve.yaml`.

### 2. Magentic-One venv (Python 3.12)

```bash
cd datagen/TraceElephant/code/agent_system/Magentic-One
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install \
  autogen-agentchat "autogen-ext[openai,file-surfer,docker]" \
  python-dotenv "playwright>=1.48.0"
cd -
```

The `playwright` **client only** — the browser binary and its system libs live
in the Docker image (step 4). The shipped `requirements.txt` is a full pip
freeze (tensorflow, vllm 0.7, ray) and is not what this driver needs.

### 3. Captain-Agent venv (Python 3.10)

```bash
cd datagen/TraceElephant/code/agent_system/Captain-Agent
uv venv --python 3.10 .venv
# vendored autogen 0.2 deps + autobuild retriever
VIRTUAL_ENV=.venv uv pip install \
  "openai>=1.3" diskcache termcolor flaml "numpy>=1.17,<2" python-dotenv \
  tiktoken "pydantic>=1.10,<3,!=2.6.0" docker sentence-transformers chromadb pandas
# tool-library imports (arxiv, markdownify, pptx, textract, …)
VIRTUAL_ENV=.venv uv pip install \
  mammoth markdownify arxiv pymupdf wikipedia-api python-pptx pandas scipy \
  sympy pillow textract openpyxl
VIRTUAL_ENV=.venv uv pip install "pip<24.1"   # REQUIRED — see below
cd -
```

`pip<24.1` is mandatory: Captain's code executor pip-installs the tool
library's requirements before every run, and `textract==1.6.5` declared an
invalid requirement that pip ≥24.1 refuses, poisoning every pip call in the
venv. (uv venvs also ship no `pip` at all, which fails the same way.)

### 4. Browser container

```bash
docker pull mcr.microsoft.com/playwright:v1.61.0-noble
```

The tag **must match** the `playwright` client in the Magentic-One venv
(1.61.0). Starting and using it is covered in SCRIPTS.md §2b.

### 5. Configuration

- **Serper key (Captain web search).** Put `SERPER_API_KEY=<key>` in
  `datagen/TraceElephant/code/agent_system/Captain-Agent/.env`. Without it
  `perform_web_search` raises and teams fall back to parametric knowledge.
- **Derived files.** After editing `serve.yaml`, regenerate the patched
  tokenizer, patched chat template and Captain's `OAI_CONFIG_LIST`
  (SCRIPTS.md §2). These are gitignored and must exist before serving.

Verify the whole stack with `python datagen/serve/smoke.py --all --wait` (models)
and `curl -s http://127.0.0.1:9222/json/version` (browser).

## Status

Phase 1 complete. Four cells collected over `gaia` + `assistantbench`
(198 tasks each, 790 runs):

| Cell | median steps | p90 | tool agents used | multi-agent | errors |
|---|---|---|---|---|---|
| magentic-qwen35b | 26 | 45 | 99% | 100% | 4% |
| magentic-qwen9b | 19 | 45 | 96% | 99% | 21% |
| captain-qwen9b | 14 | 20 | 81% | 77% | 0% |
| captain-qwen35b | 10 | 20 | 97% | 83% | 0% |

For reference, the benchmarks being emulated: ww/algorithm-generated is capped
at 10 turns, traceelephant/captain median 13, traceelephant/magentic median 28.

Phase 2 (`inject/`, `replay/` — error injection and counterfactual replay for
labeled validation data) is not started.

## Layout

```
configs/     serve.yaml (endpoints — the single source of truth for every model
             role) + one collect config per (backbone, agent system)
serve/       launch script, capability smoke test, tokenizer and chat-template
             patchers, OAI_CONFIG_LIST generator
pools/       prepare.py -> pools/data/<pool>.jsonl
collect/     run_batch.py — matrix expansion, one subprocess per task
judge/       rejudge.py + checkers.py
convert.py   run dirs -> data/synthetic/<subset>/<N>.json
TraceElephant/  vendored harness clone; the only additions are the two
                run_pool.py drivers and each harness's own .venv
```

## Question pools

Uniform schema per task, so a new benchmark means a loader and nothing else:

```json
{"id": "gsm8k-test-00017", "question": "...", "answer": "72", "pool": "gsm8k",
 "level": null, "answer_type": "numeric", "file_path": null, "raw_id": null}
```

`answer_type` drives judging: `numeric` (normalized equality), `mcq` (option
letter), `exact` (EM / token-F1), `open` (LLM judge). MCQ pools are rendered to
lettered text at prep time so the gold answer is always a letter.

| Pool | n | answer_type | source |
|---|---|---|---|
| gaia | 165 | open | `smolagents/GAIA-annotated` 2023 val |
| assistantbench | 33 | open | `AssistantBench/AssistantBench` val |
| gsm8k | 1,319 | numeric | `openai/gsm8k` test |
| math500 | 500 | numeric | `HuggingFaceH4/MATH-500` |
| arc | 1,172 | mcq | `allenai/ai2_arc` ARC-Challenge test |
| mmlu-pro | 12,032 | mcq | `TIGER-Lab/MMLU-Pro` test |
| hotpotqa | 7,405 | exact | `hotpotqa/hotpot_qa` distractor val |
| 2wikimqa | 12,576 | exact | `voidful/2WikiMultihopQA` dev.json |
| musique | 2,412 | exact | `dgslibisey/MuSiQue` val |

Only `gaia` + `assistantbench` are active in the configs. The rest are prepared
and runnable via `--pools` (see SCRIPTS.md); they are deprioritized because
gsm8k/math500/arc/mmlu-pro are largely self-contained (shallow trajectories,
arc maxes at 6 steps) and hotpotqa/2wikimqa/musique score poorly only because
`prepare.py` drops their supporting-context paragraphs, turning reading
comprehension into closed-book trivia. Re-enable those once context is added.

Two sourcing detours: `xanhho/2WikiMultihopQA` ships a loading script that
`datasets` 4.x refuses to run, so 2wikimqa reads raw `dev.json` from a mirror;
and `gaia-benchmark/GAIA` is gated, so GAIA comes from the ungated `smolagents`
mirror, which also carries the 38 attachment files.

## Trajectory schemas

### Raw run directory

`runs/<harness>/<backbone>/<pool>/<task_id>-r<rep>/`

| File | Written by | Content |
|---|---|---|
| `summary.json` | driver | the conversation trace (**load-bearing**) |
| `judge.json` | driver | in-run correctness guess — untrusted |
| `verdict.json` | `rejudge.py` | authoritative `{is_correct, method, extracted, reason}` |
| `llm_steps/step_N.json` | driver | one file per LLM call — provenance |
| `driver.log` | orchestrator | stdout/stderr of the run |
| `workspace/` | driver | scratch dir for executed code |

`summary.json` carries `history`, `question`, `ground_truth`, `question_ID`,
`pool`, `extracted_answer`, `backbone`, `elapsed_s`. The one difference between
harnesses is the turn shape: Magentic-One emits `{role, content}` with
Who&When-style roles (`human`, `Orchestrator (thought)`,
`Orchestrator (-> Agent)`, the bare agent name,
`Orchestrator (termination condition)`); Captain-Agent emits `{content, name}`,
which the converter maps to `role`.

### Converted trajectory

`data/synthetic/<subset>/<N>.json`, the schema
`src.data.trajectory.load_dataset` consumes:

```json
{"question_ID": "gaia/gaia-val-00017-r0", "question": "...",
 "ground_truth": "...", "history": [{"role": "...", "content": "..."}],
 "mistake_agent": "", "mistake_step": -1, "mistake_reason": "",
 "level": -1, "system": "...", "subset": "magentic-qwen9b", "pool": "gaia"}
```

Subsets encode `<harness>-<backbone>`, because the generator backbone is a
different axis from the manifest `models:` field, which lists the *proxy*
extraction models. `--mixed` materializes a merged subset so a combined SVD fit
needs no loader changes; every subset gets a `filename_map.csv`.

`mistake_step = -1` marks unlabeled. Those are **fit-only**: `compute_metrics`
skips unlabeled trajectories when counting hits but still counts them in the
denominator (`src/utils/utils.py:290-299`), so they must never sit in a metric
split. Phase-2 injected trajectories will carry the real triple, with the
invariant `history[mistake_step]["role"] == mistake_agent`.

Cost note: fit-only data needs **activations but not attention** (attention is
consumed only by rescoring on val/test), so only the Phase-2 labeled set will
need `extract_attention.sh`.

## Environment behaviour worth knowing

### Two shipped checkpoint files are wrong

Both are patched into corrected copies; the checkpoints are never modified.

- **DeepSeek tokenizer** — declares `tokenizer_class=LlamaTokenizerFast` beside
  a ByteLevel-BPE `tokenizer.json`. transformers 5.x then skips the ByteLevel
  decoder, so text round-trips corrupted (`"Hello world"` → `"Helloworld"`) and
  served completions return raw byte tokens (`Ċ`, `Ġ`). `smoke.py` hard-fails
  on stray byte markers so it cannot regress silently. **The same bare
  `AutoTokenizer.from_pretrained` at `src/models/base.py:71` hits this bug**, so
  the extraction pipeline is affected too — deliberately not changed here.
- **Qwen3.5 chat template** — raises `No user query found in messages.` for any
  message list without a user turn, which autogen 0.2 (Captain) sends on every
  expert turn after the first. Each returned HTTP 400, retried 3×, and produced
  an empty turn.

### WebSurfer runs headless in a container, in text mode

The host has no browser and no root to install one, so Chromium runs in the
Playwright image and the driver attaches over CDP, passing `playwright` and
`context` into `MultimodalWebSurfer` so its `_lazy_init` skips every local
launch path. Text mode (`vision=False`) reads pages from the DOM rather than
screenshots — web pages are HTML, so nothing is lost by not rendering to pixels
and OCR-ing back. About 6% of GAIA has image attachments where vision would
genuinely help; that is the deliberate trade. `use_ocr` in the library is dead
code (assigned, never read).

Note the surfer attaches a screenshot to its reply **unconditionally**, even in
text mode (`_multimodal_web_surfer.py:817`). `content_to_text` in the driver
replaces those with `[screenshot omitted]`; without it every WebSurfer turn
embeds a `<…Image object at 0x7f19…>` memory address that differs per run.

### Captain-Agent constraints

- Delegates **only** through the `seek_experts_help` native tool call
  (`meta_agent.py:187`, no code-block fallback), so a backbone served without a
  tool-call parser cannot drive it at all.
- Its schema omits a `required` array, so a model may drop `group_name` and
  crash `_run_autobuild`. Both the schema and the signature default are patched.
- Web access comes from its own tool library and needs `SERPER_API_KEY`; team
  building and web search track together (81% on the 9B, 97% on the 35B).
- Expert names are autobuilt and **drift** (`Python_Programming_Expert` vs
  `PythonProgramming_Expert`; 8–10 distinct names), so agent-level scoring
  should normalize them.
- Its venv needs `pip<24.1`: the code executor pip-installs the tool library's
  requirements before every run, and `textract==1.6.5` declares an invalid
  requirement that pip ≥24.1 refuses, poisoning every pip call.

### Thread exhaustion at high concurrency

Libraries in the harness stacks size thread pools from the visible CPU count —
chromadb's tokio runtime opened one worker per core, ~682 threads per Captain
process on this 240-core box, hitting the cgroup pid ceiling and failing with
EAGAIN. `cpus_per_task` (default 4) pins each task with `taskset`, capping every
such pool at once: 682 → 18 threads.

### The prompt-echo trap

Both harnesses' prompts contain a literal `## ANSWER\n[concise final answer]`
block, and both orchestrators echo the task to their agents. A naive reverse
scan for `## ANSWER` extracts the placeholder — observed in real runs from both.
Both drivers reject known placeholders and keep scanning. Re-check if a prompt
template is edited.

### Code executes locally

Both harnesses run LLM-generated code with `use_docker: False` in a per-run
`workspace/`, auto-approved. One early run grepped the host filesystem into its
trajectory (a 675 MB `summary.json`); 94 runs embedded host paths. No
credentials leaked, but switching the executor to Docker is the obvious
hardening step, and the container stack is already in place.
