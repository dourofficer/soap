# Running datagen — Phase 1

Generate agentic trajectories with local backbones, judge them, and convert the
failures into an unlabeled corpus at `data/synthetic/` for SVD fitting.

Run everything **from the repo root**. There are no wrapper scripts: each stage
is one Python CLI (plus `launch_vllm.sh`), and every one takes `--help`.

```
1. pools/prepare.py       question pools        once
2. serve/launch_vllm.sh   vLLM endpoints        once, must stay up
2b. docker run …          browser container     once, must stay up
3. collect/run_batch.py   run the harnesses     long; once per (backbone, system)
4. judge/rejudge.py       success/failure verdicts
5. convert.py             -> data/synthetic/
```

## The short version

```bash
python datagen/pools/prepare.py --all              # ~10 min, once
./datagen/serve/launch_vllm.sh qwen3.5-9b qwen3.5-35b-a3b
python datagen/serve/smoke.py --all --wait         # blocks until ready
docker start datagen-browser                       # see §2b for first-time setup

python datagen/collect/run_batch.py --config collect-qwen35b-magentic
python datagen/collect/run_batch.py --config collect-qwen35b-captain
python datagen/collect/run_batch.py --config collect-qwen9b-magentic
python datagen/collect/run_batch.py --config collect-qwen9b-captain

python datagen/judge/rejudge.py
python datagen/convert.py --outcome fail --mixed
```

Then onboard `data/synthetic/` as a dataset (manifest + thin stage configs) and
run the normal extraction pipeline — see `scripts/README.md` at the repo root.

---

## 1. Question pools

```bash
python datagen/pools/prepare.py --all                # every pool
python datagen/pools/prepare.py --pool gsm8k --pool gaia
python datagen/pools/prepare.py --stats              # what exists, with counts
python datagen/pools/prepare.py --pool gsm8k --force # re-download
```

Writes one uniform jsonl per pool to `datagen/pools/data/`. Idempotent —
prepared pools are skipped unless `--force`. GAIA also downloads its 38
attachment files. Nine pools are available; the active configs use `gaia` and
`assistantbench` (198 tasks), see §3 for running the others.

## 2. Serving

`configs/serve.yaml` is the single source of truth for every model role
(agent backbones, judge, and later the Phase-2 injector/verifier): ports, GPUs,
context length, parser flags.

```bash
./datagen/serve/launch_vllm.sh --all           # start every model, detached
./datagen/serve/launch_vllm.sh qwen3.5-9b      # just one
DRY_RUN=1 ./datagen/serve/launch_vllm.sh --all # print commands only
./datagen/serve/launch_vllm.sh --stop

python datagen/serve/smoke.py --all --wait     # block until ready, then verify
```

Startup takes minutes (weights + CUDA graphs), so always gate on
`smoke.py --wait` rather than guessing. Logs/pidfiles: `datagen/serve/logs/`.

`smoke.py` checks each capability the harnesses depend on separately — model
listing, multi-line chat, JSON mode, tool calling — so a failure names the
broken thing instead of surfacing mid-collection. It also fails on undecoded
byte tokens, which is how a mis-declared tokenizer shows up.

The 35B MoE needs a GPU largely to itself (65 GiB of weights,
`gpu_memory_utilization: 0.80`); anything below ~0.55 cannot load it.

Three files are **derived** from `serve.yaml` — regenerate them after editing
ports, model names or checkpoints:

```bash
python datagen/serve/fix_tokenizer.py --check-all      # audit; --model X to write
python datagen/serve/fix_chat_template.py --check-all
python datagen/serve/gen_oai_config.py                 # Captain's OAI_CONFIG_LIST
```

Both patchers exist because a shipped checkpoint file is wrong, and both leave
the checkpoint untouched (they write a corrected copy that `serve.yaml` points
at):

- **Tokenizer** — DeepSeek-R1-Distill declares `LlamaTokenizerFast` beside a
  ByteLevel-BPE `tokenizer.json`; transformers 5.x then skips the ByteLevel
  decoder and every completion returns raw byte tokens (`Ċ`, `Ġ`).
- **Chat template** — Qwen3.5's template raises `No user query found in
  messages.` for any list without a user turn, which autogen 0.2 (Captain)
  sends on every expert turn after the first.

## 2b. Browser container (required for WebSurfer)

The Magentic-One configs run `agents: fs,coder,executor,surfer`, and
MultimodalWebSurfer needs a real Chromium. This host has **no browser and no
root** to install one (`libatk`, `libgbm`, `libxkbcommon` missing, uid ≠ 0), so
Chromium runs in the Playwright image and the driver attaches over CDP.

```bash
docker run -d --name datagen-browser --shm-size=2g --network host \
  --entrypoint /ms-playwright/chromium-1228/chrome-linux64/chrome \
  mcr.microsoft.com/playwright:v1.61.0-noble \
  --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --remote-debugging-port=9222 --user-data-dir=/tmp/prof about:blank

curl -s http://127.0.0.1:9222/json/version    # sanity check
docker start datagen-browser                  # after a reboot
docker rm -f datagen-browser                  # remove
```

- **`--network host` is required.** Chromium ignores
  `--remote-debugging-address` and binds CDP to loopback, so a published port
  (`-p 9222:9222`) sees nothing.
- The image tag must match the `playwright` client in the Magentic-One venv
  (1.61.0). The client is installed there; the browser exists only in the image.
- **One shared container serves every task** — each run opens its own
  `BrowserContext` (isolated cookies/storage) and closes it on exit. Override
  the endpoint with `--cdp-url` or `DATAGEN_CDP_URL`.
- WebSurfer runs in **text mode** (`model_info["vision"]=False`): it reads pages
  from the DOM plus an aria-labelled target list, so there is no vision serving
  path and no 1105-token screenshot per step. It hard-requires function
  calling, so it is qwen-only.

Captain reaches the web differently — through its own tool library, which needs
`SERPER_API_KEY` in the harness `.env`. Without it `perform_web_search` raises
and teams fall back to parametric knowledge.

## 3. Collection

One config per **(backbone, agent system)**, so cells schedule, retry and
compare independently:

| Config | Backbone | System | Subset |
|---|---|---|---|
| `collect-qwen9b-magentic.yaml` | qwen3.5-9b | Magentic-One | `magentic-qwen9b` |
| `collect-qwen9b-captain.yaml` | qwen3.5-9b | Captain-Agent | `captain-qwen9b` |
| `collect-qwen35b-magentic.yaml` | qwen3.5-35b-a3b | Magentic-One | `magentic-qwen35b` |
| `collect-qwen35b-captain.yaml` | qwen3.5-35b-a3b | Captain-Agent | `captain-qwen35b` |

Each defines only its own harness, so naming the other in `matrix` is a hard
error rather than a silent cross-run. All share `out_root: datagen/runs`, so
everything lands in one tree keyed by `harness/backbone/pool` and is judged and
converted in a single pass. `collect-deepseek.yaml` is **stale** — kept for
reference, not maintained.

```bash
python datagen/collect/run_batch.py --config collect-qwen35b-magentic --dry-run
python datagen/collect/run_batch.py --config collect-qwen35b-magentic
```

`--config` is required — there is no default. `--dry-run` prints per-cell job
counts and the first ten rendered commands.

**Resumable.** A task is done iff its `summary.json` exists, and the drivers
write it last, so re-running the same command after an interruption skips
finished work. There is no resume flag; just run it again.

### Running other datasets without editing a config

The active configs cover `gaia` + `assistantbench`. `--pools` **replaces** the
matrix pool list, so any prepared pool can be run against any config:

```bash
# any dataset, no config edit
python datagen/collect/run_batch.py --config collect-qwen9b-magentic \
    --pools gsm8k,math500 --limit 100

# every prepared pool
python datagen/collect/run_batch.py --config collect-qwen35b-captain \
    --pools gaia,assistantbench,gsm8k,math500,arc,mmlu-pro,hotpotqa,2wikimqa,musique
```

Other overrides — all leave the config file untouched:

```bash
--pool gaia                 # narrow to a pool the matrix ALREADY names
--harness magentic          # narrow when a config defines several
--limit 20                  # cap tasks per pool
--set concurrency=8         # any dot-path key
--set out_root=datagen/runs-pilot     # keep experiments out of the main tree
--set 'timeout_s={default: 1800}'     # values parse as YAML
```

`--pool` only *filters*; it cannot reach a pool the matrix omits (the run exits
with a message saying so). Use `--pools` for that.

Make it permanent by editing the config's `matrix` — the non-web tiers are
commented out in-place, ready to restore.

Runs land in `datagen/runs/<harness>/<backbone>/<pool>/<task_id>-r<rep>/`, with
a `manifest.jsonl` ledger at the tree root. Per-run artifacts and the trajectory
schema are in `README.md`.

## 4. Judging

```bash
python datagen/judge/rejudge.py                    # all runs
python datagen/judge/rejudge.py --pool gaia --force
python datagen/judge/rejudge.py --no-llm           # programmatic checkers only
python datagen/judge/rejudge.py --backbone qwen3.5-9b
```

Writes `verdict.json` per run — the authoritative success/failure signal. The
harnesses' own `judge.json` is a permissive substring match and is never used
(it scored `**$70,000 profit**` wrong against gold `70000`). Dispatch is by the
pool's `answer_type`: `numeric` / `mcq` / `exact` are checked programmatically,
`open` and borderline exact-match cases go to an LLM judge. Re-runnable;
already-judged runs are skipped unless `--force`.

`--config` defaults to `collect-qwen9b-magentic`, but only its `judge` and
`out_root` are read and all configs share those.

## 5. Conversion

For the SVD-fit corpus, convert **every valid run regardless of outcome** —
fitting is unsupervised, so successes carry signal too and no rejudge pass is
needed first:

```bash
python datagen/convert.py --outcome all --mixed --dry-run   # counts first
python datagen/convert.py --outcome all --mixed             # -> data/synthetic/
```

Other selections:

```bash
python datagen/convert.py --outcome fail --mixed        # failure-only (needs verdicts)
python datagen/convert.py --outcome success --out-root data/synthetic-ok   # Phase-2 candidates
python datagen/convert.py --exclude-pools gaia --mixed
```

`--outcome fail`/`success` require a verdict per run, so run `rejudge.py`
first; `--outcome all` does not.

**Provenance.** Every output subset is named `<harness>-<backbone>`
(`magentic-qwen9b`, `captain-qwen35b`, …), and every trajectory JSON carries
`pool`, `backbone`, `harness` and `outcome` fields (the loader ignores unknown
keys), so its origin — task pool, agent system, generator backbone, run result
— survives any downstream merge. Each subset also gets a `filename_map.csv`
(`file, question_ID, pool, backbone, harness, outcome, run_dir`) mapping every
`<N>.json` back to its raw run directory. `outcome` comes from `verdict.json`
when present, else the in-run `judge.json` — the latter is the permissive
substring judge, so treat it as approximate until `rejudge.py` has run
(re-converting afterwards refreshes the field).

`--mixed` also materializes a merged subset so a combined SVD fit needs no
loader changes.

Unlabeled trajectories carry `mistake_step = -1`, which makes them
**fit-only**: `compute_metrics` skips them when counting hits but still counts
them in the denominator, so they must never sit in a split that reports metrics.

---

## Checking results

```bash
python datagen/judge/rejudge.py       # prints the overall failure rate
python datagen/convert.py --dry-run   # per-subset counts + step spread
```

Two things to eyeball: the **failure rate** (a pool near 0 % or 100 % is
useless) and the **step spread**. Trajectory richness tracks task difficulty —
gsm8k yields 4 steps with no delegation; gaia yields 19–26. Compare against the
benchmarks being emulated: ww/algorithm-generated is capped at 10 turns,
traceelephant/captain has median 13, traceelephant/magentic median 28.

Triage the run tree:

```bash
python - <<'PY'
import json, glob, os, collections
c = collections.Counter()
for d in glob.glob('datagen/runs/*/*/*/*'):
    if not os.path.isdir(d): continue
    s = os.path.join(d, 'summary.json')
    if not os.path.exists(s):
        c['partial (re-runs automatically)'] += 1
    else:
        j = json.load(open(s))
        c['broken (delete to retry)' if j.get('error') and len(j.get('history', [])) < 2
          else 'usable'] += 1
for k, v in c.most_common(): print(f'{v:6d}  {k}')
PY
```

## Interrupted and broken runs

**Killed midway (no `summary.json`)** — nothing to do. Resume keys on that
file, and the drivers wipe `llm_steps/` and `workspace/` on start, so a re-run
never inherits a dead attempt's files. Just re-run the collect command.

**Finished but broken (`summary.json` with an `error` and empty history)** —
these look done and are skipped forever. They do not pollute `data/synthetic/`
(the converter drops empty histories), but the task is never retried. Delete,
then re-run:

```bash
python - <<'PY'
import json, glob, shutil
n = 0
for f in glob.glob('datagen/runs/*/*/*/*/summary.json'):
    d = json.load(open(f))
    if d.get('error') and len(d.get('history', [])) < 2:
        shutil.rmtree(f.rsplit('/', 1)[0]); n += 1
print('removed', n, 'broken run dirs')
PY
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` at startup | ports in `serve.yaml` taken; this box has other services in the 8000s |
| Undecoded byte tokens (`Ċ`, `Ġ`) | run `fix_tokenizer.py`; `smoke.py` fails loudly on this |
| `No user query found in messages.` (Captain) | run `fix_chat_template.py` |
| Captain: `Pip install failed` | its venv needs `pip<24.1` (textract 1.6.5 declares an invalid requirement) |
| Captain: no expert team, 2-step runs | missing `SERPER_API_KEY`, or a backbone with no tool-call parser |
| WebSurfer never acts / connection refused | browser container down — `docker start datagen-browser` |
| `KeyError: 'is_request_satisfied'` | the backbone emitted valid JSON with the wrong ledger schema; a model limit, ~21 % on the 9B and ~4 % on the 35B |
| `no jobs after filtering` | `--pool` cannot reach a pool the matrix omits — use `--pools` |
| Many `timeout` rows in the ledger | raise `timeout_s` for that pool |
| chromadb `Resource temporarily unavailable` | thread exhaustion; `cpus_per_task` (default 4) caps every `num_cpus`-sized pool — 682 → 18 threads per process |
