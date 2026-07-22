#!/usr/bin/env bash
# Launch vLLM endpoints for the datagen pipeline.
#
# Every knob (checkpoint path, port, GPUs, context length, parser flags) comes
# from datagen/configs/serve.yaml — this script only renders and runs the
# command, so the registry stays the single source of truth.
#
#   ./datagen/serve/launch_vllm.sh qwen3.5-9b              # one model
#   ./datagen/serve/launch_vllm.sh qwen3.5-9b deepseek-8b  # both
#   ./datagen/serve/launch_vllm.sh --all
#   DRY_RUN=1 ./datagen/serve/launch_vllm.sh --all         # print, don't run
#   ./datagen/serve/launch_vllm.sh --stop                  # stop what we started
#
# Servers run detached; logs land in the configured log_dir and PIDs in
# <log_dir>/<model>.pid. Startup is slow (weights + CUDA graphs) — use
#   python datagen/serve/smoke.py --model <name> --wait
# to block until an endpoint is actually answering.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
CONFIG="${CONFIG:-datagen/configs/serve.yaml}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_BIN="$(python - "$CONFIG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["vllm_bin"])
EOF
)"
LOG_DIR="$(python - "$CONFIG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("log_dir", "datagen/serve/logs"))
EOF
)"
mkdir -p "$LOG_DIR"

all_models() {
  python - "$CONFIG" <<'EOF'
import sys, yaml
print(" ".join(yaml.safe_load(open(sys.argv[1]))["models"]))
EOF
}

stop_all() {
  shopt -s nullglob
  for pidfile in "$LOG_DIR"/*.pid; do
    pid="$(cat "$pidfile")"
    name="$(basename "$pidfile" .pid)"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping $name (pid $pid)"
      kill "$pid"
    else
      echo "$name (pid $pid) not running"
    fi
    rm -f "$pidfile"
  done
}

case "${1:-}" in
  --stop) stop_all; exit 0 ;;
  --all)  MODELS=($(all_models)) ;;
  "")     echo "usage: $0 <model-key>... | --all | --stop" >&2; exit 2 ;;
  *)      MODELS=("$@") ;;
esac

for model in "${MODELS[@]}"; do
  # Render the full argv for this model from the registry.
  mapfile -t ARGS < <(python - "$CONFIG" "$model" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
name = sys.argv[2]
if name not in cfg["models"]:
    sys.exit(f"unknown model {name!r}; known: {sorted(cfg['models'])}")
m = cfg["models"][name]
args = [
    "serve", m["path"],
    "--served-model-name", m.get("served_model_name", name),
]
if m.get("tokenizer"):          # patched tokenizer dir, see fix_tokenizer.py
    args += ["--tokenizer", m["tokenizer"]]
if m.get("chat_template"):      # patched template, see fix_chat_template.py
    args += ["--chat-template", m["chat_template"]]
args += [
    "--port", str(m["port"]),
    "--tensor-parallel-size", str(m.get("tensor_parallel_size", 1)),
    "--max-model-len", str(m["max_model_len"]),
    "--gpu-memory-utilization", str(m.get("gpu_memory_utilization", 0.9)),
]
args += [str(a) for a in m.get("extra_args", [])]
print("\n".join(args))
print(m.get("gpus", ""))          # last line: CUDA_VISIBLE_DEVICES
EOF
  )
  GPUS="${ARGS[-1]}"
  unset 'ARGS[-1]'

  LOG="$LOG_DIR/$model.log"
  echo "── $model  (GPUs=$GPUS, log=$LOG)"
  echo "   CUDA_VISIBLE_DEVICES=$GPUS $VLLM_BIN ${ARGS[*]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  if [[ -f "$LOG_DIR/$model.pid" ]] && kill -0 "$(cat "$LOG_DIR/$model.pid")" 2>/dev/null; then
    echo "   already running (pid $(cat "$LOG_DIR/$model.pid")) — skipping"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" nohup "$VLLM_BIN" "${ARGS[@]}" \
    > "$LOG" 2>&1 &
  echo $! > "$LOG_DIR/$model.pid"
  echo "   started pid $!"
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "Wait for readiness:  python datagen/serve/smoke.py --model ${MODELS[0]} --wait"
  echo "Tail a log:          tail -f $LOG_DIR/${MODELS[0]}.log"
fi
