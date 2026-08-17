#!/usr/bin/env bash
# Evaluate a server that is already running: a smoke test across the reasoning
# modes, then a throughput/latency benchmark. Both are client-side -- nothing
# here starts, configures or stops the server.
#
#   ./benchmark.sh                       # smoke test, then the benchmark
#   ./benchmark.sh --smoke-only          # just the round trips
#   ./benchmark.sh --bench-only          # just the throughput numbers
#   ./benchmark.sh --all                 # include Think Max in the smoke test
#   ./benchmark.sh NUM_PROMPTS=128 CONCURRENCY=32
#   ./benchmark.sh BASE_URL=http://gpu-01:8000/v1
#
# The recipe does not publish H200 numbers, so the benchmark measures your node
# rather than checking it against a target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

usage() {
  cat <<'EOF'
Usage: benchmark.sh [--smoke-only | --bench-only] [--all] [KEY=value ...]

Runs against a server that is already up; nothing here starts or stops one.

  --smoke-only  just the round trips across reasoning modes
  --bench-only  just the throughput/latency numbers
  --all         include Think Max in the smoke test
  -h, --help    this message

Any setting can be overridden as an argument, which outranks the environment,
.env and defaults.env:

  ./benchmark.sh NUM_PROMPTS=128 CONCURRENCY=32
  ./benchmark.sh BASE_URL=http://gpu-01:8000/v1
  ./benchmark.sh RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024
EOF
}

load_config "$@"

RUN_SMOKE=1
RUN_BENCH=1
SMOKE_ARGS=()

set -- ${CONFIG_ARGV[@]+"${CONFIG_ARGV[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
  --smoke-only) RUN_BENCH=0; shift ;;
  --bench-only) RUN_SMOKE=0; shift ;;
  --all) SMOKE_ARGS+=(--all); shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "unknown argument: $1 (settings are passed as KEY=value)" >&2; exit 2 ;;
  esac
done

if [ "$RUN_SMOKE" -eq 0 ] && [ "$RUN_BENCH" -eq 0 ]; then
  echo "--smoke-only and --bench-only are mutually exclusive" >&2
  exit 2
fi

require_config MODEL_ID

# The repo-level venv is where model_serving/install.sh puts things; fall back to
# whatever python is on PATH so benchmark/ still works on a plain client box with
# `pip install -r requirements.txt`.
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
  [ -n "$PYTHON_BIN" ] || { echo "no python3 found" >&2; exit 1; }
fi

if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
  echo "No healthy server at $HEALTH_URL -- start one with ../serve_model.sh first." >&2
  exit 1
fi

if [ "$RUN_SMOKE" -eq 1 ]; then
  printf '\n\033[1mSmoke test\033[0m\n'
  "$PYTHON_BIN" "$SCRIPT_DIR/smoke_test.py" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}
fi

if [ "$RUN_BENCH" -eq 0 ]; then
  exit 0
fi

printf '\n\033[1mThroughput\033[0m\n'

if [ ! -x "$VLLM_BIN" ]; then
  echo "vllm not installed at $VLLM_BIN -- 'vllm bench serve' comes from the" >&2
  echo "serving install (model_serving/install.sh). Use --smoke-only without it." >&2
  exit 1
fi

# Concurrency is capped by the server's --max-num-seqs (16 in the recipe);
# asking for more just queues.
if [ "$CONCURRENCY" -gt "$MAX_NUM_SEQS" ]; then
  echo "note: concurrency $CONCURRENCY exceeds --max-num-seqs $MAX_NUM_SEQS, requests will queue"
fi

exec "$VLLM_BIN" bench serve \
  --backend "$BACKEND" \
  --endpoint "$BENCH_ENDPOINT" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --dataset-name random \
  --random-input-len "$RANDOM_INPUT_LEN" \
  --random-output-len "$RANDOM_OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --percentile-metrics "$PERCENTILE_METRICS"
