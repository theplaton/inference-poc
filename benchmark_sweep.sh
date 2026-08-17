#!/usr/bin/env bash
# Concurrency sweep. For each level -- 1, 2, 4, ... up to MAX_CONCURR -- serve
# the model with that batch limit, benchmark it at that client concurrency, tear
# it down, and append the numbers to two CSVs.
#
#   ./benchmark_sweep.sh                  # the full sweep, ~2h: a reload per level
#   ./benchmark_sweep.sh --plan           # the plan and the per-level commands
#   ./benchmark_sweep.sh MAX_CONCURR=4    # a shorter sweep: 1, 2, 4
#   ./benchmark_sweep.sh RESULTS_DIR=out/run7
#
# Both halves of every level are the repo's own entrypoints, unchanged:
# ./serve_model.sh with MAX_NUM_SEQS set to the level, then ./benchmark.sh
# --bench-only with CONCURRENCY set to the same number. Nothing here reaches
# past them into model_serving/ or benchmark/, except to clear stragglers.
#
# A level that fails does not stop the sweep: it is reported at the end and the
# CSVs keep every level that did finish.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The sweep runs the server, so it configures itself from the server's layers.
# shellcheck source=model_serving/config.sh
source "$REPO_ROOT/model_serving/config.sh"

usage() {
  cat <<'EOF'
Usage: benchmark_sweep.sh [--plan] [KEY=value ...]

Sweeps concurrency by powers of two, reloading the model at each level.

  --plan        print the plan and the exact per-level commands, then exit
  -h, --help    this message

Settings follow the same four layers as the tools it drives -- an argument
outranks the environment, which outranks model_serving/.env, which outranks
model_serving/defaults.env. The sweep's own settings:

  MAX_CONCURR=16          highest level; the sweep doubles 1, 2, 4 ... up to it
  PROMPTS_PER_LEVEL=8     requests per level = this x the level
  RANDOM_INPUT_LEN=2048   ISL
  RANDOM_OUTPUT_LEN=512   OSL
  RESULTS_DIR=...         defaults to results/sweep-<timestamp>
  SETTLE_SECONDS=15       pause after teardown for the GPUs to come back

Everything the server understands is also settable, so a sweep can be retuned
without touching a file:

  ./benchmark_sweep.sh MAX_MODEL_LEN=131072 STRATEGY=dep
  ./benchmark_sweep.sh MAX_CONCURR=8 PROMPTS_PER_LEVEL=16 RANDOM_OUTPUT_LEN=1024

Requests scale with the level so every level runs the same number of batch
rounds (PROMPTS_PER_LEVEL of them) -- level 1 sends 8 requests, level 16 sends
128. Comparing levels that ran two rounds against levels that ran fifty would
compare warm-up against steady state.
EOF
}

load_config "$@"

PLAN_ONLY=0
set -- ${CONFIG_ARGV[@]+"${CONFIG_ARGV[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
  --plan) PLAN_ONLY=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "unknown argument: $1 (settings are passed as KEY=value)" >&2; exit 2 ;;
  esac
done

require_config MODEL_ID

# The sweep's own defaults, below every layer above -- same rule as the derived
# defaults at the end of config.sh.
: "${MAX_CONCURR:=16}"
: "${PROMPTS_PER_LEVEL:=8}"
: "${RANDOM_INPUT_LEN:=2048}"
: "${RANDOM_OUTPUT_LEN:=512}"
: "${SETTLE_SECONDS:=15}"
: "${RESULTS_DIR:=$REPO_ROOT/results/sweep-$(date +%Y%m%d-%H%M%S)}"
# Repeated from model_serving/defaults.env so a deleted defaults file degrades to
# a working sweep instead of an unbound-variable abort, the same way
# benchmark/config.sh repeats the endpoint.
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${HEALTH_TIMEOUT:=1800}"
: "${SHUTDOWN_TIMEOUT:=30}"
# A stuck engine should not hang the sweep: the server gets SHUTDOWN_TIMEOUT to
# release its GPUs, and this much again before it is killed outright.
: "${STOP_TIMEOUT:=$((SHUTDOWN_TIMEOUT + 30))}"

case "$MAX_CONCURR" in
'' | *[!0-9]*) echo "MAX_CONCURR must be a positive integer, got '$MAX_CONCURR'" >&2; exit 2 ;;
esac
[ "$MAX_CONCURR" -ge 1 ] || { echo "MAX_CONCURR must be at least 1" >&2; exit 2; }

# Double until the cap, then include the cap itself -- so a MAX_CONCURR that is
# not a power of two (12) still gets measured: 1, 2, 4, 8, 12.
LEVELS=()
level=1
while [ "$level" -lt "$MAX_CONCURR" ]; do
  LEVELS+=("$level")
  level=$((level * 2))
done
LEVELS+=("$MAX_CONCURR")

# HOST is a bind address; 0.0.0.0 is not something you can connect to.
health_host="$HOST"
case "$health_host" in
0.0.0.0 | :: | '') health_host=localhost ;;
esac
HEALTH_URL="http://$health_host:$PORT/health"

SUMMARY_CSV="$RESULTS_DIR/benchmark_sweep.csv"
DETAILED_CSV="$RESULTS_DIR/benchmark_sweep_detailed.csv"

# benchmark/ falls back to a plain python3 for the same reason: the venv is the
# serving install's, and it may not be the interpreter this box has.
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
  [ -n "$PYTHON_BIN" ] || { echo "no python3 found -- needed to write the CSVs" >&2; exit 1; }
fi

# --- plan ---------------------------------------------------------------------

printf '\033[1mConcurrency sweep\033[0m\n'
printf '  model       %s\n' "$MODEL_ID"
printf '  levels      %s\n' "${LEVELS[*]}"
printf '  requests    %s per level (%s x level)\n' \
  "$(for n in "${LEVELS[@]}"; do printf '%s ' "$((n * PROMPTS_PER_LEVEL))"; done)" "$PROMPTS_PER_LEVEL"
printf '  ISL/OSL     %s / %s tokens\n' "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN"
printf '  results     %s\n' "$RESULTS_DIR"
printf '  each level reloads the checkpoint (~10-20 min), so budget for it\n\n'

if [ "$PLAN_ONLY" -eq 1 ]; then
  for n in "${LEVELS[@]}"; do
    printf '\033[1m-- concurrency %s\033[0m\n' "$n"
    printf '  %s/serve_model.sh MAX_NUM_SEQS=%s\n' "$REPO_ROOT" "$n"
    printf '  %s/benchmark.sh --bench-only HOST=%s PORT=%s CONCURRENCY=%s MAX_NUM_SEQS=%s NUM_PROMPTS=%s RANDOM_INPUT_LEN=%s RANDOM_OUTPUT_LEN=%s RESULT_FILE=%s\n\n' \
      "$REPO_ROOT" "$health_host" "$PORT" "$n" "$n" "$((n * PROMPTS_PER_LEVEL))" \
      "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" "$RESULTS_DIR/result-c$n.json"
  done
  exit 0
fi

mkdir -p "$RESULTS_DIR"

# --- the server, one level at a time -------------------------------------------

SERVE_PID=""

stop_server() {
  [ -n "$SERVE_PID" ] || return 0
  local pid="$SERVE_PID" deadline
  SERVE_PID=""

  # serve_model.sh owns its own teardown: TERM makes it signal the engine's
  # whole process group and wait for the GPUs to be released. Give it that long
  # plus a margin before insisting.
  kill -TERM "$pid" 2>/dev/null || true
  deadline=$((SECONDS + STOP_TIMEOUT))
  while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  # A level that ended badly can leave workers holding VRAM, and the next level
  # needs all of it. This is host-wide, which on a single-tenant node is what
  # you want.
  "$REPO_ROOT/model_serving/cleanup_vllm.sh" >/dev/null 2>&1 || true
}

on_signal() {
  printf '\nSignal received -- stopping the sweep.\n'
  stop_server
  exit 130
}

trap on_signal INT TERM
trap stop_server EXIT

# Launch, then poll /health ourselves rather than reading serve_model.sh's log:
# the log is for the human reading it afterwards, the poll is the control flow.
start_server() {
  local n="$1" log="$2" began deadline last_report

  "$REPO_ROOT/serve_model.sh" MAX_NUM_SEQS="$n" >"$log" 2>&1 &
  SERVE_PID=$!

  printf 'Loading the checkpoint (up to %ss), log: %s\n' "$HEALTH_TIMEOUT" "$log"
  began=$SECONDS
  deadline=$((began + HEALTH_TIMEOUT))
  last_report=$began

  until curl -sf "$HEALTH_URL" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      wait "$SERVE_PID" 2>/dev/null || true
      SERVE_PID=""
      echo "  server exited before answering /health -- last 20 lines:" >&2
      tail -20 "$log" >&2
      return 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "  timed out after ${HEALTH_TIMEOUT}s waiting for $HEALTH_URL" >&2
      echo "  still loading? raise it: ./benchmark_sweep.sh HEALTH_TIMEOUT=3600" >&2
      return 1
    fi
    if [ "$((SECONDS - last_report))" -ge 60 ]; then
      last_report=$SECONDS
      printf '  ... %sm elapsed\n' "$(((SECONDS - began) / 60))"
    fi
    sleep 5
  done

  printf 'Healthy after %ss at max-num-seqs %s\n' "$((SECONDS - began))" "$n"
}

# --- CSVs ----------------------------------------------------------------------

# Two files on purpose: the summary is the answer to "how does this node scale",
# the detailed one is everything the run measured, for when the summary raises a
# question. Both are appended per level, so an interrupted sweep still leaves
# the levels it finished.
append_csv() {
  local json="$1" n="$2" prompts="$3"
  "$PYTHON_BIN" - "$json" "$n" "$prompts" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" \
    "$SUMMARY_CSV" "$DETAILED_CSV" <<'PY'
import csv, json, os, sys

result_path, level, prompts, isl, osl, summary_csv, detailed_csv = sys.argv[1:8]

with open(result_path) as f:
    r = json.load(f)

def num(key, digits=2):
    v = r.get(key)
    return round(v, digits) if isinstance(v, (int, float)) else ""

def sec(key, digits=3):
    """A latency vLLM reports in ms, in seconds -- readable at 2k/512."""
    v = r.get(key)
    return round(v / 1000, digits) if isinstance(v, (int, float)) else ""

summary = {
    "concurrency": level,
    "mean_request_latency_s": sec("mean_e2el_ms"),
    "total_throughput_tok_s": num("total_token_throughput"),
}

detailed = {
    "concurrency": level,
    "max_num_seqs": level,
    "num_prompts": prompts,
    "completed": r.get("completed", ""),
    "input_len": isl,
    "output_len": osl,
    "duration_s": num("duration"),
    "request_throughput_req_s": num("request_throughput", 4),
    "output_throughput_tok_s": num("output_throughput"),
    "total_throughput_tok_s": num("total_token_throughput"),
    "mean_request_latency_s": sec("mean_e2el_ms"),
    "median_request_latency_s": sec("median_e2el_ms"),
    "p99_request_latency_s": sec("p99_e2el_ms"),
    "mean_ttft_ms": num("mean_ttft_ms"),
    "p99_ttft_ms": num("p99_ttft_ms"),
    "mean_tpot_ms": num("mean_tpot_ms"),
    "p99_tpot_ms": num("p99_tpot_ms"),
    "mean_itl_ms": num("mean_itl_ms"),
    "p99_itl_ms": num("p99_itl_ms"),
    "spec_decode_acceptance_rate": num("spec_decode_acceptance_rate", 4),
    "model_id": r.get("model_id", ""),
    "date": r.get("date", ""),
}

for path, row in ((summary_csv, summary), (detailed_csv, detailed)):
    is_new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
PY
}

# --- sweep ---------------------------------------------------------------------

FAILED=()
sweep_start=$SECONDS

for n in "${LEVELS[@]}"; do
  prompts=$((n * PROMPTS_PER_LEVEL))
  serve_log="$RESULTS_DIR/serve-c$n.log"
  bench_log="$RESULTS_DIR/bench-c$n.log"
  result_json="$RESULTS_DIR/result-c$n.json"

  printf '\n\033[1m== concurrency %s (%s requests, %s/%s ISL/OSL)\033[0m\n' \
    "$n" "$prompts" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN"

  if ! start_server "$n" "$serve_log"; then
    FAILED+=("$n (server)")
    stop_server
    sleep "$SETTLE_SECONDS"
    continue
  fi

  # HOST is passed explicitly because the server's bind address (0.0.0.0) is
  # inherited by this shell and is not a connectable one.
  if "$REPO_ROOT/benchmark.sh" --bench-only \
    HOST="$health_host" PORT="$PORT" \
    CONCURRENCY="$n" MAX_NUM_SEQS="$n" NUM_PROMPTS="$prompts" \
    RANDOM_INPUT_LEN="$RANDOM_INPUT_LEN" RANDOM_OUTPUT_LEN="$RANDOM_OUTPUT_LEN" \
    RESULT_FILE="$result_json" 2>&1 | tee "$bench_log"; then
    append_csv "$result_json" "$n" "$prompts"
  else
    echo "  benchmark failed at concurrency $n -- see $bench_log" >&2
    FAILED+=("$n (benchmark)")
  fi

  stop_server
  # The next level cannot allocate what this one has not finished releasing.
  sleep "$SETTLE_SECONDS"
done

# --- report --------------------------------------------------------------------

printf '\n\033[1mSweep finished in %sm\033[0m\n' "$(((SECONDS - sweep_start) / 60))"

if [ -f "$SUMMARY_CSV" ]; then
  printf '\n\033[1mbenchmark_sweep.csv\033[0m\n'
  if command -v column >/dev/null 2>&1; then
    column -s, -t <"$SUMMARY_CSV"
  else
    cat "$SUMMARY_CSV"
  fi
  printf '\n%s\n%s  (every metric the run reported)\n' "$SUMMARY_CSV" "$DETAILED_CSV"
else
  echo "No level produced a result -- nothing was written." >&2
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nFailed levels: %s\n' "${FAILED[*]}" >&2
  exit 1
fi
