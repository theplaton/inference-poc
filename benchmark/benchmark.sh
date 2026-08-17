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
#   ./benchmark.sh RESULT_FILE=run.json   # also save the numbers as JSON
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
  ./benchmark.sh RESULT_FILE=run.json
  ./benchmark.sh IGNORE_EOS=0 RANDOM_RANGE_RATIO=0.3

RESULT_FILE writes the run's numbers to that path as JSON in addition to the
table on stdout, which is what ../benchmark_sweep.sh reads to build its CSVs.
IGNORE_EOS (default 1) makes every request generate the full output length;
RANDOM_RANGE_RATIO (default 0.0) samples lengths around ISL/OSL instead of
fixing them.

GPU_THERMALS_CSV appends one row of per-GPU average temperature and power,
sampled every GPU_POLL_INTERVAL seconds while the throughput run is in flight
and at no other time:

  ./benchmark.sh --bench-only GPU_THERMALS_CSV=runs/gpu_thermals.csv
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

# Concurrency above the width the server was started with does not add
# parallelism, it just queues.
if [ "$CONCURRENCY" -gt "$DEFAULT_NUM_SEQS" ]; then
  echo "note: concurrency $CONCURRENCY exceeds the server's $DEFAULT_NUM_SEQS sequences, requests will queue"
fi

# Stdout is for reading; RESULT_FILE is for a program to read afterwards. Unset
# (the default) means the run leaves nothing behind but its output.
SAVE_ARGS=()
if [ -n "${RESULT_FILE:-}" ]; then
  mkdir -p "$(dirname "$RESULT_FILE")"
  SAVE_ARGS+=(--save-result --result-filename "$RESULT_FILE")
  echo "note: saving the result JSON to $RESULT_FILE"
fi

# A model that stops early turns a 8192-token run into something shorter without
# saying so, and the number that comes out is then not the number you asked for.
EOS_ARGS=()
case "$(printf '%s' "${IGNORE_EOS:-}" | tr '[:upper:]' '[:lower:]')" in
1 | true | yes | on) EOS_ARGS+=(--ignore-eos) ;;
esac

# --- GPU thermals ---------------------------------------------------------------
# What the hardware was doing while the numbers above were being produced. The
# window is this benchmark and nothing else: the server outlives any one
# measurement, so a poller that ran with it would fold the idle gaps between
# runs into every average.
#
# Repeated from defaults.env so a deleted defaults file degrades to a working
# poll instead of an unbound-variable abort, the same way the endpoint is.
: "${GPU_POLL_INTERVAL:=2}"

GPU_SAMPLES=""
GPU_POLL_PID=""
GPU_POLL_BEGAN=0
GPU_SAMPLED_S=0

# Averaging and appending in python for the same reason the sweep's CSV writer
# is: one place builds the row, so the header and the values cannot drift apart.
GPU_CSV_PY="$(
  cat <<'PY'
import csv
import os
import sys

samples_path, csv_path = sys.argv[1:3]
run, concurrency, isl, osl, num_prompts, sampled_s = sys.argv[3:9]

# index -> [samples, temperature sum, power sum]
totals = {}
with open(samples_path) as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index, temp, power = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue  # "[N/A]" from a GPU that did not answer that round
        seen = totals.setdefault(index, [0, 0.0, 0.0])
        seen[0] += 1
        seen[1] += temp
        seen[2] += power

if not totals:
    sys.exit("warning: no usable GPU samples, so no thermals row was written")

row = {
    "run": run,
    "concurrency": concurrency,
    "isl": isl,
    "osl": osl,
    "num_prompts": num_prompts,
    "sampled_s": sampled_s,
    # The thinnest average in the row, so a figure from three samples is not
    # read with the confidence of one from three hundred.
    "samples": min(seen[0] for seen in totals.values()),
}
for index in sorted(totals):
    count, temp_sum, power_sum = totals[index]
    row[f"GPU_{index}_avg_temp"] = round(temp_sum / count, 1)
    row[f"GPU_{index}_avg_power"] = round(power_sum / count, 1)

is_new = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
if not is_new:
    with open(csv_path, newline="") as f:
        have = next(csv.reader(f), [])
    if have != list(row):
        # A different GPU count, or a different CUDA_VISIBLE_DEVICES. Not worth
        # failing a benchmark that has already produced good numbers.
        sys.exit(
            f"warning: {csv_path} has the columns of a different node, so this "
            f"run's thermals were not appended.\n  in the file: {','.join(have)}"
            f"\n  this run writes: {','.join(row)}\nMove or delete it, or point "
            f"GPU_THERMALS_CSV somewhere else."
        )

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if is_new:
        writer.writeheader()
    writer.writerow(row)
PY
)"

start_gpu_poll() {
  [ -n "${GPU_THERMALS_CSV:-}" ] || return 0

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "note: no nvidia-smi here, so no GPU thermals -- the benchmark is unaffected"
    GPU_THERMALS_CSV=""
    return 0
  fi

  # A zero or non-numeric interval would spin nvidia-smi in a tight loop and
  # perturb the very run it is measuring.
  case "$GPU_POLL_INTERVAL" in
  '' | 0 | 0.0 | . | *[!0-9.]*)
    echo "note: GPU_POLL_INTERVAL='$GPU_POLL_INTERVAL' is not a positive number, using 2" >&2
    GPU_POLL_INTERVAL=2
    ;;
  esac

  mkdir -p "$(dirname "$GPU_THERMALS_CSV")"
  GPU_SAMPLES="$(mktemp)"

  # One nvidia-smi per sample rather than `nvidia-smi -l`: each invocation
  # closes the file, so a poller killed mid-run still leaves whole samples.
  # The sleep is backgrounded and waited on so a TERM lands between samples
  # instead of after one, however long the interval is.
  (
    trap 'exit 0' TERM
    while :; do
      nvidia-smi --query-gpu=index,temperature.gpu,power.draw \
        --format=csv,noheader,nounits >>"$GPU_SAMPLES" 2>/dev/null || exit 0
      sleep "$GPU_POLL_INTERVAL" &
      wait $!
    done
  ) &
  GPU_POLL_PID=$!
  GPU_POLL_BEGAN=$SECONDS
  echo "note: sampling GPU temperature and power every ${GPU_POLL_INTERVAL}s -> $GPU_THERMALS_CSV"
}

stop_gpu_poll() {
  [ -n "$GPU_POLL_PID" ] || return 0
  GPU_SAMPLED_S=$((SECONDS - GPU_POLL_BEGAN))
  kill "$GPU_POLL_PID" 2>/dev/null || true
  wait "$GPU_POLL_PID" 2>/dev/null || true
  GPU_POLL_PID=""
}

# Called from the EXIT trap too, so it has to survive being run twice.
discard_gpu_samples() {
  stop_gpu_poll
  [ -n "$GPU_SAMPLES" ] && rm -f "$GPU_SAMPLES"
  GPU_SAMPLES=""
}

on_signal() {
  discard_gpu_samples
  exit 130
}

trap discard_gpu_samples EXIT
trap on_signal INT TERM

start_gpu_poll

# --metadata rides along into the result JSON, so a saved run says which of
# these two knobs were set rather than leaving a reader to guess. It goes last:
# the flag takes a list, and anything after it would be eaten as another pair.
#
# Run rather than exec, so the poller above can be stopped and its samples
# averaged once the benchmark is over. The status is re-raised at the end:
# ../benchmark_sweep.sh reads it to decide whether the run counts.
BENCH_STATUS=0
"$VLLM_BIN" bench serve \
  ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} \
  ${EOS_ARGS[@]+"${EOS_ARGS[@]}"} \
  --backend "$BACKEND" \
  --endpoint "$BENCH_ENDPOINT" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --dataset-name random \
  --random-input-len "$RANDOM_INPUT_LEN" \
  --random-output-len "$RANDOM_OUTPUT_LEN" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --percentile-metrics "$PERCENTILE_METRICS" \
  --metadata "ignore_eos=${IGNORE_EOS:-0}" "random_range_ratio=$RANDOM_RANGE_RATIO" ||
  BENCH_STATUS=$?

stop_gpu_poll

# Only a run that finished gets a row: a failed one's partial averages would sit
# in the CSV looking like a measurement of something.
if [ "$BENCH_STATUS" -eq 0 ] && [ -n "${GPU_THERMALS_CSV:-}" ] && [ -n "$GPU_SAMPLES" ]; then
  "$PYTHON_BIN" -c "$GPU_CSV_PY" "$GPU_SAMPLES" "$GPU_THERMALS_CSV" \
    "${RUN_LABEL:-}" "$CONCURRENCY" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" \
    "$NUM_PROMPTS" "$GPU_SAMPLED_S" ||
    true # its own message says what went wrong; the numbers above still stand
fi

discard_gpu_samples
exit "$BENCH_STATUS"
