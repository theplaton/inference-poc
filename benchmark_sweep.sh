#!/usr/bin/env bash
# Concurrency sweep. For every request shape and concurrency level in
# benchmark_sweep_config.json, serve the model with that batch limit, benchmark
# it at that client concurrency, tear it down, and append the numbers to CSVs.
#
#   ./benchmark_sweep.sh                  # everything in the config file
#   ./benchmark_sweep.sh --plan           # the plan and the per-run commands
#   ./benchmark_sweep.sh SWEEP_CONFIG=quick.json
#   ./benchmark_sweep.sh RESULTS_DIR=out/run7
#
# Which shapes and concurrencies are worth measuring is a judgement about this
# model on these GPUs, so it lives in the config file rather than in a rule
# here. Editing that file is how you change the sweep.
#
# The shape is a client-side parameter, so every shape at a given concurrency
# is measured against the same server: the checkpoint is loaded once per
# distinct level, not once per (level, shape).
#
# Both halves of every run are the repo's own entrypoints, unchanged:
# ./serve_model.sh with DEFAULT_NUM_SEQS set to the level, then ./benchmark.sh
# --bench-only with CONCURRENCY set to the same number. Nothing here reaches
# past them into model_serving/ or benchmark/, except to clear stragglers.
#
# A run that fails does not stop the sweep: it is reported at the end and the
# CSVs keep everything that did finish.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The sweep runs the server, so it configures itself from the server's layers.
# shellcheck source=model_serving/config.sh
source "$REPO_ROOT/model_serving/config.sh"

usage() {
  cat <<'EOF'
Usage: benchmark_sweep.sh [--plan] [KEY=value ...]

Measures every (concurrency, request shape) pair in the config file, loading
the checkpoint once per distinct concurrency level.

  --plan        print the plan and the exact per-run commands, then exit
  -h, --help    this message

The sweep is described by benchmark_sweep_config.json:

  {"sweeps": [
     {"isl": 2048, "osl": 512,  "concurrency_levels": [1, 8, 64, 128, 164]},
     {"isl": 2048, "osl": 8192, "concurrency_levels": [1, 8, 40],
      "prompts_per_level": 4}
  ]}

Each entry may set prompts_per_level; without it PROMPTS_PER_LEVEL applies. The
older {"concurrency_levels": [...]} form, and a bare [1, 8, 64] array, are read
as a single sweep at RANDOM_INPUT_LEN/RANDOM_OUTPUT_LEN.

Settings follow the same four layers as the tools it drives -- an argument
outranks the environment, which outranks model_serving/.env, which outranks
model_serving/defaults.env. The sweep's own:

  SWEEP_CONFIG=...        the sweep description; defaults to
                          benchmark_sweep_config.json
  PROMPTS_PER_LEVEL=8     requests per run = this x the level
  RANDOM_INPUT_LEN=2048   ISL, for a config that does not name one
  RANDOM_OUTPUT_LEN=512   OSL, likewise
  RESULTS_DIR=...         defaults to the next number in results/
  ROLLUP_CSV=...          every sweep's rows; defaults to the parent of
                          RESULTS_DIR
  SETTLE_SECONDS=15       pause after teardown for the GPUs to come back

Everything the server understands is also settable, so a sweep can be retuned
without touching a file:

  ./benchmark_sweep.sh MAX_MODEL_LEN=131072 STRATEGY=dep
  ./benchmark_sweep.sh IGNORE_EOS=0 RANDOM_RANGE_RATIO=0.3

Requests scale with the level so every run does the same number of batch rounds
(prompts_per_level of them) -- level 1 sends 8 requests, level 164 sends 1312.
Comparing a run of two rounds against one of fifty would compare warm-up
against steady state.
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
: "${SWEEP_CONFIG:=$REPO_ROOT/benchmark_sweep_config.json}"
: "${PROMPTS_PER_LEVEL:=8}"
: "${RANDOM_INPUT_LEN:=2048}"
: "${RANDOM_OUTPUT_LEN:=512}"
: "${SETTLE_SECONDS:=15}"
: "${RESULTS_ROOT:=$REPO_ROOT/results}"
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

# Runs are numbered: results/1, results/2, ... Everything one invocation
# measures -- every shape at every level -- lands in one of them. The number is
# read off the directory that does not exist yet; claiming it is mkdir's job
# below, so two sweeps started at once cannot land on the same one.
next_run_number() {
  local n=1
  while [ -e "$RESULTS_ROOT/$n" ]; do n=$((n + 1)); done
  printf '%s' "$n"
}

claim_run_dir() {
  local n
  n="$(next_run_number)"
  mkdir -p "$RESULTS_ROOT"
  until mkdir "$RESULTS_ROOT/$n" 2>/dev/null; do
    [ -e "$RESULTS_ROOT/$n" ] || return 1 # not "taken" -- permissions, a full disk
    n=$((n + 1))
  done
  printf '%s/%s' "$RESULTS_ROOT" "$n"
}

: "${RESULTS_DIR:=$RESULTS_ROOT/$(next_run_number)}"
RUN_ID="$(basename "$RESULTS_DIR")"

SUMMARY_CSV="$RESULTS_DIR/benchmark_sweep.csv"
DETAILED_CSV="$RESULTS_DIR/benchmark_sweep_detailed.csv"
# Written by benchmark.sh rather than from here: only it knows when a
# measurement starts and stops, which is the only window worth sampling.
THERMALS_CSV="$RESULTS_DIR/gpu_thermals.csv"
# One row per run per sweep, next to the numbered folders rather than inside
# one: the point of it is comparing sweeps that differ in strategy, context or
# checkpoint, which no single folder can hold.
: "${ROLLUP_CSV:=$(dirname "$RESULTS_DIR")/benchmark_sweep_all.csv}"

# HOST is a bind address; 0.0.0.0 is not something you can connect to.
health_host="$HOST"
case "$health_host" in
0.0.0.0 | :: | '') health_host=localhost ;;
esac
HEALTH_URL="http://$health_host:$PORT/health"

# benchmark/ falls back to a plain python3 for the same reason: the venv is the
# serving install's, and it may not be the interpreter this box has.
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
  [ -n "$PYTHON_BIN" ] || { echo "no python3 found -- needed to write the CSVs" >&2; exit 1; }
fi

# --- what to measure -----------------------------------------------------------

# The plan is a list, not a rule: which shapes and concurrencies are worth an
# hour of this node is a judgement, so it is written down in one file. Parsing
# and validation live in python because that is what reads the JSON anyway; it
# prints one "level isl osl prompts" line per run, level-ascending so a level's
# shapes are adjacent and can share a server.
RUNS_TEXT="$("$PYTHON_BIN" - "$SWEEP_CONFIG" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" \
  "$PROMPTS_PER_LEVEL" <<'PY'
import json
import sys

path, default_isl, default_osl, default_prompts = sys.argv[1:5]

try:
    with open(path) as f:
        doc = json.load(f)
except OSError as e:
    sys.exit(f"cannot read the sweep config: {e}")
except ValueError as e:
    sys.exit(f"{path} is not valid JSON: {e}")


def whole(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        sys.exit(f"{path}: {what} must be a positive whole number, got {value!r}")
    return value


# Three spellings, one meaning: a list of sweeps, the older bare level list, or
# a bare array of levels. The last two describe one sweep at the ISL/OSL
# settings, which is what they meant before shapes existed.
if isinstance(doc, list):
    entries = [{"concurrency_levels": doc}]
elif isinstance(doc, dict) and "sweeps" in doc:
    entries = doc["sweeps"]
    if not isinstance(entries, list) or not entries:
        sys.exit(f'{path}: "sweeps" must be a non-empty array')
elif isinstance(doc, dict) and "concurrency_levels" in doc:
    entries = [doc]
else:
    sys.exit(f'{path}: expected a "sweeps" array or a "concurrency_levels" array')

runs = []
for i, entry in enumerate(entries):
    if not isinstance(entry, dict):
        sys.exit(f"{path}: sweep {i} must be an object, got {entry!r}")

    levels = entry.get("concurrency_levels")
    if not isinstance(levels, list) or not levels:
        sys.exit(f'{path}: sweep {i} needs a non-empty "concurrency_levels" array')

    isl = whole(entry.get("isl", int(default_isl)), f"sweep {i} isl")
    osl = whole(entry.get("osl", int(default_osl)), f"sweep {i} osl")
    per_level = whole(
        entry.get("prompts_per_level", int(default_prompts)),
        f"sweep {i} prompts_per_level",
    )

    for level in levels:
        whole(level, f"sweep {i} concurrency level")
        runs.append((level, isl, osl, level * per_level))

# Level-ascending, and stable within a level so the config's order survives.
# Ascending because a level that cannot fit should fail after the smaller ones
# have already been recorded, not before.
runs.sort(key=lambda r: r[0])

for level, isl, osl, prompts in runs:
    print(level, isl, osl, prompts)
PY
)" || exit 1

RUN_LEVELS=() RUN_ISLS=() RUN_OSLS=() RUN_PROMPTS=()
while read -r level isl osl prompts; do
  [ -n "$level" ] || continue
  RUN_LEVELS+=("$level")
  RUN_ISLS+=("$isl")
  RUN_OSLS+=("$osl")
  RUN_PROMPTS+=("$prompts")
done <<<"$RUNS_TEXT"

DISTINCT_LEVELS=()
for level in "${RUN_LEVELS[@]}"; do
  if [ "${#DISTINCT_LEVELS[@]}" -eq 0 ] || [ "${DISTINCT_LEVELS[-1]}" != "$level" ]; then
    DISTINCT_LEVELS+=("$level")
  fi
done

# --- CSVs ----------------------------------------------------------------------

# Three files, each answering a different question: the summary is "how does
# this node scale", the detailed one is everything the run measured for when the
# summary raises a question, and the rollup is "how does this sweep compare to
# the last one". All three are appended per run, so an interrupted sweep still
# leaves what it finished.
#
# One script, two entry points: `check` refuses to append to a file whose
# columns came from an older version of this script, and `write` appends a row.
# Both build their rows from the same code, so the columns cannot drift apart.
CSV_PY="$(
  cat <<'PY'
import csv
import json
import os
import sys


def build_rows(r, ctx):
    def num(key, digits=2):
        v = r.get(key)
        return round(v, digits) if isinstance(v, (int, float)) else ""

    def sec(key, digits=3):
        """A latency vLLM reports in ms, in seconds -- easier to compare."""
        v = r.get(key)
        return round(v / 1000, digits) if isinstance(v, (int, float)) else ""

    # What share of the engine's KV cache this run's requests want at once.
    # Over 100% means the level could not have run at full width whatever the
    # numbers say, because the scheduler was queueing.
    kv_fit = ""
    try:
        pool = int(ctx["kv_tokens"])
        wanted = int(ctx["level"]) * (int(ctx["isl"]) + int(ctx["osl"]))
        kv_fit = round(100 * wanted / pool, 1) if pool else ""
    except (KeyError, TypeError, ValueError):
        pass

    summary = {
        "concurrency": ctx["level"],
        "isl": ctx["isl"],
        "osl": ctx["osl"],
        "mean_request_latency_s": sec("mean_e2el_ms"),
        "total_throughput_tok_s": num("total_token_throughput"),
    }

    # No separate column for the server's batch width: the sweep always starts
    # the server at the concurrency it then drives, so it would repeat the first
    # column in every row.
    detailed = {
        "concurrency": ctx["level"],
        "num_prompts": ctx["prompts"],
        "completed": r.get("completed", ""),
        "input_len": ctx["isl"],
        "output_len": ctx["osl"],
        "total_output_tokens": r.get("total_output_tokens", ""),
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
        "kv_pool_tokens": ctx["kv_tokens"],
        "kv_fit_pct": kv_fit,
        "ignore_eos": r.get("ignore_eos", ""),
        "random_range_ratio": r.get("random_range_ratio", ""),
        "model_id": r.get("model_id", ""),
        "date": r.get("date", ""),
    }

    # Carries the settings that make two sweeps different -- without them a row
    # from a dep/131072 sweep is indistinguishable from a tep/200000 one.
    rollup = {
        "run": ctx["run_id"],
        "concurrency": ctx["level"],
        "isl": ctx["isl"],
        "osl": ctx["osl"],
        "strategy": ctx["strategy"],
        "max_model_len": ctx["max_model_len"],
        "mean_request_latency_s": sec("mean_e2el_ms"),
        "total_throughput_tok_s": num("total_token_throughput"),
        "output_throughput_tok_s": num("output_throughput"),
        "p99_request_latency_s": sec("p99_e2el_ms"),
        "kv_fit_pct": kv_fit,
        "ignore_eos": r.get("ignore_eos", ""),
        "random_range_ratio": r.get("random_range_ratio", ""),
        "model_id": r.get("model_id", ""),
    }

    return summary, detailed, rollup


BLANK_CTX = dict.fromkeys(
    ("level", "isl", "osl", "prompts", "kv_tokens", "run_id", "strategy", "max_model_len"),
    "",
)

def header_of(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return None
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


mode = sys.argv[1]

# A rollup written by an older version of this script has fewer columns. Widen
# it in place rather than refusing to append: the file exists to accumulate,
# and old rows simply have nothing to say about the new columns.
if mode == "migrate":
    for path, row in zip(sys.argv[2:5], build_rows({}, BLANK_CTX)):
        have = header_of(path)
        if have is None or have == list(row):
            continue
        added = [c for c in row if c not in have]
        if not added:
            continue
        with open(path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        merged = have + added
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged)
            writer.writeheader()
            for old in old_rows:
                writer.writerow({c: old.get(c, "") for c in merged})
        print(f"note: added {', '.join(added)} to {path}; earlier rows left blank there")
    sys.exit(0)

(result_path, summary_csv, detailed_csv, rollup_csv) = sys.argv[2:6]
ctx = dict(
    zip(
        ("level", "isl", "osl", "prompts", "kv_tokens", "run_id", "strategy", "max_model_len"),
        sys.argv[6:14],
    )
)

with open(result_path) as f:
    result = json.load(f)

for path, row in zip(
    (summary_csv, detailed_csv, rollup_csv), build_rows(result, ctx)
):
    # Follow the file's own column order when it already has one, so a widened
    # rollup keeps its shape and a row missing a column writes an empty cell
    # instead of shifting every value one place left.
    header = header_of(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=header or list(row), extrasaction="ignore"
        )
        if header is None:
            writer.writeheader()
            writer.writerow(row)
        else:
            writer.writerow({c: row.get(c, "") for c in header})
PY
)"

# Before the first checkpoint load, not after: widening the rollup is worth ten
# seconds now rather than a surprise twenty minutes in.
"$PYTHON_BIN" -c "$CSV_PY" migrate "$SUMMARY_CSV" "$DETAILED_CSV" "$ROLLUP_CSV" || exit 1

append_csv() {
  local json="$1" level="$2" isl="$3" osl="$4" prompts="$5" kv_tokens="$6"
  "$PYTHON_BIN" -c "$CSV_PY" write "$json" \
    "$SUMMARY_CSV" "$DETAILED_CSV" "$ROLLUP_CSV" \
    "$level" "$isl" "$osl" "$prompts" "$kv_tokens" "$RUN_ID" \
    "${STRATEGY:-}" "${MAX_MODEL_LEN:-}"
}

# --- plan ---------------------------------------------------------------------

printf '\033[1mConcurrency sweep\033[0m\n'
printf '  model       %s\n' "$MODEL_ID"
printf '  from        %s\n' "$SWEEP_CONFIG"
printf '  runs        %s across %s checkpoint loads (levels %s)\n' \
  "${#RUN_LEVELS[@]}" "${#DISTINCT_LEVELS[@]}" "${DISTINCT_LEVELS[*]}"
printf '  results     %s\n' "$RESULTS_DIR"
printf '  a load is ~10-20 min; shapes at the same level share one\n\n'

for i in "${!RUN_LEVELS[@]}"; do
  printf '  c=%-4s %6s/%-5s %5s requests\n' \
    "${RUN_LEVELS[$i]}" "${RUN_ISLS[$i]}" "${RUN_OSLS[$i]}" "${RUN_PROMPTS[$i]}"
done
printf '\n'

if [ "$PLAN_ONLY" -eq 1 ]; then
  for level in "${DISTINCT_LEVELS[@]}"; do
    printf '\033[1m-- concurrency %s\033[0m\n' "$level"
    printf '  %s/serve_model.sh DEFAULT_NUM_SEQS=%s\n' "$REPO_ROOT" "$level"
    for i in "${!RUN_LEVELS[@]}"; do
      [ "${RUN_LEVELS[$i]}" = "$level" ] || continue
      printf '  %s/benchmark.sh --bench-only HOST=%s PORT=%s CONCURRENCY=%s DEFAULT_NUM_SEQS=%s NUM_PROMPTS=%s RANDOM_INPUT_LEN=%s RANDOM_OUTPUT_LEN=%s RESULT_FILE=%s GPU_THERMALS_CSV=%s RUN_LABEL=%s\n' \
        "$REPO_ROOT" "$health_host" "$PORT" "$level" "$level" "${RUN_PROMPTS[$i]}" \
        "${RUN_ISLS[$i]}" "${RUN_OSLS[$i]}" \
        "$RESULTS_DIR/result-c$level-i${RUN_ISLS[$i]}o${RUN_OSLS[$i]}.json" \
        "$THERMALS_CSV" "$RUN_ID"
    done
    printf '\n'
  done
  exit 0
fi

# Claim the number now rather than at the top, so --plan leaves no folder behind
# and a sweep that aborts on a bad config does not burn one either.
if [ ! -d "$RESULTS_DIR" ]; then
  if [ "$RESULTS_DIR" = "$RESULTS_ROOT/$RUN_ID" ]; then
    RESULTS_DIR="$(claim_run_dir)" ||
      { echo "cannot create a run folder under $RESULTS_ROOT" >&2; exit 1; }
    RUN_ID="$(basename "$RESULTS_DIR")"
    SUMMARY_CSV="$RESULTS_DIR/benchmark_sweep.csv"
    DETAILED_CSV="$RESULTS_DIR/benchmark_sweep_detailed.csv"
    THERMALS_CSV="$RESULTS_DIR/gpu_thermals.csv"
    printf 'Run %s: %s\n' "$RUN_ID" "$RESULTS_DIR"
  else
    mkdir -p "$RESULTS_DIR"
  fi
fi
mkdir -p "$(dirname "$ROLLUP_CSV")"

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

  "$REPO_ROOT/serve_model.sh" DEFAULT_NUM_SEQS="$n" >"$log" 2>&1 &
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

# vLLM sizes its KV cache from whatever the weights left behind, and prints the
# result. Reading it back is how a row can say whether the requests it sent
# could all be resident, rather than leaving that to arithmetic done later.
kv_pool_tokens() {
  sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' "$1" |
    head -1 | tr -d ','
}

# --- sweep ---------------------------------------------------------------------

FAILED=()
sweep_start=$SECONDS

for level in "${DISTINCT_LEVELS[@]}"; do
  serve_log="$RESULTS_DIR/serve-c$level.log"

  printf '\n\033[1m== concurrency %s\033[0m\n' "$level"

  if ! start_server "$level" "$serve_log"; then
    for i in "${!RUN_LEVELS[@]}"; do
      [ "${RUN_LEVELS[$i]}" = "$level" ] &&
        FAILED+=("c$level ${RUN_ISLS[$i]}/${RUN_OSLS[$i]} (server)")
    done
    stop_server
    sleep "$SETTLE_SECONDS"
    continue
  fi

  kv_tokens="$(kv_pool_tokens "$serve_log")"
  [ -n "$kv_tokens" ] && printf 'KV cache: %s tokens\n' "$kv_tokens"

  # Every shape at this level shares the server that is already up -- the
  # expensive part happened once, above.
  for i in "${!RUN_LEVELS[@]}"; do
    [ "${RUN_LEVELS[$i]}" = "$level" ] || continue
    isl="${RUN_ISLS[$i]}" osl="${RUN_OSLS[$i]}" prompts="${RUN_PROMPTS[$i]}"
    shape="i${isl}o${osl}"
    bench_log="$RESULTS_DIR/bench-c$level-$shape.log"
    result_json="$RESULTS_DIR/result-c$level-$shape.json"

    printf '\n\033[1m-- %s/%s, %s requests\033[0m\n' "$isl" "$osl" "$prompts"

    # HOST is passed explicitly because the server's bind address (0.0.0.0) is
    # inherited by this shell and is not a connectable one.
    if "$REPO_ROOT/benchmark.sh" --bench-only \
      HOST="$health_host" PORT="$PORT" \
      CONCURRENCY="$level" DEFAULT_NUM_SEQS="$level" NUM_PROMPTS="$prompts" \
      RANDOM_INPUT_LEN="$isl" RANDOM_OUTPUT_LEN="$osl" \
      RESULT_FILE="$result_json" \
      GPU_THERMALS_CSV="$THERMALS_CSV" RUN_LABEL="$RUN_ID" \
      2>&1 | tee "$bench_log"; then
      append_csv "$result_json" "$level" "$isl" "$osl" "$prompts" "$kv_tokens"
    else
      echo "  benchmark failed at c$level $isl/$osl -- see $bench_log" >&2
      FAILED+=("c$level $isl/$osl (benchmark)")
    fi
  done

  stop_server
  # The next level cannot allocate what this one has not finished releasing.
  sleep "$SETTLE_SECONDS"
done

# --- report --------------------------------------------------------------------

printf '\n\033[1mSweep finished in %sm\033[0m\n' "$(((SECONDS - sweep_start) / 60))"

if [ -f "$SUMMARY_CSV" ]; then
  printf '\n\033[1mrun %s -- benchmark_sweep.csv\033[0m\n' "$RUN_ID"
  if command -v column >/dev/null 2>&1; then
    column -s, -t <"$SUMMARY_CSV"
  else
    cat "$SUMMARY_CSV"
  fi
  printf '\n%s\n%s  (every metric the run reported)\n' "$SUMMARY_CSV" "$DETAILED_CSV"
  [ -f "$THERMALS_CSV" ] &&
    printf '%s  (average temperature and power per GPU)\n' "$THERMALS_CSV"
  if [ -f "$ROLLUP_CSV" ]; then
    printf '%s  (%s rows, every sweep so far)\n' \
      "$ROLLUP_CSV" "$(($(wc -l <"$ROLLUP_CSV") - 1))"
  fi
else
  echo "Nothing was measured -- no CSV was written." >&2
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nFailed: %s\n' "${FAILED[*]}" >&2
  exit 1
fi
