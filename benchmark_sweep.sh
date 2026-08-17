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
# The shape is a client-side parameter, so every shape at a given server and
# concurrency is measured against the same server: the checkpoint is loaded once
# per (profile, tp, dp, level), not once per run.
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
     {"profile": "deepseek_v4_speculative", "tp": 8, "dp": 1,
      "isl": 2048, "osl": 512, "concurrency_levels": [1, 8, 64, 128]},
     {"profile": "deepseek_v4_speculative", "tp": 4, "dp": 2,
      "isl": 2048, "osl": 512, "concurrency_levels": [1, 8, 64, 128]}
  ]}

An entry may set profile, tp, dp, isl, osl and prompts_per_level; each falls
back to this run's setting for it. profile chooses the checkpoint and engine
flags, tp/dp the parallel layout -- so the pair above measures one ladder
against two servers. An unknown key is an error rather than an oversight.

The checkpoint reloads whenever profile, tp, dp or the level changes, so the
number of loads is levels x layouts. --plan prints that count first.

The older {"concurrency_levels": [...]} form, and a bare [1, 8, 64] array, are
read as a single sweep at this run's settings.

Settings follow the same layers as the tools it drives -- an argument
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

  ./benchmark_sweep.sh MAX_MODEL_LEN=131072 TP_SIZE=4 DP_SIZE=2
  ./benchmark_sweep.sh IGNORE_EOS=0 RANDOM_RANGE_RATIO=0.3

These are the defaults a config entry may leave out, so setting one here retunes
every entry that does not name its own.

Requests scale with the level so every run does the same number of batch rounds
(prompts_per_level of them) -- level 1 sends 8 requests, level 164 sends 1312.
Comparing a run of two rounds against one of fifty would compare warm-up
against steady state.
EOF
}

# Settings a profile owns. The sweep resolves its own from whichever profile it
# was started under and exports them, and an exported value outranks a profile
# -- so a group naming a different profile would be served this run's checkpoint
# under that profile's name. They are cleared for the children below, which then
# resolve each one from the profile they were given.
PROFILE_OWNED=(
  MODEL_ID MODEL_REVISION
  TP_SIZE DP_SIZE EXPERT_PARALLEL GPU_COUNT
  MAX_MODEL_LEN GPU_MEMORY_UTILIZATION DEFAULT_NUM_SEQS HEALTH_TIMEOUT
  CHECKPOINT_VRAM_GB CHECKPOINT_DISK_GB
  SMOKE_MODES SMOKE_MODES_ALL TEMPERATURE TOP_P
)

# What this run set for itself, asked before load_config fills the same names in
# from a profile -- afterwards the two are indistinguishable. An explicit
# setting is a deliberate override and survives into every group; the rest are
# just this run's profile showing through, and must not.
EXPLICIT_SETTINGS=" "
for _n in "${PROFILE_OWNED[@]}"; do
  [ -n "${!_n:-}" ] && EXPLICIT_SETTINGS="$EXPLICIT_SETTINGS$_n "
done
for _arg in "$@"; do
  case "$_arg" in
  [A-Za-z_]*=*) EXPLICIT_SETTINGS="$EXPLICIT_SETTINGS${_arg%%=*} " ;;
  esac
done

# The layout to fall back on when a config entry names none: this run's, if it
# was told one, and otherwise the marker that means "ask the entry's profile".
TP_ARG="-" DP_ARG="-"
case "$EXPLICIT_SETTINGS" in *" TP_SIZE "*) TP_ARG="$TP_SIZE" ;; esac
case "$EXPLICIT_SETTINGS" in *" DP_SIZE "*) DP_ARG="$DP_SIZE" ;; esac

ENV_CLEAR=()
for _n in "${PROFILE_OWNED[@]}"; do
  case "$EXPLICIT_SETTINGS" in *" $_n "*) continue ;; esac
  ENV_CLEAR+=(-u "$_n")
done

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
TELEMETRY_CSV="$RESULTS_DIR/gpu_telemetry.csv"
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

# The plan is a list, not a rule: which shapes, concurrencies and server layouts
# are worth an hour of this node is a judgement, so it is written down in one
# file rather than derived here. benchmark/sweep_config.py reads and validates it
# and prints one "profile tp dp level isl osl prompts" line per run, sorted so
# that every run able to share a server is adjacent.
#
# The sweep's own resolved settings are the defaults a config entry may leave
# out, which is why an entry naming nothing but levels still describes a server.
RUNS_TEXT="$("$PYTHON_BIN" "$REPO_ROOT/benchmark/sweep_config.py" \
  "$SWEEP_CONFIG" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" "$PROMPTS_PER_LEVEL" \
  "$PROFILE" "$TP_ARG" "$DP_ARG")" || exit 1

# One key out of a profile, following PROFILE_BASE the way load_config does.
# Needed because an entry may name a profile without naming a layout, and that
# profile's layout is the one it means -- granite is one GPU whatever the run
# that mentions it was otherwise set to.
profile_setting() {
  local profile="$1" key="$2" seen="" file value
  while [ -n "$profile" ]; do
    case " $seen " in *" $profile "*) return 0 ;; esac
    seen="$seen $profile"
    file="$REPO_ROOT/model_serving/profiles/$profile.env"
    [ -f "$file" ] || return 0
    value="$(_config_value_from "$file" "$key")"
    [ -n "$value" ] && { printf '%s' "$value"; return 0; }
    profile="$(_config_value_from "$file" PROFILE_BASE)"
  done
}

RUN_PROFILES=() RUN_TPS=() RUN_DPS=()
RUN_LEVELS=() RUN_ISLS=() RUN_OSLS=() RUN_PROMPTS=()
while read -r profile tp dp level isl osl prompts; do
  [ -n "$profile" ] || continue
  RUN_PROFILES+=("$profile")
  RUN_TPS+=("$tp")
  RUN_DPS+=("$dp")
  RUN_LEVELS+=("$level")
  RUN_ISLS+=("$isl")
  RUN_OSLS+=("$osl")
  RUN_PROMPTS+=("$prompts")
done <<<"$RUNS_TEXT"

# Resolve the "use the profile's own" marker before anything groups on it, so
# every later step -- grouping, filenames, the CSV -- sees concrete numbers.
for i in "${!RUN_LEVELS[@]}"; do
  [ "${RUN_TPS[$i]}" = "-" ] && RUN_TPS[$i]="$(profile_setting "${RUN_PROFILES[$i]}" TP_SIZE)"
  [ "${RUN_DPS[$i]}" = "-" ] && RUN_DPS[$i]="$(profile_setting "${RUN_PROFILES[$i]}" DP_SIZE)"
  if [ -z "${RUN_TPS[$i]}" ] || [ -z "${RUN_DPS[$i]}" ]; then
    printf 'error: profile "%s" sets no TP_SIZE/DP_SIZE, and %s does not either.\n' \
      "${RUN_PROFILES[$i]}" "$SWEEP_CONFIG" >&2
    exit 1
  fi
done

# What a run needs a server to be. Everything else -- the shape, the request
# count -- is a client-side parameter that costs nothing to vary, so one load
# serves every run sharing this key.
run_group() { printf '%s %s %s %s' "${RUN_PROFILES[$1]}" "${RUN_TPS[$1]}" "${RUN_DPS[$1]}" "${RUN_LEVELS[$1]}"; }

# Filenames have to carry the same key: a sweep across two layouts would
# otherwise write both to result-c64-i2048o512.json and keep only the second.
run_slug() { printf '%s-tp%sdp%s-c%s' "${RUN_PROFILES[$1]}" "${RUN_TPS[$1]}" "${RUN_DPS[$1]}" "${RUN_LEVELS[$1]}"; }

# Dedup against every key kept so far, not just the last one. The sort put
# equal keys next to each other, but only as they were written: an entry that
# named tp/dp and one that let the profile supply the same numbers sort apart
# and resolve together, and taking both would load the identical server twice
# and give the two runs one filename between them.
SERVER_GROUPS=()
for i in "${!RUN_LEVELS[@]}"; do
  key="$(run_group "$i")"
  seen=0
  for have in ${SERVER_GROUPS[@]+"${SERVER_GROUPS[@]}"}; do
    [ "$have" = "$key" ] && { seen=1; break; }
  done
  [ "$seen" -eq 0 ] && SERVER_GROUPS+=("$key")
done

# A profile that does not exist is a typo worth catching now rather than after
# the first load: both halves need a file, and the client's is easy to forget.
for key in "${SERVER_GROUPS[@]}"; do
  read -r g_profile _ _ _ <<<"$key"
  for side in model_serving benchmark; do
    [ -f "$REPO_ROOT/$side/profiles/$g_profile.env" ] && continue
    printf 'error: %s names profile "%s", but %s/profiles/%s.env does not exist.\n' \
      "$SWEEP_CONFIG" "$g_profile" "$side" "$g_profile" >&2
    printf 'Available: %s\n' \
      "$(ls "$REPO_ROOT/$side/profiles" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')" >&2
    exit 1
  done
done

# --- CSVs ----------------------------------------------------------------------

# Three files, each answering a different question: the summary is "how does
# this node scale", the detailed one is everything the run measured for when the
# summary raises a question, and the rollup is "how does this sweep compare to
# the last one". All three are appended per run, so an interrupted sweep still
# leaves what it finished.
#
# benchmark/sweep_csv.py owns their columns: `migrate` widens a CSV written by
# an older version of it, `write` appends a row, and both build the row from the
# same code so the columns cannot drift apart.

# Before the first checkpoint load, not after: widening the rollup is worth ten
# seconds now rather than a surprise twenty minutes in.
"$PYTHON_BIN" "$REPO_ROOT/benchmark/sweep_csv.py" migrate \
  "$SUMMARY_CSV" "$DETAILED_CSV" "$ROLLUP_CSV" || exit 1

append_csv() {
  local json="$1" level="$2" isl="$3" osl="$4" prompts="$5" kv_tokens="$6"
  local profile="$7" tp="$8" dp="$9"
  "$PYTHON_BIN" "$REPO_ROOT/benchmark/sweep_csv.py" write "$json" \
    "$SUMMARY_CSV" "$DETAILED_CSV" "$ROLLUP_CSV" \
    "$level" "$isl" "$osl" "$prompts" "$kv_tokens" "$RUN_ID" \
    "${MAX_MODEL_LEN:-}" "$profile" "$tp" "$dp"
}

# --- plan ---------------------------------------------------------------------

printf '\033[1mConcurrency sweep\033[0m\n'
printf '  from        %s\n' "$SWEEP_CONFIG"
printf '  runs        %s across %s checkpoint loads\n' \
  "${#RUN_LEVELS[@]}" "${#SERVER_GROUPS[@]}"
printf '  results     %s\n' "$RESULTS_DIR"
printf '  a load is ~10-20 min; every run of one server+level shares one\n\n'

# Grouped by server, because that is what the sweep costs: a layout the config
# mentions once still reloads the checkpoint at every level it asks for, so the
# bill is levels x layouts and is worth seeing before committing to it.
printf '\033[1m  %-30s %-9s %s\033[0m\n' "profile" "parallel" "levels"
last_sig="" line=""
for key in "${SERVER_GROUPS[@]}"; do
  read -r g_profile g_tp g_dp g_level <<<"$key"
  sig="$g_profile tp${g_tp}dp${g_dp}"
  if [ "$sig" != "$last_sig" ]; then
    [ -n "$last_sig" ] && printf '%s\n' "$line"
    line="$(printf '  %-30s %-9s %s' "$g_profile" "tp${g_tp}dp${g_dp}" "$g_level")"
    last_sig="$sig"
  else
    line="$line $g_level"
  fi
done
[ -n "$last_sig" ] && printf '%s\n' "$line"
printf '\n'

for i in "${!RUN_LEVELS[@]}"; do
  printf '  %-26s c=%-5s %6s/%-5s %5s requests\n' \
    "${RUN_PROFILES[$i]} tp${RUN_TPS[$i]}dp${RUN_DPS[$i]}" \
    "${RUN_LEVELS[$i]}" "${RUN_ISLS[$i]}" "${RUN_OSLS[$i]}" "${RUN_PROMPTS[$i]}"
done
printf '\n'

if [ "$PLAN_ONLY" -eq 1 ]; then
  for key in "${SERVER_GROUPS[@]}"; do
    read -r g_profile g_tp g_dp g_level <<<"$key"
    printf '\033[1m-- %s tp%sdp%s, concurrency %s\033[0m\n' \
      "$g_profile" "$g_tp" "$g_dp" "$g_level"
    printf '  %s/serve_model.sh PROFILE=%s TP_SIZE=%s DP_SIZE=%s DEFAULT_NUM_SEQS=%s\n' \
      "$REPO_ROOT" "$g_profile" "$g_tp" "$g_dp" "$g_level"
    for i in "${!RUN_LEVELS[@]}"; do
      [ "$(run_group "$i")" = "$key" ] || continue
      printf '  %s/benchmark.sh --bench-only PROFILE=%s HOST=%s PORT=%s CONCURRENCY=%s DEFAULT_NUM_SEQS=%s NUM_PROMPTS=%s RANDOM_INPUT_LEN=%s RANDOM_OUTPUT_LEN=%s RESULT_FILE=%s GPU_TELEMETRY_CSV=%s RUN_LABEL=%s\n' \
        "$REPO_ROOT" "$g_profile" "$health_host" "$PORT" "$g_level" "$g_level" \
        "${RUN_PROMPTS[$i]}" "${RUN_ISLS[$i]}" "${RUN_OSLS[$i]}" \
        "$RESULTS_DIR/result-$(run_slug "$i")-i${RUN_ISLS[$i]}o${RUN_OSLS[$i]}.json" \
        "$TELEMETRY_CSV" "$RUN_ID"
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
    TELEMETRY_CSV="$RESULTS_DIR/gpu_telemetry.csv"
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
  local n="$1" log="$2" profile="$3" tp="$4" dp="$5" began deadline last_report free_by

  # Refuse to adopt a server this sweep did not start. The health poll below
  # cannot tell ours from a survivor of an earlier run: if the port is taken,
  # our vLLM exits on the bind and the poll succeeds against the survivor. With
  # granite that shows up as 404s, but every deepseek profile serves the same
  # model name, so a stale one answers every request and its numbers get written
  # under this profile -- wrong data, and wrong in exactly the comparison these
  # profiles exist for.
  #
  # A teardown still in flight is not that, so give it SETTLE_SECONDS to finish
  # before calling the port occupied.
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    free_by=$((SECONDS + SETTLE_SECONDS))
    while curl -sf "$HEALTH_URL" >/dev/null 2>&1 && [ "$SECONDS" -lt "$free_by" ]; do
      sleep 1
    done
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      echo "  something is already serving on $HEALTH_URL -- refusing to benchmark it" >&2
      echo "  stop it, or clear stragglers: $REPO_ROOT/model_serving/cleanup_vllm.sh" >&2
      return 1
    fi
  fi

  # Passed as arguments rather than exported, so each group gets exactly the
  # server its config entry asked for and nothing leaks into the next one. The
  # env clearing is the other half of that: without it this sweep's own profile
  # would still be showing through every name it did not pass here.
  env "${ENV_CLEAR[@]+"${ENV_CLEAR[@]}"}" "$REPO_ROOT/serve_model.sh" \
    PROFILE="$profile" TP_SIZE="$tp" DP_SIZE="$dp" DEFAULT_NUM_SEQS="$n" \
    >"$log" 2>&1 &
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

for key in "${SERVER_GROUPS[@]}"; do
  read -r g_profile g_tp g_dp level <<<"$key"
  g_label="$g_profile tp${g_tp}dp${g_dp}"
  # The group's own slug, from the first run that belongs to it -- every run in
  # the group shares it, since it is built from exactly the grouping key.
  for i in "${!RUN_LEVELS[@]}"; do
    [ "$(run_group "$i")" = "$key" ] && { g_slug="$(run_slug "$i")"; break; }
  done
  serve_log="$RESULTS_DIR/serve-$g_slug.log"

  printf '\n\033[1m== %s, concurrency %s\033[0m\n' "$g_label" "$level"

  if ! start_server "$level" "$serve_log" "$g_profile" "$g_tp" "$g_dp"; then
    for i in "${!RUN_LEVELS[@]}"; do
      [ "$(run_group "$i")" = "$key" ] &&
        FAILED+=("$g_label c$level ${RUN_ISLS[$i]}/${RUN_OSLS[$i]} (server)")
    done
    stop_server
    sleep "$SETTLE_SECONDS"
    continue
  fi

  kv_tokens="$(kv_pool_tokens "$serve_log")"
  [ -n "$kv_tokens" ] && printf 'KV cache: %s tokens\n' "$kv_tokens"

  # Every shape at this server and level shares the one that is already up --
  # the expensive part happened once, above.
  for i in "${!RUN_LEVELS[@]}"; do
    [ "$(run_group "$i")" = "$key" ] || continue
    isl="${RUN_ISLS[$i]}" osl="${RUN_OSLS[$i]}" prompts="${RUN_PROMPTS[$i]}"
    shape="i${isl}o${osl}"
    bench_log="$RESULTS_DIR/bench-$(run_slug "$i")-$shape.log"
    result_json="$RESULTS_DIR/result-$(run_slug "$i")-$shape.json"

    printf '\n\033[1m-- %s/%s, %s requests\033[0m\n' "$isl" "$osl" "$prompts"

    # HOST is passed explicitly because the server's bind address (0.0.0.0) is
    # inherited by this shell and is not a connectable one. PROFILE goes with it
    # so the client resolves the same checkpoint this server was started with.
    if env "${ENV_CLEAR[@]+"${ENV_CLEAR[@]}"}" "$REPO_ROOT/benchmark.sh" --bench-only \
      PROFILE="$g_profile" HOST="$health_host" PORT="$PORT" \
      CONCURRENCY="$level" DEFAULT_NUM_SEQS="$level" NUM_PROMPTS="$prompts" \
      RANDOM_INPUT_LEN="$isl" RANDOM_OUTPUT_LEN="$osl" \
      RESULT_FILE="$result_json" \
      GPU_TELEMETRY_CSV="$TELEMETRY_CSV" RUN_LABEL="$RUN_ID" \
      2>&1 | tee "$bench_log"; then
      append_csv "$result_json" "$level" "$isl" "$osl" "$prompts" "$kv_tokens" \
        "$g_profile" "$g_tp" "$g_dp"
    else
      echo "  benchmark failed at $g_label c$level $isl/$osl -- see $bench_log" >&2
      FAILED+=("$g_label c$level $isl/$osl (benchmark)")
    fi
  done

  stop_server
  # The next server cannot allocate what this one has not finished releasing.
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
  [ -f "$TELEMETRY_CSV" ] &&
    printf '%s  (average temperature, power and utilization per GPU)\n' "$TELEMETRY_CSV"
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
