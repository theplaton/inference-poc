#!/usr/bin/env bash
# Launch the vLLM server for DeepSeek-V4-Pro on an 8x H200 node, per the
# official recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro/hw/h200.json
#
# This script does four things and nothing else: launch vLLM, wait until it
# answers /health, shut it down cleanly on a signal, and report failures.
# Installing the runtime, downloading the checkpoint, validating the host and
# clearing stragglers are separate tools next to this one -- see README.md.
#
# vLLM's output is this script's output. Keep it if you want it kept:
#   ./serve_model.sh > serve.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

usage() {
  cat <<'EOF'
Usage: serve_model.sh [--dry-run] [KEY=value ...]

Launches vLLM in the foreground and stays up until it exits or you interrupt it.

  --dry-run     print the exact launch command and exit
  -h, --help    this message

Any setting can be overridden as an argument, which outranks the environment,
.env, the profile and defaults.env:

  ./serve_model.sh PROFILE=granite         the small model, one GPU, ~1 min
  ./serve_model.sh STRATEGY=dep            data + expert parallel
  ./serve_model.sh PORT=8001 MODEL_ID=...  different endpoint and checkpoint
  ./serve_model.sh MAX_MODEL_LEN=131072    shorter context, more KV headroom
  ./serve_model.sh HEALTH_TIMEOUT=3600     allow a slower cold start

PROFILE picks the model: which checkpoint, how it is sharded, how much VRAM it
needs and which engine flags it takes. See profiles/ for what exists.

Run ./preflight.sh first; it fails in seconds on a node that cannot hold the
checkpoint, where the engine takes minutes to reach the same conclusion.
EOF
}

load_config "$@"

DRY_RUN=0
set -- ${CONFIG_ARGV[@]+"${CONFIG_ARGV[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
  --dry-run) DRY_RUN=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "unknown argument: $1 (settings are passed as KEY=value)" >&2; exit 2 ;;
  esac
done

require_config MODEL_ID

# TP splits dense layers across the GPUs; DP replicates them per rank. Both add
# --enable-expert-parallel so a MoE's experts are sharded, not copied. solo is
# one GPU and no sharding at all, which is what a model small enough to fit on
# one card wants.
case "$STRATEGY" in
tep) PARALLEL_ARGS=(--enable-expert-parallel --tensor-parallel-size "$GPU_COUNT") ;;
dep) PARALLEL_ARGS=(--enable-expert-parallel --data-parallel-size "$GPU_COUNT") ;;
solo) PARALLEL_ARGS=(--tensor-parallel-size "$GPU_COUNT") ;;
*) echo "STRATEGY must be tep, dep or solo, got '$STRATEGY'" >&2; exit 2 ;;
esac

# PROFILE_ARGS is whatever the -- lines of profiles/$PROFILE.env and its bases
# came to; config.sh collected them. Nothing here knows which flags any model
# needs, which is why adding a model is a file rather than a branch.
SERVE_ARGS=(
  "$MODEL_ID"
  "${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"}"
  "${PARALLEL_ARGS[@]}"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$DEFAULT_NUM_SEQS"
  --host "$HOST"
  --port "$PORT"
)

CMD=("$VLLM_BIN" serve "${SERVE_ARGS[@]}")

# A .env that pins MODEL_ID outranks the profile, which is the intended
# precedence but a surprising way to serve the wrong checkpoint. Say so rather
# than let twenty minutes of loading answer the question.
profile_model="$(_config_value_from "$PROFILE_FILE" MODEL_ID)"
if [ -n "$profile_model" ] && [ "$profile_model" != "$MODEL_ID" ]; then
  printf 'note: PROFILE=%s names %s, but MODEL_ID is set to %s\n\n' \
    "$PROFILE" "$profile_model" "$MODEL_ID"
fi

printf 'Launching %s (%s):\n\n' "$PROFILE" "$STRATEGY"
printf '%q ' "${CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

if [ ! -x "$VLLM_BIN" ]; then
  echo "vllm not installed at $VLLM_BIN -- run $SCRIPT_DIR/install.sh" >&2
  exit 1
fi

# --- signals ------------------------------------------------------------------

SERVE_PID=""

stop_server() {
  [ -n "$SERVE_PID" ] || return 0
  local pid="$SERVE_PID" deadline
  SERVE_PID=""

  # Signal the whole process group, not just the process we launched: vLLM forks
  # EngineCore workers that hold VRAM and outlive a TERM aimed at the parent
  # alone. `set -m` at launch is what put them in a group of their own; fall
  # back to the bare pid if that failed.
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  # Give the engine time to release GPU memory before escalating; a half torn
  # down engine leaves the GPUs unusable for the next run.
  deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
  while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
    sleep 1
  done
  kill -KILL -"$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  # Anything that escaped that group -- a re-parented worker, a stale run --
  # is what cleanup_vllm.sh is for. It is host-wide: this node is single-tenant,
  # so clearing every vLLM on the way out is intended.
  "$SCRIPT_DIR/cleanup_vllm.sh" >/dev/null 2>&1 || true
}

on_signal() {
  printf '\nSignal received -- stopping the server (up to %ss).\n' "$SHUTDOWN_TIMEOUT"
  stop_server
  exit 130
}

trap on_signal INT TERM
trap stop_server EXIT

# --- launch -------------------------------------------------------------------

# vLLM inherits this script's stdout and stderr, so its output is simply our
# output: no log file to own, no tee to orphan, nothing to clean up. Redirect
# the whole script if you want the run kept. `set -m` gives the job its own
# process group, which is what lets stop_server reach the workers vLLM forks.
set -m
"${CMD[@]}" &
SERVE_PID=$!
set +m

# --- readiness ----------------------------------------------------------------

# HOST is a bind address; 0.0.0.0 is not something you can connect to.
health_host="$HOST"
case "$health_host" in
0.0.0.0 | :: | '') health_host=localhost ;;
esac
HEALTH_URL="http://$health_host:$PORT/health"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found -- skipping the readiness check, watch the output instead"
else
  # Loading ~893 GB of shards and compiling takes many minutes; the first health
  # check will fail long before the server is actually up. vLLM's own log is the
  # progress indicator, so this prints once and then keeps quiet.
  printf 'Waiting up to %ss for %s (weights load + compile is ~10-20 min)\n' \
    "$HEALTH_TIMEOUT" "$HEALTH_URL"
  wait_start=$SECONDS
  deadline=$((wait_start + HEALTH_TIMEOUT))

  until curl -sf "$HEALTH_URL" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      status=0
      wait "$SERVE_PID" 2>/dev/null || status=$?
      SERVE_PID=""
      echo "Server exited (status $status) before becoming healthy -- see above." >&2
      exit "$((status == 0 ? 1 : status))"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "Timed out after ${HEALTH_TIMEOUT}s waiting for $HEALTH_URL." >&2
      echo "Still loading? Raise the budget: serve_model.sh HEALTH_TIMEOUT=3600" >&2
      exit 1
    fi
    sleep 5
  done

  printf '\nReady after %ss: %s on http://%s:%s\n' \
    "$((SECONDS - wait_start))" "$MODEL_ID" "$health_host" "$PORT"
  printf 'Verify it from another shell: ../benchmark.sh --smoke-only\n'
  printf 'Ctrl-C here stops the server.\n\n'
fi

# --- run ----------------------------------------------------------------------

status=0
wait "$SERVE_PID" || status=$?
SERVE_PID=""

if [ "$status" -ne 0 ]; then
  echo "Server exited with status $status." >&2
fi
exit "$status"
