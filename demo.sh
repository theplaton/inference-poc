#!/usr/bin/env bash
# End-to-end demo: install -> preflight -> download -> serve -> smoke test.
#
#   ./demo.sh                    # full run, tears the server down at the end
#   ./demo.sh --keep             # leave the server up when the smoke test passes
#   ./demo.sh --skip-install     # reuse the existing venv
#   ./demo.sh --skip-download    # weights are already in the cache
#   ./demo.sh --bench            # also run bench.sh once the server is healthy
#
# Serving is the only step that does not return on its own, so it runs in the
# background and we poll /health. Loading ~893 GB of shards takes 10-20 minutes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$REPO_ROOT/source"
# shellcheck source=source/recipe.env
source "$SRC/recipe.env"
: "${MODEL_ID:?not set -- run 'cp .env.example .env' in the repo root and edit it}"

PY="$REPO_ROOT/venv/bin/python"
SERVE_LOG="$REPO_ROOT/serve.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

SKIP_INSTALL=0
SKIP_DOWNLOAD=0
RUN_BENCH=0
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-install)  SKIP_INSTALL=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --bench)         RUN_BENCH=1; shift ;;
    --keep)          KEEP=1; shift ;;
    -h|--help)       sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

STEP=0
step() { STEP=$((STEP + 1)); printf '\n\033[1m[%d/5] %s\033[0m\n' "$STEP" "$1"; }

SERVE_PID=""
cleanup() {
  # Always stop the server and its workers on failure or Ctrl-C; on success
  # --keep clears SERVE_PID and leaves them running.
  if [ -n "$SERVE_PID" ]; then
    "$SRC/cleanup_vllm.sh" || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Clearing any vLLM processes left by an earlier run"
"$SRC/cleanup_vllm.sh"

step "Install"
if [ "$SKIP_INSTALL" -eq 1 ]; then
  echo "skipped (--skip-install)"
else
  "$SRC/install.sh"
fi

step "Preflight"
# Run it up front so a bad node fails in seconds rather than after the download.
"$SRC/preflight.sh"

step "Download"
if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
  echo "skipped (--skip-download)"
else
  "$PY" "$SRC/download.py"
fi

step "Serve"
echo "Launching in the background, logging to $SERVE_LOG"
"$SRC/serve.sh" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!

printf 'Waiting up to %ss for http://localhost:%s/health' "$HEALTH_TIMEOUT" "$PORT"
wait_start=$SECONDS
deadline=$((wait_start + HEALTH_TIMEOUT))
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    printf '\n'
    echo "Server exited before becoming healthy. Last 30 lines of $SERVE_LOG:" >&2
    tail -30 "$SERVE_LOG" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    printf '\n'
    echo "Timed out after ${HEALTH_TIMEOUT}s. Still loading? Watch: tail -f $SERVE_LOG" >&2
    exit 1
  fi
  printf '.'
  sleep 5
done
printf '\nHealthy after %ss\n' "$((SECONDS - wait_start))"

step "Smoke test"
"$PY" "$SRC/smoke_test.py"

if [ "$RUN_BENCH" -eq 1 ]; then
  printf '\n\033[1mBench\033[0m\n'
  "$SRC/bench.sh"
fi

printf '\n\033[1mDone.\033[0m %s on http://localhost:%s\n' "$MODEL_ID" "$PORT"
if [ "$KEEP" -eq 1 ]; then
  SERVE_PID=""  # defuse the trap so the server outlives this script
  echo "Server left running. Stop it with: $SRC/cleanup_vllm.sh"
  echo "Logs: $SERVE_LOG"
fi
