#!/usr/bin/env bash
# Stop every vLLM process on this host. Send TERM first so engines can release
# resources cleanly, then force-stop anything that remains after the timeout.
#
#   ./cleanup_vllm.sh                            # 15s grace period
#   ./cleanup_vllm.sh VLLM_CLEANUP_TIMEOUT=60    # longer
#
# serve_model.sh runs this itself when it shuts down. Run it standalone to clear
# workers that outlived an earlier run and are still holding VRAM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
load_config "$@"

if [ "${#CONFIG_ARGV[@]}" -gt 0 ]; then
  echo "unknown argument: ${CONFIG_ARGV[0]} (settings are passed as KEY=value)" >&2
  exit 2
fi

TIMEOUT="$VLLM_CLEANUP_TIMEOUT"
case "$TIMEOUT" in
'' | *[!0-9]*) echo "VLLM_CLEANUP_TIMEOUT must be a non-negative integer" >&2; exit 2 ;;
esac

find_vllm_pids() {
  # vLLM names its server and worker processes with vllm/VLLM. Inspect the
  # process name rather than the full command line so a shell that merely
  # mentions this script is never selected.
  ps -eo pid=,stat=,comm= | awk -v self="$$" '
    $1 != self && $2 !~ /^Z/ && tolower($3) ~ /vllm/ { print $1 }
  '
}

mapfile -t pids < <(find_vllm_pids)
if [ "${#pids[@]}" -eq 0 ]; then
  echo "No vLLM processes found."
  exit 0
fi

echo "Stopping vLLM process(es): ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true

deadline=$((SECONDS + TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  mapfile -t pids < <(find_vllm_pids)
  [ "${#pids[@]}" -eq 0 ] && { echo "All vLLM processes stopped."; exit 0; }
  sleep 1
done

mapfile -t pids < <(find_vllm_pids)
if [ "${#pids[@]}" -gt 0 ]; then
  echo "Force-stopping remaining vLLM process(es): ${pids[*]}"
  kill -KILL "${pids[@]}" 2>/dev/null || true
fi

mapfile -t pids < <(find_vllm_pids)
if [ "${#pids[@]}" -gt 0 ]; then
  echo "Could not stop vLLM process(es): ${pids[*]}" >&2
  exit 1
fi

echo "All vLLM processes stopped."
