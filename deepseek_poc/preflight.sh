#!/usr/bin/env bash
# Check that this box can actually hold DeepSeek-V4-Pro before you wait out a
# ~900 GB download or a multi-minute engine start that ends in OOM.
#
# Run it directly, or let serve.sh source it (RECIPE_SOURCED=1 suppresses the
# summary and returns instead of exiting).
set -uo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipe.env
source "$RECIPE_DIR/recipe.env"

FAILURES=0
WARNINGS=0

pass() { printf '  ok    %s\n' "$1"; }
warn() { printf '  warn  %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

# `sort -V` gives us semver ordering without pulling in python.
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

check_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found -- this recipe needs an NVIDIA H200 node"
    return
  fi

  local mems count total_mib total_gb min_mib
  mapfile -t mems < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  count=${#mems[@]}

  if [ "$count" -ne "$GPU_COUNT" ]; then
    fail "found $count GPU(s), the recipe is built for $GPU_COUNT"
  else
    pass "$count GPUs visible"
  fi

  total_mib=0
  min_mib=${mems[0]:-0}
  for m in "${mems[@]}"; do
    total_mib=$((total_mib + m))
    [ "$m" -lt "$min_mib" ] && min_mib=$m
  done
  total_gb=$((total_mib / 1024))

  # H200 SXM reports ~140 GiB usable out of its nominal 141 GB.
  if [ "$min_mib" -lt 140000 ]; then
    warn "smallest GPU has $((min_mib / 1024)) GiB -- H200 SXM should report ~140 GiB"
  fi

  if [ "$total_gb" -lt "$CHECKPOINT_VRAM_GB" ]; then
    fail "total VRAM ${total_gb} GB < ${CHECKPOINT_VRAM_GB} GB of weights -- the checkpoint will not fit"
  else
    pass "total VRAM ${total_gb} GB, ~$((total_gb - CHECKPOINT_VRAM_GB)) GB left for KV cache after weights"
  fi
}

check_shm() {
  # vLLM's multiproc executor puts its RPC ring buffers in POSIX shared memory:
  # one broadcast queue (~160 MiB) plus a 240 MiB response queue per worker. The
  # 64 MiB that containers get by default dies at engine init, ~40 s in.
  local required_mib avail_mib
  required_mib=$((160 + 240 * GPU_COUNT))
  avail_mib=$(($(df -Pk /dev/shm | awk 'NR==2 {print $4}') / 1024))

  if [ "$avail_mib" -lt "$required_mib" ]; then
    fail "/dev/shm has ${avail_mib} MiB, need >= ${required_mib} MiB for ${GPU_COUNT} workers -- remount it or add --shm-size / --ipc=host (k8s: an emptyDir with medium: Memory at /dev/shm)"
  elif [ "$avail_mib" -lt $((required_mib * 2)) ]; then
    warn "/dev/shm has ${avail_mib} MiB, just over the ${required_mib} MiB floor -- NCCL and torch also draw on it"
  else
    pass "/dev/shm has $((avail_mib / 1024)) GiB (need ~$((required_mib / 1024)) GiB)"
  fi
}

check_disk() {
  local cache avail_gb
  cache="${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}"
  cache="${cache:-$HOME/.cache/huggingface}"
  # Walk up to the nearest existing parent; the cache dir may not exist yet.
  while [ ! -d "$cache" ] && [ "$cache" != "/" ]; do cache="$(dirname "$cache")"; done

  avail_gb=$(($(df -Pk "$cache" | awk 'NR==2 {print $4}') / 1024 / 1024))
  if [ "$avail_gb" -lt "$CHECKPOINT_DISK_GB" ]; then
    fail "$cache has ${avail_gb} GB free, need ~${CHECKPOINT_DISK_GB} GB for the shards"
  else
    pass "$cache has ${avail_gb} GB free"
  fi
}

check_vllm() {
  if [ "${RECIPE_RUNTIME:-native}" = "docker" ]; then
    if command -v docker >/dev/null 2>&1; then
      pass "docker present (image $VLLM_DOCKER_IMAGE pinned in recipe.env)"
    else
      fail "docker not found but RECIPE_RUNTIME=docker"
    fi
    return
  fi

  if ! command -v vllm >/dev/null 2>&1; then
    fail "vllm not on PATH -- see install.sh"
    return
  fi

  local version
  version="$(vllm --version 2>/dev/null | tr -d '[:space:]')"
  if [ -z "$version" ]; then
    warn "could not read 'vllm --version'; need >= $MIN_VLLM_VERSION"
  elif version_ge "$version" "$MIN_VLLM_VERSION"; then
    pass "vllm $version (>= $MIN_VLLM_VERSION)"
  else
    # The DSpark draft module in the 0813 checkpoint is the hard requirement.
    fail "vllm $version < $MIN_VLLM_VERSION required by the fused 0813 checkpoint"
  fi
}

check_context() {
  # Think Max truncates below 384K, but 384K of KV does not fit next to ~960 GB
  # of weights on this node. Surface the tradeoff rather than letting it surprise.
  if [ "$MAX_MODEL_LEN" -lt 393216 ]; then
    pass "MAX_MODEL_LEN=$MAX_MODEL_LEN (Think Max needs >= 393216 and will truncate)"
  else
    warn "MAX_MODEL_LEN=$MAX_MODEL_LEN enables Think Max but likely OOMs without KV offload"
  fi
}

printf 'Preflight: DeepSeek-V4-Pro on %sx H200 (strategy=%s)\n' "$GPU_COUNT" "$STRATEGY"
check_gpus
check_shm
check_disk
check_vllm
check_context

if [ "${RECIPE_SOURCED:-0}" = "1" ]; then
  return "$FAILURES"
fi

printf '\n%d failure(s), %d warning(s)\n' "$FAILURES" "$WARNINGS"
exit $((FAILURES > 0 ? 1 : 0))
