#!/usr/bin/env bash
# Serve DeepSeek-V4-Pro on an 8x H200 node, per the official vLLM recipe:
#   https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro/hw/h200.json
#
#   ./serve.sh                     # recipe default: tensor + expert parallel
#   ./serve.sh --strategy dep      # data + expert parallel
#   ./serve.sh --docker            # run the pinned vllm/vllm-openai image
#   ./serve.sh --dry-run           # print the command and exit
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipe.env
source "$RECIPE_DIR/recipe.env"

RUNTIME=native
DRY_RUN=0
SKIP_PREFLIGHT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --docker) RUNTIME=docker; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# TP splits dense layers across all 8 GPUs; DP replicates them per rank. Both
# add --enable-expert-parallel so the 1.6T of experts are sharded, not copied.
case "$STRATEGY" in
  tep) PARALLEL_ARGS=(--enable-expert-parallel --tensor-parallel-size "$GPU_COUNT") ;;
  dep) PARALLEL_ARGS=(--enable-expert-parallel --data-parallel-size "$GPU_COUNT") ;;
  *) echo "STRATEGY must be tep or dep, got '$STRATEGY'" >&2; exit 2 ;;
esac

if [ "$SKIP_PREFLIGHT" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  RECIPE_SOURCED=1 RECIPE_RUNTIME="$RUNTIME" source "$RECIPE_DIR/preflight.sh"
  if [ "$FAILURES" -gt 0 ]; then
    echo "Preflight failed ($FAILURES). Fix the above, or pass --skip-preflight." >&2
    exit 1
  fi
  echo
fi

# Flags below are verbatim from the recipe's h200.json. The four after
# --max-model-len are the Hopper overrides: FP4/FP8 kernels on Hopper need the
# conservative compile mode, and flashinfer autotune is off because it does not
# pay for itself here.
SERVE_ARGS=(
  "$MODEL_ID"
  --trust-remote-code
  --kv-cache-dtype fp8
  --block-size 256
  "${PARALLEL_ARGS[@]}"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --no-enable-flashinfer-autotune
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
  --host "$HOST"
  --port "$PORT"
)

if [ "$RUNTIME" = "docker" ]; then
  CMD=(
    docker run --gpus all --privileged --ipc=host
    -p "$PORT:$PORT"
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    "$VLLM_DOCKER_IMAGE"
    "${SERVE_ARGS[@]}"
  )
else
  CMD=(vllm serve "${SERVE_ARGS[@]}")
fi

printf 'Launching (%s, %s):\n\n' "$RUNTIME" "$STRATEGY"
printf '%q ' "${CMD[@]}"
printf '\n\n'

[ "$DRY_RUN" -eq 1 ] && exit 0

# Loading ~893 GB of shards and compiling takes many minutes; the first health
# check will fail long before the server is actually up.
echo "Weights load + compile takes ~10-20 min. Readiness: curl localhost:$PORT/health"
exec "${CMD[@]}"
