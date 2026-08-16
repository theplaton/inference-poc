#!/usr/bin/env bash
# Throughput/latency against a already-running server. Optional -- the recipe
# does not publish H200 numbers, so this is for measuring your own node.
#
#   ./bench.sh                # 32 prompts, 8 concurrent
#   ./bench.sh 128 32         # prompts, concurrency
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipe.env
source "$RECIPE_DIR/recipe.env"
: "${MODEL_ID:?not set -- run 'cp .env.example .env' in the repo root and edit it}"

NUM_PROMPTS="${1:-32}"
CONCURRENCY="${2:-8}"

if ! curl -sf "http://localhost:$PORT/health" >/dev/null; then
  echo "No healthy server on port $PORT -- start serve.sh first." >&2
  exit 1
fi

# Concurrency is capped by the server's --max-num-seqs (16 in the recipe);
# asking for more just queues.
if [ "$CONCURRENCY" -gt "$MAX_NUM_SEQS" ]; then
  echo "note: concurrency $CONCURRENCY exceeds --max-num-seqs $MAX_NUM_SEQS, requests will queue"
fi

exec "$VLLM_BIN" bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --host localhost \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --percentile-metrics ttft,tpot,itl,e2el
