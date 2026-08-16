#!/usr/bin/env bash
# Install the whole python environment into venv/ at the repo root, from
# source/requirements.txt -- vllm, the openai client, and huggingface_hub.
#
# The recipe's own install block is:
#   uv venv && source .venv/bin/activate && uv pip install -U vllm --torch-backend auto
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipe.env
source "$RECIPE_DIR/recipe.env"

VENV="$REPO_ROOT/venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ -d "$VENV" ]; then
  # Re-running should not throw away a multi-GB torch build.
  echo "Reusing $VENV"
else
  echo "Creating $VENV"
  uv venv "$VENV"
fi

# One install from requirements.txt: vllm (>= $MIN_VLLM_VERSION, the floor
# declared there), the openai client, and -r ../requirements.txt for the Hub.
# --torch-backend auto lets uv resolve the CUDA wheel that matches the driver.
echo "Installing from $RECIPE_DIR/requirements.txt (vllm >= $MIN_VLLM_VERSION)"
VIRTUAL_ENV="$VENV" uv pip install -U -r "$RECIPE_DIR/requirements.txt" --torch-backend auto

echo
echo "Done. Activate with:  source $VENV/bin/activate"
"$VENV/bin/vllm" --version
