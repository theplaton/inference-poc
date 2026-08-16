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
  echo "uv not found." >&2
  exit 1
fi

if [ -d "$VENV" ]; then
  echo "Reusing $VENV"
else
  echo "Creating $VENV"
  uv venv "$VENV"
fi

# One install from requirements.txt: vllm (>= $MIN_VLLM_VERSION, the floor
# declared there), the openai client, and -r ../requirements.txt for the Hub.
echo "Installing from $RECIPE_DIR/requirements.txt (vllm >= $MIN_VLLM_VERSION)"
VIRTUAL_ENV="$VENV" uv pip install -U -r "$RECIPE_DIR/requirements.txt"

echo
echo "Done. Project scripts use $VENV automatically; activation is not required."
"$VENV/bin/python" - <<'PY'
import torch
import torchaudio
import torchvision

packages = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
}
print("Torch stack:", ", ".join(f"{name}={version}" for name, version in packages.items()))
print("PyTorch CUDA:", torch.version.cuda)
PY
"$VENV/bin/vllm" --version
