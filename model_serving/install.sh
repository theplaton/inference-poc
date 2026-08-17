#!/usr/bin/env bash
# Install the serving runtime into venv/ at the repo root, from
# model_serving/requirements.txt -- vllm, the openai client and huggingface_hub.
#
#   ./install.sh                     # create or reuse the repo-level venv/
#   ./install.sh VENV=/opt/vllm-env  # somewhere else
#
# The recipe's own install block is:
#   uv venv && source .venv/bin/activate && uv pip install -U vllm --torch-backend auto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
load_config "$@"

if [ "${#CONFIG_ARGV[@]}" -gt 0 ]; then
  echo "unknown argument: ${CONFIG_ARGV[0]} (settings are passed as KEY=value)" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found -- install it with dev/install_system.sh" >&2
  exit 1
fi

if [ -d "$VENV" ]; then
  echo "Reusing $VENV"
else
  echo "Creating $VENV"
  uv venv "$VENV"
fi

# One install from requirements.txt: vllm (>= $MIN_VLLM_VERSION, the floor
# declared there), the openai client, and huggingface_hub for the download.
echo "Installing from $SCRIPT_DIR/requirements.txt (vllm >= $MIN_VLLM_VERSION)"
VIRTUAL_ENV="$VENV" uv pip install -U -r "$SCRIPT_DIR/requirements.txt"

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
