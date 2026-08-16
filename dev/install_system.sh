#!/usr/bin/env bash
# Installs the system packages and uv needed to run this PoC.
# Ubuntu/Debian only. Idempotent: safe to re-run.
set -euo pipefail

PACKAGES=(
  git
  curl
  wget
  vim
  python3
  python3-pip
  python3-venv
  build-essential
)

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  command -v sudo >/dev/null || { echo "need root or sudo" >&2; exit 1; }
  SUDO="sudo"
fi

echo "==> apt-get update"
$SUDO apt-get update

echo "==> apt-get install: ${PACKAGES[*]}"
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "${PACKAGES[@]}"

if command -v uv >/dev/null; then
  echo "==> uv already installed: $(uv --version)"
else
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops these in ~/.local/bin. Exporting PATH here would only
  # affect this script, so expose them in a system PATH directory as well.
  for bin in uv uvx; do
    if [[ -x "$HOME/.local/bin/$bin" ]]; then
      $SUDO ln -sfn "$HOME/.local/bin/$bin" "/usr/local/bin/$bin"
    fi
  done
  echo "==> uv installed: $(uv --version)"
fi

echo "==> done"
