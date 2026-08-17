# Configuration for the model_serving tools. Source it, then call
# `load_config "$@"`; arguments that are not KEY=value are left in CONFIG_ARGV
# for the calling script's own flag parser.
#
# Precedence, highest wins:
#
#   1. KEY=value arguments      ./serve_model.sh PORT=8001
#   2. exported environment     PORT=8001 ./serve_model.sh
#   3. model_serving/.env       local, gitignored, holds tokens
#   4. model_serving/defaults.env   committed defaults
#
# Each layer only fills in keys no higher layer has set, so the order the
# layers are applied in *is* the precedence. Values that depend on where the
# repo lives cannot be written in a plain KEY=value file, so they are derived
# at the end -- still below every layer above.
#
# model_serving/ carries its own copy of this loader, .env and defaults so it
# stays runnable on its own; benchmark/ has an independent set.

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CONFIG_DIR/.." && pwd)"

CONFIG_ARGV=()

_config_is_assignment() { [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; }

# Plain KEY=VALUE only -- no quoting, no shell expansion, no inline comments.
# A key that already has a value came from a higher layer, so it is skipped.
_config_read_file() {
  local file="$1" line key
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in '' | '#'*) continue ;; esac
    _config_is_assignment "$line" || continue
    key="${line%%=*}"
    [ -n "${!key+x}" ] && continue
    export "$key=${line#*=}"
  done <"$file"
}

load_config() {
  CONFIG_ARGV=()
  local arg

  for arg in "$@"; do
    if _config_is_assignment "$arg"; then
      export "${arg%%=*}=${arg#*=}"
    else
      CONFIG_ARGV+=("$arg")
    fi
  done

  # Layer 2 is the environment we already inherited, so there is nothing to do
  # between the arguments above and the files below.
  _config_read_file "$CONFIG_DIR/.env"
  _config_read_file "$CONFIG_DIR/defaults.env"

  # --- derived defaults -------------------------------------------------------
  # The recipe's hard floor lives in requirements.txt so the version install.sh
  # installs and the version preflight.sh checks cannot drift apart.
  : "${MIN_VLLM_VERSION:=$(sed -n 's/^vllm>=\([0-9][0-9.]*\).*/\1/p' "$CONFIG_DIR/requirements.txt" 2>/dev/null)}"
  : "${MIN_VLLM_VERSION:?could not read a 'vllm>=X.Y.Z' line from $CONFIG_DIR/requirements.txt}"
  export MIN_VLLM_VERSION

  # download_model.py lands the shards in the repo-local cache, but vLLM reads
  # HF_HUB_CACHE and otherwise falls back to ~/.cache/huggingface/hub -- a
  # different directory, so serving would re-download all ~893 GB. Pin both.
  : "${HF_HUB_CACHE:=$REPO_ROOT/.hf-cache/hub}"
  export HF_HUB_CACHE

  # install.sh owns this environment. Put its tools on the scripts' PATH so JIT
  # subprocesses (FlashInfer invokes `ninja`) work without shell activation.
  : "${VENV:=$REPO_ROOT/venv}"
  : "${VENV_BIN:=$VENV/bin}"
  export VENV VENV_BIN
  case ":$PATH:" in
  *":$VENV_BIN:"*) ;;
  *) export PATH="$VENV_BIN:$PATH" ;;
  esac
  : "${VLLM_BIN:=$VENV_BIN/vllm}"
  : "${PYTHON_BIN:=$VENV_BIN/python}"
  export VLLM_BIN PYTHON_BIN
}

# Abort with an actionable message when a required setting has no value.
require_config() {
  local key missing=()
  for key in "$@"; do
    [ -n "${!key:-}" ] || missing+=("$key")
  done
  [ "${#missing[@]}" -eq 0 ] && return 0

  printf 'error: required setting(s) not set: %s\n' "${missing[*]}" >&2
  printf '\nSet each one in any of these, highest precedence first:\n' >&2
  printf '  1. an argument:      %s %s=...\n' "$(basename "$0")" "${missing[0]}" >&2
  printf '  2. the environment:  export %s=...\n' "${missing[0]}" >&2
  printf '  3. %s/.env  (cp .env.example .env)\n' "$CONFIG_DIR" >&2
  exit 1
}
