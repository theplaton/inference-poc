# Configuration for the benchmark tools. Source it, then call
# `load_config "$@"`; arguments that are not KEY=value are left in CONFIG_ARGV
# for the calling script's own flag parser.
#
# Precedence, highest wins:
#
#   1. KEY=value arguments      ./benchmark.sh NUM_PROMPTS=128
#   2. exported environment     NUM_PROMPTS=128 ./benchmark.sh
#   3. benchmark/.env           local, gitignored
#   4. benchmark/profiles/$PROFILE.env   everything model-specific
#   5. benchmark/defaults.env   committed defaults, the same for any model
#
# Each layer only fills in keys no higher layer has set, so the order the
# layers are applied in *is* the precedence. Values that depend on where the
# repo lives, or on other settings, are derived at the end -- still below every
# layer above.
#
# benchmark/ carries its own copy of this loader, .env and defaults so it stays
# runnable against any OpenAI-compatible endpoint, with or without
# model_serving/ present.

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CONFIG_DIR/.." && pwd)"

CONFIG_ARGV=()

_config_is_assignment() { [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; }

# One key out of a settings file, without applying the rest of it. PROFILE has
# to be resolved before the profile layer can be read, and reading defaults.env
# early would let it outrank the profile it is supposed to sit below.
_config_value_from() {
  local file="$1" key="$2" line
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in "$key="*) printf '%s' "${line#*=}"; return 0 ;; esac
  done <"$file"
}

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

  # The client's half of the profile: which checkpoint to name in requests, and
  # what the chat template understands. The sweep exports PROFILE when it drives
  # this script, so the two halves stay in step without being wired together.
  [ -n "${PROFILE:-}" ] || PROFILE="$(_config_value_from "$CONFIG_DIR/defaults.env" PROFILE)"
  : "${PROFILE:=deepseek_v4}"
  export PROFILE
  PROFILE_FILE="$CONFIG_DIR/profiles/$PROFILE.env"
  export PROFILE_FILE
  if [ ! -f "$PROFILE_FILE" ]; then
    printf 'error: no profile "%s" -- %s does not exist.\n' "$PROFILE" "$PROFILE_FILE" >&2
    printf 'Available: %s\n' \
      "$(ls "$CONFIG_DIR/profiles" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')" >&2
    exit 1
  fi
  _config_read_file "$PROFILE_FILE"

  _config_read_file "$CONFIG_DIR/defaults.env"

  # --- derived defaults -------------------------------------------------------
  # One endpoint spelled two ways: the vLLM bench CLI wants host and port, the
  # OpenAI client wants a base URL. Set BASE_URL explicitly to point the client
  # somewhere the host/port pair cannot describe (https, a path prefix, a proxy).
  # The two fallbacks repeat defaults.env so a deleted defaults file degrades to
  # a working localhost endpoint instead of an unbound-variable abort.
  : "${HOST:=localhost}"
  : "${PORT:=8000}"
  : "${BASE_URL:=http://$HOST:$PORT/v1}"
  # /health sits next to /v1, not under it.
  : "${HEALTH_URL:=${BASE_URL%/v1}/health}"
  export HOST PORT BASE_URL HEALTH_URL

  # `vllm bench serve` comes from the serving install; the smoke test only needs
  # the openai client. Both live in the repo-level venv when it exists.
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
