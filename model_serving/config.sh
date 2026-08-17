# Configuration for the model_serving tools. Source it, then call
# `load_config "$@"`; arguments that are not KEY=value are left in CONFIG_ARGV
# for the calling script's own flag parser.
#
# Precedence, highest wins:
#
#   1. KEY=value arguments      ./serve_model.sh PORT=8001
#   2. exported environment     PORT=8001 ./serve_model.sh
#   3. model_serving/.env       local, gitignored, holds tokens
#   4. model_serving/profiles/$PROFILE.env   everything model-specific
#   5. model_serving/defaults.env   committed defaults, the same for any model
#
# Each layer only fills in keys no higher layer has set, so the order the
# layers are applied in *is* the precedence. Values that depend on where the
# repo lives cannot be written in a plain KEY=value file, so they are derived
# at the end -- still below every layer above.
#
# A profile carries one thing the other layers do not: its engine arguments, as
# lines beginning with --. load_config leaves them in PROFILE_ARGS for
# serve_model.sh to splice into its command line. They are not settings and no
# layer above can override one, because a flag a model needs is a property of
# the model rather than of a run.
#
# model_serving/ carries its own copy of this loader, .env and defaults so it
# stays runnable on its own; benchmark/ has an independent set.

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

# The other kind of line a profile may hold: an engine argument, written exactly
# as it appears on the command line. A flag and its value are split on the first
# run of whitespace and no further, so a JSON value needs no quoting and no
# escaping -- the rest of the line is one argument however many spaces it has.
#
# Collected first-seen-wins, like every setting: the profile is read before its
# base, so a derived profile restating a flag overrides it, and the value `off`
# drops it. Nothing here knows what any particular flag means.
_config_read_args() {
  local file="$1" line flag value i
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in '--'*) ;; *) continue ;; esac
    flag="${line%%[[:space:]]*}"
    value="${line#"$flag"}"
    value="${value#"${value%%[![:space:]]*}"}"
    for i in ${_profile_arg_names[@]+"${!_profile_arg_names[@]}"}; do
      if [ "${_profile_arg_names[$i]}" = "$flag" ]; then
        continue 2 # a nearer layer already decided this flag
      fi
    done
    _profile_arg_names+=("$flag")
    _profile_arg_values+=("$value")
  done <"$file"
  return 0
}

# Flatten what the profile chain collected into the array serve_model.sh splices
# into its command line.
_config_build_profile_args() {
  local i flag value
  PROFILE_ARGS=()
  for i in ${_profile_arg_names[@]+"${!_profile_arg_names[@]}"}; do
    flag="${_profile_arg_names[$i]}"
    value="${_profile_arg_values[$i]}"
    if [ "$value" = off ]; then
      continue # set by a base, switched off here
    fi
    PROFILE_ARGS+=("$flag")
    if [ -n "$value" ]; then
      PROFILE_ARGS+=("$value")
    fi
  done
  return 0
}

load_config() {
  CONFIG_ARGV=()
  local arg
  _profile_arg_names=()
  _profile_arg_values=()

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

  # The profile is what makes this repo serve something other than V4-Pro: the
  # engine flags, the memory envelope and the checkpoint footprint all differ
  # per model, and none of them belong in a defaults file shared by all of them.
  # It is resolved from the layers already applied, then from defaults.env.
  [ -n "${PROFILE:-}" ] || PROFILE="$(_config_value_from "$CONFIG_DIR/defaults.env" PROFILE)"
  : "${PROFILE:=deepseek_v4_tp8dp1_speculative}"
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
  _config_read_args "$PROFILE_FILE"

  # A profile may name another as its base, and takes from it everything it does
  # not set itself -- the same rule as every layer above, so a variant profile
  # holds only what makes it a variant. That matters when the variant exists to
  # be compared against its base: parity is then structural rather than a
  # promise that two files stay in step. Read from the file rather than the
  # environment, so an inherited PROFILE_BASE cannot redirect an unrelated run.
  local base seen base_file
  base="$(_config_value_from "$PROFILE_FILE" PROFILE_BASE)"
  seen=" $PROFILE "
  while [ -n "$base" ]; do
    case "$seen" in
    *" $base "*)
      printf 'error: profile "%s" inherits in a cycle (%s)\n' "$PROFILE" "$base" >&2
      exit 1
      ;;
    esac
    seen="$seen$base "
    base_file="$CONFIG_DIR/profiles/$base.env"
    if [ ! -f "$base_file" ]; then
      printf 'error: profile "%s" names base "%s", but %s does not exist.\n' \
        "$PROFILE" "$base" "$base_file" >&2
      exit 1
    fi
    _config_read_file "$base_file"
    _config_read_args "$base_file"
    base="$(_config_value_from "$base_file" PROFILE_BASE)"
  done

  _config_build_profile_args

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
