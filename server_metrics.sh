#!/usr/bin/env bash
# Live view of a running vLLM server: what it was launched with, what its queue
# and KV cache are doing, and a line per GPU from nvidia-smi.
#
#   ./server_metrics.sh [SECONDS]     poll every SECONDS (default 1)
#   ./server_metrics.sh --once        print one frame and exit
#
# HOST, PORT and METRICS_URL override where the metrics come from; the defaults
# match model_serving/defaults.env, so a server started by ./serve_model.sh on
# this node needs no arguments.
#
# With no server up it says so and keeps showing the GPUs, so it can be started
# before a serve and left running across it. Read-only either way: starting or
# stopping it never disturbs a run.
set -uo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
METRICS_URL="${METRICS_URL:-http://$HOST:$PORT/metrics}"
MODELS_URL="${MODELS_URL:-${METRICS_URL%/metrics}/v1/models}"
CURL_TIMEOUT="${CURL_TIMEOUT:-2}"

INTERVAL=1
ONCE=0

usage() {
  awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --once) ONCE=1 ;;
    *)
      # A bare positive number is the poll interval; anything else is a typo
      # worth stopping for rather than silently polling at the default rate.
      if [[ "$arg" =~ ^[0-9]+([.][0-9]+)?$ ]] && [ "${arg//[0.]/}" != "" ]; then
        INTERVAL="$arg"
      else
        echo "server_metrics.sh: unrecognized argument '$arg'" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

# Everything about the server comes out of /metrics, apart from the context
# length, which only /v1/models reports.
#
# Values are read from $NF rather than $2 because every metric line carries its
# labels in between, and a label value may itself contain a space. One node can
# serve several engines -- data parallel ranks each export their own series -- so
# counts are summed over them and KV cache use is averaged, giving one figure for
# the server rather than whichever rank happened to be printed first.
#
# vLLM renamed two of these metrics along the way: gpu_cache_usage_perc became
# kv_cache_usage_perc, and time_per_output_token became inter_token_latency.
# Both spellings are read and the newer one wins where both are present.
#
# The launch settings are labels on vllm:cache_config_info, a gauge that exists
# only to carry them. Speculation is not in there, so it is inferred from the
# spec_decode counters: present means the server is drafting, and their ratios
# give the draft length it was configured with and how much of it lands.
vllm_block() {
  local metrics="$1" ctx="$2"
  printf '%s\n' "$metrics" | awk -v ctx="$ctx" '
    # One label out of a metric line. Anchored on the { or , in front of the
    # name so block_size cannot be answered with mamba_block_size.
    function label(line, key,   lead, at, rest) {
      lead = length(key) + 3
      at = index(line, "," key "=\"")
      if (!at) { at = index(line, "{" key "=\""); if (!at) return "" }
      rest = substr(line, at + lead)
      return substr(rest, 1, index(rest, "\"") - 1)
    }
    function yesno(v) { return v == "True" ? "on" : (v == "False" ? "off" : v) }

    /^vllm:num_requests_running[{ ]/            { running += $NF; seen = 1 }
    /^vllm:num_requests_waiting[{ ]/            { waiting += $NF; seen = 1 }
    /^vllm:kv_cache_usage_perc[{ ]/             { kv += $NF; kvn++; seen = 1 }
    /^vllm:gpu_cache_usage_perc[{ ]/            { okv += $NF; okvn++; seen = 1 }
    /^vllm:time_to_first_token_seconds_sum[{ ]/ { ttft_s += $NF }
    /^vllm:time_to_first_token_seconds_count[{ ]/ { ttft_n += $NF }
    /^vllm:inter_token_latency_seconds_sum[{ ]/   { itl_s += $NF; itl = 1 }
    /^vllm:inter_token_latency_seconds_count[{ ]/ { itl_n += $NF; itl = 1 }
    /^vllm:time_per_output_token_seconds_sum[{ ]/   { tpot_s += $NF }
    /^vllm:time_per_output_token_seconds_count[{ ]/ { tpot_n += $NF }

    /^vllm:spec_decode_num_drafts_total[{ ]/          { drafts += $NF; spec = 1 }
    /^vllm:spec_decode_num_draft_tokens_total[{ ]/    { draft_tok += $NF; spec = 1 }
    /^vllm:spec_decode_num_accepted_tokens_total[{ ]/ { accepted += $NF; spec = 1 }

    # Identical across ranks, so the last one read stands for the server.
    /^vllm:cache_config_info[{ ]/ { cfg = $0; seen = 1 }
    /model_name="/ { if (!model) model = label($0, "model_name") }
    /engine="/     { engines[label($0, "engine")] = 1 }

    END {
      if (!seen) exit 3
      if (!itl) { itl_s = tpot_s; itl_n = tpot_n }
      if (!kvn) { kv = okv; kvn = okvn }

      nengines = 0
      for (e in engines) nengines++

      printf "Model            : %s\n", model ? model : "unknown"
      if (ctx) printf "Context length   : %s tokens\n", ctx
      else     printf "Context length   : unknown\n"
      if (nengines > 1) printf "Engines          : %d data-parallel ranks\n", nengines

      if (cfg) {
        tokens = label(cfg, "kv_cache_size_tokens")
        blocks = label(cfg, "num_gpu_blocks")
        bsize = label(cfg, "block_size")
        if (tokens == "" && blocks != "" && bsize != "") tokens = blocks * bsize
        size = "unknown"
        if (tokens != "") size = tokens " tokens"
        printf "KV cache size    : %s  (%s, block %s, prefix caching %s)\n",
               size, label(cfg, "cache_dtype"), bsize,
               yesno(label(cfg, "enable_prefix_caching"))
        printf "GPU mem fraction : %s\n", label(cfg, "gpu_memory_utilization")
      }

      if (!spec) {
        printf "Speculation      : off\n"
      } else if (drafts > 0) {
        printf "Speculation      : on, %.0f draft tokens, %.1f%% accepted\n",
               draft_tok / drafts, 100 * accepted / draft_tok
      } else {
        printf "Speculation      : on, nothing drafted yet\n"
      }

      printf "%s\n", "--------------------------------------------------------------"
      printf "Running requests : %d\n", running
      printf "Waiting requests : %d\n", waiting
      if (kvn > 1) {
        printf "KV cache used    : %.1f%%  (mean of %d engines)\n", 100 * kv / kvn, kvn
      } else if (kvn == 1) {
        printf "KV cache used    : %.1f%%\n", 100 * kv
      } else {
        printf "KV cache used    : n/a\n"
      }
      if (ttft_n > 0) printf "Mean TTFT        : %.1f ms\n", 1000 * ttft_s / ttft_n
      else            printf "Mean TTFT        : n/a\n"
      if (itl_n > 0)  printf "Mean ITL         : %.1f ms\n", 1000 * itl_s / itl_n
      else            printf "Mean ITL         : n/a\n"
    }
  '
}

# memory.used and memory.total come back in MiB and are shown in GiB; a GPU that
# does not answer a field reports "[N/A]", which prints as a dash rather than as
# a zero that would read like an idle GPU.
gpu_block() {
  local query=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu
  local rows
  rows="$(nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits 2>/dev/null)"
  if [ -z "$rows" ]; then
    echo "no GPU readings (nvidia-smi unavailable)"
    return
  fi
  printf '%s\n' "$rows" | awk -F' *, *' '
    function num(v) { return (v + 0 == v) ? v + 0 : "" }

    BEGIN {
      row = "%3s  %6s  %6s  %18s  %13s  %6s\n"
      printf row, "GPU", "Core", "Mem", "Memory", "Power", "Temp"
      printf "%s\n", "--------------------------------------------------------------"
    }
    {
      core = num($2); memutil = num($3)
      used = num($4); total = num($5)
      draw = num($6); cap = num($7); temp = num($8)

      c = "-"; if (core != "") c = sprintf("%d%%", core)
      m = "-"; if (memutil != "") m = sprintf("%d%%", memutil)
      g = "-"
      if (used != "" && total != "")
        g = sprintf("%.1f / %.1f GB", used / 1024, total / 1024)
      p = "-"
      if (draw != "" && cap != "") p = sprintf("%.0f / %.0f W", draw, cap)
      t = "-"; if (temp != "") t = sprintf("%d C", temp)

      printf row, $1, c, m, g, p, t
    }
  '
}

# Three states worth telling apart: nothing listening, something listening that
# has not reported yet -- a V4-Pro load answers /metrics long before it serves --
# and a server with metrics to show.
frame() {
  local metrics ctx block

  printf 'vLLM Server  %s\n' "$METRICS_URL"
  printf '%s\n' "--------------------------------------------------------------"

  if ! metrics="$(curl -s --max-time "$CURL_TIMEOUT" "$METRICS_URL")"; then
    printf 'vLLM server is not running -- nothing answered at %s\n' "$METRICS_URL"
  else
    ctx="$(curl -s --max-time "$CURL_TIMEOUT" "$MODELS_URL" |
      grep -o '"max_model_len":[0-9]*' | head -1 | cut -d: -f2)"
    block="$(vllm_block "$metrics" "$ctx")"
    if [ -z "$block" ]; then
      printf 'vLLM server is answering but has not reported any metrics yet\n'
    else
      printf '%s\n' "$block"
    fi
  fi

  printf '\n'
  gpu_block
}

if [ "$ONCE" = 1 ]; then
  frame
  exit 0
fi

# Each frame is built first and then written over the last one -- home the
# cursor, print every line with the rest of that line erased, then erase
# whatever is left below -- so the display neither blinks the way a
# clear-then-draw loop does nor leaves the tails of longer previous lines
# behind, which is what a plain redraw does to a screen with a shell prompt or
# an earlier frame on it.
tty=0
[ -t 1 ] && tty=1
if [ "$tty" = 1 ]; then
  printf '\033[?25l'                                  # hide the cursor
  trap 'printf "\033[?25h\n"; exit 0' INT TERM
fi

while true; do
  out="$(frame)"
  if [ "$tty" = 1 ]; then
    printf '\033[H'
    while IFS= read -r line; do
      printf '%s\033[K\n' "$line"
    done <<< "$out"
    printf '\033[0J'
  else
    printf '%s\n\n' "$out"
  fi
  sleep "$INTERVAL"
done
