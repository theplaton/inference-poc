#!/usr/bin/env bash
# Single-node NVIDIA GPU preflight: inventory -> BF16 GEMM -> NCCL AllReduce.
# Exit 0 = PASS, 1 = FAIL.
set -uo pipefail

RESULTS="${RESULTS_DIR:-/results}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-}"
MIN_GEMM_TFLOPS="${MIN_GEMM_TFLOPS:-}"
MIN_NCCL_BUSBW_GBPS="${MIN_NCCL_BUSBW_GBPS:-}"
RELATIVE_GPU_FLOOR="${RELATIVE_GPU_FLOOR:-0.90}"

GEMM_BENCH="${GEMM_BENCH:-gemm_bench}"
ALL_REDUCE_PERF="${ALL_REDUCE_PERF:-all_reduce_perf}"

mkdir -p "$RESULTS" || exit 1
REPORT="$RESULTS/report.txt"
: > "$REPORT"

OVERALL=PASS
log()  { printf '>> %s\n' "$*" >&2; }
out()  { printf '%s\n' "$*" >> "$REPORT"; }
row()  {
    if [[ -n ${3:-} ]]; then printf '  %-20s %-26s %s\n' "$1" "$2" "$3" >> "$REPORT"
    else printf '  %-20s %s\n' "$1" "$2" >> "$REPORT"; fi
}
fail() { OVERALL=FAIL; }
# ge A B -> true if A >= B (float)
ge()   { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }

# ---------------------------------------------------------------- inventory
log "collecting GPU inventory"
{
    nvidia-smi -L
    echo
    echo "index,uuid,name,memory.total,pci.bus_id,temperature.gpu"
    nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id,temperature.gpu \
        --format=csv,noheader
} > "$RESULTS/nvidia-smi.txt" 2>&1 || { log "nvidia-smi failed"; fail; }

nvidia-smi topo -m > "$RESULTS/topology.txt" 2>&1 || log "nvidia-smi topo -m failed"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
GPU_COUNT=${#GPU_NAMES[@]}
GPU_MODEL="${GPU_NAMES[0]:-unknown}"

out "===================================================="
out " NVIDIA GPU NODE VALIDATION"
out "===================================================="
out ""
out "System"

if [[ $GPU_COUNT -eq 0 ]]; then
    row "GPU count:" "0" "FAIL"
    row "GPU model:" "-" "FAIL"
    fail
    out ""
    out "----------------------------------------------------"
    out " OVERALL: FAIL (no GPUs visible)"
    out "===================================================="
    cat "$REPORT"
    exit 1
fi

if [[ -n $EXPECTED_GPU_COUNT ]]; then
    if [[ $GPU_COUNT -eq $EXPECTED_GPU_COUNT ]]; then
        row "GPU count:" "$GPU_COUNT" "PASS"
    else
        row "GPU count:" "$GPU_COUNT (want $EXPECTED_GPU_COUNT)" "FAIL"
        fail
    fi
else
    row "GPU count:" "$GPU_COUNT" "-"
fi

if [[ -n $EXPECTED_GPU_NAME ]]; then
    NAME_STATUS=PASS
    for n in "${GPU_NAMES[@]}"; do
        [[ $n == *"$EXPECTED_GPU_NAME"* ]] || NAME_STATUS=FAIL
    done
    row "GPU model:" "$GPU_MODEL" "$NAME_STATUS"
    [[ $NAME_STATUS == PASS ]] || fail
else
    row "GPU model:" "$GPU_MODEL" "-"
fi

# ------------------------------------------------------------------- gemm
log "running BF16 GEMM on $GPU_COUNT GPU(s)"
: > "$RESULTS/gemm.txt"
GEMM_OK=1
for ((i = 0; i < GPU_COUNT; i++)); do
    if ! "$GEMM_BENCH" --device "$i" >> "$RESULTS/gemm.txt" 2>&1; then
        log "gemm_bench failed on GPU $i (see gemm.txt)"
        GEMM_OK=0
    fi
done
cat "$RESULTS/gemm.txt" >&2

mapfile -t TFLOPS < <(awk -F'TFLOPS=' '/TFLOPS=/{split($2,a," "); print a[1]}' \
    "$RESULTS/gemm.txt")

out ""
out "Per-GPU Compute"
if [[ $GEMM_OK -eq 0 || ${#TFLOPS[@]} -ne $GPU_COUNT ]]; then
    row "GEMM benchmark:" "incomplete" "FAIL"
    fail
fi

if [[ ${#TFLOPS[@]} -gt 0 ]]; then
    MEDIAN=$(printf '%s\n' "${TFLOPS[@]}" | sort -n | awk '
        {v[NR]=$1}
        END{ if (NR%2) print v[(NR+1)/2];
             else printf "%.1f", (v[NR/2]+v[NR/2+1])/2 }')

    for ((i = 0; i < ${#TFLOPS[@]}; i++)); do
        t=${TFLOPS[$i]}
        rel=$(awk -v t="$t" -v m="$MEDIAN" 'BEGIN{printf "%.0f", 100*t/m}')
        status=PASS
        ge "$rel" "$(awk -v f="$RELATIVE_GPU_FLOOR" 'BEGIN{printf "%.4f", 100*f}')" \
            || status=FAIL
        if [[ -n $MIN_GEMM_TFLOPS ]]; then
            ge "$t" "$MIN_GEMM_TFLOPS" || status=FAIL
        fi
        row "GPU$i:" "$t TFLOPS ($rel% median)" "$status"
        [[ $status == PASS ]] || fail
    done

    out ""
    row "Median:" "$MEDIAN TFLOPS"
    row "Relative floor:" "$(awk -v f="$RELATIVE_GPU_FLOOR" 'BEGIN{printf "%.0f%%", 100*f}')"
    [[ -n $MIN_GEMM_TFLOPS ]] && row "Absolute floor:" "$MIN_GEMM_TFLOPS TFLOPS"
fi

# ------------------------------------------------------------------- nccl
out ""
out "Multi-GPU"
out "  NCCL AllReduce"
if [[ $GPU_COUNT -lt 2 ]]; then
    row "busbw:" "n/a" "SKIP (1 GPU)"
    : > "$RESULTS/nccl.txt"
else
    log "running NCCL AllReduce across $GPU_COUNT GPUs"
    "$ALL_REDUCE_PERF" -b 8M -e 1G -f 2 -g "$GPU_COUNT" \
        > "$RESULTS/nccl.txt" 2>&1
    NCCL_RC=$?
    tail -n 20 "$RESULTS/nccl.txt" >&2

    # Data rows: size count type redop root time algbw busbw #wrong (x2 in/out-of-place)
    read -r BUSBW WRONG < <(awk '$1 ~ /^[0-9]+$/ && NF >= 12 {b=$8; w=$9} END{print b+0, w}' \
        "$RESULTS/nccl.txt")

    if [[ $NCCL_RC -ne 0 || -z ${BUSBW:-} ]] || ge 0 "$BUSBW"; then
        row "busbw:" "n/a" "FAIL (nccl error, rc=$NCCL_RC)"
        fail
    elif [[ ${WRONG:-0} != "0" && ${WRONG:-0} != "N/A" ]]; then
        row "busbw:" "$BUSBW GB/s" "FAIL (correctness: $WRONG wrong)"
        fail
    elif [[ -n $MIN_NCCL_BUSBW_GBPS ]]; then
        if ge "$BUSBW" "$MIN_NCCL_BUSBW_GBPS"; then
            row "busbw:" "$BUSBW GB/s" "PASS"
        else
            row "busbw:" "$BUSBW GB/s (want $MIN_NCCL_BUSBW_GBPS)" "FAIL"
            fail
        fi
    else
        row "busbw:" "$BUSBW GB/s" "PASS"
    fi
fi

# ----------------------------------------------------------------- verdict
out ""
out "----------------------------------------------------"
out " OVERALL: $OVERALL"
out "===================================================="

cat "$REPORT"
[[ $OVERALL == PASS ]] && exit 0 || exit 1
