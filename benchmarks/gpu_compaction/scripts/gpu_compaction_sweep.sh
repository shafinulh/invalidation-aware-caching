#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/gpu_sweep_common.sh"

HOST_METRICS_INTERVAL_SEC="${HOST_METRICS_INTERVAL_SEC:-}"
HOST_METRICS_DEVICE="${HOST_METRICS_DEVICE:-}"
GRAPH_DIR="${GRAPH_DIR:-}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
HOST_METRICS_COLLECTOR="${REPO_ROOT}/benchmarks/common/collect_host_metrics.py"

gpu_sweep_parse_common_args "$@"
set -- "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --plot) PLOT_RESULTS="1"; shift ;;
        --no_plot) PLOT_RESULTS="0"; shift ;;
        --host_metrics_interval_sec) HOST_METRICS_INTERVAL_SEC="$2"; shift 2 ;;
        --host_metrics_device) HOST_METRICS_DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${HOST_METRICS_INTERVAL_SEC}" ]]; then
    if [[ "${GPU_ONLY}" == "1" ]]; then
        HOST_METRICS_INTERVAL_SEC="0.1"
    else
        HOST_METRICS_INTERVAL_SEC="0"
    fi
fi

resolve_block_device_for_path() {
    local target_path="$1"
    local source_device=""
    local physical_device=""

    if [[ -n "${HOST_METRICS_DEVICE}" ]]; then
        printf '%s\n' "${HOST_METRICS_DEVICE}"
        return 0
    fi

    source_device="$(df --output=source "${target_path}" 2>/dev/null | tail -n 1 | tr -d ' ')"
    if [[ -z "${source_device}" ]]; then
        return 1
    fi

    if [[ "${source_device}" == /dev/* ]] && command -v lsblk >/dev/null 2>&1; then
        physical_device="$(lsblk -no pkname "${source_device}" 2>/dev/null | head -n 1 | tr -d ' ')"
        if [[ -n "${physical_device}" ]]; then
            printf '%s\n' "${physical_device}"
            return 0
        fi
    fi

    printf '%s\n' "$(basename "${source_device}")"
}

start_host_metrics_collection() {
    local bench_pid="$1"
    local host_metrics_dir="$2"
    local device=""

    if [[ "${GPU_ONLY}" != "1" ]]; then
        return 0
    fi
    if [[ "${HOST_METRICS_INTERVAL_SEC}" == "0" || "${HOST_METRICS_INTERVAL_SEC}" == "0.0" ]]; then
        return 0
    fi
    if [[ ! -f "${HOST_METRICS_COLLECTOR}" ]]; then
        echo "warning: host metrics collector missing: ${HOST_METRICS_COLLECTOR}" >&2
        return 0
    fi

    device="$(resolve_block_device_for_path "${SCRIPT_DIR}" || true)"
    if [[ -z "${device}" ]]; then
        echo "warning: unable to resolve block device for ${SCRIPT_DIR}; skipping host metrics" >&2
        return 0
    fi

    mkdir -p "${host_metrics_dir}"
    python3 "${HOST_METRICS_COLLECTOR}" \
        --pid "${bench_pid}" \
        --device "${device}" \
        --interval-sec "${HOST_METRICS_INTERVAL_SEC}" \
        --output-dir "${host_metrics_dir}" &
    HOST_METRICS_COLLECTOR_PID="$!"
}

stop_host_metrics_collection() {
    local collector_pid="${HOST_METRICS_COLLECTOR_PID:-}"

    if [[ -z "${collector_pid}" ]]; then
        return 0
    fi
    if kill -0 "${collector_pid}" 2>/dev/null; then
        kill -TERM "${collector_pid}" 2>/dev/null || true
        wait "${collector_pid}" || true
    fi
    unset HOST_METRICS_COLLECTOR_PID
}

run_bench_with_optional_host_metrics() {
    local log_path="$1"
    local host_metrics_dir="$2"
    shift 2

    if [[ "${GPU_ONLY}" == "1" ]]; then
        rm -rf "${host_metrics_dir}"
        mkdir -p "${host_metrics_dir}"
    fi

    set +e
    "${BENCH_BIN}" "$@" > "${log_path}" 2>&1 &
    local bench_pid=$!
    start_host_metrics_collection "${bench_pid}" "${host_metrics_dir}"
    wait "${bench_pid}"
    local bench_rc="$?"
    set -e

    stop_host_metrics_collection
    return "${bench_rc}"
}

# ── Setup ─────────────────────────────────────────────────────────
gpu_sweep_save_originals
CURRENT_SSTS=$(gpu_sweep_current_ssts)
CURRENT_SST_SIZE_MB=$(gpu_sweep_current_sst_size_mb)

if [[ -z "$LABEL" ]]; then
    LABEL="${CURRENT_SST_SIZE_MB}mb-sst_${CURRENT_SSTS}sst"
fi
if [[ "${PROFILE_ONLY}" == "1" && "${LABEL}" != *gpu-profile-only* ]]; then
    LABEL="${LABEL}_gpu-profile-only"
elif [[ "${GPU_ONLY}" == "1" && "${LABEL}" != *gpu-only* ]]; then
    LABEL="${LABEL}_gpu-only"
fi

OUTDIR="${OUT_ROOT}/${LABEL}"
GRAPH_DIR="${GRAPH_DIR:-${OUTDIR}/graphs}"
TEMP_ROOT="$OUTDIR/temp"
HOST_METRICS_ROOT="${OUTDIR}/host_metrics"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
if [[ "${PLOT_RESULTS}" == "1" ]]; then
    mkdir -p "$GRAPH_DIR"
fi

cleanup() {
    rm -rf "$TEMP_ROOT"
    gpu_sweep_restore_originals
}
trap cleanup EXIT

echo "========================================================="
echo " GPComp Execution Sweep"
echo " comparing q/c compaction with and without planning"
if [[ "${GPU_ONLY}" == "1" ]]; then
    if [[ "${PROFILE_ONLY}" == "1" ]]; then
        echo " collecting sampled host CPU and block-device IO metrics for GPU profile-only compaction"
    else
        echo " collecting throughput plus sampled host CPU and block-device IO metrics"
    fi
else
    echo " collecting throughput only"
fi
echo " storage IO mode: direct IO for SST reads and writes"
echo " label: $LABEL"
echo " output saved to: $OUTDIR"
if [[ "${PLOT_RESULTS}" == "1" ]]; then
    echo " graphs saved to: $GRAPH_DIR"
else
    echo " graphs: skipped (--no_plot)"
fi
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
echo " gpu modes: $MODES_STR"
echo " gpu-only benchmark mode: $GPU_ONLY"
echo " profile-only benchmark mode: $PROFILE_ONLY"
if [[ "${GPU_ONLY}" == "1" ]]; then
    echo " host metrics interval sec: $HOST_METRICS_INTERVAL_SEC"
    if [[ -n "${HOST_METRICS_DEVICE}" ]]; then
        echo " host metrics device override: $HOST_METRICS_DEVICE"
    fi
fi
echo " input SSTs: $CURRENT_SSTS"
echo " target SST size: ${CURRENT_SST_SIZE_MB} MB"
echo " dataset distribution: uniform"
echo " dataset user key space: $DATASET_USER_KEY_SPACE"
echo "========================================================="

# ── Main loop ─────────────────────────────────────────────────────
for VAL in $VALUES_STR; do
    gpu_sweep_build_and_datagen "${VAL}" "${OUTDIR}"

    mkdir -p "$TEMP_ROOT"
    for MODE in $MODES_STR; do
        MODE_OUTDIR="$TEMP_ROOT/${MODE}"
        LOG_PATH="$OUTDIR/result_val${VAL}B_${MODE}.log"
        HOST_METRICS_DIR="${HOST_METRICS_ROOT}/val${VAL}B_${MODE}"
        echo "Running ${MODE}..."
        rm -rf "$MODE_OUTDIR"
        gpu_sweep_bench_args "${DATASET_DIR}" "${MODE_OUTDIR}" "${RUNS}" "${MODE}"

        if ! run_bench_with_optional_host_metrics "$LOG_PATH" "$HOST_METRICS_DIR" "${BENCH_ARGS[@]}"; then
            echo "error: benchmark failed for mode=${MODE} value=${VAL}" >&2
            exit 1
        fi
    done

    echo "  Value Size  : ${VAL} B"
    for MODE in $MODES_STR; do
        LOG_PATH="$OUTDIR/result_val${VAL}B_${MODE}.log"
        if [[ "${GPU_ONLY}" == "1" ]]; then
            HOST_METRICS_DIR="${HOST_METRICS_ROOT}/val${VAL}B_${MODE}"
            if [[ -f "${HOST_METRICS_DIR}/summary.json" ]]; then
                python3 - <<PY
import json
from pathlib import Path
summary = json.loads(Path(${HOST_METRICS_DIR@Q} + "/summary.json").read_text())
print(f"  [${MODE}]  Avg process CPU: {summary.get('avg_process_cpu_pct', 0.0):.1f}%  Avg device util: {summary.get('avg_device_util_pct', 0.0):.1f}%")
PY
            else
                echo "  [${MODE}]  Host metrics summary missing"
            fi
        else
            SPEEDUP="$(grep "Speedup:" "$LOG_PATH" | awk '{print $2}')"
            echo "  [${MODE}]  Speedup: ${SPEEDUP}"
        fi
    done
    rm -rf "$TEMP_ROOT"
done

if [[ "${PLOT_RESULTS}" == "1" ]]; then
    plot_args=(
        --sweep_dir "$OUTDIR"
        --graphs_dir "$GRAPH_DIR"
        --label "$LABEL"
    )
    if [[ "${GPU_ONLY}" == "1" ]]; then
        plot_args+=(--gpu_only)
    fi
    if [[ "${PROFILE_ONLY}" == "1" ]]; then
        plot_args+=(--profile_only)
    fi
    python3 "${BENCH_DIR}/plotting/plot_results.py" "${plot_args[@]}"
fi

echo ""
echo "Done! All results saved in $OUTDIR"
