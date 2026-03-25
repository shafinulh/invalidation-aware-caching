#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULTS_FILE="${SCRIPT_DIR}/final_speedup_gpu_defaults.sh"
cd "${SCRIPT_DIR}"

if [[ ! -f "${DEFAULTS_FILE}" ]]; then
    echo "error: missing ${DEFAULTS_FILE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${DEFAULTS_FILE}"

RUNS="${RUNS:-${FINAL_GPU_RUNS:-${FINAL_SPEEDUP_GPU_DEFAULT_RUNS}}}"
VALUES_STR="${VALUES:-${FINAL_VALUE_SIZES:-${FINAL_SPEEDUP_GPU_DEFAULT_VALUE_SIZES}}}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
MODES_STR="${MODES:-${FINAL_GPU_MODES:-${FINAL_SPEEDUP_GPU_DEFAULT_MODES}}}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-${FINAL_GPU_ZIPF_ALPHA:-${FINAL_SPEEDUP_GPU_DEFAULT_ZIPF_ALPHA}}}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-${FINAL_GPU_USER_KEY_SPACE:-${FINAL_SPEEDUP_GPU_DEFAULT_USER_KEY_SPACE}}}"
GRAPH_DIR="${GRAPH_DIR:-${FINAL_GRAPH_DIR:-${FINAL_SPEEDUP_GPU_DEFAULT_GRAPH_DIR}}}"
OUT_ROOT="${OUT_ROOT:-./sweep_results}"
GPU_ONLY="${GPU_ONLY:-0}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"
HOST_METRICS_INTERVAL_SEC="${HOST_METRICS_INTERVAL_SEC:-}"
HOST_METRICS_DEVICE="${HOST_METRICS_DEVICE:-}"
NUM_SSTS=""
LABEL=""
BENCH_BIN="${SCRIPT_DIR}/gpcomp_bench"
DATAGEN_BIN="${SCRIPT_DIR}/gpcomp_datagen"
HOST_METRICS_COLLECTOR="${REPO_ROOT}/benchmarks/cpu/python/collect_host_metrics.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --modes) MODES_STR="$2"; shift 2 ;;
        --gpu_only) GPU_ONLY="1"; shift ;;
        --with_cpu_baseline) GPU_ONLY="0"; shift ;;
        --profile_only) PROFILE_ONLY="1"; GPU_ONLY="1"; shift ;;
        --no_profile_only) PROFILE_ONLY="0"; shift ;;
        --zipf_alpha) DATASET_ZIPF_ALPHA="$2"; shift 2 ;;
        --user_key_space) DATASET_USER_KEY_SPACE="$2"; shift 2 ;;
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --out_root) OUT_ROOT="$2"; shift 2 ;;
        --host_metrics_interval_sec) HOST_METRICS_INTERVAL_SEC="$2"; shift 2 ;;
        --host_metrics_device) HOST_METRICS_DEVICE="$2"; shift 2 ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
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

ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)
ORIG_VALUE_BYTES=$(grep -oP 'GP_VALUE_BYTES\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)

# If --num_ssts given, patch GP_NUM_INPUT_SSTS
if [[ -n "$NUM_SSTS" ]]; then
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${NUM_SSTS};/g" gpcomp_common.cuh
fi

# Read current NUM_INPUT_SSTS for labeling
CURRENT_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh)
if [[ -z "$LABEL" ]]; then
    LABEL="8mb-sst_${CURRENT_SSTS}sst"
fi
if [[ "${PROFILE_ONLY}" == "1" && "${LABEL}" != *gpu-profile-only* ]]; then
    LABEL="${LABEL}_gpu-profile-only"
elif [[ "${GPU_ONLY}" == "1" && "${LABEL}" != *gpu-only* ]]; then
    LABEL="${LABEL}_gpu-only"
fi

OUTDIR="${OUT_ROOT}/sweep_${LABEL}"
TEMP_ROOT="$OUTDIR/temp"
HOST_METRICS_ROOT="${OUTDIR}/host_metrics"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$GRAPH_DIR"

cleanup() {
    rm -rf "$TEMP_ROOT"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${ORIG_VALUE_BYTES};/g" gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${ORIG_NUM_SSTS};/g" gpcomp_common.cuh
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
echo " graphs saved to: $GRAPH_DIR"
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
echo " dataset distribution: uniform"
echo " dataset user key space: $DATASET_USER_KEY_SPACE"
echo "========================================================="

# Loop over value sizes
for VAL in $VALUES_STR; do
    echo ""
    echo "--- Building for Value Size: ${VAL} B ---"

    # Patch the value size in gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${VAL};/g" gpcomp_common.cuh
    make clean > /dev/null
    make gpcomp_datagen gpcomp_bench -j > /dev/null

    # Generate dataset for this value size
    DATASET_DIR="${DATASET_PREFIX}${VAL}"
    echo "Generating dataset in $DATASET_DIR..."
    rm -rf "$DATASET_DIR"
    "${DATAGEN_BIN}" \
        --out_dir "$DATASET_DIR" \
        --seed 42 \
        --zipf_alpha "$DATASET_ZIPF_ALPHA" \
        --user_key_space "$DATASET_USER_KEY_SPACE" \
        > "$OUTDIR/dataset_val${VAL}B_datagen.log"

    mkdir -p "$TEMP_ROOT"
    for MODE in $MODES_STR; do
        MODE_OUTDIR="$TEMP_ROOT/${MODE}"
        LOG_PATH="$OUTDIR/result_val${VAL}B_${MODE}.log"
        HOST_METRICS_DIR="${HOST_METRICS_ROOT}/val${VAL}B_${MODE}"
        echo "Running ${MODE}..."
        rm -rf "$MODE_OUTDIR"
        bench_args=(
            --dataset "$DATASET_DIR"
            --out_dir "$MODE_OUTDIR"
            --runs "$RUNS"
            --gpu_mode "$MODE"
        )
        if [[ "${GPU_ONLY}" == "1" ]]; then
            bench_args+=(--gpu_only)
        fi
        if [[ "${PROFILE_ONLY}" == "1" ]]; then
            bench_args+=(--profile_only)
        fi

        if ! run_bench_with_optional_host_metrics "$LOG_PATH" "$HOST_METRICS_DIR" "${bench_args[@]}"; then
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
python3 ./plot_results.py "${plot_args[@]}"

echo ""
echo "Done! All results saved in $OUTDIR"
