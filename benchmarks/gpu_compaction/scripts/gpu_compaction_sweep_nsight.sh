#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ONLY_WAS_SET="${GPU_ONLY+x}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/gpu_sweep_common.sh"

# Nsight-specific defaults
if [[ -z "${GPU_ONLY_WAS_SET}" ]]; then
    GPU_ONLY="1"
fi
TOOL="${TOOL:-nsys}"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
NSYS_GPU_METRICS="${NSYS_GPU_METRICS:-0}"
NSYS_GPU_METRICS_DEVICES="${NSYS_GPU_METRICS_DEVICES:-cuda-visible}"
NSYS_GPU_METRICS_SET="${NSYS_GPU_METRICS_SET:-}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-10000}"
GRAPH_DIR="${GRAPH_DIR:-}"
PLOT_RESULTS="${PLOT_RESULTS:-0}"
TIMESTAMP_OUTPUT="${TIMESTAMP_OUTPUT:-0}"

gpu_sweep_parse_common_args "$@"
set -- "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool) TOOL="$2"; shift 2 ;;
        --cpuctxsw) NSYS_CPUCTXSW="$2"; shift 2 ;;
        --gpu_metrics) NSYS_GPU_METRICS="1"; shift ;;
        --no_gpu_metrics) NSYS_GPU_METRICS="0"; shift ;;
        --gpu_metrics_devices) NSYS_GPU_METRICS_DEVICES="$2"; shift 2 ;;
        --gpu_metrics_set) NSYS_GPU_METRICS_SET="$2"; shift 2 ;;
        --gpu_metrics_frequency) NSYS_GPU_METRICS_FREQUENCY="$2"; shift 2 ;;
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --plot) PLOT_RESULTS="1"; shift ;;
        --no_plot) PLOT_RESULTS="0"; shift ;;
        --timestamp_output) TIMESTAMP_OUTPUT="1"; shift ;;
        --no_timestamp_output) TIMESTAMP_OUTPUT="0"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TOOL" != "nsys" && "$TOOL" != "ncu" ]]; then
    echo "Unsupported --tool value: $TOOL (expected nsys or ncu)"
    exit 1
fi

resolve_nsys_bin

TOOL_CMD="$TOOL"
if [[ "$TOOL" == "nsys" ]]; then
    TOOL_CMD="$NSYS_BIN"
fi

if ! command -v "$TOOL_CMD" >/dev/null 2>&1; then
    echo "$TOOL_CMD is not in PATH"
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────
gpu_sweep_save_originals
CURRENT_SSTS=$(gpu_sweep_current_ssts)
CURRENT_SST_SIZE_MB=$(gpu_sweep_current_sst_size_mb)

PROFILE_SUFFIX=""
if [[ "$PROFILE_ONLY" == "1" ]]; then
    PROFILE_SUFFIX="_gpu-profile-only"
fi
if [[ -z "$LABEL" ]]; then
    LABEL="nsight_${TOOL}_${CURRENT_SST_SIZE_MB}mb-sst_${CURRENT_SSTS}sst${PROFILE_SUFFIX}"
elif [[ "$PROFILE_ONLY" == "1" && "$LABEL" != *"${PROFILE_SUFFIX}" ]]; then
    LABEL="${LABEL}${PROFILE_SUFFIX}"
fi

OUTDIR="${OUT_ROOT}/${LABEL}"
if [[ "$TIMESTAMP_OUTPUT" == "1" ]]; then
    STAMP="$(date '+%Y%m%d_%H%M%S')"
    OUTDIR="${OUTDIR}_${STAMP}"
fi
GRAPH_DIR="${GRAPH_DIR:-${OUTDIR}/graphs}"
TEMP_ROOT="${OUTDIR}/temp"
PROFILES_DIR="${OUTDIR}/profiles"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$TEMP_ROOT" "$PROFILES_DIR"
if [[ "$PLOT_RESULTS" == "1" ]]; then
    mkdir -p "$GRAPH_DIR"
fi

cleanup() {
    rm -rf "$TEMP_ROOT"
    gpu_sweep_restore_originals
}
trap cleanup EXIT

MANIFEST="${OUTDIR}/nsight_manifest.csv"
echo "value_bytes,mode,tool,dataset,profile_prefix,report_kind,report_path,kernel_csv,api_csv,bench_log,run_status,exit_code" > "$MANIFEST"

echo "========================================================="
echo " Lightweight Nsight Sweep"
echo " tool: $TOOL"
if [[ "$TOOL" == "nsys" ]]; then
    echo " tool path: $TOOL_CMD"
    echo " cpu context switches: $NSYS_CPUCTXSW"
    echo " gpu metrics: $NSYS_GPU_METRICS"
    if [[ "$NSYS_GPU_METRICS" == "1" ]]; then
        echo " gpu metrics devices: $NSYS_GPU_METRICS_DEVICES"
        echo " gpu metrics set: ${NSYS_GPU_METRICS_SET:-<nsys default>}"
        echo " gpu metrics frequency: $NSYS_GPU_METRICS_FREQUENCY"
    fi
fi
echo " label: $LABEL"
echo " output: $OUTDIR"
echo " profiles: $PROFILES_DIR"
if [[ "$PLOT_RESULTS" == "1" ]]; then
    echo " graphs: $GRAPH_DIR"
else
    echo " graphs: skipped"
fi
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
echo " gpu modes: $MODES_STR"
echo " gpu-only benchmark mode: $GPU_ONLY"
echo " profile-only benchmark mode: $PROFILE_ONLY"
echo " storage IO mode: direct IO for SST reads and writes"
echo " input SSTs: $CURRENT_SSTS"
echo " target SST size: ${CURRENT_SST_SIZE_MB} MB"
echo " dataset user key space: $DATASET_USER_KEY_SPACE"
echo "========================================================="

# ── Main loop ─────────────────────────────────────────────────────
for VAL in $VALUES_STR; do
    gpu_sweep_build_and_datagen "${VAL}" "${OUTDIR}"

    for MODE in $MODES_STR; do
        BENCH_OUTDIR="${TEMP_ROOT}/${MODE}"
        PREFIX="${PROFILES_DIR}/${TOOL}_val${VAL}B_${MODE}"
        LOG_PATH="${OUTDIR}/result_val${VAL}B_${MODE}.log"
        REPORT_KIND="none"
        REPORT_PATH=""
        KERNEL_CSV=""
        API_CSV=""
        RUN_STATUS="ok"
        EXIT_CODE="0"

        echo "Profiling ${MODE} at ${VAL} B..."
        rm -rf "$BENCH_OUTDIR"
        gpu_sweep_bench_args "${DATASET_DIR}" "${BENCH_OUTDIR}" "${RUNS}" "${MODE}"

        if [[ "$TOOL" == "nsys" ]]; then
            NSYS_EXTRA_ARGS=()
            if [[ "$NSYS_GPU_METRICS" == "1" ]]; then
                NSYS_EXTRA_ARGS+=(
                    --gpu-metrics-devices="$NSYS_GPU_METRICS_DEVICES"
                    --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY"
                )
                if [[ -n "$NSYS_GPU_METRICS_SET" ]]; then
                    NSYS_EXTRA_ARGS+=(--gpu-metrics-set="$NSYS_GPU_METRICS_SET")
                fi
            fi
            set +e
            "$TOOL_CMD" profile \
                --force-overwrite true \
                -o "$PREFIX" \
                --trace=cuda,osrt,nvtx \
                --sample=none \
                --cpuctxsw="$NSYS_CPUCTXSW" \
                "${NSYS_EXTRA_ARGS[@]}" \
                "$BENCH_BIN" "${BENCH_ARGS[@]}" \
                > "$LOG_PATH" 2>&1
            EXIT_CODE="$?"
            set -e

            if [[ "$EXIT_CODE" != "0" ]]; then
                RUN_STATUS="failed"
                echo "WARNING: run failed for mode=${MODE} value=${VAL} exit_code=${EXIT_CODE}" >> "$LOG_PATH"
                echo "WARNING: run failed for mode=${MODE} value=${VAL} exit_code=${EXIT_CODE}"
            fi

            if [[ -f "${PREFIX}.nsys-rep" ]]; then
                REPORT_KIND="nsys-rep"
                REPORT_PATH="${PREFIX}.nsys-rep"

                if KERNEL_CSV_PATH=$(NSYS_BIN="$TOOL_CMD" run_nsys_stats_report "${REPORT_PATH}" "${PREFIX}_kernel_stats_gpukernsum.csv" gpukernsum cuda_gpu_kern_sum); then
                    KERNEL_CSV="${KERNEL_CSV_PATH}"
                fi
                if API_CSV_PATH=$(NSYS_BIN="$TOOL_CMD" run_nsys_stats_report "${REPORT_PATH}" "${PREFIX}_api_stats_cudaapisum.csv" cudaapisum cuda_api_sum); then
                    API_CSV="${API_CSV_PATH}"
                fi
            elif [[ -f "${PREFIX}.qdstrm" ]]; then
                REPORT_KIND="qdstrm"
                REPORT_PATH="${PREFIX}.qdstrm"
            fi
        else
            set +e
            ncu \
                --set full \
                --target-processes all \
                --export "$PREFIX" \
                "$BENCH_BIN" "${BENCH_ARGS[@]}" \
                > "$LOG_PATH" 2>&1
            EXIT_CODE="$?"
            set -e

            if [[ "$EXIT_CODE" != "0" ]]; then
                RUN_STATUS="failed"
                echo "WARNING: run failed for mode=${MODE} value=${VAL} exit_code=${EXIT_CODE}" >> "$LOG_PATH"
                echo "WARNING: run failed for mode=${MODE} value=${VAL} exit_code=${EXIT_CODE}"
            fi

            if [[ -f "${PREFIX}.ncu-rep" ]]; then
                REPORT_KIND="ncu-rep"
                REPORT_PATH="${PREFIX}.ncu-rep"
            fi
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$VAL" "$MODE" "$TOOL" "$DATASET_DIR" "$PREFIX" "$REPORT_KIND" "$REPORT_PATH" "$KERNEL_CSV" "$API_CSV" "$LOG_PATH" "$RUN_STATUS" "$EXIT_CODE" \
            >> "$MANIFEST"
    done

done

if [[ "$PLOT_RESULTS" == "1" ]]; then
    python3 "${BENCH_DIR}/plotting/plot_nsight_sweep.py" --manifest "$MANIFEST" --graphs_dir "$GRAPH_DIR" --out_dir "$OUTDIR"
fi

echo ""
echo "Done. Sweep outputs are in: $OUTDIR"
echo "Profiles: $PROFILES_DIR"
echo "Manifest: $MANIFEST"
if grep -q ',qdstrm,' "$MANIFEST"; then
    echo "Note: some runs only generated .qdstrm files. Figures still use benchmark logs; Nsight-kernel figures require .nsys-rep + stats CSV."
fi
