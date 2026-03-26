#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULTS_FILE="${SCRIPT_DIR}/final_speedup_gpu_defaults.sh"
cd "$SCRIPT_DIR"

TOOL="${TOOL:-nsys}"
DEFAULT_LOCAL_NSYS_BIN="$HOME/.local/nsight-systems-2026.2.1/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
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
GPU_ONLY="${GPU_ONLY:-1}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-${FINAL_GPU_ZIPF_ALPHA:-${FINAL_SPEEDUP_GPU_DEFAULT_ZIPF_ALPHA}}}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-${FINAL_GPU_USER_KEY_SPACE:-${FINAL_SPEEDUP_GPU_DEFAULT_USER_KEY_SPACE}}}"
OUT_ROOT="${OUT_ROOT:-./sweep_results}"
GRAPH_DIR="${GRAPH_DIR:-${FINAL_GRAPH_DIR:-${FINAL_SPEEDUP_GPU_DEFAULT_GRAPH_DIR}}}"
PLOT_RESULTS="${PLOT_RESULTS:-0}"
TIMESTAMP_OUTPUT="${TIMESTAMP_OUTPUT:-0}"
NUM_SSTS=""
LABEL=""
DATAGEN_BIN="$SCRIPT_DIR/gpcomp_datagen"
BENCH_BIN="$SCRIPT_DIR/gpcomp_bench"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool) TOOL="$2"; shift 2 ;;
        --cpuctxsw) NSYS_CPUCTXSW="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --modes) MODES_STR="$2"; shift 2 ;;
        --gpu_only) GPU_ONLY="1"; shift ;;
        --with_cpu_baseline) GPU_ONLY="0"; shift ;;
        --profile_only) PROFILE_ONLY="1"; shift ;;
        --no_profile_only) PROFILE_ONLY="0"; shift ;;
        --zipf_alpha) DATASET_ZIPF_ALPHA="$2"; shift 2 ;;
        --user_key_space) DATASET_USER_KEY_SPACE="$2"; shift 2 ;;
        --out_root) OUT_ROOT="$2"; shift 2 ;;
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --plot) PLOT_RESULTS="1"; shift ;;
        --no_plot) PLOT_RESULTS="0"; shift ;;
        --timestamp_output) TIMESTAMP_OUTPUT="1"; shift ;;
        --no_timestamp_output) TIMESTAMP_OUTPUT="0"; shift ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TOOL" != "nsys" && "$TOOL" != "ncu" ]]; then
    echo "Unsupported --tool value: $TOOL (expected nsys or ncu)"
    exit 1
fi

if [[ -z "$NSYS_BIN" && -x "$DEFAULT_LOCAL_NSYS_BIN" ]]; then
    NSYS_BIN="$DEFAULT_LOCAL_NSYS_BIN"
fi
if [[ -z "$NSYS_BIN" ]]; then
    NSYS_BIN="nsys"
fi

TOOL_CMD="$TOOL"
if [[ "$TOOL" == "nsys" ]]; then
    TOOL_CMD="$NSYS_BIN"
fi

if ! command -v "$TOOL_CMD" >/dev/null 2>&1; then
    echo "$TOOL_CMD is not in PATH"
    exit 1
fi

run_nsys_stats_report() {
    local report_path="$1"
    local canonical_path="$2"
    shift 2
    local nsys_bin="${NSYS_BIN:-nsys}"
    local report=""
    local candidate=""
    local output_prefix="${canonical_path%.csv}.tmp"

    for report in "$@"; do
        rm -f "${output_prefix}"*.csv
        "$nsys_bin" stats --force-export=true --report "$report" --format csv --output "$output_prefix" "$report_path" >/dev/null 2>&1 || true
        candidate="${output_prefix}_${report}.csv"
        if [[ -f "$candidate" ]]; then
            mv -f "$candidate" "$canonical_path"
            printf '%s\n' "$canonical_path"
            return 0
        fi
    done

    return 1
}

ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)
ORIG_VALUE_BYTES=$(grep -oP 'GP_VALUE_BYTES\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)

if [[ -n "$NUM_SSTS" ]]; then
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${NUM_SSTS};/g" gpcomp_common.cuh
fi

CURRENT_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh)
PROFILE_SUFFIX=""
if [[ "$PROFILE_ONLY" == "1" ]]; then
    PROFILE_SUFFIX="_gpu-profile-only"
fi
if [[ -z "$LABEL" ]]; then
    LABEL="nsight_${TOOL}_8mb-sst_${CURRENT_SSTS}sst${PROFILE_SUFFIX}"
elif [[ "$PROFILE_ONLY" == "1" && "$LABEL" != *"${PROFILE_SUFFIX}" ]]; then
    LABEL="${LABEL}${PROFILE_SUFFIX}"
fi

OUTDIR="${OUT_ROOT}/sweep_${LABEL}"
if [[ "$TIMESTAMP_OUTPUT" == "1" ]]; then
    STAMP="$(date '+%Y%m%d_%H%M%S')"
    OUTDIR="${OUTDIR}_${STAMP}"
fi
TEMP_ROOT="${OUTDIR}/temp"
PROFILES_DIR="${OUTDIR}/profiles"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$TEMP_ROOT" "$PROFILES_DIR"
if [[ "$PLOT_RESULTS" == "1" ]]; then
    mkdir -p "$GRAPH_DIR"
fi

cleanup() {
    rm -rf "$TEMP_ROOT"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${ORIG_VALUE_BYTES};/g" gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${ORIG_NUM_SSTS};/g" gpcomp_common.cuh
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
echo " dataset user key space: $DATASET_USER_KEY_SPACE"
echo "========================================================="

for VAL in $VALUES_STR; do
    echo ""
    echo "--- Building for Value Size: ${VAL} B ---"

    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${VAL};/g" gpcomp_common.cuh
    make clean > /dev/null
    make gpcomp_datagen gpcomp_bench -j > /dev/null

    DATASET_DIR="${DATASET_PREFIX}${VAL}"
    echo "Generating dataset in $DATASET_DIR..."
    rm -rf "$DATASET_DIR"
    "$DATAGEN_BIN" \
        --out_dir "$DATASET_DIR" \
        --seed 42 \
        --zipf_alpha "$DATASET_ZIPF_ALPHA" \
        --user_key_space "$DATASET_USER_KEY_SPACE" \
        > "${OUTDIR}/dataset_val${VAL}B_datagen.log"

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

        bench_args=(
            --dataset "$DATASET_DIR"
            --out_dir "$BENCH_OUTDIR"
            --runs "$RUNS"
            --gpu_mode "$MODE"
        )
        if [[ "$GPU_ONLY" == "1" ]]; then
            bench_args+=(--gpu_only)
        fi
        if [[ "$PROFILE_ONLY" == "1" ]]; then
            bench_args+=(--profile_only)
        fi

        if [[ "$TOOL" == "nsys" ]]; then
            set +e
            "$TOOL_CMD" profile \
                --force-overwrite true \
                -o "$PREFIX" \
                --trace=cuda,osrt,nvtx \
                --sample=none \
                --cpuctxsw="$NSYS_CPUCTXSW" \
                "$BENCH_BIN" "${bench_args[@]}" \
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
                "$BENCH_BIN" "${bench_args[@]}" \
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
    python3 ./plot_nsight_sweep.py --manifest "$MANIFEST" --graphs_dir "$GRAPH_DIR" --out_dir "$OUTDIR"
fi

echo ""
echo "Done. Sweep outputs are in: $OUTDIR"
echo "Profiles: $PROFILES_DIR"
echo "Manifest: $MANIFEST"
if grep -q ',qdstrm,' "$MANIFEST"; then
    echo "Note: some runs only generated .qdstrm files. Figures still use benchmark logs; Nsight-kernel figures require .nsys-rep + stats CSV."
fi
