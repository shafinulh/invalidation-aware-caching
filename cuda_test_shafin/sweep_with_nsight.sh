#!/usr/bin/env bash
set -euo pipefail

TOOL="${TOOL:-nsys}"
RUNS="${RUNS:-1}"
VALUES_STR="${VALUES:-32 64 128}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
MODES_STR="${MODES:-q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-0.0}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-10000000}"
OUT_ROOT="${OUT_ROOT:-./nsight_sweep_results}"
GRAPH_DIR="${GRAPH_DIR:-./graphs}"
NUM_SSTS=""
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool) TOOL="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --modes) MODES_STR="$2"; shift 2 ;;
        --zipf_alpha) DATASET_ZIPF_ALPHA="$2"; shift 2 ;;
        --user_key_space) DATASET_USER_KEY_SPACE="$2"; shift 2 ;;
        --out_root) OUT_ROOT="$2"; shift 2 ;;
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TOOL" != "nsys" && "$TOOL" != "ncu" ]]; then
    echo "Unsupported --tool value: $TOOL (expected nsys or ncu)"
    exit 1
fi

if ! command -v "$TOOL" >/dev/null 2>&1; then
    echo "$TOOL is not in PATH"
    exit 1
fi

ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)
ORIG_VALUE_BYTES=$(grep -oP 'GP_VALUE_BYTES\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)

if [[ -n "$NUM_SSTS" ]]; then
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${NUM_SSTS};/g" gpcomp_common.cuh
fi

CURRENT_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh)
if [[ -z "$LABEL" ]]; then
    LABEL="nsight_${TOOL}_${CURRENT_SSTS}sst"
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUTDIR="${OUT_ROOT}/sweep_${LABEL}_${STAMP}"
TEMP_ROOT="${OUTDIR}/temp"
mkdir -p "$OUTDIR" "$TEMP_ROOT" "$GRAPH_DIR"

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
echo " label: $LABEL"
echo " output: $OUTDIR"
echo " graphs: $GRAPH_DIR"
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
echo " gpu modes: $MODES_STR"
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
    ./gpcomp_datagen \
        --out_dir "$DATASET_DIR" \
        --seed 42 \
        --zipf_alpha "$DATASET_ZIPF_ALPHA" \
        --user_key_space "$DATASET_USER_KEY_SPACE" \
        > "${OUTDIR}/dataset_val${VAL}B_datagen.log"

    for MODE in $MODES_STR; do
        BENCH_OUTDIR="${TEMP_ROOT}/${MODE}"
        PREFIX="${OUTDIR}/${TOOL}_val${VAL}B_${MODE}"
        LOG_PATH="${OUTDIR}/result_val${VAL}B_${MODE}.log"
        REPORT_KIND="none"
        REPORT_PATH=""
        KERNEL_CSV=""
        API_CSV=""
        RUN_STATUS="ok"
        EXIT_CODE="0"

        echo "Profiling ${MODE} at ${VAL} B..."
        rm -rf "$BENCH_OUTDIR"

        if [[ "$TOOL" == "nsys" ]]; then
            set +e
            nsys profile \
                --force-overwrite true \
                -o "$PREFIX" \
                --trace=cuda,osrt,nvtx \
                --sample=none \
                ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$BENCH_OUTDIR" --runs "$RUNS" --gpu_mode "$MODE" \
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

                nsys stats --report cuda_gpu_kern_sum --format csv --output "${PREFIX}_cuda_gpu_kern_sum" "${REPORT_PATH}" >/dev/null 2>&1 || true
                nsys stats --report cuda_api_sum --format csv --output "${PREFIX}_cuda_api_sum" "${REPORT_PATH}" >/dev/null 2>&1 || true

                [[ -f "${PREFIX}_cuda_gpu_kern_sum.csv" ]] && KERNEL_CSV="${PREFIX}_cuda_gpu_kern_sum.csv"
                [[ -f "${PREFIX}_cuda_api_sum.csv" ]] && API_CSV="${PREFIX}_cuda_api_sum.csv"
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
                ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$BENCH_OUTDIR" --runs "$RUNS" --gpu_mode "$MODE" \
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

python3 ./plot_nsight_sweep.py --manifest "$MANIFEST" --graphs_dir "$GRAPH_DIR" --out_dir "$OUTDIR"

echo ""
echo "Done. Sweep outputs are in: $OUTDIR"
echo "Manifest: $MANIFEST"
if grep -q ',qdstrm,' "$MANIFEST"; then
    echo "Note: some runs only generated .qdstrm files. Figures still use benchmark logs; Nsight-kernel figures require .nsys-rep + stats CSV."
fi
