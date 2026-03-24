#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-5}"
VALUES_STR="${VALUES:-32 64 128 256 512 1024}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
MODES_STR="${MODES:-q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-0.0}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-20000000}"
GRAPH_DIR="${GRAPH_DIR:-./graphs}"
NUM_SSTS=""
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --modes) MODES_STR="$2"; shift 2 ;;
        --zipf_alpha) DATASET_ZIPF_ALPHA="$2"; shift 2 ;;
        --user_key_space) DATASET_USER_KEY_SPACE="$2"; shift 2 ;;
        --graph_dir) GRAPH_DIR="$2"; shift 2 ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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

OUTDIR="./sweep_results/sweep_${LABEL}"
TEMP_ROOT="$OUTDIR/temp"
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
echo " collecting throughput, CPU utilization, and SSD read/write bandwidth"
echo " label: $LABEL"
echo " output saved to: $OUTDIR"
echo " graphs saved to: $GRAPH_DIR"
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
echo " gpu modes: $MODES_STR"
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
    ./gpcomp_datagen \
        --out_dir "$DATASET_DIR" \
        --seed 42 \
        --zipf_alpha "$DATASET_ZIPF_ALPHA" \
        --user_key_space "$DATASET_USER_KEY_SPACE" \
        > "$OUTDIR/dataset_val${VAL}B_datagen.log"

    mkdir -p "$TEMP_ROOT"
    for MODE in $MODES_STR; do
        MODE_OUTDIR="$TEMP_ROOT/${MODE}"
        LOG_PATH="$OUTDIR/result_val${VAL}B_${MODE}.log"
        echo "Running ${MODE}..."
        rm -rf "$MODE_OUTDIR"
        ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$MODE_OUTDIR" --runs "$RUNS" --gpu_mode "$MODE" > "$LOG_PATH"
    done

    echo "  Value Size  : ${VAL} B"
    for MODE in $MODES_STR; do
        LOG_PATH="$OUTDIR/result_val${VAL}B_${MODE}.log"
        SPEEDUP=$(grep "Speedup:" "$LOG_PATH" | awk '{print $2}')
        GPU_PIPE=$(grep "GPU pipeline" "$LOG_PATH" | sed -n 's/.*utilization: \([0-9.]*\)%.*/\1/p')
        echo "  [${MODE}]  Speedup: ${SPEEDUP}  GPU CPU-util: ${GPU_PIPE}%"
    done
    rm -rf "$TEMP_ROOT"
done

python3 ./plot_results.py --sweep_dir "$OUTDIR" --graphs_dir "$GRAPH_DIR" --label "$LABEL"

echo ""
echo "Done! All results saved in $OUTDIR"
