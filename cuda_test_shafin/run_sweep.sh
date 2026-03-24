#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-5}"
VALUES_STR="${VALUES:-32 64 128}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
NUM_SSTS=""
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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
TEMP_WITH_PLAN="$OUTDIR/temp_with_plan"
TEMP_WITHOUT_PLAN="$OUTDIR/temp_without_plan"
mkdir -p "$OUTDIR"

ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh)

cleanup() {
    rm -rf "$TEMP_WITH_PLAN" "$TEMP_WITHOUT_PLAN"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = 32;/g" gpcomp_common.cuh
    # Restore original NUM_INPUT_SSTS if we changed it
    if [[ -n "$NUM_SSTS" ]]; then
        sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = 4;/g" gpcomp_common.cuh
    fi
}

trap cleanup EXIT

echo "========================================================="
echo " GPComp Execution Sweep"
echo " comparing with_plan vs without_plan"
echo " measuring CPU utilization alongside speedup"
echo " label: $LABEL"
echo " output saved to: $OUTDIR"
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
echo " input SSTs: $CURRENT_SSTS"
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
    ./gpcomp_datagen --out_dir "$DATASET_DIR" --seed 42 > /dev/null

    # Run with_plan
    LOG_WITH_PLAN="$OUTDIR/result_val${VAL}B_with_plan.log"
    echo "Running q_paper_with_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITH_PLAN" --runs "$RUNS" --gpu_mode q_paper_with_plan > "$LOG_WITH_PLAN"

    # Run without_plan
    LOG_WITHOUT_PLAN="$OUTDIR/result_val${VAL}B_without_plan.log"
    echo "Running q_paper_without_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITHOUT_PLAN" --runs "$RUNS" --gpu_mode q_paper_without_plan > "$LOG_WITHOUT_PLAN"

    # Run with_plan utilization (no instrumentation)
    LOG_WITH_PLAN_UTIL="$OUTDIR/result_val${VAL}B_with_plan_util.log"
    echo "Running q_paper_with_plan_profile..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITH_PLAN" --runs "$RUNS" --gpu_mode q_paper_with_plan_profile > "$LOG_WITH_PLAN_UTIL"

    # Run without_plan utilization (no instrumentation)
    LOG_WITHOUT_PLAN_UTIL="$OUTDIR/result_val${VAL}B_without_plan_util.log"
    echo "Running q_paper_without_plan_profile..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITHOUT_PLAN" --runs "$RUNS" --gpu_mode q_paper_without_plan_profile > "$LOG_WITHOUT_PLAN_UTIL"

    # Print summary to terminal
    echo "  Value Size  : ${VAL} B"

    WITH_SPD=$(grep "Speedup:" "$LOG_WITH_PLAN" | awk '{print $2}')
    WITHOUT_SPD=$(grep "Speedup:" "$LOG_WITHOUT_PLAN" | awk '{print $2}')

    # Profile mode: pipeline-only utilization (I/O excluded, no instrumentation)
    WITH_PROF_PIPE=$(grep "GPU pipeline" "$LOG_WITH_PLAN_UTIL" | sed -n 's/.*utilization: \([0-9.]*\)%.*/\1/p')
    WITHOUT_PROF_PIPE=$(grep "GPU pipeline" "$LOG_WITHOUT_PLAN_UTIL" | sed -n 's/.*utilization: \([0-9.]*\)%.*/\1/p')
    CPU_PIPE=$(grep "CPU pipeline" "$LOG_WITH_PLAN_UTIL" | sed -n 's/.*utilization: \([0-9.]*\)%.*/\1/p')

    echo "  [With Plan]     Speedup: $WITH_SPD"
    echo "  [Without Plan]  Speedup: $WITHOUT_SPD"
    echo "  Pipeline CPU Util (I/O excluded):  CPU: ${CPU_PIPE}%  GPU-WP: ${WITH_PROF_PIPE}%  GPU-WoP: ${WITHOUT_PROF_PIPE}%"

    rm -rf "$TEMP_WITH_PLAN" "$TEMP_WITHOUT_PLAN"
done

echo ""
echo "Done! All results saved in $OUTDIR"
