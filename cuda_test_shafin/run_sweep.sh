#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-5}"
VALUES_STR="${VALUES:-32 64 128}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
OUTDIR="${OUTDIR:-./results/sweep_$(date '+%Y-%m-%d_%H-%M-%S')}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir) OUTDIR="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --values) VALUES_STR="$2"; shift 2 ;;
        --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TEMP_WITH_PLAN="$OUTDIR/temp_with_plan"
TEMP_WITHOUT_PLAN="$OUTDIR/temp_without_plan"
mkdir -p "$OUTDIR"

cleanup() {
    rm -rf "$TEMP_WITH_PLAN" "$TEMP_WITHOUT_PLAN"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = 32;/g" gpcomp_common.cuh
}

trap cleanup EXIT

echo "========================================================="
echo " GPComp Execution Sweep"
echo " comparing with_plan vs without_plan"
echo " measuring CPU utilization alongside speedup"
echo " output saved to: $OUTDIR"
echo " runs per mode: $RUNS"
echo " value sizes: $VALUES_STR"
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
    LOG_WITH_PLAN="$OUTDIR/result_val${VAL}B_with_plan.txt"
    echo "Running q_paper_with_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITH_PLAN" --runs "$RUNS" --gpu_mode q_paper_with_plan > "$LOG_WITH_PLAN"

    # Run without_plan
    LOG_WITHOUT_PLAN="$OUTDIR/result_val${VAL}B_without_plan.txt"
    echo "Running q_paper_without_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$TEMP_WITHOUT_PLAN" --runs "$RUNS" --gpu_mode q_paper_without_plan > "$LOG_WITHOUT_PLAN"

    # Print summary to terminal
    echo "  Value Size  : ${VAL} B"
    
    WITH_SPD=$(grep "Speedup:" "$LOG_WITH_PLAN" | awk '{print $2}')
    
    WITH_CPU_CPU_TIME=$(grep "CPU total (CPU-Time):" "$LOG_WITH_PLAN" | awk '{print $4}')
    WITH_CPU_WALL=$(grep "CPU total (Wall):" "$LOG_WITH_PLAN" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
    WITH_CPU_PCT=$(awk -v cpu="$WITH_CPU_CPU_TIME" -v wall="$WITH_CPU_WALL" 'BEGIN { if (wall > 0) printf "%.1f%%", (cpu/wall)*100; else print "N/A" }')

    WITH_GPU_CPU_TIME=$(grep "GPU total (CPU-Time):" "$LOG_WITH_PLAN" | awk '{print $4}')
    WITH_GPU_WALL=$(grep "GPU total (Wall):" "$LOG_WITH_PLAN" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
    WITH_GPU_PCT=$(awk -v cpu="$WITH_GPU_CPU_TIME" -v wall="$WITH_GPU_WALL" 'BEGIN { if (wall > 0) printf "%.1f%%", (cpu/wall)*100; else print "N/A" }')

    WITHOUT_SPD=$(grep "Speedup:" "$LOG_WITHOUT_PLAN" | awk '{print $2}')
    
    WITHOUT_CPU_CPU_TIME=$(grep "CPU total (CPU-Time):" "$LOG_WITHOUT_PLAN" | awk '{print $4}')
    WITHOUT_CPU_WALL=$(grep "CPU total (Wall):" "$LOG_WITHOUT_PLAN" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
    WITHOUT_CPU_PCT=$(awk -v cpu="$WITHOUT_CPU_CPU_TIME" -v wall="$WITHOUT_CPU_WALL" 'BEGIN { if (wall > 0) printf "%.1f%%", (cpu/wall)*100; else print "N/A" }')

    WITHOUT_GPU_CPU_TIME=$(grep "GPU total (CPU-Time):" "$LOG_WITHOUT_PLAN" | awk '{print $4}')
    WITHOUT_GPU_WALL=$(grep "GPU total (Wall):" "$LOG_WITHOUT_PLAN" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
    WITHOUT_GPU_PCT=$(awk -v cpu="$WITHOUT_GPU_CPU_TIME" -v wall="$WITHOUT_GPU_WALL" 'BEGIN { if (wall > 0) printf "%.1f%%", (cpu/wall)*100; else print "N/A" }')


    echo "  [With Plan]     Speedup: $WITH_SPD  CPU-run Utils (CPU_Time/Wall): $WITH_CPU_PCT,  GPU-run Utils: $WITH_GPU_PCT"
    echo "  [Without Plan]  Speedup: $WITHOUT_SPD  CPU-run Utils (CPU_Time/Wall): $WITHOUT_CPU_PCT,  GPU-run Utils: $WITHOUT_GPU_PCT"

    rm -rf "$TEMP_WITH_PLAN" "$TEMP_WITHOUT_PLAN"
done

echo ""
echo "Done! All results saved in $OUTDIR"
