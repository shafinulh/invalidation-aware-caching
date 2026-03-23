#!/usr/bin/env bash
set -euo pipefail

OUTDIR="./results/sweep_$(date '+%Y-%m-%d_%H-%M-%S')"
mkdir -p "$OUTDIR"

echo "========================================================="
echo " GPComp Execution Sweep"
echo " comparing with_plan vs without_plan"
echo " measuring CPU utilization alongside speedup"
echo " output saved to: $OUTDIR"
echo "========================================================="

# Loop over value sizes
for VAL in 32 64 128; do
    echo ""
    echo "--- Building for Value Size: ${VAL} B ---"
    
    # Patch the value size in gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${VAL};/g" gpcomp_common.cuh
    make clean > /dev/null
    make gpcomp_datagen gpcomp_bench -j > /dev/null

    # Generate dataset for this value size
    DATASET_DIR="dataset_shafin_V${VAL}"
    echo "Generating dataset in $DATASET_DIR..."
    ./gpcomp_datagen --out_dir "$DATASET_DIR" --seed 42 > /dev/null

    # Run with_plan
    LOG_WITH_PLAN="$OUTDIR/result_val${VAL}B_with_plan.txt"
    echo "Running q_paper_with_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$OUTDIR/temp_with_plan" --runs 5 --gpu_mode q_paper_with_plan > "$LOG_WITH_PLAN"

    # Run without_plan
    LOG_WITHOUT_PLAN="$OUTDIR/result_val${VAL}B_without_plan.txt"
    echo "Running q_paper_without_plan..."
    ./gpcomp_bench --dataset "$DATASET_DIR" --out_dir "$OUTDIR/temp_without_plan" --runs 5 --gpu_mode q_paper_without_plan > "$LOG_WITHOUT_PLAN"

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

    rm -rf "$OUTDIR/temp_with_plan" "$OUTDIR/temp_without_plan"
done

echo ""
echo "Done! All results saved in $OUTDIR"
# Restore standard 32B just in case
sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = 32;/g" gpcomp_common.cuh
