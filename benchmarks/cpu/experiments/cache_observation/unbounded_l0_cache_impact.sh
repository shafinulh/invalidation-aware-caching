#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Same cache-observation experiment but with unbounded L0 (FPGA-style).
# No write stalls, so compaction frequency is lower.  Compare how
# cache invalidation differs when the LSM tree is less bounded.

RUN_ID=cache-obs-unbounded_50w \
THREADS=4 \
OPEN_FILES=512 \
DIRECT_IO=false \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 4" \
WRITE_RATIOS="50" \
NUM_LOADS=20000000 \
MIX_READS=30000000 \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=100 \
METRICS_INTERVAL_MS=100 \
./benchmarks/cpu/scripts/run_readwritemix.sh
