#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Bounded L0 (GPComp-style defaults) with readrandomwriterandom mix.
# Write stalls may still occur, but the read-heavy mix is less
# affected by compaction speed.

### GOAL ###
# Measure impact of subcompaction parallelism on mixed read/write
# throughput under bounded L0 conditions.

# WRITE_BUFFER_SIZE=33554432 \
# TARGET_FILE_SIZE_BASE=33554432 \
# MAX_BYTES_FOR_LEVEL_BASE=134217728 \
# WRITE_BUFFER_SIZE=67108864 \
# TARGET_FILE_SIZE_BASE=67108864 \
# MAX_BYTES_FOR_LEVEL_BASE=268435456 \
# LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
# LEVEL0_SLOWDOWN_WRITES_TRIGGER=6 \
# LEVEL0_STOP_WRITES_TRIGGER=8 \
# MAX_BACKGROUND_FLUSHES=4 \
# BLOOM_BITS=0 \

RUN_ID=l0-8mb_page-cache-reads \
THREADS=1 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4" \
WRITE_RATIOS="99" \
MAX_BACKGROUND_FLUSHES=4 \
WRITE_BUFFER_SIZE=67108864 \
TARGET_FILE_SIZE_BASE=67108864 \
MAX_BYTES_FOR_LEVEL_BASE=268435456 \
./benchmarks/cpu/scripts/run_readwritemix.sh
