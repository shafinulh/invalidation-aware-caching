#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Unbounded L0 (FPGA-style) with parallel background compactions.
# Running multiple parallel compactions (as done in the FPGA paper)
# causes CPU contention: more time on background compactions means
# less CPU for foreground writes.

### GOAL ###
# Demonstrate that parallel background compactions cause CPU contention
# and thus worse write throughput.

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes
# and have fastest theoretical write throughput.
# 200M Key Space, 40M writes per thread = 400M total writes.

RUN_ID=fpga-cpu-contention-fr_l0-16mb_ \
NUM_KEYS=200000000 \
WRITES=40000000 \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4 8 16 32" \
BG_COMP_THREADS_LIST="16" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=100 \
./benchmarks/cpu/scripts/run_fillrandom.sh
