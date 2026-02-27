#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Bounded L0 (GPComp-style): slowdown=8, stop=12.
# Write stalls occur when L0 files accumulate, so faster compactions
# (via subcompactions) should significantly improve throughput by
# alleviating those stalls.

### GOAL ###
# Demonstrate throughput gains from faster single-compaction execution
# (subcompactions) in a bounded LSM tree.

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes
# and have fastest theoretical write throughput.
# 200M Key Space, 40M writes per thread = 400M total writes.

# Also experiment with setting larger memtable. Larger memtables result in lower write stalls,
# but more expensive compactions that may potentially benefit more from multithreading.
# WRITE_BUFFER_SIZE=67108864 \
# TARGET_FILE_SIZE_BASE=67108864 \
# MAX_BYTES_FOR_LEVEL_BASE=268435456 \
# LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
# LEVEL0_SLOWDOWN_WRITES_TRIGGER=6 \
# LEVEL0_STOP_WRITES_TRIGGER=8 \

RUN_ID=gpcomp-fr \
NUM_KEYS=200000000 \
WRITES=40000000 \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4 8 16" \
MAX_BACKGROUND_FLUSHES=4 \
./benchmarks/cpu/scripts/run_fillrandom.sh
