#!/usr/bin/env bash
set -euo pipefail

### Context ###
# GPComp demonstrates better throughput with faster compactions
# (multithreading or GPU) by reducing write stalls. 

# GPComp focuses only on speeding up single compactions with GPU,
# so to make comparison with CPU multithreading, we set max_background_compactions=1 
# and vary subcompactions which splits individual compactions into multiple threads.
# Multiple compactions are not scheduled at the same time.

### GOAL ###
# This benchmark serves to demonstrate how throughput increases as compaction
# throughput increases with more subcompactions. We expect to see significant improvements

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes 
# and have fastest theoretical write throughput

# 200M Key Space, 40M writes per thread = 400M total writes

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
