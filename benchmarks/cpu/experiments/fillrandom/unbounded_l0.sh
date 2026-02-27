#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Unbounded L0 (FPGA-style): slowdown/stop triggers set very high so
# write stalls never occur.  Faster compactions still reduce L0 file
# build-up which impacts read amplification.

# With only single-compaction multithreading (no parallel compactions),
# FillRandom throughput should not change significantly since there are
# no write stalls to alleviate.

### GOAL ###
# Demonstrate how L0 file accumulation changes with faster compactions
# when write stalls are disabled.

### Setup ###
# 10 threads to saturate i5-12900k cores with foreground writes
# and have fastest theoretical write throughput.
# 200M Key Space, 40M writes per thread = 400M total writes.

RUN_ID=fpga-fr_l0-16mb \
NUM_KEYS=200000000 \
WRITES=40000000 \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4 8 16 32" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=4 \
./benchmarks/cpu/scripts/run_fillrandom.sh
