#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Unbounded L0 (FPGA-style): no write stalls.
# Uses readwhilewriting benchmark (dedicated writer thread) instead of
# readrandomwriterandom.  Tests parallel background compaction threads
# rather than subcompactions alone.

### GOAL ###
# Measure impact of background compaction parallelism on read
# throughput while a separate thread is writing.

RUN_ID=gpcomp-rwr-test_2th_sparse20m_db-compacted \
MIX_BENCH=readwhilewriting \
NUM_KEYS=200000000 \
NUM_LOADS=20000000 \
LOAD_BENCH=fillrandom \
PRELOAD_DIR_NAME=rwr_preload_sparse20m_random \
COMPACT_PRELOAD_ON_CREATE=1 \
MIX_READS=2500000 \
THREADS=2 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 8" \
BG_COMP_THREADS_LIST="1" \
WRITE_RATIOS="50" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=8 \
LEVEL0_STOP_WRITES_TRIGGER=12 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=100 \
./benchmarks/cpu/scripts/run_readwritemix.sh
