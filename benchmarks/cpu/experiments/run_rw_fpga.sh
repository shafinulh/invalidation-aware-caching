#!/usr/bin/env bash
set -euo pipefail

# bloom enabled by default
# measure how many files accumulated by the end of the run, and how that impacts read performance
# might need read while writing, but experiment with that later

RUN_ID=fpga-rw-test_8th \
NUM_LOADS=20000000 \
MIX_READS=30000000 \
THREADS=8 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 2 4" \
WRITE_RATIOS="99" \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=1000000 \
LEVEL0_STOP_WRITES_TRIGGER=1000000 \
WRITE_BUFFER_SIZE=16777216 \
TARGET_FILE_SIZE_BASE=16777216 \
MAX_BYTES_FOR_LEVEL_BASE=100663296 \
MAX_BACKGROUND_FLUSHES=4 \
MAX_WRITE_BUFFER_NUMBER=100 \
./benchmarks/cpu/scripts/run_readwritemix.sh