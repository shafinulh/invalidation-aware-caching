#!/usr/bin/env bash
set -euo pipefail

### Sample monitored FillRandom experiment ###
# Demonstrates:
# - value size 32
# - subcompactions 8 vs 16
# - 60s cooldown between runs
# - host IO / CPU / thread CPU capture without RocksDB metrics.csv

RUN_ID=bounded-fr-l0-monitored-sample-retry \
NUM_KEYS=200000000 \
WRITES=-1 \
WRITES_BY_VALUE_SIZE="32:20000000" \
THREADS=10 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES="32" \
SUBCOMP_THREADS_LIST="8 16" \
BG_COMP_THREADS_LIST="1" \
DISABLE_AUTO_COMPACTIONS=0 \
LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 \
LEVEL0_SLOWDOWN_WRITES_TRIGGER=8 \
LEVEL0_STOP_WRITES_TRIGGER=12 \
MAX_BACKGROUND_FLUSHES=4 \
HOST_METRICS_INTERVAL_SEC=1 \
REPORT_BG_IO_STATS=1 \
RUN_PAUSE_SECONDS=60 \
RUN_QUIET=1 \
./benchmarks/cpu/scripts/run_fillrandom.sh
