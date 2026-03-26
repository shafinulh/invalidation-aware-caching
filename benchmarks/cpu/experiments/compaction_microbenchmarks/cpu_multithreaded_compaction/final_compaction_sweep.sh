#!/usr/bin/env bash
set -euo pipefail

### Final isolated compaction-parallelism sweep ###
# Knobs aligned with the CPU compaction study:
# - input data: 32 .. 2048 MB
# - key size: 16 B
# - value size: 32 .. 1024 B
# - SST size: 8, 16, 32, 64 MB
# - subcompactions: 1, 2, 4, 8, 16, 32, 64
#
# The supplied sweep uses compactall because db_bench's compact0 is a no-op on
# the pure L0-only preload shape used here. The analyzer still reports the
# actual L0-input compactions from RocksDB's event log.

RUN_ID=final-compactall-parallelism \
KEY_SIZE=16 \
VALUE_SIZES="32 64 128 256 512 1024" \
INPUT_DATA_MB_LIST="32 256" \
SST_SIZE_MB_LIST="8" \
SUBCOMP_THREADS_LIST="1 2 4 8 16 32" \
THREADS=1 \
PRELOAD_THREADS=1 \
PRELOAD_KEYSPACE_MULTIPLIER=20 \
PRELOAD_MIN_NUM_KEYS=200000000 \
COMPACTION_BENCH=compactall \
COMPACTION_BG_THREADS=1 \
COMPACTION_PERF_LEVEL=5 \
OPEN_FILES=512 \
DIRECT_IO=true \
METRICS_INTERVAL_MS=100 \
HOST_METRICS_INTERVAL_SEC=0.1 \
REPORT_BG_IO_STATS=1 \
RUN_QUIET=1 \
RUN_PAUSE_SECONDS=2 \
./benchmarks/cpu/scripts/run_compaction_parallelism.sh
