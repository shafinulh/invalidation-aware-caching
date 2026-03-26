#!/usr/bin/env bash
set -euo pipefail

### Sample isolated compaction-parallelism experiment ###
# Smaller matrix for quick validation and iteration.
#
# Analyze a finished run with:
# python3 benchmarks/cpu/python/analyze_compaction_parallelism.py \
#   /path/to/bench_results/cpu/compaction_parallelism/$RUN_ID

RUN_ID=sample-compactall-parallelism \
VALUE_SIZES="32 256 1024" \
INPUT_DATA_MB_LIST="64 256 1024" \
SST_SIZE_MB_LIST="8 32" \
SUBCOMP_THREADS_LIST="1 4 16 64" \
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
./benchmarks/cpu/scripts/run_compaction_parallelism.sh
