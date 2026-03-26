#!/usr/bin/env bash
set -euo pipefail

### Single GPU compactall base case ###
# This creates L0 files with fillrandom, then runs compactall with the
# in-process GPU compaction service enabled for the L0/L1 -> L1 step.
#
# Result path:
#   $OUTPUT_DIR/compaction_parallelism/$RUN_ID/sst_8MB/input_32MB/value_32/subcomp_1/compact.log

RUN_ID=gpu-compactall-base-case \
KEY_SIZE=16 \
VALUE_SIZES="32" \
INPUT_DATA_MB_LIST="32" \
SST_SIZE_MB_LIST="8" \
SUBCOMP_THREADS_LIST="1" \
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
RUN_PAUSE_SECONDS=0 \
LOAD_DB_BENCH_EXTRA_FLAGS="--use_gpu_compaction_service --gpu_compaction_cuda_device=0 --gpu_compaction_fallback_on_unsupported_input=0 --compaction_verify_record_count=0" \
COMPACTION_DB_BENCH_EXTRA_FLAGS="--use_gpu_compaction_service --gpu_compaction_cuda_device=0 --gpu_compaction_fallback_on_unsupported_input=0 --compaction_verify_record_count=0" \
./benchmarks/cpu/scripts/run_compaction_parallelism.sh
