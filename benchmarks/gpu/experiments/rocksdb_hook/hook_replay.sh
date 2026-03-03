#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Dummy GPU compaction hook benchmark for RocksDB.
#
# This drives a manual compaction through RocksDB's CompactionService hook,
# stages a ground-truth compaction output, then replays those SST files through
# a cuFile-backed GPU pass-through copy helper before RocksDB installs them.

### Run ###

RUN_ID="${RUN_ID:-rocksdb-hook-replay}" \
ALIGNMENT="${ALIGNMENT:-4096}" \
INPUT_SST_MB="${INPUT_SST_MB:-8}" \
NUM_REPS="${NUM_REPS:-1}" \
./benchmarks/gpu/scripts/run_rocksdb_gpu_hook_bench.sh
