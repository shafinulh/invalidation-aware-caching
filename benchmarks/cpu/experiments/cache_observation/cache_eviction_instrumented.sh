#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Instrumented run to measure block cache eviction caused by compaction.
#
# Uses the same Zipfian workload as bounded_l0_zipfian_cache.sh but with
# RocksDB-level instrumentation that logs [CACHE-EVICT] messages showing:
#   - Per-file: how many data blocks from each obsoleted SST are in the cache
#   - Per-compaction: total blocks evicted, total cache entries, eviction %
#
# The instrumentation output appears in the RocksDB LOG file, which is
# copied to the run directory's metadata/ folder after the mix phase.
#
# To extract the instrumentation data from the results:
#   grep 'CACHE-EVICT' <run_dir>/metadata/rocksdb_LOG_after_mix.txt

### Setup ###
# Identical parameters to bounded_l0_zipfian_cache.sh.
# Smaller run to iterate quickly on instrumentation validation.

RUN_ID=cache-evict-instrumented \
THREADS=4 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1" \
WRITE_RATIOS="25" \
NUM_KEYS=20000000 \
NUM_LOADS=20000000 \
LOAD_BENCH=fillseqdeterministic \
PRELOAD_DIR_NAME=zipfian_readwritemix_preload \
COMPACT_PRELOAD_ON_CREATE=1 \
MIX_READS=10000000 \
MAX_BACKGROUND_FLUSHES=4 \
METRICS_INTERVAL_MS=50 \
KEY_DIST=zipfian \
ZIPF_ALPHA=0.6 \
CACHE_SIZE=1073741824 \
./benchmarks/cpu/scripts/run_readwritemix.sh
