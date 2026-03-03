#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Observe block-cache behaviour with Zipfian key distribution.
#
# Uniform random reads spread access across all keys equally, producing
# ~1% cache hit rate with a 32 MB block cache.  A 128 MB cache (default)
# is large enough to hold the hot working set.  Zipfian distribution
# (alpha=0.99, YCSB default) concentrates reads on a small hot set,
# creating the locality needed to actually exercise the block cache.
#
# Key questions this experiment answers:
#   1. How much does the cache hit rate increase with Zipfian reads?
#   2. Does compaction-driven cache invalidation become visible when
#      the cache is actually populated with hot blocks?
#   3. How does read latency behave around compaction events when
#      there is meaningful cache utilisation?
#
# Compare with bounded_l0_cache_impact.sh (uniform) to isolate the
# effect of key distribution on cache behaviour.

### Setup ###
# Same as bounded_l0_cache_impact.sh but with Zipfian key distribution.
# Preload a full 20M-key table deterministically, compact that preload
# once, then run the mixed phase with normal background compactions.
# 50ms metrics polling to catch transient dips.  25% writes keeps both
# reads and writes active without burying cache effects in write stalls.

RUN_ID=cache-obs-zipfian-0.6-1024mb-test \
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
