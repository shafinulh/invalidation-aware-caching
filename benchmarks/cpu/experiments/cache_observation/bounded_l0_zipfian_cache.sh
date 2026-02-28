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
# Bounded L0 (GPComp-style) so compactions are frequent under write
# pressure.  50ms metrics polling to catch transient dips.
# 50% write ratio so both reads and writes are active.

RUN_ID=cache-obs-zipfian-0.6-2048mb \
THREADS=4 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1" \
WRITE_RATIOS="25" \
NUM_LOADS=20000000 \
MIX_READS=10000000 \
MAX_BACKGROUND_FLUSHES=4 \
METRICS_INTERVAL_MS=50 \
KEY_DIST=zipfian \
ZIPF_ALPHA=0.6 \
CACHE_SIZE=2147483648 \
./benchmarks/cpu/scripts/run_readwritemix.sh
