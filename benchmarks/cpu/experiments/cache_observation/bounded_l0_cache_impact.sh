#!/usr/bin/env bash
set -euo pipefail

### Context ###
# Observe block-cache behaviour during a mixed read/write workload.
#
# Key questions this experiment answers:
#   1. Does the block-cache hit rate drop after a compaction?
#   2. Do read P95 latencies spike around compaction events?
#   3. Does real-time throughput dip when compactions invalidate
#      cached blocks?
#
# The metrics CSV produced by --metrics_interval_ms contains
# per-interval deltas for cache hit/miss, latency histograms, and
# compaction byte counters.  Compaction events show up as non-zero
# compact_read_bytes / compact_write_bytes columns.

### Setup ###
# Bounded L0 (GPComp-style) so compactions are frequent under write
# pressure.  100ms metrics polling to catch transient dips.
# Moderate write ratio (50%) so both reads and writes are active.

RUN_ID=cache-obs-bounded \
THREADS=4 \
OPEN_FILES=512 \
DIRECT_IO=true \
VALUE_SIZES=32 \
SUBCOMP_THREADS_LIST="1 4" \
WRITE_RATIOS="50" \
NUM_LOADS=20000000 \
MIX_READS=10000000 \
MAX_BACKGROUND_FLUSHES=4 \
METRICS_INTERVAL_MS=50 \
./benchmarks/cpu/scripts/run_readwritemix.sh
