#!/usr/bin/env bash
set -euo pipefail

# Inline compaction profiling during a fillrandom workload.
#
# Runs fillrandom with auto-compactions enabled and --report_bg_io_stats=true
# so every background compaction emits wall-clock, CPU, and write-IO timing.
#
# Analyse results with:
#   python3 benchmarks/cpu_compaction/plotting/parse_compaction_profile.py <RUN_DIR>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "${SCRIPT_DIR}/../.." && pwd)/common/benchmark_common.sh"

# ── Knobs ───────────────────────────────────────────────────────────
NUM_KEYS="${NUM_KEYS:-200000000}"
WRITES="${WRITES:-20000000}"
COMPACT_SUBCOMP="${COMPACT_SUBCOMP:-1}"
COMPACT_BG_THREADS="${COMPACT_BG_THREADS:-1}"

# ── Run ─────────────────────────────────────────────────────────────
for value_size in ${VALUE_SIZES}; do

  RUN_DIR="${OUTPUT_DIR}/compaction_profile/value_${value_size}/fillrandom_inline/${RUN_ID}"
  DB_DIR="${DB_BASE_DIR}/compaction_profile_fr/value_${value_size}"
  WAL_DIR="${WAL_BASE_DIR}/compaction_profile_fr/value_${value_size}"

  echo "================================================================"
  echo "  Inline Compaction Profiling — fillrandom, value_size=${value_size}"
  echo "  RUN_DIR: ${RUN_DIR}"
  echo "================================================================"

  cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
  mkdir -p "${RUN_DIR}/metadata" "${DB_DIR}" "${WAL_DIR}"
  write_run_config "${RUN_DIR}" "run_fillrandom_profile.sh"

  run_db_bench "${RUN_DIR}/db_bench.log" \
    --benchmarks=fillrandom,stats \
    --disable_auto_compactions=0 \
    --num="${NUM_KEYS}" \
    --writes="${WRITES}" \
    --value_size="${value_size}" \
    --threads="${THREADS}" \
    --db="${DB_DIR}" \
    --wal_dir="${WAL_DIR}" \
    --report_file="${RUN_DIR}/report.csv" \
    --metrics_file="${RUN_DIR}/metrics.csv" \
    --use_existing_db=0 \
    --subcompactions="${COMPACT_SUBCOMP}" \
    --max_background_compactions="${COMPACT_BG_THREADS}" \
    --report_bg_io_stats=true \
    --perf_level=5 \
    "${COMMON_FLAGS[@]}"

  copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
  copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_fillrandom"
  cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"

  echo ""
  echo "✓ Done. Results in: ${RUN_DIR}"
  echo "  python3 benchmarks/cpu_compaction/plotting/parse_compaction_profile.py ${RUN_DIR}"
  echo ""

done
