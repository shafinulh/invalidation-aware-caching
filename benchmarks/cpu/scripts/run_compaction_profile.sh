#!/usr/bin/env bash
set -euo pipefail

# Isolated compaction profiling workload.
#
# Two-phase approach:
#   Phase 1 – LOAD:  fillrandom with auto-compactions disabled so L0 SSTs
#             accumulate without being merged.
#   Phase 2 – COMPACT: compactall with --report_bg_io_stats=true.
#             RocksDB logs per-compaction wall-clock, CPU, and write-IO nanos.
#
# Analyse results with:
#   python3 benchmarks/cpu/python/parse_compaction_profile.py <RUN_DIR>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/benchmark_common.sh"

# ── Knobs ───────────────────────────────────────────────────────────
NUM_KEYS="${NUM_KEYS:-200000000}"
NUM_LOADS="${NUM_LOADS:-20000000}"
LOAD_BENCH="${LOAD_BENCH:-fillrandom}"
COMPACT_THREADS="${COMPACT_THREADS:-1}"
COMPACT_SUBCOMP="${COMPACT_SUBCOMP:-1}"

# ── Run ─────────────────────────────────────────────────────────────
for value_size in ${VALUE_SIZES}; do

  RUN_DIR="${OUTPUT_DIR}/compaction_profile/value_${value_size}/${RUN_ID}"
  DB_DIR="${DB_BASE_DIR}/compaction_profile/value_${value_size}"
  WAL_DIR="${WAL_BASE_DIR}/compaction_profile/value_${value_size}"

  echo "================================================================"
  echo "  Compaction IO Breakdown — value_size=${value_size}"
  echo "  RUN_DIR: ${RUN_DIR}"
  echo "================================================================"

  cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"
  mkdir -p "${RUN_DIR}/metadata" "${DB_DIR}" "${WAL_DIR}"
  write_run_config "${RUN_DIR}" "run_compaction_profile.sh"

  # ── Phase 1: LOAD (auto-compactions disabled) ────────────────────
  echo ""
  echo "▶ Phase 1: Loading ${NUM_LOADS} keys (auto-compactions OFF)..."
  echo ""

  run_db_bench "${RUN_DIR}/load.log" \
    --benchmarks="${LOAD_BENCH}" \
    --disable_auto_compactions=1 \
    --num="${NUM_KEYS}" \
    --writes="${NUM_LOADS}" \
    --value_size="${value_size}" \
    --db="${DB_DIR}" \
    --wal_dir="${WAL_DIR}" \
    --report_file="${RUN_DIR}/load_report.csv" \
    --use_existing_db=0 \
    --subcompactions=1 \
    --max_background_compactions=1 \
    --report_bg_io_stats=false \
    "${COMMON_FLAGS[@]}"

  copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_load"
  copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_load"

  # Record L0 file count before compaction
  L0_COUNT=$(find "${DB_DIR}" -maxdepth 1 -name "*.sst" 2>/dev/null | wc -l)
  echo "  SST files after load: ${L0_COUNT}"
  echo "L0_SST_COUNT_AFTER_LOAD=${L0_COUNT}" >> "${RUN_DIR}/metadata/run_config.env"

  # ── Phase 2: COMPACT (with IO stats) ─────────────────────────────
  echo ""
  echo "▶ Phase 2: Running compactall (report_bg_io_stats=true)..."
  echo ""

  run_db_bench "${RUN_DIR}/compact.log" \
    --benchmarks=compactall,stats \
    --disable_auto_compactions=0 \
    --num="${NUM_KEYS}" \
    --value_size="${value_size}" \
    --db="${DB_DIR}" \
    --wal_dir="${WAL_DIR}" \
    --report_file="${RUN_DIR}/compact_report.csv" \
    --use_existing_db=1 \
    --subcompactions="${COMPACT_SUBCOMP}" \
    --max_background_compactions="${COMPACT_THREADS}" \
    --report_bg_io_stats=true \
    --perf_level=5 \
    "${COMMON_FLAGS[@]}"

  copy_latest_rocksdb_options "${DB_DIR}" "${RUN_DIR}" "after_compact"
  copy_rocksdb_log_file "${DB_DIR}" "${RUN_DIR}" "after_compact"
  cleanup_db_wal_dirs "${DB_DIR}" "${WAL_DIR}"

  echo ""
  echo "✓ Done. Results in: ${RUN_DIR}"
  echo "  python3 benchmarks/cpu/python/parse_compaction_profile.py ${RUN_DIR}"
  echo ""

done
