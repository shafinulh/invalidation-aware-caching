#!/usr/bin/env bash
set -euo pipefail

# set env vars
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCKSDB_DIR="${ROOT_DIR}/rocksdb-gpu"
DB_BENCH="${DB_BENCH:-${ROCKSDB_DIR}/db_bench}"

# set these to nvme storage
DB_BASE_DIR="${DB_BASE_DIR:-${DB_DIR:-/tmp/rocksdb_data}}"
WAL_BASE_DIR="${WAL_BASE_DIR:-${WAL_DIR:-/tmp/rocksdb_wal}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/bench_results}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
THREADS="${THREADS:-8}"
KEY_SIZE="${KEY_SIZE:-16}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"

COMMON_FLAGS=(
  --key_size="${KEY_SIZE}"
  --compression_type="${COMPRESSION_TYPE}"
  --statistics
  --stats_interval_seconds=60
  --stats_dump_period_sec=60
  --report_interval_seconds=1
  --threads="${THREADS}"
)

run_db_bench() {
  local log_file="$1"
  shift
  "${DB_BENCH}" "$@" 2>&1 | tee "${log_file}"
}
