#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CPU_ENV_FILE="${REPO_ROOT}/benchmarks/cpu/config/.env.local"

if [[ ! -f "${CPU_ENV_FILE}" ]]; then
  echo "error: missing ${CPU_ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CPU_ENV_FILE}"
set +a

SST_SIZE_MB=""
INPUT_SSTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sst-size-mb)
      SST_SIZE_MB="$2"
      shift 2
      ;;
    --input-ssts)
      INPUT_SSTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SST_SIZE_MB}" || -z "${INPUT_SSTS}" ]]; then
  echo "usage: $0 --sst-size-mb <mb> --input-ssts <count>" >&2
  exit 1
fi

FINAL_VALUE_SIZES="${FINAL_VALUE_SIZES:-32 64 128 256 512 1024}"
FINAL_SUBCOMP_THREADS_LIST="${FINAL_SUBCOMP_THREADS_LIST:-1 2 4 8 16 32}"
FINAL_GPU_MODES="${FINAL_GPU_MODES:-q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan}"
FINAL_GPU_RUNS="${FINAL_GPU_RUNS:-5}"
FINAL_CPU_RUNS="${FINAL_CPU_RUNS:-5}"

FINAL_PRELOAD_KEYSPACE_MULTIPLIER="${FINAL_PRELOAD_KEYSPACE_MULTIPLIER:-1}"
FINAL_PRELOAD_MIN_NUM_KEYS="${FINAL_PRELOAD_MIN_NUM_KEYS:-200000000}"
FINAL_GPU_USER_KEY_SPACE="${FINAL_GPU_USER_KEY_SPACE:-200000000}"
FINAL_GPU_ZIPF_ALPHA="${FINAL_GPU_ZIPF_ALPHA:-0.0}"

FINAL_CPU_RUN_QUIET="${FINAL_CPU_RUN_QUIET:-1}"
FINAL_RUN_PAUSE_SECONDS="${FINAL_RUN_PAUSE_SECONDS:-2}"
FINAL_GRAPH_DIR="${FINAL_GRAPH_DIR:-./graphs/final_speedup}"

CASE_LABEL="${SST_SIZE_MB}mb-sst_${INPUT_SSTS}sst"
CPU_RUN_ID="${CPU_RUN_ID:-final-speedup-cpu-${CASE_LABEL}}"
GPU_LABEL="${GPU_LABEL:-${CASE_LABEL}}"
PRELOAD_DIR_NAME="${PRELOAD_DIR_NAME:-final_speedup_preload_${CASE_LABEL}}"

CPU_EXPERIMENT_ROOT="${OUTPUT_DIR}/compaction_parallelism/${CPU_RUN_ID}"
GPU_SWEEP_DIR="${REPO_ROOT}/cuda_test_shafin/sweep_results/sweep_${GPU_LABEL}"
COMPARE_OUTPUT_DIR="${OUTPUT_DIR}/compaction_parallelism/final_speedup/${GPU_LABEL}"
CPU_PER_RUN_LOG_DIR="${CPU_EXPERIMENT_ROOT}/per_run_logs"

assert_release_db_bench() {
  if strings "${DB_BENCH}" | grep -q "Assertions are enabled"; then
    echo "error: DB_BENCH is an assertions-enabled build: ${DB_BENCH}" >&2
    echo "hint: rebuild RocksDB db_bench in release mode, e.g. make db_bench DEBUG_LEVEL=0 -j" >&2
    exit 1
  fi
}

copy_gpu_logs_into_cpu_per_run_logs() {
  mkdir -p "${CPU_PER_RUN_LOG_DIR}"
  rm -f "${CPU_PER_RUN_LOG_DIR}"/gpcomp_bench_v*.log

  shopt -s nullglob
  local copied_count=0
  local src_path=""
  for src_path in "${GPU_SWEEP_DIR}"/result_val*B_*.log; do
    local base_name
    base_name="$(basename "${src_path}")"
    if [[ "${base_name}" =~ ^result_val([0-9]+)B_(.+)\.log$ ]]; then
      cp "${src_path}" "${CPU_PER_RUN_LOG_DIR}/gpcomp_bench_v${BASH_REMATCH[1]}_${BASH_REMATCH[2]}.log"
      copied_count=$((copied_count + 1))
    fi
  done
  shopt -u nullglob

  if (( copied_count == 0 )); then
    echo "error: no GPU result logs found under ${GPU_SWEEP_DIR}" >&2
    exit 1
  fi
}

echo "========================================================="
echo " Final CPU/GPU Compaction Speedup Case"
echo " case label: ${CASE_LABEL}"
echo " CPU run id: ${CPU_RUN_ID}"
echo " CPU output: ${CPU_EXPERIMENT_ROOT}"
echo " GPU output: ${GPU_SWEEP_DIR}"
echo " Compare output: ${COMPARE_OUTPUT_DIR}"
echo " value sizes: ${FINAL_VALUE_SIZES}"
echo " subcompactions: ${FINAL_SUBCOMP_THREADS_LIST}"
echo " cpu repeats per subcomp: ${FINAL_CPU_RUNS}"
echo " gpu modes: ${FINAL_GPU_MODES}"
echo " uniform key space: CPU min=${FINAL_PRELOAD_MIN_NUM_KEYS}, GPU user_key_space=${FINAL_GPU_USER_KEY_SPACE}"
echo "========================================================="

if [[ "${FINAL_SKIP_CPU:-0}" != "1" ]]; then
  assert_release_db_bench
  (
    cd "${REPO_ROOT}"
    RUN_ID="${CPU_RUN_ID}" \
    KEY_SIZE=16 \
    VALUE_SIZES="${FINAL_VALUE_SIZES}" \
    INPUT_SST_COUNT_LIST="${INPUT_SSTS}" \
    SST_SIZE_MB_LIST="${SST_SIZE_MB}" \
    SUBCOMP_THREADS_LIST="${FINAL_SUBCOMP_THREADS_LIST}" \
    COMPACTION_RUNS="${FINAL_CPU_RUNS}" \
    THREADS=1 \
    PRELOAD_THREADS=1 \
    PRELOAD_DIR_NAME="${PRELOAD_DIR_NAME}" \
    PRELOAD_KEYSPACE_MULTIPLIER="${FINAL_PRELOAD_KEYSPACE_MULTIPLIER}" \
    PRELOAD_MIN_NUM_KEYS="${FINAL_PRELOAD_MIN_NUM_KEYS}" \
    LOAD_PERF_LEVEL=1 \
    COMPACTION_BENCH=compactall \
    COMPACTION_BG_THREADS=1 \
    COMPACTION_PERF_LEVEL=5 \
    OPEN_FILES=512 \
    DIRECT_IO=true \
    METRICS_INTERVAL_MS=100 \
    HOST_METRICS_INTERVAL_SEC=0.1 \
    REPORT_BG_IO_STATS=1 \
    RUN_QUIET="${FINAL_CPU_RUN_QUIET}" \
    RUN_PAUSE_SECONDS="${FINAL_RUN_PAUSE_SECONDS}" \
    ./benchmarks/cpu/scripts/run_compaction_parallelism.sh
  )

  (
    cd "${REPO_ROOT}"
    python3 ./benchmarks/cpu/python/analyze_compaction_parallelism.py "${CPU_EXPERIMENT_ROOT}"
  )
fi

if [[ "${FINAL_SKIP_GPU:-0}" != "1" ]]; then
  (
    cd "${REPO_ROOT}/cuda_test_shafin"
    bash ./run_sweep.sh \
      --num_ssts "${INPUT_SSTS}" \
      --label "${GPU_LABEL}" \
      --values "${FINAL_VALUE_SIZES}" \
      --runs "${FINAL_GPU_RUNS}" \
      --modes "${FINAL_GPU_MODES}" \
      --zipf_alpha "${FINAL_GPU_ZIPF_ALPHA}" \
      --user_key_space "${FINAL_GPU_USER_KEY_SPACE}" \
      --graph_dir "${FINAL_GRAPH_DIR}"
  )
fi

if [[ "${FINAL_SKIP_GPU:-0}" != "1" || -d "${GPU_SWEEP_DIR}" ]]; then
  copy_gpu_logs_into_cpu_per_run_logs
fi

if [[ "${FINAL_SKIP_COMPARE:-0}" != "1" ]]; then
  (
    cd "${REPO_ROOT}"
    python3 ./benchmarks/cpu/python/compare_cpu_gpu_compaction_speedups.py \
      --cpu-experiment-root "${CPU_EXPERIMENT_ROOT}" \
      --gpu-sweep-dir "${GPU_SWEEP_DIR}" \
      --output-dir "${COMPARE_OUTPUT_DIR}" \
      --sst-size-mb "${SST_SIZE_MB}" \
      --input-sst-count "${INPUT_SSTS}" \
      --subcomp-threads "${FINAL_SUBCOMP_THREADS_LIST}" \
      --gpu-modes "${FINAL_GPU_MODES}"
  )
fi

echo ""
echo "Case complete."
echo "CPU experiment root: ${CPU_EXPERIMENT_ROOT}"
echo "GPU sweep root: ${GPU_SWEEP_DIR}"
echo "Comparison output: ${COMPARE_OUTPUT_DIR}"
