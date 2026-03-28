#!/usr/bin/env bash
set -euo pipefail

# CPU vs GPU compaction comparison orchestrator.
#
# Runs 4 phases with matching parameters, collecting all results under
# a single experiment directory:
#
#   Phase 1 — CPU RocksDB compaction sweep
#             Multithreaded subcompaction sweep with host metrics (CPU/IO).
#             Uses cpu_compaction_sweep.sh.
#
#   Phase 2 — GPU compaction with stage breakdown
#             Default mode (CPU baseline + GPU), produces per-stage timing
#             breakdown (read, unpack, sort, gc, plan, bloom, pack, write).
#             Uses gpu_compaction_sweep.sh.
#
#   Phase 3 — GPU compaction with host metrics
#             GPU-only mode with host CPU/IO metrics collection (no CPU
#             baseline overhead). Uses gpu_compaction_sweep.sh --gpu_only.
#
#   Phase 4 — GPU compaction with nvidia-smi GPU metrics
#             GPU utilization, power, temperature, memory via nvidia-smi
#             sampling. Uses gpu_compaction_sweep_metrics.sh.
#
# Usage:
#   Called by ../run_sweep.sh which loads config and passes --sst-size-mb / --input-ssts.
#
# Skip individual phases:
#   SKIP_CPU=1              skip phase 1
#   SKIP_GPU_BREAKDOWN=1    skip phase 2
#   SKIP_GPU_HOST_METRICS=1 skip phase 3
#   SKIP_GPU_DEVICE_METRICS=1 skip phase 4
#   SKIP_COMPARE=1          skip final comparison

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CPU_ENV_FILE="${REPO_ROOT}/benchmarks/common/config/.env.local"
GPU_DEFAULTS_FILE="${REPO_ROOT}/benchmarks/gpu_compaction/config/defaults.sh"

if [[ ! -f "${CPU_ENV_FILE}" ]]; then
  echo "error: missing ${CPU_ENV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${GPU_DEFAULTS_FILE}" ]]; then
  echo "error: missing ${GPU_DEFAULTS_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CPU_ENV_FILE}"
set +a

# shellcheck disable=SC1090
source "${GPU_DEFAULTS_FILE}"

# ── CLI arguments ──────────────────────────────────────────────────
SST_SIZE_MB=""
INPUT_SSTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sst-size-mb) SST_SIZE_MB="$2"; shift 2 ;;
    --input-ssts)  INPUT_SSTS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${SST_SIZE_MB}" || -z "${INPUT_SSTS}" ]]; then
  echo "usage: $0 --sst-size-mb <mb> --input-ssts <count>" >&2
  exit 1
fi

# ── Parameters (set by config, fallback to shared defaults) ────────
VALUE_SIZES="${VALUE_SIZES:-${COMPACTION_DEFAULT_VALUE_SIZES}}"
SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST:-${COMPACTION_DEFAULT_SUBCOMP_THREADS_LIST}}"
GPU_MODES="${GPU_MODES:-${COMPACTION_DEFAULT_GPU_MODES}}"
GPU_RUNS="${GPU_RUNS:-${COMPACTION_DEFAULT_RUNS}}"
CPU_RUNS="${CPU_RUNS:-${COMPACTION_DEFAULT_RUNS}}"
USER_KEY_SPACE="${USER_KEY_SPACE:-${COMPACTION_DEFAULT_USER_KEY_SPACE}}"
ZIPF_ALPHA="${ZIPF_ALPHA:-${COMPACTION_DEFAULT_ZIPF_ALPHA}}"
RUN_PAUSE_SECONDS="${RUN_PAUSE_SECONDS:-2}"

# ── Output layout ─────────────────────────────────────────────────
CASE_LABEL="${SST_SIZE_MB}mb-sst_${INPUT_SSTS}sst"
RUN_ID="${RUN_ID:-${CASE_LABEL}}"
EXPERIMENT_DIR="${OUTPUT_DIR}/initial_gpu_cpu_compaction_microbenchmark/${RUN_ID}"
CPU_RUN_ID="cpu-${CASE_LABEL}"
GPU_BREAKDOWN_LABEL="gpu_breakdown"
GPU_HOST_METRICS_LABEL="gpu_host_metrics"
GPU_DEVICE_METRICS_LABEL="gpu_device_metrics"
PRELOAD_DIR_NAME="comparison_preload_${CASE_LABEL}"

CPU_EXPERIMENT_ROOT="${EXPERIMENT_DIR}/cpu"
# GPU sweep creates ${OUT_ROOT}/${LABEL}, so these resolve to EXPERIMENT_DIR/gpu_*
GPU_BREAKDOWN_SWEEP="${EXPERIMENT_DIR}/${GPU_BREAKDOWN_LABEL}"
GPU_HOST_METRICS_SWEEP="${EXPERIMENT_DIR}/${GPU_HOST_METRICS_LABEL}_gpu-profile-only"
GPU_DEVICE_METRICS_SWEEP="${EXPERIMENT_DIR}/${GPU_DEVICE_METRICS_LABEL}_gpu-only"
COMPARE_DIR="${EXPERIMENT_DIR}/comparison"

mkdir -p "${EXPERIMENT_DIR}"

assert_release_db_bench() {
  if strings "${DB_BENCH}" | grep -q "Assertions are enabled"; then
    echo "error: DB_BENCH is an assertions-enabled build: ${DB_BENCH}" >&2
    echo "hint: rebuild RocksDB db_bench in release mode, e.g. make db_bench DEBUG_LEVEL=0 -j" >&2
    exit 1
  fi
}

echo "========================================================="
echo " CPU/GPU Compaction Comparison"
echo " case label:       ${CASE_LABEL}"
echo " experiment dir:   ${EXPERIMENT_DIR}"
echo " SST size:         ${SST_SIZE_MB} MB"
echo " input SSTs:       ${INPUT_SSTS}"
echo " value sizes:      ${VALUE_SIZES}"
echo " subcompactions:   ${SUBCOMP_THREADS_LIST}"
echo " CPU runs/config:  ${CPU_RUNS}"
echo " GPU modes:        ${GPU_MODES}"
echo " GPU runs/mode:    ${GPU_RUNS}"
echo " key space:        ${USER_KEY_SPACE}"
echo "========================================================="

# ── Phase 1: CPU RocksDB compaction sweep ─────────────────────────
if [[ "${SKIP_CPU:-0}" != "1" ]]; then
  echo ""
  echo "═══ Phase 1: CPU RocksDB compaction sweep ═══"
  assert_release_db_bench
  (
    cd "${REPO_ROOT}"
    RUN_ID="${CPU_RUN_ID}" \
    KEY_SIZE=16 \
    VALUE_SIZES="${VALUE_SIZES}" \
    INPUT_SST_COUNT_LIST="${INPUT_SSTS}" \
    SST_SIZE_MB_LIST="${SST_SIZE_MB}" \
    SUBCOMP_THREADS_LIST="${SUBCOMP_THREADS_LIST}" \
    COMPACTION_RUNS="${CPU_RUNS}" \
    THREADS=1 \
    PRELOAD_THREADS=1 \
    PRELOAD_DIR_NAME="${PRELOAD_DIR_NAME}" \
    PRELOAD_KEYSPACE_MULTIPLIER=1 \
    PRELOAD_MIN_NUM_KEYS="${USER_KEY_SPACE}" \
    LOAD_PERF_LEVEL=1 \
    COMPACTION_BENCH=compactall \
    COMPACTION_BG_THREADS=1 \
    COMPACTION_PERF_LEVEL=5 \
    OPEN_FILES=512 \
    DIRECT_IO=true \
    METRICS_INTERVAL_MS=100 \
    HOST_METRICS_INTERVAL_SEC=0.1 \
    REPORT_BG_IO_STATS=1 \
    RUN_QUIET=1 \
    RUN_PAUSE_SECONDS="${RUN_PAUSE_SECONDS}" \
    EXPERIMENT_ROOT="${CPU_EXPERIMENT_ROOT}" \
    ./benchmarks/cpu_compaction/scripts/cpu_compaction_sweep.sh
  )

  echo ""
  echo "--- CPU analysis ---"
  (
    cd "${REPO_ROOT}"
    python3 ./benchmarks/cpu_compaction/plotting/analyze_compaction_parallelism.py \
      "${CPU_EXPERIMENT_ROOT}"
  )
else
  echo ""
  echo "═══ Phase 1: SKIPPED (SKIP_CPU=1) ═══"
fi

# ── Phase 2: GPU compaction with stage breakdown ──────────────────
if [[ "${SKIP_GPU_BREAKDOWN:-0}" != "1" ]]; then
  echo ""
  echo "═══ Phase 2: GPU compaction with stage breakdown ═══"
  (
    bash "${REPO_ROOT}/benchmarks/gpu_compaction/scripts/gpu_compaction_sweep.sh" \
      --sst-size-mb "${SST_SIZE_MB}" \
      --num_ssts "${INPUT_SSTS}" \
      --label "${GPU_BREAKDOWN_LABEL}" \
      --values "${VALUE_SIZES}" \
      --runs "${GPU_RUNS}" \
      --modes "${GPU_MODES}" \
      --zipf_alpha "${ZIPF_ALPHA}" \
      --user_key_space "${USER_KEY_SPACE}" \
      --no_plot \
      --out_root "${EXPERIMENT_DIR}"
  )
else
  echo ""
  echo "═══ Phase 2: SKIPPED (SKIP_GPU_BREAKDOWN=1) ═══"
fi

# ── Phase 3: GPU compaction with host metrics ─────────────────────
if [[ "${SKIP_GPU_HOST_METRICS:-0}" != "1" ]]; then
  echo ""
  echo "═══ Phase 3: GPU compaction with host CPU/IO metrics ═══"
  (
    bash "${REPO_ROOT}/benchmarks/gpu_compaction/scripts/gpu_compaction_sweep.sh" \
      --sst-size-mb "${SST_SIZE_MB}" \
      --num_ssts "${INPUT_SSTS}" \
      --label "${GPU_HOST_METRICS_LABEL}" \
      --values "${VALUE_SIZES}" \
      --runs "${GPU_RUNS}" \
      --modes "${GPU_MODES}" \
      --profile_only \
      --host_metrics_interval_sec 0.1 \
      --zipf_alpha "${ZIPF_ALPHA}" \
      --user_key_space "${USER_KEY_SPACE}" \
      --no_plot \
      --out_root "${EXPERIMENT_DIR}"
  )
else
  echo ""
  echo "═══ Phase 3: SKIPPED (SKIP_GPU_HOST_METRICS=1) ═══"
fi

# ── Phase 4: GPU compaction with nvidia-smi device metrics ────────
if [[ "${SKIP_GPU_DEVICE_METRICS:-0}" != "1" ]]; then
  echo ""
  echo "═══ Phase 4: GPU compaction with nvidia-smi device metrics ═══"
  (
    bash "${REPO_ROOT}/benchmarks/gpu_compaction/scripts/gpu_compaction_sweep_metrics.sh" \
      --sst-size-mb "${SST_SIZE_MB}" \
      --num_ssts "${INPUT_SSTS}" \
      --label "${GPU_DEVICE_METRICS_LABEL}" \
      --values "${VALUE_SIZES}" \
      --runs "${GPU_RUNS}" \
      --modes "${GPU_MODES}" \
      --gpu_only \
      --zipf_alpha "${ZIPF_ALPHA}" \
      --user_key_space "${USER_KEY_SPACE}" \
      --out_root "${EXPERIMENT_DIR}"
  )
else
  echo ""
  echo "═══ Phase 4: SKIPPED (SKIP_GPU_DEVICE_METRICS=1) ═══"
fi

# ── Comparison: CPU vs GPU speedup analysis ───────────────────────
if [[ "${SKIP_COMPARE:-0}" != "1" ]]; then
  if [[ -d "${CPU_EXPERIMENT_ROOT}" && -d "${GPU_BREAKDOWN_SWEEP}" ]]; then
    echo ""
    echo "═══ Comparison: CPU vs GPU speedup analysis ═══"

    # Copy GPU logs into CPU per_run_logs for the comparison script
    CPU_PER_RUN_LOG_DIR="${CPU_EXPERIMENT_ROOT}/per_run_logs"
    mkdir -p "${CPU_PER_RUN_LOG_DIR}"
    shopt -s nullglob
    for src_path in "${GPU_BREAKDOWN_SWEEP}"/result_val*B_*.log; do
      base_name="$(basename "${src_path}")"
      if [[ "${base_name}" =~ ^result_val([0-9]+)B_(.+)\.log$ ]]; then
        cp "${src_path}" "${CPU_PER_RUN_LOG_DIR}/gpcomp_bench_v${BASH_REMATCH[1]}_${BASH_REMATCH[2]}.log"
      fi
    done
    shopt -u nullglob

    (
      cd "${REPO_ROOT}"
      python3 ./benchmarks/cpu_compaction/plotting/compare_cpu_gpu_compaction_speedups.py \
        --cpu-experiment-root "${CPU_EXPERIMENT_ROOT}" \
        --gpu-sweep-dir "${GPU_BREAKDOWN_SWEEP}" \
        --output-dir "${COMPARE_DIR}" \
        --sst-size-mb "${SST_SIZE_MB}" \
        --input-sst-count "${INPUT_SSTS}" \
        --subcomp-threads "${SUBCOMP_THREADS_LIST}" \
        --gpu-modes "${GPU_MODES}"
    )
  else
    echo ""
    echo "═══ Comparison: SKIPPED (missing CPU or GPU breakdown results) ═══"
  fi
else
  echo ""
  echo "═══ Comparison: SKIPPED (SKIP_COMPARE=1) ═══"
fi

# ── Analysis: summary tables and throughput/utilization figure ─────
if [[ "${SKIP_COMPARE:-0}" != "1" ]]; then
  echo ""
  echo "═══ Analysis: generating summary tables and figures ═══"
  (
    cd "${REPO_ROOT}"
    python3 ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/plotting/analyze_comparison.py \
      "${EXPERIMENT_DIR}"
  )
fi

echo ""
echo "========================================================="
echo " Comparison complete."
echo " Experiment dir:     ${EXPERIMENT_DIR}"
echo " CPU results:        ${CPU_EXPERIMENT_ROOT}"
echo " GPU breakdown:      ${GPU_BREAKDOWN_SWEEP}"
echo " GPU host metrics:   ${GPU_HOST_METRICS_SWEEP}"
echo " GPU device metrics: ${GPU_DEVICE_METRICS_SWEEP}"
echo " Comparison:         ${COMPARE_DIR}"
echo "========================================================="
