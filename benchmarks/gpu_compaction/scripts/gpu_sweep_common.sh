#!/usr/bin/env bash
# Shared infrastructure for GPU compaction sweep scripts.
# Source this file; do not execute directly.

# ── Path setup ────────────────────────────────────────────────────
# Caller must have set SCRIPT_DIR before sourcing.
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../.." && pwd)"
GPU_COMPACTION_DIR="${REPO_ROOT}/gpu_compaction"
DEFAULTS_FILE="${BENCH_DIR}/config/defaults.sh"
COMMON_BENCH_FILE="${REPO_ROOT}/benchmarks/common/benchmark_common.sh"

if [[ ! -f "${DEFAULTS_FILE}" ]]; then
    echo "error: missing ${DEFAULTS_FILE}" >&2
    exit 1
fi

# Source shared machine-local env (benchmarks/common/config/.env.local) if available.
if [[ -f "${COMMON_BENCH_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${COMMON_BENCH_FILE}"
fi

# shellcheck disable=SC1090
source "${DEFAULTS_FILE}"

cd "${GPU_COMPACTION_DIR}"

# ── Common defaults ───────────────────────────────────────────────
RUNS="${RUNS:-${COMPACTION_DEFAULT_RUNS:-5}}"
VALUES_STR="${VALUES:-${COMPACTION_DEFAULT_VALUE_SIZES:-32 64 128 256 512 1024}}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_V}"
MODES_STR="${MODES:-${COMPACTION_DEFAULT_GPU_MODES:-q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan}}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-${COMPACTION_DEFAULT_ZIPF_ALPHA:-0.0}}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-${COMPACTION_DEFAULT_USER_KEY_SPACE:-200000000}}"
GPU_COMPACTION_OUT_ROOT="${GPU_COMPACTION_OUT_ROOT:-${REPO_ROOT}/local_benchmark_artifacts/gpu_compaction/sweep_results}"
GPU_COMPACTION_DATASET_ROOT="${GPU_COMPACTION_DATASET_ROOT:-${REPO_ROOT}/local_benchmark_artifacts/gpu_compaction/datasets}"
DATASET_ROOT="${DATASET_ROOT:-${GPU_COMPACTION_DATASET_ROOT}}"
OUT_ROOT="${OUT_ROOT:-${GPU_COMPACTION_OUT_ROOT}}"
GPU_ONLY="${GPU_ONLY:-0}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"
NUM_SSTS="${NUM_SSTS:-}"
LABEL="${LABEL:-}"
BENCH_BIN="${GPU_COMPACTION_DIR}/gpcomp_bench"
DATAGEN_BIN="${GPU_COMPACTION_DIR}/gpcomp_datagen"

DEFAULT_LOCAL_NSYS_BIN="$HOME/.local/nsight-systems-2026.2.1/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"

mkdir -p "${DATASET_ROOT}" "${OUT_ROOT}"

# ── Common CLI argument parser ────────────────────────────────────
# Parses arguments shared across all GPU sweep scripts.
# Returns remaining (unrecognised) arguments in EXTRA_ARGS array.
EXTRA_ARGS=()
gpu_sweep_parse_common_args() {
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --runs) RUNS="$2"; shift 2 ;;
            --values) VALUES_STR="$2"; shift 2 ;;
            --dataset_prefix) DATASET_PREFIX="$2"; shift 2 ;;
            --modes) MODES_STR="$2"; shift 2 ;;
            --gpu_only) GPU_ONLY="1"; shift ;;
            --with_cpu_baseline) GPU_ONLY="0"; shift ;;
            --profile_only) PROFILE_ONLY="1"; GPU_ONLY="1"; shift ;;
            --no_profile_only) PROFILE_ONLY="0"; shift ;;
            --zipf_alpha) DATASET_ZIPF_ALPHA="$2"; shift 2 ;;
            --user_key_space) DATASET_USER_KEY_SPACE="$2"; shift 2 ;;
            --out_root) OUT_ROOT="$2"; shift 2 ;;
            --num_ssts) NUM_SSTS="$2"; shift 2 ;;
            --label) LABEL="$2"; shift 2 ;;
            *) EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
}

# ── Compile-time constant patching ────────────────────────────────
ORIG_NUM_SSTS=""
ORIG_VALUE_BYTES=""

gpu_sweep_save_originals() {
    ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)
    ORIG_VALUE_BYTES=$(grep -oP 'GP_VALUE_BYTES\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)

    if [[ -n "${NUM_SSTS}" ]]; then
        sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${NUM_SSTS};/g" gpcomp_common.cuh
    fi
}

gpu_sweep_restore_originals() {
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${ORIG_VALUE_BYTES};/g" gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${ORIG_NUM_SSTS};/g" gpcomp_common.cuh
}

gpu_sweep_current_ssts() {
    grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh
}

# ── Build + datagen per value size ────────────────────────────────
# Patches value size, rebuilds, and generates dataset.
# Sets DATASET_DIR for the caller.
DATASET_DIR=""

gpu_sweep_build_and_datagen() {
    local val="$1"
    local outdir="$2"

    echo ""
    echo "--- Building for Value Size: ${val} B ---"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${val};/g" gpcomp_common.cuh
    make clean > /dev/null
    make gpcomp_datagen gpcomp_bench -j > /dev/null

    DATASET_DIR="${DATASET_ROOT%/}/${DATASET_PREFIX}${val}"
    echo "Generating dataset in ${DATASET_DIR}..."
    rm -rf "${DATASET_DIR}"
    "${DATAGEN_BIN}" \
        --out_dir "${DATASET_DIR}" \
        --seed 42 \
        --zipf_alpha "${DATASET_ZIPF_ALPHA}" \
        --user_key_space "${DATASET_USER_KEY_SPACE}" \
        > "${outdir}/dataset_val${val}B_datagen.log"
}

# ── Build bench_args array ────────────────────────────────────────
# Sets BENCH_ARGS for the caller.
BENCH_ARGS=()

gpu_sweep_bench_args() {
    local dataset="$1"
    local out_dir="$2"
    local runs="$3"
    local mode="$4"

    BENCH_ARGS=(
        --dataset "${dataset}"
        --out_dir "${out_dir}"
        --runs "${runs}"
        --gpu_mode "${mode}"
    )
    if [[ "${GPU_ONLY}" == "1" ]]; then
        BENCH_ARGS+=(--gpu_only)
    fi
    if [[ "${PROFILE_ONLY}" == "1" ]]; then
        BENCH_ARGS+=(--profile_only)
    fi
}

# ── Nsight stats extraction (shared by metrics + nsight scripts) ──
run_nsys_stats_report() {
    local report_path="$1"
    local canonical_path="$2"
    shift 2
    local nsys_bin="${NSYS_BIN:-nsys}"
    local report=""
    local candidate=""
    local output_prefix="${canonical_path%.csv}.tmp"

    for report in "$@"; do
        rm -f "${output_prefix}"*.csv
        "${nsys_bin}" stats --force-export=true --report "${report}" --format csv --output "${output_prefix}" "${report_path}" >/dev/null 2>&1 || true
        candidate="${output_prefix}_${report}.csv"
        if [[ -f "${candidate}" ]]; then
            mv -f "${candidate}" "${canonical_path}"
            printf '%s\n' "${canonical_path}"
            return 0
        fi
    done

    return 1
}

# ── Nsys binary resolution ────────────────────────────────────────
resolve_nsys_bin() {
    NSYS_BIN="${NSYS_BIN:-}"
    if [[ -z "${NSYS_BIN}" && -x "${DEFAULT_LOCAL_NSYS_BIN}" ]]; then
        NSYS_BIN="${DEFAULT_LOCAL_NSYS_BIN}"
    fi
    if [[ -z "${NSYS_BIN}" ]]; then
        NSYS_BIN="nsys"
    fi
}
