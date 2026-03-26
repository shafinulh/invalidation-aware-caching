#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULTS_FILE="${SCRIPT_DIR}/final_speedup_gpu_defaults.sh"
DEFAULT_LOCAL_NSYS_BIN="$HOME/.local/nsight-systems-2026.2.1/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"
cd "${SCRIPT_DIR}"

if [[ ! -f "${DEFAULTS_FILE}" ]]; then
    echo "error: missing ${DEFAULTS_FILE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${DEFAULTS_FILE}"

RUNS="${RUNS:-${FINAL_GPU_RUNS:-${FINAL_SPEEDUP_GPU_DEFAULT_RUNS}}}"
VALUES_STR="${VALUES:-${FINAL_VALUE_SIZES:-${FINAL_SPEEDUP_GPU_DEFAULT_VALUE_SIZES}}}"
DATASET_PREFIX="${DATASET_PREFIX:-dataset_shafin_V}"
MODES_STR="${MODES:-${FINAL_GPU_MODES:-${FINAL_SPEEDUP_GPU_DEFAULT_MODES}}}"
GPU_ONLY="${GPU_ONLY:-1}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"
DATASET_ZIPF_ALPHA="${DATASET_ZIPF_ALPHA:-${FINAL_GPU_ZIPF_ALPHA:-${FINAL_SPEEDUP_GPU_DEFAULT_ZIPF_ALPHA}}}"
DATASET_USER_KEY_SPACE="${DATASET_USER_KEY_SPACE:-${FINAL_GPU_USER_KEY_SPACE:-${FINAL_SPEEDUP_GPU_DEFAULT_USER_KEY_SPACE}}}"
OUT_ROOT="${OUT_ROOT:-./sweep_results}"
TIMESTAMP_OUTPUT="${TIMESTAMP_OUTPUT:-0}"
GPU_ID="${GPU_ID:-0}"
QUERY_INTERVAL_MS="${QUERY_INTERVAL_MS:-100}"
DMON_INTERVAL_SEC="${DMON_INTERVAL_SEC:-1}"
COLLECT_NSYS="${COLLECT_NSYS:-0}"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
NSYS_PROFILE_RUNS="${NSYS_PROFILE_RUNS:-1}"
NSYS_GPU_METRICS="${NSYS_GPU_METRICS:-0}"
NSYS_GPU_METRICS_DEVICE="${NSYS_GPU_METRICS_DEVICE:-}"
NSYS_GPU_METRICS_SET="${NSYS_GPU_METRICS_SET:-}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-}"
COLLECT_NCU="${COLLECT_NCU:-0}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="${NCU_SET:-default}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"
NCU_PROFILE_RUNS="${NCU_PROFILE_RUNS:-1}"
FORCE_FIRST_RUN_WARMUP="${FORCE_FIRST_RUN_WARMUP:-0}"
PRESERVE_OUTDIR="${PRESERVE_OUTDIR:-0}"
NUM_SSTS=""
LABEL=""
DATAGEN_BIN="${SCRIPT_DIR}/gpcomp_datagen"
BENCH_BIN="${SCRIPT_DIR}/gpcomp_bench"
SUMMARY_BIN="${SCRIPT_DIR}/summarize_gpu_metrics.py"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
GPU_QUERY_SAMPLER_PID=""
GPU_DMON_SAMPLER_PID=""

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
        --timestamp_output) TIMESTAMP_OUTPUT="1"; shift ;;
        --no_timestamp_output) TIMESTAMP_OUTPUT="0"; shift ;;
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --query_interval_ms) QUERY_INTERVAL_MS="$2"; shift 2 ;;
        --dmon_interval_sec|--gpm_interval_sec) DMON_INTERVAL_SEC="$2"; shift 2 ;;
        --collect_nsys) COLLECT_NSYS="1"; shift ;;
        --no_collect_nsys) COLLECT_NSYS="0"; shift ;;
        --nsys_bin) NSYS_BIN="$2"; shift 2 ;;
        --nsys_cpuctxsw|--cpuctxsw) NSYS_CPUCTXSW="$2"; shift 2 ;;
        --nsys_profile_runs) NSYS_PROFILE_RUNS="$2"; shift 2 ;;
        --nsys_gpu_metrics|--nsys-gpu-metrics) NSYS_GPU_METRICS="1"; shift ;;
        --no_nsys_gpu_metrics|--no-nsys-gpu-metrics) NSYS_GPU_METRICS="0"; shift ;;
        --nsys_gpu_metrics_device|--nsys-gpu-metrics-device) NSYS_GPU_METRICS_DEVICE="$2"; shift 2 ;;
        --nsys_gpu_metrics_set|--nsys-gpu-metrics-set) NSYS_GPU_METRICS_SET="$2"; shift 2 ;;
        --nsys_gpu_metrics_frequency|--nsys-gpu-metrics-frequency) NSYS_GPU_METRICS_FREQUENCY="$2"; shift 2 ;;
        --collect_ncu) COLLECT_NCU="1"; shift ;;
        --no_collect_ncu) COLLECT_NCU="0"; shift ;;
        --ncu_set) NCU_SET="$2"; shift 2 ;;
        --ncu_target_processes) NCU_TARGET_PROCESSES="$2"; shift 2 ;;
        --ncu_replay_mode) NCU_REPLAY_MODE="$2"; shift 2 ;;
        --ncu_profile_runs) NCU_PROFILE_RUNS="$2"; shift 2 ;;
        --warmup_first_run) FORCE_FIRST_RUN_WARMUP="1"; shift ;;
        --no_warmup_first_run) FORCE_FIRST_RUN_WARMUP="0"; shift ;;
        --preserve_outdir) PRESERVE_OUTDIR="1"; shift ;;
        --replace_outdir) PRESERVE_OUTDIR="0"; shift ;;
        --num_ssts) NUM_SSTS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
    echo "error: ${NVIDIA_SMI_BIN} is not in PATH" >&2
    exit 1
fi
if [[ ! -f "${SUMMARY_BIN}" ]]; then
    echo "error: missing ${SUMMARY_BIN}" >&2
    exit 1
fi
if [[ -z "${NSYS_BIN}" && -x "${DEFAULT_LOCAL_NSYS_BIN}" ]]; then
    NSYS_BIN="${DEFAULT_LOCAL_NSYS_BIN}"
fi
if [[ -z "${NSYS_BIN}" ]]; then
    NSYS_BIN="nsys"
fi
if [[ "${COLLECT_NSYS}" == "1" ]] && ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
    echo "error: ${NSYS_BIN} is not in PATH" >&2
    exit 1
fi
if [[ "${COLLECT_NCU}" == "1" ]] && ! command -v "${NCU_BIN}" >/dev/null 2>&1; then
    echo "error: ${NCU_BIN} is not in PATH" >&2
    exit 1
fi

start_gpu_metrics_collection() {
    local query_csv="$1"
    local dmon_log="$2"

    mkdir -p "$(dirname "${query_csv}")"
    : > "${query_csv}"
    : > "${dmon_log}"

    "${NVIDIA_SMI_BIN}" \
        --id "${GPU_ID}" \
        --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate \
        --format=csv,noheader,nounits \
        --loop-ms "${QUERY_INTERVAL_MS}" \
        > "${query_csv}" 2>/dev/null &
    GPU_QUERY_SAMPLER_PID="$!"

    "${NVIDIA_SMI_BIN}" dmon \
        -i "${GPU_ID}" \
        -d "${DMON_INTERVAL_SEC}" \
        -s um \
        -o DT \
        -f "${dmon_log}" \
        >/dev/null 2>&1 &
    GPU_DMON_SAMPLER_PID="$!"
}

stop_one_sampler() {
    local sampler_pid="$1"
    if [[ -z "${sampler_pid}" ]]; then
        return 0
    fi
    if kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
}

stop_gpu_metrics_collection() {
    stop_one_sampler "${GPU_QUERY_SAMPLER_PID:-}"
    stop_one_sampler "${GPU_DMON_SAMPLER_PID:-}"
    GPU_QUERY_SAMPLER_PID=""
    GPU_DMON_SAMPLER_PID=""
}

run_bench_with_gpu_metrics() {
    local log_path="$1"
    local query_csv="$2"
    local dmon_log="$3"
    shift 3

    start_gpu_metrics_collection "${query_csv}" "${dmon_log}"
    set +e
    "${BENCH_BIN}" "$@" > "${log_path}" 2>&1
    local bench_rc="$?"
    set -e
    stop_gpu_metrics_collection
    return "${bench_rc}"
}

run_ncu_profile() {
    local report_prefix="$1"
    local log_path="$2"
    shift 2

    set +e
    "${NCU_BIN}" \
        --set "${NCU_SET}" \
        --target-processes "${NCU_TARGET_PROCESSES}" \
        --replay-mode "${NCU_REPLAY_MODE}" \
        -f \
        --export "${report_prefix}" \
        "${BENCH_BIN}" "$@" > "${log_path}" 2>&1
    local ncu_rc="$?"
    set -e
    return "${ncu_rc}"
}

run_nsys_profile() {
    local report_prefix="$1"
    local log_path="$2"
    shift 2
    local -a nsys_gpu_metrics_args=()

    if [[ "${NSYS_GPU_METRICS}" == "1" ]]; then
        local metrics_device="${NSYS_GPU_METRICS_DEVICE:-${GPU_ID}}"
        nsys_gpu_metrics_args+=(--gpu-metrics-device "${metrics_device}")
        if [[ -n "${NSYS_GPU_METRICS_SET}" ]]; then
            nsys_gpu_metrics_args+=(--gpu-metrics-set "${NSYS_GPU_METRICS_SET}")
        fi
        if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" ]]; then
            nsys_gpu_metrics_args+=(--gpu-metrics-frequency "${NSYS_GPU_METRICS_FREQUENCY}")
        fi
    fi

    set +e
    "${NSYS_BIN}" profile \
        --force-overwrite true \
        -o "${report_prefix}" \
        --trace=cuda,osrt,nvtx \
        --sample=none \
        --cpuctxsw="${NSYS_CPUCTXSW}" \
        "${nsys_gpu_metrics_args[@]}" \
        "${BENCH_BIN}" "$@" > "${log_path}" 2>&1
    local nsys_rc="$?"
    set -e
    return "${nsys_rc}"
}

run_nsys_stats_report() {
    local report_path="$1"
    local canonical_path="$2"
    shift 2
    local report=""
    local candidate=""
    local output_prefix="${canonical_path%.csv}.tmp"

    for report in "$@"; do
        rm -f "${output_prefix}"*.csv
        "${NSYS_BIN}" stats --force-export=true --report "${report}" --format csv --output "${output_prefix}" "${report_path}" >/dev/null 2>&1 || true
        candidate="${output_prefix}_${report}.csv"
        if [[ -f "${candidate}" ]]; then
            mv -f "${candidate}" "${canonical_path}"
            printf '%s\n' "${canonical_path}"
            return 0
        fi
    done

    return 1
}

ORIG_NUM_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)
ORIG_VALUE_BYTES=$(grep -oP 'GP_VALUE_BYTES\s*=\s*\K[0-9]+' gpcomp_common.cuh | head -n 1)

if [[ -n "${NUM_SSTS}" ]]; then
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${NUM_SSTS};/g" gpcomp_common.cuh
fi

CURRENT_SSTS=$(grep -oP 'GP_NUM_INPUT_SSTS\s*=\s*\K[0-9]+' gpcomp_common.cuh)
PROFILE_SUFFIX=""
if [[ "${PROFILE_ONLY}" == "1" ]]; then
    PROFILE_SUFFIX="_gpu-profile-only"
elif [[ "${GPU_ONLY}" == "1" ]]; then
    PROFILE_SUFFIX="_gpu-only"
fi
if [[ -z "${LABEL}" ]]; then
    if [[ "${COLLECT_NSYS}" == "1" || "${COLLECT_NCU}" == "1" ]]; then
        LABEL="nsight_ncu_8mb-sst_${CURRENT_SSTS}sst${PROFILE_SUFFIX}"
    else
        LABEL="gpu_metrics_8mb-sst_${CURRENT_SSTS}sst${PROFILE_SUFFIX}"
    fi
elif [[ -n "${PROFILE_SUFFIX}" && "${LABEL}" != *"${PROFILE_SUFFIX}" ]]; then
    LABEL="${LABEL}${PROFILE_SUFFIX}"
fi

OUTDIR="${OUT_ROOT}/sweep_${LABEL}"
if [[ "${TIMESTAMP_OUTPUT}" == "1" ]]; then
    STAMP="$(date '+%Y%m%d_%H%M%S')"
    OUTDIR="${OUTDIR}_${STAMP}"
fi
TEMP_ROOT="${OUTDIR}/temp"
GPU_METRICS_ROOT="${OUTDIR}/gpu_metrics"
PROFILES_DIR="${OUTDIR}/profiles"
if [[ "${PRESERVE_OUTDIR}" != "1" ]]; then
    rm -rf "${OUTDIR}"
fi
mkdir -p "${OUTDIR}" "${TEMP_ROOT}" "${GPU_METRICS_ROOT}" "${PROFILES_DIR}"

cleanup() {
    stop_gpu_metrics_collection
    rm -rf "${TEMP_ROOT}"
    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${ORIG_VALUE_BYTES};/g" gpcomp_common.cuh
    sed -i -E "s/static constexpr int GP_NUM_INPUT_SSTS     = [0-9]+;/static constexpr int GP_NUM_INPUT_SSTS     = ${ORIG_NUM_SSTS};/g" gpcomp_common.cuh
}
trap cleanup EXIT

MANIFEST="${OUTDIR}/gpu_metrics_manifest.csv"
echo "value_bytes,mode,dataset,bench_log,query_csv,dmon_log,iteration_csv,summary_json,run_status,exit_code,nsys_profile_runs,nsys_log,nsys_report_path,nsys_kernel_csv,nsys_api_csv,nsys_run_status,nsys_exit_code,ncu_set,ncu_profile_runs,ncu_log,ncu_report_path,ncu_run_status,ncu_exit_code" > "${MANIFEST}"

echo "========================================================="
echo " GPU Metrics Sweep"
echo " output: ${OUTDIR}"
echo " manifest: ${MANIFEST}"
echo " runs per mode: ${RUNS}"
echo " value sizes: ${VALUES_STR}"
echo " gpu modes: ${MODES_STR}"
echo " gpu-only benchmark mode: ${GPU_ONLY}"
echo " profile-only benchmark mode: ${PROFILE_ONLY}"
echo " gpu id: ${GPU_ID}"
echo " query interval ms: ${QUERY_INTERVAL_MS}"
echo " dmon interval sec: ${DMON_INTERVAL_SEC}"
echo " preserve existing output dir: ${PRESERVE_OUTDIR}"
echo " Nsight Systems capture: ${COLLECT_NSYS}"
if [[ "${COLLECT_NSYS}" == "1" ]]; then
    echo " Nsight Systems binary: ${NSYS_BIN}"
    echo " Nsight Systems cpu context switches: ${NSYS_CPUCTXSW}"
    echo " Nsight Systems benchmark runs: ${NSYS_PROFILE_RUNS}"
    echo " Nsight Systems GPU metrics: ${NSYS_GPU_METRICS}"
    if [[ "${NSYS_GPU_METRICS}" == "1" ]]; then
        echo " Nsight Systems GPU metrics device: ${NSYS_GPU_METRICS_DEVICE:-${GPU_ID}}"
        if [[ -n "${NSYS_GPU_METRICS_SET}" ]]; then
            echo " Nsight Systems GPU metrics set: ${NSYS_GPU_METRICS_SET}"
        else
            echo " Nsight Systems GPU metrics set: default-for-device"
        fi
        if [[ -n "${NSYS_GPU_METRICS_FREQUENCY}" ]]; then
            echo " Nsight Systems GPU metrics frequency: ${NSYS_GPU_METRICS_FREQUENCY}"
        else
            echo " Nsight Systems GPU metrics frequency: nsys-default"
        fi
    fi
fi
echo " Nsight Compute capture: ${COLLECT_NCU}"
if [[ "${COLLECT_NCU}" == "1" ]]; then
    echo " Nsight Compute binary: ${NCU_BIN}"
    echo " Nsight Compute set: ${NCU_SET}"
    echo " Nsight Compute target processes: ${NCU_TARGET_PROCESSES}"
    echo " Nsight Compute replay mode: ${NCU_REPLAY_MODE}"
    echo " Nsight Compute benchmark runs: ${NCU_PROFILE_RUNS}"
    echo " Nsight Compute reports: ${PROFILES_DIR}"
fi
echo " force first run warmup in summary: ${FORCE_FIRST_RUN_WARMUP}"
echo " input SSTs: ${CURRENT_SSTS}"
echo " dataset user key space: ${DATASET_USER_KEY_SPACE}"
echo "========================================================="

for VAL in ${VALUES_STR}; do
    echo ""
    echo "--- Building for Value Size: ${VAL} B ---"

    sed -i -E "s/static constexpr int GP_VALUE_BYTES        = [0-9]+;/static constexpr int GP_VALUE_BYTES        = ${VAL};/g" gpcomp_common.cuh
    make clean > /dev/null
    make gpcomp_datagen gpcomp_bench -j > /dev/null

    DATASET_DIR="${DATASET_PREFIX}${VAL}"
    echo "Generating dataset in ${DATASET_DIR}..."
    rm -rf "${DATASET_DIR}"
    "${DATAGEN_BIN}" \
        --out_dir "${DATASET_DIR}" \
        --seed 42 \
        --zipf_alpha "${DATASET_ZIPF_ALPHA}" \
        --user_key_space "${DATASET_USER_KEY_SPACE}" \
        > "${OUTDIR}/dataset_val${VAL}B_datagen.log"

    for MODE in ${MODES_STR}; do
        BENCH_OUTDIR="${TEMP_ROOT}/${MODE}"
        METRICS_DIR="${GPU_METRICS_ROOT}/val${VAL}B_${MODE}"
        LOG_PATH="${OUTDIR}/result_val${VAL}B_${MODE}.log"
        QUERY_CSV="${METRICS_DIR}/query_gpu.csv"
        DMON_LOG="${METRICS_DIR}/dmon.log"
        ITERATION_CSV="${METRICS_DIR}/iteration_summary.csv"
        SUMMARY_JSON="${METRICS_DIR}/summary.json"
        RUN_STATUS="ok"
        EXIT_CODE="0"
        NSYS_LOG_PATH=""
        NSYS_REPORT_PATH=""
        NSYS_KERNEL_CSV=""
        NSYS_API_CSV=""
        NSYS_RUN_STATUS="skipped"
        NSYS_EXIT_CODE=""
        NCU_LOG_PATH=""
        NCU_REPORT_PATH=""
        NCU_RUN_STATUS="skipped"
        NCU_EXIT_CODE=""

        echo "Collecting GPU metrics for ${MODE} at ${VAL} B..."
        rm -rf "${BENCH_OUTDIR}" "${METRICS_DIR}"
        mkdir -p "${METRICS_DIR}"

        bench_args=(
            --dataset "${DATASET_DIR}"
            --out_dir "${BENCH_OUTDIR}"
            --runs "${RUNS}"
            --gpu_mode "${MODE}"
        )
        if [[ "${GPU_ONLY}" == "1" ]]; then
            bench_args+=(--gpu_only)
        fi
        if [[ "${PROFILE_ONLY}" == "1" ]]; then
            bench_args+=(--profile_only)
        fi

        set +e
        run_bench_with_gpu_metrics "${LOG_PATH}" "${QUERY_CSV}" "${DMON_LOG}" "${bench_args[@]}"
        EXIT_CODE="$?"
        set -e

        if [[ "${EXIT_CODE}" != "0" ]]; then
            RUN_STATUS="failed"
            echo "WARNING: run failed for mode=${MODE} value=${VAL} exit_code=${EXIT_CODE}" | tee -a "${LOG_PATH}"
        fi

        summary_args=(
            --bench_log "${LOG_PATH}"
            --query_csv "${QUERY_CSV}"
            --out_csv "${ITERATION_CSV}"
            --out_json "${SUMMARY_JSON}"
            --query_interval_ms "${QUERY_INTERVAL_MS}"
            --dmon_interval_sec "${DMON_INTERVAL_SEC}"
        )
        summary_args+=(--dmon_log "${DMON_LOG}")
        if [[ "${FORCE_FIRST_RUN_WARMUP}" == "1" ]]; then
            summary_args+=(--force_first_run_warmup)
        fi
        python3 "${SUMMARY_BIN}" "${summary_args[@]}"

        if [[ "${COLLECT_NSYS}" == "1" ]]; then
            NSYS_PREFIX="${PROFILES_DIR}/nsys_val${VAL}B_${MODE}"
            NSYS_LOG_PATH="${OUTDIR}/nsys_val${VAL}B_${MODE}.log"
            NSYS_BENCH_OUTDIR="${TEMP_ROOT}/${MODE}_nsys"
            NSYS_RUN_STATUS="ok"
            NSYS_EXIT_CODE="0"

            echo "Collecting Nsight Systems profile for ${MODE} at ${VAL} B..."
            rm -rf "${NSYS_BENCH_OUTDIR}"
            nsys_bench_args=(
                --dataset "${DATASET_DIR}"
                --out_dir "${NSYS_BENCH_OUTDIR}"
                --runs "${NSYS_PROFILE_RUNS}"
                --gpu_mode "${MODE}"
            )
            if [[ "${GPU_ONLY}" == "1" ]]; then
                nsys_bench_args+=(--gpu_only)
            fi
            if [[ "${PROFILE_ONLY}" == "1" ]]; then
                nsys_bench_args+=(--profile_only)
            fi

            if run_nsys_profile "${NSYS_PREFIX}" "${NSYS_LOG_PATH}" "${nsys_bench_args[@]}"; then
                NSYS_EXIT_CODE="0"
            else
                NSYS_EXIT_CODE="$?"
                NSYS_RUN_STATUS="failed"
                echo "WARNING: Nsight Systems run failed for mode=${MODE} value=${VAL} exit_code=${NSYS_EXIT_CODE}" | tee -a "${NSYS_LOG_PATH}"
            fi
            if [[ -f "${NSYS_PREFIX}.nsys-rep" ]]; then
                NSYS_REPORT_PATH="${NSYS_PREFIX}.nsys-rep"
                if NSYS_KERNEL_CSV_PATH=$(run_nsys_stats_report "${NSYS_REPORT_PATH}" "${NSYS_PREFIX}_kernel_stats_gpukernsum.csv" gpukernsum cuda_gpu_kern_sum); then
                    NSYS_KERNEL_CSV="${NSYS_KERNEL_CSV_PATH}"
                fi
                if NSYS_API_CSV_PATH=$(run_nsys_stats_report "${NSYS_REPORT_PATH}" "${NSYS_PREFIX}_api_stats_cudaapisum.csv" cudaapisum cuda_api_sum); then
                    NSYS_API_CSV="${NSYS_API_CSV_PATH}"
                fi
            fi
        fi

        if [[ "${COLLECT_NCU}" == "1" ]]; then
            NCU_PREFIX="${PROFILES_DIR}/ncu_val${VAL}B_${MODE}"
            NCU_LOG_PATH="${OUTDIR}/ncu_val${VAL}B_${MODE}.log"
            NCU_BENCH_OUTDIR="${TEMP_ROOT}/${MODE}_ncu"
            NCU_RUN_STATUS="ok"
            NCU_EXIT_CODE="0"

            echo "Collecting Nsight Compute profile for ${MODE} at ${VAL} B..."
            rm -rf "${NCU_BENCH_OUTDIR}"
            ncu_bench_args=(
                --dataset "${DATASET_DIR}"
                --out_dir "${NCU_BENCH_OUTDIR}"
                --runs "${NCU_PROFILE_RUNS}"
                --gpu_mode "${MODE}"
            )
            if [[ "${GPU_ONLY}" == "1" ]]; then
                ncu_bench_args+=(--gpu_only)
            fi
            if [[ "${PROFILE_ONLY}" == "1" ]]; then
                ncu_bench_args+=(--profile_only)
            fi

            if run_ncu_profile "${NCU_PREFIX}" "${NCU_LOG_PATH}" "${ncu_bench_args[@]}"; then
                NCU_EXIT_CODE="0"
            else
                NCU_EXIT_CODE="$?"
                NCU_RUN_STATUS="failed"
                echo "WARNING: Nsight Compute run failed for mode=${MODE} value=${VAL} exit_code=${NCU_EXIT_CODE}" | tee -a "${NCU_LOG_PATH}"
            fi
            if [[ -f "${NCU_PREFIX}.ncu-rep" ]]; then
                NCU_REPORT_PATH="${NCU_PREFIX}.ncu-rep"
            fi
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${VAL}" "${MODE}" "${DATASET_DIR}" "${LOG_PATH}" "${QUERY_CSV}" "${DMON_LOG}" "${ITERATION_CSV}" "${SUMMARY_JSON}" "${RUN_STATUS}" "${EXIT_CODE}" "${NSYS_PROFILE_RUNS}" "${NSYS_LOG_PATH}" "${NSYS_REPORT_PATH}" "${NSYS_KERNEL_CSV}" "${NSYS_API_CSV}" "${NSYS_RUN_STATUS}" "${NSYS_EXIT_CODE}" "${NCU_SET}" "${NCU_PROFILE_RUNS}" "${NCU_LOG_PATH}" "${NCU_REPORT_PATH}" "${NCU_RUN_STATUS}" "${NCU_EXIT_CODE}" \
            >> "${MANIFEST}"
    done
done

echo ""
echo "Done. Sweep outputs are in: ${OUTDIR}"
echo "Manifest: ${MANIFEST}"
echo "GPU metric summaries: ${GPU_METRICS_ROOT}"
if [[ "${COLLECT_NSYS}" == "1" ]]; then
    echo "Nsight Systems reports: ${PROFILES_DIR}"
fi
if [[ "${COLLECT_NCU}" == "1" ]]; then
    echo "Nsight Compute reports: ${PROFILES_DIR}"
fi
