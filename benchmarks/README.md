# Benchmarks

## Prerequisites

### Build `db_bench` (release mode)

```bash
cd rocksdb-gpu
make -j"$(nproc)" DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 db_bench
```

### Build `gpcomp_bench` / `gpcomp_datagen`

```bash
cd gpu_compaction
make gpcomp_datagen gpcomp_bench -j
```

> The GPU sweep scripts automatically rebuild `gpcomp_bench` per value size (compile-time constant), so manual builds are only needed for standalone use.

### Configure machine-local paths

```bash
cp benchmarks/common/config/.env.example benchmarks/common/config/.env.local
```

Required in `.env.local`:

| Variable      | Description                         |
|---------------|-------------------------------------|
| `DB_BENCH`    | Absolute path to `db_bench` binary  |
| `DB_BASE_DIR` | Directory for RocksDB data files    |
| `WAL_BASE_DIR`| Directory for WAL files             |
| `OUTPUT_DIR`  | Root directory for benchmark results|
| `CPU_COMPACTION_OUT_ROOT` | Root directory for standalone CPU sweep outputs |
| `CPU_COMPACTION_PRELOAD_ROOT` | Root directory for standalone CPU preload artifacts |
| `GPU_COMPACTION_OUT_ROOT` | Root directory for GPU sweep outputs |
| `GPU_COMPACTION_DATASET_ROOT` | Root directory for generated GPU datasets |

---

## CPU Compaction (`cpu_compaction/`)

Measures RocksDB subcompaction scaling on CPU. Preloads a database with `fillrandom` (auto-compactions disabled), then triggers manual compaction (`compactall`) while sweeping subcompaction thread counts.

### Parameters

| Env Variable | Default | Description |
|---|---|---|
| `VALUE_SIZES` | `32 64 128 256 512 1024` | Value sizes in bytes |
| `SST_SIZE_MB_LIST` | `8` | SST target file sizes in MB |
| `INPUT_DATA_MB_LIST` | `32 256` | Logical input data sizes in MB |
| `INPUT_SST_COUNT_LIST` | _(unset)_ | Alternative: specify SST count directly (mutually exclusive with `INPUT_DATA_MB_LIST`) |
| `SUBCOMP_THREADS_LIST` | `1 2 4 8 16 32` | Subcompaction thread counts to sweep |
| `COMPACTION_BG_THREADS` | `1` | Background compaction job threads (`--max_background_compactions`) |
| `COMPACTION_RUNS` | `5` | Repetitions per configuration |
| `COMPACTION_BENCH` | `compactall` | Benchmark type: `compactall`, `compact0`, `compact1` |
| `HOST_METRICS_INTERVAL_SEC` | `0.1` | Host CPU/IO sampling interval (0 = disabled) |
| `METRICS_INTERVAL_MS` | `100` | RocksDB metrics interval (0 = disabled) |

### Standalone usage

```bash
# Quick validation
./benchmarks/cpu_compaction/run_sweep.sh --config quick_test

# Full sweep
./benchmarks/cpu_compaction/run_sweep.sh --config full_sweep

# Custom parameters
VALUE_SIZES="64 256" \
SUBCOMP_THREADS_LIST="1 4 16" \
COMPACTION_RUNS=3 \
./benchmarks/cpu_compaction/run_sweep.sh --config full_sweep
```

Standalone CPU runs now default to:

```text
${CPU_COMPACTION_OUT_ROOT}/<run-id>/
```

For example, the quick test writes to `local_benchmark_artifacts/cpu_compaction/sweep_results/quick-test/`
when using the sample `.env.local` paths.

---

## GPU Compaction (`gpu_compaction/`)

Measures GPU compaction throughput using `gpcomp_bench`. Sweeps across value sizes and GPU modes, automatically rebuilding for each value size (compile-time constant in `gpcomp_common.cuh`).

### GPU Modes

| Mode | Description |
|---|---|
| `q_paper_with_plan` | Query-based compaction with planning |
| `q_paper_without_plan` | Query-based compaction without planning |
| `c_paper_with_plan` | Compaction-based with planning |
| `c_paper_without_plan` | Compaction-based without planning |

### Execution Modes

| Flag | Effect |
|---|---|
| _(default)_ | Runs with CPU baseline comparison |
| `--gpu_only` | GPU-only, no CPU baseline; enables host metrics at 0.1s |
| `--profile_only` | GPU-only, skips throughput measurement, focuses on resource utilization |

### Scripts

#### `gpu_compaction_sweep.sh` — throughput + optional host metrics

Primary sweep script. Collects per-stage timing breakdown (read, unpack, sort, gc, plan, bloom, pack, write).

| CLI Argument | Default | Description |
|---|---|---|
| `--values` | `32 64 128 256 512 1024` | Value sizes to sweep |
| `--modes` | all 4 modes | GPU modes to run |
| `--runs` | `5` | Iterations per mode |
| `--num_ssts` | _(from compile constant)_ | Override `GP_NUM_INPUT_SSTS` |
| `--gpu_only` | off | GPU-only mode |
| `--profile_only` | off | Profile-only mode (implies `--gpu_only`) |
| `--host_metrics_interval_sec` | auto | Host metrics sampling interval |
| `--zipf_alpha` | `0.0` | Key distribution (0.0 = uniform) |
| `--user_key_space` | `200000000` | Key space size |
| `--out_root` | `${GPU_COMPACTION_OUT_ROOT}` | Output root directory |
| `--label` | _(auto)_ | Output directory suffix |

By default, GPU sweep scripts now read `GPU_COMPACTION_OUT_ROOT` and `GPU_COMPACTION_DATASET_ROOT`
from `benchmarks/common/config/.env.local`, so generated datasets and sweep outputs no longer have
to live under `gpu_compaction/`.

Generated dataset directories default to `dataset_V<value_size>` under `GPU_COMPACTION_DATASET_ROOT`.
Standalone GPU runs default to `${GPU_COMPACTION_OUT_ROOT}/<label>/`.

#### `gpu_compaction_sweep_metrics.sh` — nvidia-smi GPU metrics + Nsight profiling

Full metrics collection with GPU device telemetry and optional Nsight Systems/Compute profiling.

Accepts all flags from `gpu_compaction_sweep.sh`, plus:

**nvidia-smi collection** (always on):

| CLI Argument | Default | Description |
|---|---|---|
| `--gpu_id` | `0` | GPU index |
| `--query_interval_ms` | `100` | nvidia-smi polling interval |
| `--dmon_interval_sec` | `1` | nvidia-smi dmon sampling interval |

Collected fields: `utilization.gpu%`, `utilization.memory%`, `memory.used`, `memory.total`, `power.draw`, `temperature.gpu`, `clocks.sm`, `clocks.mem`, `pstate`.

**Nsight Systems** (opt-in):

| CLI Argument | Default | Description |
|---|---|---|
| `--collect_nsys` | off | Enable Nsight Systems profiling |
| `--nsys_bin` | auto-detect | Path to `nsys` binary |
| `--nsys_cpuctxsw` | `none` | CPU context switch tracing mode |
| `--nsys_profile_runs` | `1` | Profiling runs per mode |
| `--nsys_gpu_metrics` | off | Enable GPU metrics in nsys |
| `--nsys_gpu_metrics_set` | _(device default)_ | GPU metrics set name |
| `--nsys_gpu_metrics_frequency` | _(nsys default)_ | GPU metrics sampling frequency (Hz) |

Traces: `cuda`, `osrt`, `nvtx`. Produces `.nsys-rep` reports and extracted CSVs (`gpukernsum`, `cudaapisum`).

**Nsight Compute** (opt-in):

| CLI Argument | Default | Description |
|---|---|---|
| `--collect_ncu` | off | Enable Nsight Compute profiling |
| `--ncu_bin` | `ncu` | Path to `ncu` binary |
| `--ncu_set` | `default` | Metrics set |
| `--ncu_replay_mode` | `kernel` | Replay strategy |
| `--ncu_profile_runs` | `1` | Profiling runs per mode |

Produces `.ncu-rep` reports (openable in Nsight Compute UI).

#### `gpu_compaction_sweep_nsight.sh` — lightweight Nsight-only sweep

Dedicated profiling sweep without nvidia-smi collection.

| CLI Argument | Default | Description |
|---|---|---|
| `--tool` | _(required)_ | `nsys` or `ncu` |
| `--cpuctxsw` | `none` | CPU context switch tracing (nsys only) |
| `--plot` | off | Generate result plots |

#### `nsight_profile.sh` — single-run profiling utility

Quick one-off profiling of a specific mode:

```bash
./scripts/nsight_profile.sh --tool nsys --dataset /abs/path/to/dataset_V32 --gpu_mode q_paper_with_plan --out_dir /abs/path/to/results/
./scripts/nsight_profile.sh --tool ncu  --dataset /abs/path/to/dataset_V32 --gpu_mode c_paper_with_plan --out_dir /abs/path/to/results/
```

### Standalone usage

```bash
# Quick validation
./benchmarks/gpu_compaction/run_sweep.sh --config quick_test

# Full sweep
./benchmarks/gpu_compaction/run_sweep.sh --config full_sweep

# GPU-only with host metrics
./benchmarks/gpu_compaction/run_sweep.sh --config full_sweep --gpu_only

# Full sweep with nvidia-smi device metrics
bash ./benchmarks/gpu_compaction/scripts/gpu_compaction_sweep_metrics.sh \
  --values "32 64 128 256 512 1024" \
  --modes "q_paper_with_plan c_paper_with_plan" \
  --runs 5 --gpu_only

# nvidia-smi + Nsight Systems + Nsight Compute
bash ./benchmarks/gpu_compaction/scripts/gpu_compaction_sweep_metrics.sh \
  --values "64 256" --modes "q_paper_with_plan" --runs 3 --gpu_only \
  --collect_nsys --collect_ncu --ncu_set default

# Lightweight nsys-only sweep
bash ./benchmarks/gpu_compaction/scripts/gpu_compaction_sweep_nsight.sh \
  --tool nsys --values "64 256" --modes "q_paper_with_plan" --runs 1
```

---

## Combined CPU + GPU Comparison (`initial_gpu_cpu_compaction_microbenchmark/`)

End-to-end orchestrator that runs CPU and GPU compaction sweeps with matched parameters and produces a comparison analysis.

### Phases

| Phase | Script Used | Description | Skip |
|---|---|---|---|
| 1 | `cpu_compaction_sweep.sh` | CPU subcompaction scaling with host metrics | `SKIP_CPU=1` |
| 2 | `gpu_compaction_sweep.sh` | GPU with per-stage timing breakdown (includes CPU baseline) | `SKIP_GPU_BREAKDOWN=1` |
| 3 | `gpu_compaction_sweep.sh --profile_only` | GPU-only with host CPU/IO metrics | `SKIP_GPU_HOST_METRICS=1` |
| 4 | `gpu_compaction_sweep_metrics.sh --gpu_only` | GPU with nvidia-smi device metrics | `SKIP_GPU_DEVICE_METRICS=1` |
| Final | comparison scripts | CPU vs GPU speedup analysis | `SKIP_COMPARE=1` |

### Parameters

| Env Variable | Default | Description |
|---|---|---|
| `RUN_ID` | `full-sweep` | Run identifier |
| `VALUE_SIZES` | `32 64 128 256 512 1024` | Value sizes in bytes |
| `SUBCOMP_THREADS_LIST` | `1 2 4 8 16 32` | CPU subcompaction threads to sweep |
| `GPU_MODES` | all 4 modes | GPU modes to benchmark |
| `GPU_RUNS` | `5` | GPU repetitions per config |
| `CPU_RUNS` | `5` | CPU repetitions per config |
| `RUN_PAUSE_SECONDS` | `2` | Pause between runs |

CLI arguments `--sst-size-mb` and `--input-ssts` control the SST geometry.

### Usage

```bash
# Quick end-to-end validation
./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh \
  --config quick_test --sst-size-mb 8 --input-ssts 32

# Full comparison sweep with all defaults
./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh \
  --config full_sweep --sst-size-mb 8 --input-ssts 32

# Custom parameters
RUN_ID="8mb_sst_32sst_full" \
VALUE_SIZES="32 64 128 256 512 1024" \
SUBCOMP_THREADS_LIST="1 2 4 8 16" \
GPU_MODES="q_paper_with_plan q_paper_without_plan c_paper_with_plan c_paper_without_plan" \
GPU_RUNS=5 CPU_RUNS=5 \
bash ./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh \
  --config full_sweep --sst-size-mb 8 --input-ssts 32

# Skip CPU phase (re-run GPU only)
SKIP_CPU=1 \
./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh \
  --config full_sweep --sst-size-mb 8 --input-ssts 32

# Only run CPU + comparison (skip all GPU phases)
SKIP_GPU_BREAKDOWN=1 SKIP_GPU_HOST_METRICS=1 SKIP_GPU_DEVICE_METRICS=1 \
./benchmarks/initial_gpu_cpu_compaction_microbenchmark/run_sweep.sh \
  --config full_sweep --sst-size-mb 8 --input-ssts 32
```

### Output structure

```
${OUTPUT_DIR}/initial_gpu_cpu_compaction_microbenchmark/${RUN_ID}/
  cpu/                           # Phase 1: CPU compaction results
  gpu_breakdown/                 # Phase 2: GPU stage breakdown
  gpu_host_metrics_gpu-profile-only/   # Phase 3: GPU host CPU/IO metrics
  gpu_device_metrics_gpu-only/         # Phase 4: GPU nvidia-smi metrics
  comparison/                    # CPU vs GPU speedup analysis
```

---

## Common (`common/`)

Shared infrastructure used by both CPU and GPU benchmarks.

### `compaction_defaults.sh`

Default parameter values sourced by all sweep scripts. Override any of these via environment variables before running.

### `config/.env.local`

Machine-local paths (`DB_BENCH`, `DB_BASE_DIR`, `WAL_BASE_DIR`, `OUTPUT_DIR`). Copy from `.env.example`.

### `collect_host_metrics.py`

Low-overhead host metrics collector spawned in the background during benchmark runs. Samples `/proc` and `/sys` at a configurable interval.

Produces per-run:
- `system_cpu.csv`, `per_cpu.csv` — system and per-core CPU utilization
- `device_io.csv` — block device throughput, utilization, queue depth
- `process_cpu.csv`, `thread_cpu.csv`, `thread_role_cpu.csv` — process/thread-level CPU
- `summary.json` — aggregated averages and maximums

### `benchmark_common.sh`

Shared shell functions for `db_bench` invocation, metadata capture, host metrics lifecycle, and directory management.
