# GPU Benchmarks

Benchmark suite for GPU-related storage experiments. It currently includes:

- **`io_bench`**: GPU IO (cuFile) vs CPU direct IO for a simulated LSM-tree
  compaction IO pattern
- **`rocksdb_hook`**: RocksDB dummy GPU compaction hook benchmark — drives a
  real compaction through RocksDB's `CompactionService` hook, invokes the GPU
  merge worker on the actual input SSTs, then validates and installs the merged
  output back into RocksDB
- **`gpu_compaction_worker`**: standalone real GPU merge worker that reads
  actual RocksDB SST files, merges/deduplicates keys on the GPU, and writes a
  merged output SST file

## Motivation

GPU-accelerated compaction (e.g., GPComp) offloads merge/sort to the GPU, but
the data must travel a longer IO path:

```
CPU compaction IO:   SSD ──read()──▸ RAM ──write()──▸ SSD

GPU compaction IO:   SSD ──▸ RAM ──▸ GPU ──▸ RAM ──▸ SSD
                         (cuFileRead)    (cuFileWrite)
```

On GPUs **without** GPUDirect Storage (GDS) — such as GeForce cards — the cuFile
API falls back to a **bounce-buffer** through host memory, adding extra PCIe hops.
This benchmark quantifies exactly how much overhead those extra hops cost.

The key question: **How much faster must the GPU merge/sort be to offset the
additional IO transfer time?**

## Simulated Compaction Pattern

Each benchmark run simulates a single L0→L1 compaction:

| Phase | Files | Size each | Total data |
|-------|-------|-----------|------------|
| Read  | 4 L0  | 8 or 64 MB | 32 or 256 MB |
| Write | 3 L1  | 8 or 64 MB | 24 or 192 MB |

This matches the typical RocksDB compaction trigger of
`level0_file_num_compaction_trigger = 4`, writing back files of the same
`target_file_size_base`.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (tested: RTX 3070)
- Fast NVMe SSD

### Software
- CUDA Toolkit ≥ 11.4 with cuFile headers & libraries (`nvcc` must be on `PATH`)
- `g++-12` or compatible C++20 host compiler
- Pre-built `librocksdb.a` in `rocksdb-gpu/` (required for `gpu_compaction_worker`
  and `rocksdb_hook`; build once with
  `make -C rocksdb-gpu USE_RTTI=1 LIB_MODE=static -j$(nproc) static_lib`)
- Python 3.8+ with `pandas` (for analysis scripts)
- Optional: `matplotlib` (for plots), `tabulate` (for tables)

## Quick Start

### 1. Configure

```bash
cp benchmarks/gpu/config/.env.example benchmarks/gpu/config/.env.local
```

Edit `.env.local` and set at minimum:

```bash
DATA_DIR=/tmp/gpu_io_bench_data   # fast NVMe scratch space
OUTPUT_DIR=/path/to/results/gpu   # where CSVs and logs land
GPU_DEVICE=0                      # CUDA device ordinal
COMPAT_MODE=true                  # set false only if GDS kernel driver is installed
```

### 2. Build

All binaries are built automatically when you run the experiment scripts. To
build manually (from `benchmarks/gpu/`):

```bash
make                        # gpu_io_bench only
make gpu_file_replay_bench  # cuFile IO replay helper
make gpu_compaction_worker  # real GPU merge worker (requires librocksdb.a)
make tools                  # all three at once
```

### 3. Run `io_bench`

From the repository root, just run the experiment script — all parameters are
set inside it:

```bash
./benchmarks/gpu/experiments/io_bench/io_sweep.sh   # 8 MB and 64 MB sweep
./benchmarks/gpu/experiments/io_bench/io_8mb.sh     # 8 MB only
./benchmarks/gpu/experiments/io_bench/io_64mb.sh    # 64 MB only
```

To override a parameter inline:

```bash
NUM_REPS=5 ./benchmarks/gpu/experiments/io_bench/io_8mb.sh
```

### 4. Run `rocksdb_hook` (real GPU compaction)

From the repository root:

```bash
./benchmarks/gpu/experiments/rocksdb_hook/hook_replay.sh
```

This script:
1. Auto-builds `gpu_file_replay_bench`, `gpu_compaction_worker`, and the
   RocksDB benchmark binary if they are out of date
2. Drives a real RocksDB compaction through the `CompactionService` hook
3. For each compaction job, invokes `gpu_compaction_worker` on the actual input
   SSTs — the worker reads them, merges and deduplicates keys on the GPU, and
   writes a merged output SST
4. Validates the output SST keys against the RocksDB ground-truth result
5. Prints per-repetition timing (SST read, GPU merge kernel, SST write) and
   `validated=PASS/FAIL`

Default parameters in `hook_replay.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_REPS` | `1` | Number of benchmark repetitions |
| `INPUT_SST_MB` | `8` | Approximate size of each input SST |
| `ALIGNMENT` | `4096` | O_DIRECT alignment (bytes) |
| `RUN_ID` | `rocksdb-hook-replay` | Label for the output directory |

To override inline:

```bash
NUM_REPS=5 INPUT_SST_MB=64 ./benchmarks/gpu/experiments/rocksdb_hook/hook_replay.sh
```

### 5. Analyse results

```bash
# io_bench
python3 benchmarks/gpu/python/plot_io_bench.py <RUN_DIR> --plot

# rocksdb_hook
python3 benchmarks/gpu/python/plot_gpu_compaction_hook_bench.py <RUN_DIR> --plot
```

---

## CPU Comparison Baseline

The CPU benchmarks live in `benchmarks/cpu/`. To run the equivalent RocksDB
`fillrandom` workload (which triggers background compactions) and collect
compaction timing:

```bash
# From the repository root:
bash benchmarks/cpu/scripts/run_fillrandom.sh
```

All parameters (`NUM_KEYS`, `VALUE_SIZES`, `SUBCOMP_THREADS_LIST`, etc.) are
set inside the script and read from `benchmarks/cpu/config/.env.local`.
See `benchmarks/cpu/config/.env.example` for the full list.

For an isolated compaction profile (load first, then force a full compaction
and record per-job IO stats):

```bash
bash benchmarks/cpu/scripts/run_compaction_profile.sh
```

### Measured Results (RTX 3070, i7-11700, NVMe SSD)

**GPU compaction worker** (`rocksdb_hook`, 5 input SSTs, ~128 merged keys):

| Phase | Time |
|-------|------|
| SST read (`sst_read_us`) | ~24 ms |
| GPU merge kernel (`merge_kernel_us`) | ~190 ms |
| SST write (`sst_write_us`) | ~26 ms |
| **Total per compaction job** | **~240 ms** |

**CPU compaction** (`fillrandom`, 1 M keys, 400 B values, 1 subcompaction thread):

| Metric | Value |
|--------|-------|
| Write throughput | 34,312 ops/sec |
| Compaction wall time P50 | ~2,275 ms |
| Compaction wall time P99 | ~4,252 ms |
| Compactions triggered | 13 |

**GPU speedup on compaction: ~9–10×** (P50 CPU wall time vs GPU total time).

---

## Output Format

### `io_bench` CSV columns

| Column | Description |
|--------|-------------|
| `path` | `cpu` or `gpu` |
| `l0_size_bytes` | Size of each L0 file |
| `l0_size_mb` | Size in MB |
| `num_l0_read` | Number of files read |
| `num_l1_write` | Number of files written |
| `rep` | Repetition index |
| `read_us` | Read phase time (µs) |
| `write_us` | Write phase time (µs) |
| `total_us` | Total IO time (µs) |
| `direct_io` | Whether O_DIRECT was used |

### `rocksdb_hook` benchmark output lines

Each repetition prints a line like:

```
GPU_COMPACTION_WORKER_BENCH rep=0 sst_read_us=24123 merge_kernel_us=189754 sst_write_us=25891 validated=PASS
```

| Field | Description |
|-------|-------------|
| `sst_read_us` | Time to read all input SSTs (µs) |
| `merge_kernel_us` | GPU merge + dedup kernel time (µs) |
| `sst_write_us` | Time to write the merged output SST (µs) |
| `validated` | `PASS` if output keys match RocksDB ground truth |

## Results Directory Structure

```
bench_results/gpu/
  io_bench/
    <run_id>/
      io_bench_8mb.csv          # raw per-rep data (8 MB L0 files)
      io_bench_64mb.csv         # raw per-rep data (64 MB L0 files)
      io_bench_8mb.log          # full console output
      io_bench_64mb.log
      io_bench_summary.csv      # mean ± std per path/size
      io_bench_overhead.csv     # GPU/CPU overhead ratios
      io_bench_comparison.png   # bar chart (if --plot)
      io_bench_boxplot.png      # box plot (if --plot)
      metadata/
        run_config.env
  rocksdb_hook/
    <run_id>/
      gpu_compaction_hook.csv        # per-repetition timing data
      gpu_compaction_hook.log        # full benchmark log
      gpu_compaction_hook_summary.csv
      gpu_compaction_hook_summary.png   # if --plot
      gpu_compaction_hook_per_rep.png   # if --plot
      metadata/
        run_config.env
```

## What to Expect

On an RTX 3070 (no GDS, bounce-buffer fallback):

- **GPU reads** are ~1.5–2× slower than CPU direct reads due to the extra
  `memcpy` from host RAM to GPU device memory.
- **GPU writes** are similarly ~1.2–1.5× slower.
- **Total IO overhead** is typically 1.3–1.8× depending on file size and SSD speed.
- **GPU merge kernel** is ~9–10× faster than CPU compaction wall time, meaning
  the compute speedup more than covers the IO tax.

## Configuration Reference

See [config/.env.example](config/.env.example) for all machine-local settings.
Workload parameters are set directly in the experiment scripts under
`benchmarks/gpu/experiments/` — you do not need to edit `.env.local` for them.
