# GPComp CUDA Testbench

GPU-accelerated RocksDB compaction benchmark.  
Measures and compares CPU vs GPU performance for the two compaction algorithms:
- **Algorithm 1 – Merge**: multi-way merge of SST key-value pairs (GPU `merge_kernel`)
- **Algorithm 2 – Bloom filter**: per-block Bloom filter construction (GPU `bloom_filter_kernel` / `bloom_filter_kernel_batched`)

---

## Directory Structure

```
cuda_test/
├── gpcomp_bench.cu          # main benchmark program (Benchmarks 1-4)
├── gpcomp_merge.cuh         # merge kernel + launcher
├── gpcomp_bloom.cuh         # bloom filter kernels (batched)
├── gpcomp_common.cuh        # shared types (KVPair), CPU helpers
├── gpcomp_cufile.cuh        # cuFile / GPUDirect Storage abstraction layer
├── gpcomp_datagen.cpp       # dataset generator
├── gpcomp_tests.cu          # unit tests
├── Makefile
├── test.sh                  # value-size sweep script
├── gds/                     # locally extracted cuFile package (no root needed)
│   ├── libcufile.deb        # runtime .deb (libcufile 1.4.0.31, CUDA 11.8)
│   ├── libcufile-dev.deb    # dev .deb (cufile.h + link stubs)
│   ├── setup_local.sh       # script: extracts debs → gds/local/
│   └── local/
│       ├── include/cufile.h # extracted header
│       └── lib/             # extracted .so + symlinks
├── dataset/                 # generated dataset (SST .bin files + dataset.meta)
└── results/                 # all sweep outputs (created by test.sh)
    └── run_YYYY-MM-DD_HH-MM-SS/
        ├── result_val32B.txt
        ├── result_val64B.txt
        ├── result_val128B.txt
        └── result_metadata.txt
```

---

## Build

```sh
make gpcomp_bench       # build benchmark binary (no GDS)
make gpcomp_datagen     # build dataset generator
make gpcomp_unit_tests  # build unit tests
make all                # build everything
```

Requires: CUDA 11+, `nvcc`, C++17.

### Build with cuFile / GPUDirect Storage (GDS)

See [GDS Setup](#gds--cufile-setup-gpudirect-storage) below for the one-time `gds/` extraction step.

```sh
make GDS=1 gpcomp_bench          # build with cuFile support
make bench-gds                   # generate dataset + run compat-mode (any GPU)
make bench-gds-native            # generate dataset + run native mode (A30/A100/H100)
```

The `Makefile` automatically points `-I`, `-L`, and `-rpath` at `gds/local/` so **no system install is needed and `LD_LIBRARY_PATH` is not required** at runtime.

---

## Generate a Dataset

```sh
./gpcomp_datagen --out dataset --num_sst 4 --keys_per_sst 81920
```

This produces `dataset/sst_0000.bin … sst_000N.bin` and `dataset/dataset.meta`.

---

## Run the Benchmark Directly

```sh
./gpcomp_bench [options]
```

| Option | Default | Description |
|---|---|---|
| `--dataset DIR` | `./dataset` | Path to the dataset directory |
| `--block_size BYTES` | `32768` | SST data block size (32 KB matches GP-Comp paper) |
| `--key_size BYTES` | `16` | Key size in bytes |
| `--value_size BYTES` | `64` | **Value size in bytes** — changes keys/block and bloom config |
| `--overhead BYTES` | `20` | Per-entry SST overhead bytes |
| `--fpr_samples N` | `10000` | Non-member samples used to measure false positive rate |
| `--runs N` | `5` | Number of timed repetitions per section (reports min/mean/stddev) |
| `--fillrandom_keys N` | `0` | Simulate a fillrandom workload of N total keys. Auto-computes number of compaction rounds. Accepts K/M/B suffixes (e.g. `10M`, `200M`, `1B`). `0` = skip. |
| `--compaction_rounds N` | `0` | Manually set number of compaction rounds. Overridden by `--fillrandom_keys`. |
| `--gds` | off | Enable **Benchmark 4**: GDS I/O + Merge via cuFile. Requires `GDS=1` build; falls back to pinned `cudaHostAlloc` + `fread` + `cudaMemcpy` if cuFile is not compiled in. |
| `--compat-mode` | off | Force cuFile **Compatibility Mode** before driver open. Use on RTX 3070 / any GPU without native GDS. Implies `--gds`. |
| `--help` | — | Print usage |

**Examples:**
```sh
./gpcomp_bench --dataset dataset --value_size 128 --runs 10

# simulate 10M key fillrandom workload (auto-computes ~31 compaction rounds)
./gpcomp_bench --dataset dataset --fillrandom_keys 10M

# simulate 200M key fillrandom workload
./gpcomp_bench --dataset dataset --fillrandom_keys 200M

# simulate 1 billion keys
./gpcomp_bench --dataset dataset --fillrandom_keys 1B

# manually set compaction rounds
./gpcomp_bench --dataset dataset --compaction_rounds 50
```

---

## Run the Value-Size Sweep (`test.sh`)

Runs `gpcomp_bench` once per value size and collects all outputs into a
timestamped results folder.

```sh
bash test.sh [options]
```

| Option | Default | Description |
|---|---|---|
| `--dataset DIR` | `./dataset` | Dataset directory |
| `--values LIST` | `32,64,128` | Comma-separated list of value sizes to sweep (bytes) |
| `--fillrandom_keys LIST` | `0` | Comma-separated total key counts to simulate. Accepts K/M/B suffixes. `0` = skip. |
| `--key_size BYTES` | `16` | Key size in bytes |
| `--overhead BYTES` | `20` | Per-entry SST overhead bytes |
| `--runs N` | `5` | Timed repetitions per section inside each benchmark run |
| `--outdir DIR` | `./results` | Parent directory for all results |
| `--help` / `-h` | — | Print usage |

The sweep is **2D**: every combination of `--values` × `--fillrandom_keys` gets its own result file.

**Examples:**
```sh
# default sweep: 32 / 64 / 128 B values, no fillrandom simulation
bash test.sh --dataset dataset

# sweep value sizes + simulate 10M key fillrandom workload
bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 10M

# full 2D sweep: 3 value sizes x 3 key counts
bash test.sh --dataset dataset --values 32,64,128 --fillrandom_keys 1M,10M,200M

# wider value sweep with more repetitions
bash test.sh --dataset dataset --values 32,64,128,256 --runs 10

# quick sanity check (1 run each)
bash test.sh --dataset dataset --runs 1
```

Each invocation creates a timestamped sub-folder:
```
results/run_2026-02-28_17-07-13/
  result_val32B.txt                   # no fillrandom (--fillrandom_keys 0)
  result_val64B_keys1M.txt            # value_size=64B, fillrandom_keys=1M
  result_val64B_keys10M.txt           # value_size=64B, fillrandom_keys=10M
  result_metadata.txt                 # parameters + combined summary table
```

---

## Output Files

### `result_val<N>B.txt`

Full benchmark output for a single value size. Contains:

```
# Run started: 2026-02-28 17:07:13
# Parameters: value_size=64B  key_size=16B  overhead=20B  block_size=32768B  runs=5
# ─────────────────────────────────────────────────────────

BENCHMARK 1 – Merge kernel (Algorithm 1)
  ...
  CPU sort          min=  8.12  mean=  9.05 ± 0.75 ms  (40.3 M keys/s at min)
  GPU kernel-only   min=  0.12  mean=  0.12 ± 0.00 ms  (2689.8 M keys/s at min)
  GPU wall          min=  1.86  mean=  1.92 ± 0.08 ms  (176.6 M keys/s at min)
  Speedup kernel vs CPU sort (min): 66.69×
  Speedup wall   vs CPU+I/O  (min): 6.30×
  Validation: PASS ✓

BENCHMARK 2 – Bloom filter kernel (Algorithm 2)
  ...
  CPU bloom (per-block)                min= 14.99 ms
  GPU kernel-only (no xfer)            min=  2.94 ms
  GPU batched wall (1×H2D+grid+1×D2H) min=  0.90 ms
  Speedup kernel  vs CPU (min): 5.11×
  Speedup batched vs CPU (min): 16.62×
  No false negatives: PASS ✓
  FPR measured: 0.8189%  vs theoretical: 0.8194%

BENCHMARK 3 – fillrandom compaction simulation (Merge + Bloom)
  Compaction model:
    keys/compaction round = 327680  (4 SSTs flushed from MemTable)
    compaction rounds     = 4
    total simulated keys  = 1310720  (1.3 M)

  Per-round timing (best-of-5):
    I/O (disk read, per round)                   53.32 ms
    CPU compute/round  (sort+bloom)              25.01 ms
    GPU wall/round     (H2D+merge+bloom+D2H)      2.90 ms

  Aggregate over 4 compaction rounds:
    CPU total  (I/O + sort + bloom)              313.4 ms   4.18 M keys/s
    GPU total  (I/O + merge + batched bloom)     224.9 ms   5.83 M keys/s
    Speedup (min): 1.39×   Time saved: 88.5 ms

# Run finished: 2026-02-28 17:07:14
```

### `result_metadata.txt`

Summary file for the entire sweep. Contains:
- Sweep start/finish timestamps and total elapsed time
- All parameters used
- List of result files produced
- Combined summary table (all value sizes in one place)

```
═══════════════════════════════════════════════════════════
  GPComp Sweep Metadata
═══════════════════════════════════════════════════════════

  Sweep started : 2026-02-28 17:07:13
  Sweep finished: 2026-02-28 17:07:45
  Elapsed       : 32s

  ── Parameters used ─────────────────────────────────────
  dataset        = dataset
  value_sizes    = 32,64,128 B
  key_size       = 16 B
  overhead       = 20 B
  block_size     = 32768 B  (32 KB, fixed)
  runs           = 5
  bench binary   = ./gpcomp_bench

  ── Summary (best-of-5, min latency) ────────────────────
  value_size    keys/block  CPU total(ms)  GPU batched(ms)  speedup
  ----------    ----------  -------------  ---------------  -------
  32B           481         30.83          2.86             3.10×
  64B           327         26.36          2.81             5.13×
  128B          199         26.81          2.84             4.77×
```

---
---

## GDS / cuFile Setup (GPUDirect Storage)

### One-time extraction (no root required)

The `gds/` directory already contains the downloaded `.deb` packages. To extract them into a local library tree:

```sh
bash gds/setup_local.sh
```

This creates:
```
gds/local/include/cufile.h
gds/local/lib/libcufile.so.0   → libcufile.so.1.4.0
gds/local/lib/libcufile.so.1   → libcufile.so.1.4.0
gds/local/lib/libcufile.so     → libcufile.so.1
```

### If your machine is missing the `.deb` files

For **CUDA 11.8** (matches our `nvcc` version):

```sh
# Create gds/ directory
mkdir -p gds && cd gds

# Download runtime and dev packages (Ubuntu 22.04 debs, binary-compatible with Debian 12)
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
curl -fsSL -o libcufile.deb     "$BASE/libcufile-11-8_1.4.0.31-1_amd64.deb"
curl -fsSL -o libcufile-dev.deb "$BASE/libcufile-dev-11-8_1.4.0.31-1_amd64.deb"
cd ..
bash gds/setup_local.sh
```

For **other CUDA versions**, replace `11-8` and `1.4.0.31` with the version matching your `nvcc --version`. Browse available packages at:
`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/`

### GPU compatibility

| GPU | GDS support | Recommended flag |
|---|---|---|
| RTX 3070, RTX 3080, RTX 4090, any consumer GPU | **Compat Mode** (bounce-buffer via cuFile driver) | `--compat-mode` |
| A30, A100, H100, data-centre GPUs with local NVMe | **Native GDS** (PCIe DMA direct to VRAM) | `--gds` (no compat flag) |

> **Note on NFS:** GDS bypasses the OS page cache using `O_DIRECT`. On NFS, this eliminates read-ahead and is typically **slower** than POSIX `fread`. GDS gives the best performance on **local NVMe** attached via PCIe. See analysis below.

### Build and run

```sh
# Build with cuFile
make GDS=1 gpcomp_bench

# Run Benchmark 4 on RTX 3070 (or any non-GDS GPU)
./gpcomp_bench --compat-mode

# Run Benchmark 4 on A30 / A100 / H100 (native GDS)
./gpcomp_bench --gds

# Combined full benchmark including GDS benchmark
./gpcomp_bench --compat-mode --fillrandom_keys 200M --runs 5
```

### Benchmark 4 – GDS I/O + Merge

This benchmark loads all SST files **directly into device memory** using `gpcomp_cufile.cuh`, then immediately runs the merge kernel — eliminating the separate POSIX I/O + `cudaMemcpy H2D` step.

| Metric | Meaning |
|---|---|
| **GDS I/O (disk→device)** | `cuFileRead` time (compat: via pinned bounce-buffer; native: PCIe DMA) |
| **GPU merge kernel-only** | CUDA events around `merge_kernel` — data already on device |
| **GDS wall** | GDS I/O + kernel + D2H end-to-end |

Output also prints a comparison table vs the traditional POSIX I/O + H2D path.

### Performance results (RTX 3070, NFS storage, 8 SSTs × 524K keys)

| Path | Min latency | Throughput |
|---|---|---|
| POSIX fread (OS-cached) | 84 ms I/O | 762 MB/s |
| POSIX fread (cold NFS) | 404 ms I/O | 158 MB/s |
| cuFile compat (O_DIRECT, NFS) | 630 ms I/O | 102 MB/s |

**Key finding:** On NFS + consumer GPU, POSIX `fread` into `cudaHostAlloc` is the correct I/O path. cuFile compat mode is 1.5–6× slower on NFS because `O_DIRECT` bypasses read-ahead caching. GDS is designed for **local NVMe**, where it delivers meaningful speedups on a GDS-capable GPU (A30+).

---
## What Each Benchmark Measures

### Benchmark 1 – Merge

| Label | What it times |
|---|---|
| **CPU sort** | `std::sort` over all merged keys (no I/O) |
| **GPU kernel-only** | CUDA events around `merge_kernel` only — pure compute time, no transfers |
| **GPU wall (H2D+k+D2H)** | Host→Device copy + kernel + Device→Host copy |

The merge kernel assigns one thread per output key and uses binary search
to find each key's source SST, so it scales with key count, not SST count.

### Benchmark 2 – Bloom Filter

| Label | What it times |
|---|---|
| **CPU bloom** | Pure CPU: `cpu_build_byte_vector` + `cpu_pack_bit_vector` for every block |
| **GPU kernel-only** | CUDA events around kernel only; data pre-transferred once before loop |
| **GPU batched wall** | 1× H2D for all blocks → `bloom_filter_kernel_batched<<<N,T>>>` → 1× D2H |

The batched kernel processes all data blocks in a single grid launch,
avoiding any per-block `cudaDeviceSynchronize` overhead.

### Benchmark 3 – fillrandom compaction simulation

Simulates an entire `db_bench fillrandom` workload spanning multiple compaction
rounds. Per-round timings (I/O + CPU compute or GPU wall) from Benchmarks 1 & 2
are scaled by the number of compaction rounds to produce aggregate CPU vs GPU
totals, speedup, and time saved.

Requires `--fillrandom_keys N` (or `--compaction_rounds N`) to be set.

| Column | Meaning |
|---|---|
| CPU total | `rounds × (I/O + sort + bloom)` |
| GPU total | `rounds × (I/O + merge wall + batched bloom wall)` |
| Speedup | `CPU total / GPU total` |
| Time saved | `CPU total − GPU total` |

If `--fillrandom_keys` or `--compaction_rounds` is provided, Benchmark 3 also
appends a **fillrandom simulation** section that projects the single-round timings
across N compaction rounds, reporting aggregate CPU vs GPU time saved.
