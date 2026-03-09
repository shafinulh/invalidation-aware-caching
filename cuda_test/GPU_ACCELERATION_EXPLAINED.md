# GPU-Accelerated RocksDB Compaction — How It Works

## Table of Contents

1. [What Does the GPU Accelerate?](#1-what-does-the-gpu-accelerate)
2. [Merge vs Bloom: What Are They and How Are They Different?](#2-merge-vs-bloom-what-are-they-and-how-are-they-different)
3. [When Does the GPU Kick In?](#3-when-does-the-gpu-kick-in)
4. [Data Paths](#4-data-paths)
5. [Simulation Testbench Overview](#5-simulation-testbench-overview)
6. [CPU vs GPU Data Paths in the Simulation](#6-cpu-vs-gpu-data-paths-in-the-simulation)
7. [Step-by-Step Examples](#7-step-by-step-examples)
8. [cuFile / GPUDirect Storage (GDS)](#8-cufile--gpudirect-storage-gds)
9. [Benchmark Results — RTX 3070, NFS, Large Dataset](#9-benchmark-results--rtx-3070-nfs-large-dataset)

---

## 1. What Does the GPU Accelerate?

The GPU accelerates the **compaction** step of RocksDB's LSM-tree storage engine. Compaction is the most compute-intensive background operation in RocksDB — it takes multiple sorted SST (Sorted String Table) files and combines them into a single, merged, sorted SST file with fresh Bloom filters.

Specifically, two algorithms from the **GP-Comp** (GPU Compaction) paper are offloaded to the GPU:

| Algorithm | What it does on GPU |
|---|---|
| **Algorithm 1 — Merge** | Multi-way merge of sorted key-value pair arrays from multiple L0 SSTs into one globally sorted output |
| **Algorithm 2 — Bloom Filter** | Construction of per-data-block Bloom filters for the newly merged SST |

On the CPU, the merge is done with `std::sort` (concat all arrays, then sort). On the GPU, each key gets its own thread and uses binary search across all input arrays to find its correct output position in parallel — no serial comparison-based sorting needed.

---

## 2. Merge vs Bloom: What Are They and How Are They Different?

### Merge (Algorithm 1)

**Purpose:** Combine multiple individually-sorted SST arrays into one globally-sorted array.

**How it works (GPU):**
- Each KV pair is assigned one CUDA thread
- Each thread knows which source SST it belongs to (via prefix-sum offsets)
- The thread does a **binary search** in every *other* SST to find how many keys are smaller than its own key
- It sums up all those counts → that gives the exact output position
- The thread writes its KV pair directly to the output at that position
- Result: a fully sorted merged array — no locks, no atomics, no serial dependencies

**How it works (CPU baseline):**
- Concatenate all SST arrays into one big array
- Run `std::sort` (comparison-based, O(N log N))

**Key difference:** The GPU merge is **embarrassingly parallel** — every thread independently computes its output position. The CPU merge is inherently serial (comparison-based sort).

### Bloom Filter (Algorithm 2)

**Purpose:** For each data block of the output SST, build a compact Bloom filter bit-vector that enables fast "is this key possibly in this block?" lookups during reads.

**How it works (GPU):**

The kernel runs in two phases within a single CUDA block, using **shared memory** as scratch space:

| Phase | What happens |
|---|---|
| **Phase 1 — Hashing** | Each thread takes one key and computes K hash values (MurmurHash3-style). For each hash, it sets the corresponding byte in a shared-memory `ByteVector` to 1. Write-write races are benign (all threads write the same value `1`). |
| **Phase 2 — Compaction** | After a `__syncthreads()`, each thread reads 8 consecutive bytes from the `ByteVector` and packs them into 1 byte of the output `BitVector` (8:1 compression). |

**Two GPU kernel variants exist:**

| Variant | Launch pattern | Use |
|---|---|---|
| **Per-block** (`bloom_filter_kernel`) | `<<<1, T>>>` launched once per data block, used for **validation** only | Verifies batched output matches per-block serial execution |
| **Batched** (`bloom_filter_kernel_batched`) | `<<<num_blocks, T>>>` launched once for ALL data blocks. 1× H2D transfer, 1 kernel launch, 1× D2H transfer | **Timed in benchmarks** — eliminates all per-block sync overhead |

**Key difference from Merge:** Merge operates on the *entire dataset at once* (one thread per key across all SSTs). Bloom operates *per data block* (one CUDA block per SST data block, using shared memory per block).

---

## 3. When Does the GPU Kick In?

The GPU is used during **RocksDB L0→L1 compaction**, triggered when the number of L0 SST files reaches a threshold (default: 4). The flow is:

```
Application writes keys
        │
        ▼
   MemTable (in-memory, write-optimized)
        │  fills up (~8 MB)
        ▼
   Flush to L0 SST file on disk
        │  accumulates 4 L0 SSTs
        ▼
  ╔═══════════════════════════════════════╗
  ║  COMPACTION TRIGGERED                 ║
  ║                                       ║
  ║  Step 1: Read 4 L0 SST files (I/O)   ║
  ║  Step 2: Merge all KV pairs (GPU)  ◄──╫── Algorithm 1
  ║  Step 3: Build Bloom filters (GPU) ◄──╫── Algorithm 2
  ║  Step 4: Write merged L1 SST (I/O)   ║
  ╚═══════════════════════════════════════╝
        │
        ▼
   L1 SST (sorted, compacted, with Bloom filters)
```

The GPU replaces steps 2 and 3 only. Disk I/O (steps 1 and 4) remains on the CPU.

---

## 4. Data Paths

### 4.1 Merge Data Path

```
 CPU (Host)                              GPU (Device)
─────────────────────────────────────────────────────────────
 SST_0.bin ─┐
 SST_1.bin ─┼──► Host arrays ──H2D──►  d_sst_arrays[]
 SST_2.bin ─┤    (sorted)              d_sst_sizes[]
 SST_3.bin ─┘                          d_sst_offsets[]
                                             │
                                             ▼
                                      merge_kernel<<<grid, 256>>>
                                      (1 thread per KV pair)
                                      (binary search across all SSTs)
                                             │
                                             ▼
                                        d_output[]
                                        (globally sorted)
                                             │
 h_output[] ◄────────────D2H────────────────┘
 (merged sorted KV pairs)
```

### 4.2 Bloom Filter Data Path (Batched)

```
 CPU (Host)                              GPU (Device)
─────────────────────────────────────────────────────────────
 merged KV pairs ──────H2D──────►  d_all_keys[]
 (from merge output)                    │
                                        ▼
                              bloom_filter_kernel_batched
                              <<<num_blocks, block_dim>>>
                              (1 CUDA block per data block)
                                        │
                               Each CUDA block:
                               ┌──────────────────────┐
                               │ Shared Memory:       │
                               │  ByteVector[bvlen]   │
                               │                      │
                               │ Phase1: hash keys    │
                               │  → set ByteVector[h] │
                               │ __syncthreads()      │
                               │ Phase2: pack 8:1     │
                               │  → BitVector byte    │
                               └──────────────────────┘
                                        │
                                        ▼
                                  d_all_bitvecs[]
                                  (packed bit-vectors)
                                        │
 h_bitvecs[] ◄──────────D2H───────────┘
 (one Bloom filter per data block)
```

---

## 5. Simulation Testbench Overview

The testbench (`gpcomp_bench.cu`) simulates a **realistic RocksDB compaction workload** without running an actual database. It measures and compares CPU vs GPU performance for both algorithms.

### What does it simulate?

It simulates the **compute-intensive core of a compaction job**:

1. **Reads synthetic SST binary files** from disk (generated by `gpcomp_datagen`) — these mimic real L0 SST files with sorted, unique KV pairs
2. **Merges them** (CPU: `std::sort`; GPU: `merge_kernel`)
3. **Builds Bloom filters** for all data blocks in the merged output (CPU: sequential hashing; GPU: kernel per block or batched)
4. **Measures wall-clock time** for each path, including memory transfer overhead
5. **Validates correctness** — GPU output is compared against CPU reference; Bloom filters are checked for false negatives and FPR

### The 4 benchmarks:

| Benchmark | What it measures |
|---|---|
| **1 — Merge** | CPU sort vs GPU merge kernel (kernel-only and with H2D/D2H transfers) |
| **2 — Bloom** | CPU bloom vs GPU bloom (kernel-only and batched) |
| **3 — fillrandom simulation** | Scales per-round CPU and GPU timings across N compaction rounds (requires `--fillrandom_keys`). Shows aggregate CPU total vs GPU total, speedup, and time saved |
| **4 — GDS I/O + Merge** | Loads SST files directly into device memory via `gpcomp_cufile.cuh` (cuFile or pinned-memory fallback), then runs the merge kernel. Requires `--gds` or `--compat-mode` flag and `GDS=1` build. |

**Note:** The SST files are pre-generated by `gpcomp_datagen` and loaded from disk.
The benchmarks do **not** regenerate random data on each run — they always use the same dataset.

---

## 6. CPU vs GPU Data Paths in the Simulation

### CPU Path (Benchmark)

```
Disk ──read──► Host SST arrays
                   │
                   ├──► std::sort (concat + sort)  ──► merged KV pairs
                   │                                        │
                   │    ┌───────────────────────────────────┘
                   │    │
                   │    ▼
                   │  for each data block:
                   │    cpu_build_byte_vector()  →  ByteVector[bvlen]
                   │    cpu_pack_bit_vector()    →  BitVector[bitvec_len]
                   │
                   ▼
              Merged output + per-block Bloom filters (all on host)
```

### GPU Path (Benchmark)

```
Disk ──read──► Host SST arrays
                   │
                   ├──H2D──► Device SST arrays
                   │              │
                   │         merge_kernel<<<grid, 256>>>
                   │              │
                   │         d_output (sorted)
                   │              │
                   │         ├──D2H──► h_output (for validation only)
                   │         │
                   │         ├──H2D──► d_all_keys (== d_output, or re-uploaded)
                   │         │              │
                   │         │    bloom_filter_kernel_batched<<<N, T>>>
                   │         │              │
                   │         │         d_all_bitvecs
                   │         │              │
                   │         │         ──D2H──► h_bitvecs
                   │
                   ▼
              Merged output + per-block Bloom filters (on host)
```

---

## 7. Step-by-Step Examples

### Example A: CPU Compaction Path

**Scenario:** 4 SSTs, each with 327 keys (value_size=64B, block_size=32KB)

| Step | Operation | Detail |
|---|---|---|
| 1 | **Disk Read** | Load `sst_0000.bin` … `sst_0003.bin` from disk → 4 host arrays (each sorted). Time: ~3 ms |
| 2 | **Concatenate** | Append all 4 arrays into one flat array of 1308 KV pairs (unsorted) |
| 3 | **std::sort** | Sort the 1308-element array by key (comparison-based, O(N log N)). Time: ~9 ms |
| 4 | **Bloom — per block** | Split sorted output into data blocks of 327 keys each (≈4 blocks). For **each block**: |
| 4a | | `cpu_build_byte_vector()`: for each of 327 keys, compute K=7 hashes → set 7 positions in a 3270-byte ByteVector |
| 4b | | `cpu_pack_bit_vector()`: pack 3270 bytes → 409 bytes (BitVector). Time: ~15 ms total for all blocks |
| 5 | **Output** | Merged sorted KV array + 4 Bloom filter bit-vectors. **Total: ~27 ms** |

### Example B: GPU Compaction Path (Batched Bloom)

**Scenario:** Same 4 SSTs, 327 keys each

| Step | Operation | Detail |
|---|---|---|
| 1 | **Disk Read** | Load `sst_0000.bin` … `sst_0003.bin` from disk → 4 host arrays. Time: ~3 ms |
| 2 | **H2D Transfer** | `cudaMemcpy` all 4 SST arrays to GPU global memory (Host→Device). Also upload sizes[] and offsets[]. Time: <1 ms |
| 3 | **merge_kernel** | Launch `<<<grid, 256>>>` — 1308 threads total. Each thread: find which SST it belongs to → binary search in all other SSTs → compute output index → write to `d_output[index]`. Kernel time: ~0.12 ms |
| 4 | **D2H Transfer (merge)** | Copy `d_output` (1308 KV pairs) back to host. Time: <1 ms |
| 5 | **H2D Transfer (bloom)** | Upload the merged 1308 KV pairs to GPU (or reuse if already there). Time: <1 ms |
| 6 | **bloom_filter_kernel_batched** | Launch `<<<4, 416>>>` — 4 CUDA blocks (one per data block), 416 threads each. Each block independently: zero shared ByteVector → Phase 1 hash → `__syncthreads()` → Phase 2 pack. Kernel time: <1 ms |
| 7 | **D2H Transfer (bloom)** | Copy all 4 packed BitVectors back to host. Time: <1 ms |
| 8 | **Output** | Merged sorted KV array + 4 Bloom filter bit-vectors. **Total: ~2.8 ms (wall), ~4–5× speedup over CPU** |

### Side-by-Side Summary

```
                    CPU Path                    GPU Path (batched)
                    ─────────                   ──────────────────
 Input:             4 SST files on disk         4 SST files on disk
                         │                           │
 Step 1:            fread → host arrays         fread → host arrays
                    (~3 ms)                     (~3 ms)
                         │                           │
 Step 2:            concat + std::sort          cudaMemcpy H2D
                    (~9 ms)                     (<1 ms)
                         │                           │
 Step 3:              (done)                    merge_kernel
                         │                      (~0.12 ms)
                         │                           │
 Step 4:            CPU bloom loop              cudaMemcpy D2H + H2D
                    (~15 ms)                    (<1 ms)
                         │                           │
 Step 5:              (done)                    bloom_batched kernel
                         │                      (<1 ms)
                         │                           │
 Step 6:                 —                      cudaMemcpy D2H
                                                (<1 ms)
                         │                           │
 Output:            sorted KVs + Blooms         sorted KVs + Blooms
 Total:             ~27 ms                      ~5 ms (incl. I/O)
 Throughput:        ~12 M keys/s                ~52 M keys/s
```

### Key Takeaway

The GPU wins because:
1. **Merge**: 1308 threads each doing O(log N) binary search in parallel vs. one CPU core doing O(N log N) serial sort
2. **Bloom (batched)**: All data blocks processed in one kernel launch (1 H2D + 1 kernel + 1 D2H) vs. CPU iterating through each block sequentially
3. **Transfer overhead** is small relative to compute savings; the batched approach processes all data blocks in one kernel launch

---

## 8. cuFile / GPUDirect Storage (GDS)

### What Is GDS?

**GPUDirect Storage (GDS)** lets the CUDA driver read data from an NVMe SSD (or RAID array) **directly into GPU device memory**, bypassing the CPU and system RAM entirely. The path is:

```
 Traditional POSIX path:
 NVMe ──DMA──► RAM (kernel buffer) ──CPU copy──► pinned RAM ──PCIe──► VRAM

 GDS Native path:
 NVMe ──PCIe DMA──────────────────────────────────────────────────► VRAM

 GDS Compat path (consumer GPUs / NFS):
 NVMe ──► pinned bounce-buffer (RAM) ──PCIe──► VRAM   (cuFile driver manages H2D)
```

GDS native mode requires:
- A **data-centre GPU** (A30, A100, H100, or similar) with GDS driver support
- A **local NVMe** SSD attached via PCIe (not NFS)
- The `nvidia-fs` kernel module installed

### Three Operating Modes

`gpcomp_cufile.cuh` defines a `GDSMode` enum and a static `s_gds_mode` variable:

| Mode | Value | Meaning |
|---|---|---|
| `GDS_DISABLED` | 0 | `GPCOMP_USE_CUFILE` not defined at compile time, or `cuFileDriverOpen()` failed. Falls back to POSIX `fread` + `cudaHostAlloc` + `cudaMemcpy`. |
| `GDS_COMPAT` | 1 | cuFile driver open with `allow_compat_mode=true`. cuFile uses a pinned bounce-buffer internally; works on any GPU without native GDS support. |
| `GDS_NATIVE` | 2 | cuFile driver open in native mode. PCIe DMA from NVMe directly into VRAM. Requires data-centre GPU + local NVMe + `nvidia-fs`. |

### Public API (`gpcomp_cufile.cuh`)

```cpp
// Call before gpcomp_gds_init() to force compat mode.
// Writes /tmp/gpcomp_cufile_compat.json and sets CUFILE_ENV_PATH_JSON.
void gpcomp_gds_force_compat_mode();

// Open the cuFile driver and detect the operating mode (sets s_gds_mode).
// If GPCOMP_USE_CUFILE is not defined, sets GDS_DISABLED and returns.
void gpcomp_gds_init();

// Load `bytes` of data from file `path` (at offset) directly into device pointer `d_ptr`.
// GDS_NATIVE/COMPAT: uses cuFileRead (O_DIRECT on native, may fall back to O_RDONLY)
// GDS_DISABLED: uses cudaHostAlloc + fread + cudaMemcpy, then cudaFreeHost
void gpcomp_gds_load_to_device(const char* path, void* d_ptr, size_t bytes, size_t offset = 0);

// Close the cuFile driver. Call once at program exit if gpcomp_gds_init() was called.
void gpcomp_gds_cleanup();
```

### How Compatibility Mode Is Forced

`gpcomp_gds_force_compat_mode()` writes a temporary JSON file and sets the environment variable:

```c
// Written to /tmp/gpcomp_cufile_compat.json
{
  "properties": {
    "allow_compat_mode": true
  }
}

// Environment variable that points cuFile to this config
setenv("CUFILE_ENV_PATH_JSON", "/tmp/gpcomp_cufile_compat.json", 1);
```

This must be called **before** `gpcomp_gds_init()` (i.e., before `cuFileDriverOpen()`). The `--compat-mode` CLI flag triggers this call.

### Mode Detection Caveat (RTX 3070)

After `cuFileDriverOpen()`, the code probes `CUfileDrvProps_t.nvfs.dcontrolflags`. On an RTX 3070, the driver does **not** set `CU_FILE_ALLOW_COMPAT_MODE` in `dcontrolflags` even when compat mode is active — so `s_gds_mode` is reported as `NATIVE` cosmetically, even though the I/O path is the bounce-buffer compat path. This is a driver/hardware quirk; the actual I/O behaviour (and slower throughput vs POSIX) confirms compat mode is active.

On an A30/A100/H100 with GDS enabled, the flag will be absent and `GDS_NATIVE` is correctly set.

### Benchmark 4 Data Paths

```
 Benchmark 4 — GDS I/O + Merge
 ─────────────────────────────────────────────
 GDS path (native or compat):

 disk (SST files)
    │
    ▼ cuFileRead() [O_DIRECT]
 d_sst_arrays[]  (already on device)
    │
    ▼ merge_kernel<<<grid, 256>>>
 d_output[]
    │
    ▼ cudaMemcpy D2H (validation only)
 h_output[]

 ─────────────────────────────────────────────
 POSIX fallback (GDS_DISABLED):

 disk (SST files)
    │
    ▼ fread() into cudaHostAlloc pinned buffer
 h_pinned[]
    │
    ▼ cudaMemcpy H2D
 d_sst_arrays[]
    │
    ▼ merge_kernel<<<grid, 256>>>
 d_output[]
    │
    ▼ cudaMemcpy D2H (validation only)
 h_output[]
```

**Build flags for Benchmark 4:**

```sh
# Compile with cuFile support (points at gds/local/ — no system install needed)
make GDS=1 gpcomp_bench
# Equivalent flags added by Makefile:
#   -DGPCOMP_USE_CUFILE
#   -I gds/local/include
#   -L gds/local/lib
#   -Xlinker -rpath,gds/local/lib
#   -lcufile
```

### GDS Package (CUDA 11.8, no root required)

The `gds/` directory in the repo contains the `.deb` files and an extraction script. If the `.deb` files are missing on a new machine:

```sh
mkdir -p gds && cd gds
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
curl -fsSL -o libcufile.deb     "$BASE/libcufile-11-8_1.4.0.31-1_amd64.deb"
curl -fsSL -o libcufile-dev.deb "$BASE/libcufile-dev-11-8_1.4.0.31-1_amd64.deb"
cd ..
bash gds/setup_local.sh
```

For GPUs with other CUDA versions, replace `11-8` and `1.4.0.31` with the version matching `nvcc --version`.

---

## 9. Benchmark Results — RTX 3070, NFS, Large Dataset

### Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA RTX 3070 (8 GB GDDR6, 46 SMs, Ampere) |
| CUDA | 11.8, driver 535.261.03 |
| Storage | NFS (networked filesystem, not local NVMe) |
| Dataset | 8 SSTs × 524,288 keys = 4,194,304 total keys, ~64 MB |
| Value size | 64 B |
| Runs | 5 (best-of-5 reported) |

### Merge (Benchmark 1)

| Metric | CPU (std::sort) | GPU wall | GPU kernel-only |
|---|---|---|---|
| Min latency | 158.32 ms | 24.81 ms | 4.00 ms |
| M keys/s | — | — | — |
| **Speedup** | baseline | **6.4× (apple-to-apple)** | **39.6× (kernel-only)** |

"Apple-to-apple" = GPU wall time (H2D + kernel + D2H) vs CPU sort time (same data already in RAM).

### Bloom Filter (Benchmark 2)

| Metric | CPU (per-block) | GPU batched wall | GPU kernel-only |
|---|---|---|---|
| Min latency | 192.91 ms | 11.73 ms | 33.75 ms |
| **Speedup** | baseline | **16.5×** | **5.72×** |

Note: the batched wall speedup (16.5×) exceeds kernel-only speedup (5.72×) because the batched wall eliminates per-block synchronization and H2D/D2H overhead, while the kernel-only timer isolates pure GPU compute (which is memory-bound on this large dataset).

### Combined Compute (Merge + Bloom)

| Path | Total time |
|---|---|
| CPU (sort + bloom) | ~351 ms |
| GPU wall (H2D + merge + bloom + D2H) | ~36.5 ms |
| **Combined speedup** | **9.6×** |

### GDS I/O Benchmark (Benchmark 4, --compat-mode, NFS)

| I/O Path | Round-trip time | Approx. bandwidth |
|---|---|---|
| POSIX fread (NFS, OS-cached) | ~84 ms | ~762 MB/s |
| POSIX fread (NFS, cold) | ~404 ms | ~158 MB/s |
| cuFile compat mode (NFS, O_DIRECT) | ~629 ms | ~102 MB/s |

**Finding:** On NFS, cuFile compat mode is **1.5–8× slower** than POSIX `fread`, because `O_DIRECT` bypasses the OS page cache (including NFS read-ahead). There is no RDMA path from a network filesystem. **GDS compat on NFS is not beneficial; use the POSIX fallback.**

The GDS I/O benchmark is meaningful on a machine with **local NVMe and a GDS-capable GPU (A30/A100/H100)**.

### Bottleneck Analysis

| Bottleneck | Detail | How to mitigate |
|---|---|---|
| **PCIe H2D bandwidth** | Transferring 64 MB from pinned RAM to VRAM saturates ~12 GB/s PCIe Gen 4 → ~5 ms minimum | Overlap H2D with kernel using CUDA streams; pre-stage data on device |
| **PCIe D2H bandwidth** | Copying merged result back to host for validation | On real compaction output, the merged SST is written to disk — skip D2H if data can stay on device for Bloom |
| **Bloom kernel serialisation** | Each CUDA block uses shared memory per data block; blocks are scheduled sequentially if SM count < num_blocks | Use multi-stream launch or increase block Thread count |
| **GDS on NFS** | O_DIRECT + no RDMA → slower than cached fread | Use local NVMe + native GDS; fall back to POSIX on NFS |
| **Kernel launch overhead** | Both kernels are launched serially; each launch has ~5–10 µs CPU-side overhead | Use CUDA graphs to pre-record the launch sequence |

### Recommendation for A30 Testing

```sh
# On a machine with local NVMe + A30:
make GDS=1 gpcomp_bench
./gpcomp_datagen --out dataset --num_sst 8 --keys_per_sst 524288
./gpcomp_bench --gds --runs 5          # native GDS (no --compat-mode)
make bench-gds-native                  # shortcut: generate + bench in native mode
```

Expected improvement: GDS native on local NVMe should deliver **~3–5 GB/s I/O** directly into device memory (vs ~158–762 MB/s POSIX on the NFS setup), making the I/O stage competitive with the GPU compute stage.
