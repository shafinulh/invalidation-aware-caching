# GPU-Accelerated RocksDB Compaction — How It Works

## Table of Contents

1. [What Does the GPU Accelerate?](#1-what-does-the-gpu-accelerate)
2. [Merge vs Bloom: What Are They and How Are They Different?](#2-merge-vs-bloom-what-are-they-and-how-are-they-different)
3. [When Does the GPU Kick In?](#3-when-does-the-gpu-kick-in)
4. [Data Paths](#4-data-paths)
5. [Simulation Testbench Overview](#5-simulation-testbench-overview)
6. [CPU vs GPU Data Paths in the Simulation](#6-cpu-vs-gpu-data-paths-in-the-simulation)
7. [Step-by-Step Examples](#7-step-by-step-examples)

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

### The 3 benchmarks:

| Benchmark | What it measures |
|---|---|
| **1 — Merge** | CPU sort vs GPU merge kernel (kernel-only and with H2D/D2H transfers) |
| **2 — Bloom** | CPU bloom vs GPU bloom (kernel-only and batched) |
| **3 — fillrandom simulation** | Scales per-round CPU and GPU timings across N compaction rounds (requires `--fillrandom_keys`). Shows aggregate CPU total vs GPU total, speedup, and time saved |

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
