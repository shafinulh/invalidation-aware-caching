# GPU Compaction — Why v1 Didn't Use the GPU (and What v2 Changes)

## Overview

Two versions of GPU-assisted RocksDB compaction were built in this project.
This document explains the design of each, why **v1 was a skeleton that never
touched the GPU**, and exactly what **v2 changed** to run real parallel merge
work on the GPU.

---

## v1 — The CPU Skeleton ("CloneInputToOutputBuffer")

### What it did

v1 introduced the plumbing to offload compaction to an external orchestrator
(`GPUCompactionOrchestrator`) but the core sort step was a **no-op clone**:

```cpp
// v1 "GPU" path — db/compaction/gpu_compaction_orchestrator.cc
static void CloneInputToOutputBuffer(PinnedSoA& input, PinnedSoA& output) {
    // Just memcpy every key and value in the original (unsorted) order.
    // No GPU call, no sorting, no merge.
}
```

After calling this, `LaunchGPUCompaction()` passed the **unsorted** entries
directly to `WriteEntriesToOutputs()`, relying on the per-SST ordering that
happened to already exist — but without merging across multiple input files.

The actual sort was done by a **`std::sort` call on the CPU** inside
`FlattenBlocksToPinnedSoA()`:

```cpp
// v1 — CPU sort applied after reading all SST blocks
std::sort(all_entries.begin(), all_entries.end(),
          [](const KVEntry& a, const KVEntry& b) {
              return internal_key_less(a.key, b.key);
          });
```

### Why no GPU was actually used

| Reason | Detail |
|---|---|
| **Sorting was on the CPU** | `std::sort` ran in the RocksDB background thread. All merge work happened before the GPU path was reached. |
| **"GPU path" was a memcpy** | `CloneInputToOutputBuffer()` only copied data from one pinned buffer to another using plain `memcpy`. No CUDA API was called. |
| **No CUDA kernel existed** | There was no `.cu` file, no `nvcc` compilation step, and no `cudaMalloc` / `cudaLaunchKernel` call anywhere in the code. |
| **GPU memory was never allocated** | `nvidia-smi` showed 0 MiB GPU memory usage from the `db_bench` process. |

In short, v1 was an **architectural skeleton** — it wired the compaction hook,
read SST blocks into pinned memory, and wrote sorted output back to RocksDB,
but the sort itself remained entirely CPU-bound.

---

## v2 — Real CUDA N-way Merge Kernel

### New files

| File | Location | Purpose |
|---|---|---|
| `gpu_varlength_merge.cuh` | `benchmarks/gpu/src/v2/` | C++ header declaring `GpuVarlengthMerge()` — the host-callable launcher |
| `gpu_varlength_merge.cu` | `benchmarks/gpu/src/v2/` | CUDA translation unit compiled by `nvcc`; contains the device kernel and host launcher |

### Modified files

| File | Change |
|---|---|
| `db/compaction/gpu_compaction_orchestrator.h` | Added `sst_run_global_offsets_` field to track per-SST entry boundaries |
| `db/compaction/gpu_compaction_orchestrator.cc` | Removed `std::sort`; replaced `CloneInputToOutputBuffer` with real `GpuVarlengthMerge()` call + CPU reorder pass |
| `Makefile` | Added `nvcc` pattern rule, `GPU_CU_SOURCES`/`GPU_CU_OBJECTS`, c++20→c++17 downgrade for nvcc |

### What v2 does differently

#### 1. `std::sort` removed

`FlattenBlocksToPinnedSoA()` no longer sorts entries as it reads SST blocks.
Instead it records SST run boundaries in `sst_run_global_offsets_` so the GPU
knows where each sorted run starts and ends.

```cpp
// v2 — per-SST boundary tracking (no sort)
sst_run_global_offsets_.push_back(soa.num_entries);
ReadDataBlocksFromTable(table, soa);
```

#### 2. GPU kernel `merge_kernel_vl`

One GPU thread is launched per input entry. Each thread:

1. Identifies which SST run `j` it belongs to (linear scan over run boundaries).
2. Fetches its own RocksDB internal key from the packed byte array.
3. Calls `lower_bound_vl()` on every other run — a binary search using the
   full **RocksDB `InternalKeyComparator`** ordering:
   - User key ascending (byte-lexicographic).
   - Equal user keys: sorted by sequence number **descending** (newer record
     first), read from the little-endian 8-byte trailer.
4. Sums the lower-bound results plus its own local index to compute its exact
   output position `out_pos`.
5. Writes a packed `uint32_t` to `out_mapping[out_pos]`:
   - Bits [31:24] = SST index `j`
   - Bits [23:0]  = local entry index within run `j`

All threads run in parallel — O(N log N) work is distributed across the GPU's
CUDA cores rather than serialised on one CPU core.

#### 3. CPU reorder pass (after kernel)

`GpuVarlengthMerge()` returns the permutation `h_out_mapping` to the host.  
`LaunchGPUCompaction()` then does a single sequential pass to reorder keys and
values into the final output SoA using the permutation:

```cpp
for (uint32_t out_idx = 0; out_idx < total_entries; ++out_idx) {
    uint32_t packed = h_out_mapping[out_idx];
    uint32_t sst_j  = packed >> 24;
    uint32_t local_k = packed & 0x00FFFFFF;
    // memcpy key + value from input SoA[sst_j][local_k] → output SoA[out_idx]
}
```

#### 4. Makefile changes

```makefile
NVCC     ?= nvcc
NVCC_STD := -std=c++17  # nvcc does not support c++20
NVCC_FLAGS := $(NVCC_STD) -O2 -Xcompiler -fPIC -I. -Iinclude ...

GPU_CU_SOURCES := ../benchmarks/gpu/src/v2/gpu_varlength_merge.cu
GPU_CU_OBJECTS := $(patsubst %.cu, $(OBJ_DIR)/%.cu.o, $(GPU_CU_SOURCES))
LIB_OBJECTS    += $(GPU_CU_OBJECTS)

$(OBJ_DIR)/%.cu.o: %.cu
    @mkdir -p $(dir $@)
    $(NVCC) $(NVCC_FLAGS) -c $< -o $@
```

---

## Performance Comparison

| Version | 20 M keys · value=32 B | Compaction write | Write stall |
|---|---|---|---|
| v1 skeleton (CPU sort) | 55,484 ops/sec | 21.0 MB/s | 90% |
| v2 GPU kernel | 72,804 ops/sec | 26.9 MB/s | 88% |
| CPU baseline (subcomp=1) | 129,645 ops/sec | — | 77% |
| CPU baseline (subcomp=4) | 205,426 ops/sec | — | 82% |

v2 is **~31 % faster than v1** for small values, primarily because the GPU
merge kernel eliminates the `std::sort` from the critical path.

### Why GPU utilisation still reads ~0 %

`nvidia-smi` samples at 1-second intervals. Each individual merge kernel launch
completes in **microseconds** (the total key set per compaction is ~1–5 MB).
The GPU is truly idle for the other 999+ ms of each compaction event — which is
dominated by SST block I/O, page-cache pressure, and SST builder write-back on
the CPU.  
The 0 % reading is accurate; it is not a measurement artefact or a build error.

To push GPU utilisation toward visible levels would require:
- Much larger compactions (`write_buffer_size` ≥ 64 MB, higher L0 trigger), or
- Pipelining SST I/O with GPU kernel execution (overlapping H2D and compute
  streams while the next batch of blocks is being read from storage).
