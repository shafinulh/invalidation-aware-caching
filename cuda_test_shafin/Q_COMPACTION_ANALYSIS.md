# Q-Compaction: Paper vs. Implementation Analysis

A thorough analysis of the GPU-accelerated Q-Compaction strategy from
**"A GPU-accelerated Compaction Strategy for LSM-based Key-Value Store System"**
(Zhou et al., MSST '24) and its implementation under `cuda_test_shafin/`.

---

## Table of Contents

1. [Paper Overview](#1-paper-overview)
2. [Kernel-by-Kernel Analysis](#2-kernel-by-kernel-analysis)
   - 2.1 [Unpack Unit (Parse)](#21-unpack-unit)
   - 2.2 [Sort Unit (Merge)](#22-sort-unit-merge)
   - 2.3 [Pack Unit (Generate SST)](#23-pack-unit)
   - 2.4 [Bloom Filter Generation](#24-bloom-filter-generation)
   - 2.5 [Supporting GPU Kernels](#25-supporting-gpu-kernels)
3. [End-to-End Pipeline Data Flow](#3-end-to-end-pipeline-data-flow)
4. [Kernel Time vs. Transfer Overhead](#4-kernel-time-vs-transfer-overhead)
5. [Concurrency Dimensions](#5-concurrency-dimensions)
6. [Paper Correspondence Matrix](#6-paper-correspondence-matrix)
7. [Deviations and Extensions](#7-deviations-and-extensions)
8. [Benchmark Results](#8-benchmark-results)

---

## 1. Paper Overview

The paper designs three GPU Compaction Units for LSM-tree compaction:

> *"We design efficient GPU compaction units for each stage of compaction, including units for parsing key-value pairs from SST files, parallel sorting of key-value pairs, and generating new SST files (encompassing data blocks, the index block, and the Bloom filter block)."*

Q-Compaction is the upper-level strategy (L0->L1, L1->L2):

> *"Q-Compaction is designed for upper-level compaction tasks, specifically L0 to L1 and L1 to L2. [...] we leverage the GPU to accelerate all processes involved in these compaction tasks, including parsing key-value pairs, sorting, and generating SST files."*

> *"To avoid impacting the performance of Q-Compaction, we deliberately exclude garbage collection for expired and deleted key-value pairs within Q-Compaction."*

The paper also describes a **Pipeline mechanism** using CUDA streams:

> *"In the pipeline mechanism, we leverage CUDA streams to exploit the parallelism of three operations: disk I/O, main memory-GPU memory copy, and GPU computation."*

And a **lazy allocation strategy**:

> *"We adopt a lazy allocation strategy -- allocating all GPU resources during key-value system initialization and deallocating them upon system closure."*

---

## 2. Kernel-by-Kernel Analysis

### 2.1 Unpack Unit

#### Paper Description

> *"The SST file comprises multiple independent data blocks, each containing several distinct groups (KV groups). Within each KV group, adjacent keys share the same prefix. Consequently, the KV group serves as the smallest independent task unit."*

> *"We employ a GPU thread to parse a KV group, with a one-dimensional thread block assigned to parsing a data block. A thread grid, comprising multiple one-dimensional thread blocks, concurrently parses key-value pairs in an SST file."*

> *"To enhance the concurrency of unpack operations, CUDA Streams are utilized to launch multiple GPU thread grids, enabling the parsing of key-value pairs in multiple SST files simultaneously and achieving grid-level parallelism."*

**CUDA Hierarchy (Paper):**
| Level | Maps To |
|-------|---------|
| GPU Thread | One KV Group (restart interval) |
| Thread Block | One Data Block |
| Thread Grid | One SST File |
| CUDA Streams | Cross-SST parallelism |

#### Implementation (`gpcomp_pack.cuh:684-735`, `gpcomp_pipeline.cuh:532-596`)

**Kernel:** `unpack_kernel`

```
Grid:  num_blocks (one block per data block in SST)
Block: ceil(max_restarts / 32) * 32 threads
```

Each GPU thread handles one KV group (restart interval = 4 KV pairs). The thread reads the restart offset from the block trailer, then sequentially decodes `shared_prefix_len`, `unshared_len`, key delta, and value for each KV in the group. The thread reconstructs full keys using prefix from the previous key within the group.

**Stream usage:** One non-blocking CUDA stream per input SST file (`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`). Each stream performs:
1. `cudaMemcpyAsync` of the SST file bytes H2D
2. `cudaMemcpyAsync` of block offsets, first_kv, num_kv arrays H2D
3. `unpack_kernel` launch on that stream
4. All streams run concurrently, synchronized with `cudaStreamSynchronize`

**Identical to paper:** Yes. One thread per KV group, one thread block per data block, one grid per SST file, CUDA streams for cross-SST parallelism. The implementation uses `cudaStreamNonBlocking` for true concurrent execution across streams.

---

### 2.2 Sort Unit (Merge)

#### Paper Description

The paper presents **Algorithm 1: Merge multiple ordered key-value pair arrays**:

> *"We assign a GPU thread to calculate the corresponding index of each key-value pair in the final key-value pair array, establishing a one-to-one correspondence between the key-value pair and the GPU thread."*

> *"For the key-value pair KV^j_idx belonging to SST_j, its index in SST_j is the GPU thread's index (i.e., idx). We use binary search to obtain the indexes I^0_idx, I^1_idx, ..., I^(j-1)_idx, I^(j+1)_idx, ..., I^n_idx when KV^j_idx is inserted into other key-value pair arrays [...] respectively. In this case, the index of KV^j_idx in the final array is I_Array = sum of I^j_idx."*

> *"Binary search inherently exhibits parallelism."*

**Algorithm pseudocode from paper:**
```
for each GPU thread do
    idx = blockIdx.x * blockDim.x + threadIdx.x
    j = Current array Id
    KV^j_idx = SST_j[idx]
    I_Array = 0
    for i = 0 to n do
        if (i == j) then I = idx
        else I = BinarySearch(KV^j_idx, SST_i)
        I_Array += I
    output[I_Array] = KV^j_idx
```

#### Implementation (`gpcomp_merge.cuh:35-60`)

**Kernel:** `merge_kernel`

```
Grid:  ceil(total_kv / 256)
Block: 256 threads
```

One thread per KV pair across all input SSTs. Each thread:
1. Determines which SST array it belongs to (linear scan of prefix-sum offsets)
2. Reads its KV pair from that array
3. For each of the N arrays, computes rank via binary search (or uses local index for own array)
4. Sums all ranks to get `final_idx`
5. Writes `output[final_idx] = pair`

**Launched via:** `launch_merge_timed_from_device` with `copy_to_host = false` -- merged data stays on device for subsequent stages.

**Identical to paper:** Yes. This is a direct implementation of Algorithm 1. One thread per KV pair, binary search for rank computation across all arrays, output written to final position.

---

### 2.3 Pack Unit

#### Paper Description

> *"Pack Unit primarily focuses on the parallel generation of these components -- data blocks, the index block, and the Bloom filter block."*

For data blocks:

> *"The process of generating a KV group involves each GPU thread obtaining the corresponding KV pairs as input according to the restart point interval and performing prefix compression on adjacent keys. A one-dimensional thread block can generate a data block, and a thread grid comprising multiple one-dimensional thread blocks can concurrently generate all data blocks of an SST file."*

> *"Similarly, we use multiple CUDA Streams to launch multiple thread grids, enabling the concurrent generation of data blocks in multiple SST files."*

For index blocks:

> *"Each GPU thread calculates the largest key in the corresponding data block and encodes its offset and length into the value using variable-length encoding, thereby generating the index block."*

**CUDA Hierarchy (Paper):**
| Level | Maps To |
|-------|---------|
| GPU Thread | One KV Group (prefix-compress) |
| Thread Block | One Data Block |
| Thread Grid | One output SST File |
| CUDA Streams | Cross-output-SST parallelism |

#### Implementation (`gpcomp_pack.cuh:534-618`, `gpcomp_pipeline.cuh:647-661`)

**Kernel:** `pack_kernel`

```
Grid:  block_count (blocks in this output SST span)
Block: ceil(max_restarts / 32) * 32 threads
Shared mem: max_restarts * 3 * sizeof(uint32_t)
```

The kernel operates in three phases within each thread block:

**Phase 1 (parallel):** Each thread computes the byte size of its KV group by iterating over the group's KV pairs, computing `key_shared_prefix` for adjacent keys, and summing `2 + (key_bytes - shared) + value_bytes` per entry. Stored in `interval_bytes[tid]`.

**Phase 2 (serial, thread 0 only):** Thread 0 computes prefix sums of group sizes to get `interval_offsets` and `restart_offsets`. Writes the `PackBlockHeader` and restart offset array to output. Records `block_sizes[block_id]`.

**Phase 3 (parallel):** Each thread prefix-compresses its KV group into the output buffer at the computed offset. Writes `shared_len | unshared_len | key_delta | value` for each KV.

**Stream usage:** One non-blocking CUDA stream per output SST file span. The pack spans are computed by `partition_output_blocks` which targets ~8MB per output file. Each span's stream runs:
1. `pack_kernel` launch
2. Async D2H copy of `block_sizes` (pinned memory via `cudaHostAlloc`)
3. Async D2H copy of packed block data (pinned memory)

**Identical to paper:** Yes for the kernel structure (thread -> KV group, block -> data block, grid -> output SST). The implementation adds per-span CUDA streams with non-blocking flags and pinned host memory for async D2H transfer, which matches the paper's description of using CUDA streams for cross-SST parallelism.

---

### 2.4 Bloom Filter Generation

#### Paper Description

The paper presents **Algorithm 2: Generate Bloom Filter block**:

> *"We utilize a GPU thread to calculate the K hash values of the corresponding key, obtain the hash positions, and store the results in a byte vector ByteVector."*

> *"Here, a byte, rather than a bit, represents a Boolean value (i.e., 0 and 1). In this case, we don't need to adopt a GPU thread synchronization strategy. Even if multiple GPU threads modify the content of the same byte, there will be no conflict since they write the same value (i.e., 1)."*

> *"Subsequently, to conserve storage space, we use one-eighth of the GPU threads to convert the byte vector ByteVector to a bit vector BitVector."*

> *"Generating the Bloom filter block in an SST file is also a highly concurrent task because the calculation of the hash functions for each key is independent. A one-dimensional thread block is responsible for generating a Bloom Filter block for all key-value pairs in a data block."*

**Algorithm pseudocode from paper:**
```
for each GPU thread do
    idx = threadIdx.x
    for i = 1 to K do
        h = Bloomhash(Array[idx])
        bytepos = h % ByteVector_Len
        ByteVector[bytepos] = 1
    _syncthreads()
    BitVector_Len = (ByteVector_Len + 7) / 8
    if (idx < BitVector_Len) then
        base = idx * 8
        for j = 0 to 7 do
            bytepos = base + j
            if (ByteVector[bytepos] == 1) then
                BitVector[idx] |= (1 << j)
```

**CUDA Hierarchy (Paper):**
| Level | Maps To |
|-------|---------|
| GPU Thread | One key (hash) / one byte->bit pack |
| Thread Block | One Data Block's bloom filter |

#### Implementation (`gpcomp_bloom.cuh:56-99`, `gpcomp_pipeline.cuh:639-644`)

**Kernel:** `bloom_filter_kernel_batched`

```
Grid:  num_blocks (ALL output data blocks in a single launch)
Block: ceil(max(max_num_kv, max_bitvec_len) / 32) * 32
Shared mem: max_byte_vector_len bytes (ByteVector in shared memory)
```

This is a **batched** version: a single kernel launch covers all output data blocks, with `blockIdx.x` selecting the data block. Each thread block:

1. Zeroes its `byte_vector` in shared memory
2. `__syncthreads()`
3. Each thread hashes its key K=7 times, writing `byte_vector[h % len] = 1`
4. `__syncthreads()`
5. First `bitvec_len` threads pack 8 bytes into 1 bit each, writing to global `all_bitvecs`

**Launched via:** `launch_bloom_filter_batched_from_device_plans` which:
1. Runs `bloom_filter_layout_kernel` (single-thread kernel) on device to compute bitvec offsets/lengths
2. Copies layout info back to host
3. Allocates device filter buffer
4. Launches `bloom_filter_kernel_batched` with CUDA events for timing
5. Copies filter bytes D2H

**Identical to paper:** Yes. The byte-vector-then-bit-vector approach exactly matches Algorithm 2. The batched launch (one grid for ALL blocks) is an extension beyond what the paper explicitly describes but does not contradict it. The paper says "a one-dimensional thread block is responsible for generating a Bloom Filter block for all key-value pairs in a data block" -- the batched kernel satisfies this with `blockIdx.x` selecting the data block.

---

### 2.5 Supporting GPU Kernels

These kernels are not explicitly described in the paper but are necessary for the implementation:

#### `compute_restart_group_sizes_kernel` (`gpcomp_pack.cuh:200-218`)

One thread per restart group. Computes the serialized byte size of a KV group by iterating over the group's KV pairs and computing shared prefix lengths. Used during the Planning stage to determine data block boundaries without transferring KV data back to host.

```
Grid:  ceil(num_groups / 256)
Block: 256
```

#### `gather_largest_keys_kernel` (`gpcomp_pipeline.cuh:63-73`)

One thread per data block. Reads the last KV pair in each block to extract the largest key (needed for index block construction). Avoids transferring the full merged KV array back to host.

```
Grid:  ceil(num_blocks / 256)
Block: 256
```

#### `bloom_filter_layout_kernel` (`gpcomp_bloom.cuh:101-118`)

Single-thread kernel that computes prefix-sum of bitvec sizes on device, so the batched bloom kernel knows where to write each block's filter output. Avoids a D2H-compute-H2D round-trip.

```
Grid:  1
Block: 1
```

#### `compute_adjacent_shared_prefix_kernel` (`gpcomp_pack.cuh:187-198`)

One thread per KV pair. Computes the shared prefix length between adjacent keys. Used by the `q_pipeline` variant for on-device planning but NOT used in the `q_paper` variant.

```
Grid:  ceil(total_kv / 256)
Block: 256
```

#### `plan_data_blocks_exact_kernel` (`gpcomp_pack.cuh:246-283`)

Single-thread kernel that greedily partitions KV pairs into data blocks on device. Sequential by nature (each block boundary depends on the previous). Used in `q_pipeline` variant only.

```
Grid:  1
Block: 1
```

#### `precompress_restart_groups_kernel` (`gpcomp_pack.cuh:220-244`)

One thread per restart group. Pre-compresses (prefix encodes) KV groups into a flat buffer on device. Used in `q_paper_overlap` variant to overlap compression with planning. Each thread writes `shared | unshared | key_delta | value` for its group.

```
Grid:  ceil(num_groups / 256)
Block: 256
```

#### `assemble_precompressed_blocks_kernel` (`gpcomp_pack.cuh:620-682`)

Like `pack_kernel` but reads from pre-compressed group payloads instead of raw KV pairs. Used in `q_paper_overlap` variant. Thread 0 assembles the block header; other threads copy pre-compressed group data to the correct offset.

---

## 3. End-to-End Pipeline Data Flow

### The `gpu_q_compaction_paper_from_parsed` function (`gpcomp_pipeline.cuh:532-667`)

This is the primary Q-Compaction implementation matching the paper. Here is the complete data flow:

```
INPUT: 4 parsed SST files (already in host memory as file_bytes)

=== STAGE 1: UNPACK  (3.93 ms wall, 1.98 ms kernel) ============================

For each input SST (i = 0..3), on its own non-blocking CUDA stream:
  [HOST]  file_bytes[i]  ─── cudaMemcpyAsync H2D ──>  [DEVICE] d_buf[i]
  [HOST]  block_offsets[i] ── cudaMemcpyAsync H2D ──>  [DEVICE] d_offsets[i]
  [HOST]  first_kv[i]  ───── cudaMemcpyAsync H2D ──>  [DEVICE] d_first_kv[i]
  [HOST]  num_kv[i]  ─────── cudaMemcpyAsync H2D ──>  [DEVICE] d_num_kv[i]

  unpack_kernel<<<num_blocks_i, block_dim, 0, stream_i>>>
    [DEVICE] d_buf[i] ──> [DEVICE] d_out[i]  (unpacked KV pairs)

All 4 streams run concurrently (non-blocking).
cudaStreamSynchronize on each stream to wait for completion.

Data on device after stage 1:
  d_out[0..3]: 4 arrays of unpacked KVPair structs

=== STAGE 2: SORT/MERGE  (2.04 ms wall, 1.97 ms kernel) ========================

Single merge kernel launch on default stream:
  merge_kernel<<<ceil(total_kv/256), 256>>>
    [DEVICE] d_out[0..3] (via pointer array) ──> [DEVICE] d_merged_output

Inputs are on device from Stage 1.  Output stays on device.
copy_to_host = false (no D2H transfer).

Data on device after stage 2:
  d_merged_output: single sorted array of ~200K KVPairs
  (d_out[0..3] freed after merge)

=== STAGE 3: PLANNING  (1.68 ms wall) ==========================================

  compute_restart_group_sizes_kernel<<<ceil(num_groups/256), 256>>>
    [DEVICE] d_merged_output ──> [DEVICE] d_group_sizes

  cudaMemcpy D2H: d_group_sizes ──> host group_sizes vector

  [CPU] plan_data_blocks_group_aligned_from_group_sizes(group_sizes)
    Sequential greedy bin-packing: assign KV groups to data blocks
    such that each block <= 32KB.  Produces 992 DataBlockPlanEntry.

  upload_plans_to_device(plans):
    cudaMemcpy H2D: first_kv[], num_kv[] ──> d_first_kv, d_num_kv

  [CPU] partition_output_blocks():
    Partition 992 blocks into ~4 output SST files (each ~8MB target).
    Produces pack_spans: [(0,249), (249,498), (498,745), (745,992)]

Data on device after stage 3:
  d_merged_output (unchanged), d_first_kv, d_num_kv (plan arrays)

=== STAGE 4: BLOOM  (0.50 ms wall, 0.09 ms kernel) =============================

  bloom_filter_layout_kernel<<<1, 1>>>
    [DEVICE] d_num_kv ──> [DEVICE] d_bitvec_offsets, d_bitvec_lengths, d_total_bytes

  cudaMemcpy D2H: d_total_bytes ──> host total_bytes

  cudaMalloc: d_filter (total_bytes for all bitvectors)

  bloom_filter_kernel_batched<<<992, block_dim, shared_mem>>>
    [DEVICE] d_merged_output + d_first_kv + d_num_kv + d_bitvec_offsets
      ──> [DEVICE] d_filter

  cudaMemcpy D2H: d_filter ──> host filter_bytes
  cudaMemcpy D2H: d_bitvec_offsets, d_bitvec_lengths ──> host vectors

Data on host after stage 4:
  filter_bytes, bitvec_offsets, bitvec_lengths (bloom data for all 992 blocks)

=== STAGE 5: PACK + ASSEMBLE  (27.44 ms wall, 1.29 ms kernel) ==================

For each output SST span (s = 0..3), on its own non-blocking CUDA stream:
  cudaMalloc: d_blocks[s], d_sizes[s]
  cudaHostAlloc: h_raw[s], h_sizes[s]  (pinned host memory)

  pack_kernel<<<span_block_count, block_dim, shared_bytes, stream_s>>>
    [DEVICE] d_merged_output + d_first_kv[span_begin..] + d_num_kv[span_begin..]
      ──> [DEVICE] d_blocks[s], d_sizes[s]

  cudaMemcpyAsync D2H: d_sizes[s] ──> h_sizes[s]  (pinned)
  cudaMemcpyAsync D2H: d_blocks[s] ──> h_raw[s]   (pinned)

Concurrently (while pack streams run):
  gather_largest_keys_kernel<<<ceil(992/256), 256>>>
    [DEVICE] d_merged_output + d_first_kv + d_num_kv ──> [DEVICE] d_largest_keys
  cudaMemcpy D2H: d_largest_keys ──> host largest_keys

cudaStreamSynchronize each pack stream.

[CPU] assemble_sst_files_targeted_from_largest_keys():
  For each output SST span:
    Concatenate: packed_data_blocks | filter_blocks | index_region |
                 filter_meta | data_meta | footer
    Uses host-side memcpy from pinned h_raw into final file_bytes vectors.

OUTPUT: SSTBuildSet with 4 output SST files (~8MB each, ~33MB total)
```

### Serial vs. Parallel Execution Timeline

```
Time ──────────────────────────────────────────────────────────────>

STAGE 1 (Unpack):
  Stream 0: [H2D SST0] [unpack_kernel SST0]
  Stream 1: [H2D SST1] [unpack_kernel SST1]    <- 4 streams concurrent
  Stream 2: [H2D SST2] [unpack_kernel SST2]
  Stream 3: [H2D SST3] [unpack_kernel SST3]
  ─── barrier (sync all streams) ───

STAGE 2 (Merge):             SERIAL
  Default:  [merge_kernel]
  ─── implicit sync ───

STAGE 3 (Planning):          SERIAL (mostly CPU)
  GPU:      [group_sizes_kernel] [D2H]
  CPU:      [plan_data_blocks]  [partition_output_blocks]
  GPU:      [H2D plans]

STAGE 4 (Bloom):             SERIAL (single batched launch)
  Default:  [layout_kernel] [bloom_batched_kernel] [D2H filter]

STAGE 5 (Pack + Assemble):
  Stream 0: [pack_kernel span0] [D2H span0]
  Stream 1: [pack_kernel span1] [D2H span1]    <- 4 streams concurrent
  Stream 2: [pack_kernel span2] [D2H span2]
  Stream 3: [pack_kernel span3] [D2H span3]
  Default:  [largest_keys_kernel] [D2H keys]   <- concurrent with pack
  ─── barrier (sync all streams) ───
  CPU:      [assemble SST files]                <- SERIAL, CPU-only
```

---

## 4. Kernel Time vs. Transfer Overhead

### Best GPU Run Breakdown (from benchmark)

| Stage | Wall Time (ms) | Kernel Time (ms) | Overhead (ms) | Overhead % |
|-------|---------------|-------------------|---------------|------------|
| read+parse | 4.14 | -- | 4.14 | 100% (CPU I/O) |
| **unpack** | **3.93** | **1.98** | **1.95** | 49.6% (H2D) |
| **sort(merge)** | **2.04** | **1.97** | **0.07** | 3.4% |
| **plan** | **1.68** | **~0.01** | **1.67** | 99.4% (CPU + D2H/H2D) |
| **bloom** | **0.50** | **0.09** | **0.41** | 82.0% (cudaMalloc + D2H) |
| **pack+assemble** | **27.44** | **1.29** | **26.15** | 95.3% (D2H + CPU) |
| write | 6.06 | -- | 6.06 | 100% (CPU I/O) |
| **GPU total** | **47.83** | **5.34** | **42.49** | **88.8%** |

### Where the Time Goes

**Kernel computation is only 11.2% of GPU total time.** The dominant costs are:

1. **Pack D2H transfer (est. ~20ms):** 992 blocks x 32KB = ~32MB transferred via 4 pinned-memory async streams. PCIe bandwidth limits this.

2. **CPU SST assembly (est. ~6ms):** After D2H, CPU concatenates packed blocks, filter blocks, index entries, metadata, and footer into final SST file byte arrays. This is pure CPU `memcpy` work.

3. **Unpack H2D transfer (~2ms):** 4 SST files (~8MB each, ~32MB total) transferred H2D via 4 concurrent streams.

4. **Planning CPU work (~1.7ms):** GPU computes group sizes quickly (~0.01ms), but CPU greedy bin-packing + output file partitioning + plan upload takes the rest.

5. **Bloom overhead (~0.4ms):** `cudaMalloc` for filter buffer + layout kernel + D2H copy of filter bytes.

6. **Disk I/O (~10ms):** read+parse (4.14ms) + write (6.06ms) are pure CPU/SSD operations.

---

## 5. Concurrency Dimensions

### Where There IS Concurrency

| Dimension | Where | Mechanism | Details |
|-----------|-------|-----------|---------|
| **Across input SSTs** | Unpack | CUDA streams | 4 non-blocking streams, one per input SST. H2D + kernel overlap across SSTs. |
| **Across data blocks (same SST)** | Unpack, Pack | Thread grid | One thread block per data block. All blocks in a grid execute concurrently on GPU SMs. |
| **Across KV groups (same block)** | Unpack, Pack | Threads in block | One thread per KV group. Groups within a block execute concurrently. |
| **Across all KV pairs** | Merge | Thread grid | One thread per KV pair. ~200K threads. Binary searches are independent. |
| **Across all data blocks** | Bloom | Thread grid | Single batched kernel: 992 thread blocks, one per output data block. |
| **Across keys (same block)** | Bloom | Threads in block | One thread per key for hashing, then one thread per bitvec byte for packing. |
| **Across output SSTs** | Pack | CUDA streams | 4 non-blocking streams, one per output SST span. Pack kernel + D2H overlap across spans. |
| **Pack + index gen** | Pack stage | Overlap | `gather_largest_keys_kernel` runs concurrently with pack streams. |

### Where There is NO Concurrency (Serial Bottlenecks)

| What | Why | Impact |
|------|-----|--------|
| **Merge kernel** | Single grid launch, all KVs at once. Cannot pipeline across SSTs because all SSTs must contribute to the global sort. | 2.04ms wall (low impact -- kernel is fast) |
| **Planning (CPU)** | Greedy bin-packing is inherently sequential: each block boundary depends on the previous. | 1.68ms (moderate -- could be GPU-ized but serial nature is fundamental) |
| **SST assembly (CPU)** | Building final file bytes requires sequential concatenation of blocks, filter, index, metadata, footer with running offsets. | ~6ms within pack stage (dominant cost, unavoidable without on-device assembly) |
| **Stage-to-stage deps** | Unpack must finish before merge. Merge must finish before plan. Plan must finish before bloom and pack. | Each stage is a barrier. Paper's Pipeline mechanism addresses only cross-compaction-job overlap, not intra-job. |

---

## 6. Paper Correspondence Matrix

| Paper Concept | Paper Section | Implementation | Match? |
|---------------|---------------|----------------|--------|
| **Unpack Unit** | III-B.1, Fig.6 | `unpack_kernel` in `gpcomp_pack.cuh:684` | Identical |
| **Thread -> KV Group** | III-B.1 | `tid` indexes restart group in unpack/pack | Identical |
| **Thread Block -> Data Block** | III-B.1 | `blockIdx.x` = data block in unpack/pack | Identical |
| **Thread Grid -> SST File** | III-B.1 | Grid launched per SST in unpack, per span in pack | Identical |
| **CUDA Streams for cross-SST** | III-B.1 | `cudaStreamCreateWithFlags(NonBlocking)` per SST | Identical (enhanced: NonBlocking) |
| **Sort Unit (Algorithm 1)** | III-B.2, Algo 1 | `merge_kernel` in `gpcomp_merge.cuh:35` | Identical |
| **Binary search rank** | III-B.2 | `binary_search_lower_bound` in `gpcomp_merge.cuh:20` | Identical |
| **One thread per KV in merge** | III-B.2 | `global_idx` = KV index in `merge_kernel` | Identical |
| **Pack Unit - data blocks** | III-B.3, Fig.7 | `pack_kernel` in `gpcomp_pack.cuh:534` | Identical |
| **Pack Unit - index block** | III-B.3 | `gather_largest_keys_kernel` + CPU assembly | Extended (GPU gathers keys, CPU assembles) |
| **Pack Unit - bloom filter** | III-B.3, Algo 2 | `bloom_filter_kernel_batched` in `gpcomp_bloom.cuh:56` | Identical (batched) |
| **ByteVector -> BitVector** | III-B.3, Algo 2 | Shared mem `byte_vector`, `__syncthreads`, bit packing | Identical |
| **Pipeline mechanism** | III-C.1, Fig.8 | `cudaMemcpyAsync` + kernel launch on streams | Partial (within-job, not cross-job) |
| **Lazy allocation** | III-D | Not implemented | Deviation |
| **Q-Compaction (no GC)** | III-A.2 | Benchmark: "garbage collection: disabled" | Identical |
| **SSD-GPU P2P (GDS)** | III-C.2 | Not implemented | Out of scope |

---

## 7. Deviations and Extensions

### 7.1 Deviations from Paper

**Lazy Allocation:**
The paper describes pre-allocating all GPU resources at system init. The implementation allocates and frees GPU memory per compaction invocation (`cudaMalloc`/`cudaFree` in each launch function). The `DevicePlanWorkspace` singleton is the only persistent allocation.

> *Why:* This is a standalone benchmark, not integrated into a KV store lifecycle. Lazy allocation would be implemented at the storage engine level.

**Pipeline Mechanism (Cross-Job):**
The paper's Pipeline mechanism (Fig. 8) overlaps I/O of SST[i+1] with GPU computation of SST[i] across compaction jobs. The implementation only pipelines within a single compaction job (streams for concurrent unpack/pack).

> *Why:* The benchmark runs a single compaction job. Cross-job pipelining requires integration with the storage engine's compaction scheduler.

**SSD-GPU P2P (GPUDirect Storage):**
Not implemented. Would require Nvidia GPUDirect Storage hardware and API.

### 7.2 Extensions Beyond Paper

**Batched Bloom Kernel:**
Paper describes per-data-block bloom generation. Implementation launches a single kernel covering all 992 data blocks (`bloom_filter_kernel_batched`), which avoids 992 separate kernel launches and reduces launch overhead.

**On-Device Group Size Computation:**
Paper doesn't explicitly describe how data block boundaries are determined. Implementation uses `compute_restart_group_sizes_kernel` to compute group sizes on GPU, then transfers only the compact group-size array (~50K groups x 4 bytes = 200KB) instead of the full KV array (~200K x 48B = 9.6MB) for CPU-side planning.

**On-Device Largest Key Extraction:**
Paper says "each GPU thread calculates the largest key in the corresponding data block." Implementation does this via `gather_largest_keys_kernel`, avoiding transfer of full merged KV array for index block construction.

**Non-Blocking CUDA Streams:**
Paper says "CUDA Streams." Implementation specifically uses `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` which prevents implicit serialization with the default stream, enabling true concurrent execution.

**Pinned Host Memory for Pack D2H:**
Implementation uses `cudaHostAlloc` for pack output buffers, enabling async D2H transfers that can overlap with pack kernel execution on other streams.

**Group-Aligned Block Planning:**
The paper mentions KV groups as the smallest unit. The implementation's `plan_data_blocks_group_aligned_from_group_sizes` ensures data block boundaries always fall on KV group boundaries, which is consistent with the paper but not explicitly stated.

---

## 8. Benchmark Results

### Configuration
- Input: 4 SST files, ~8MB each (~32MB total)
- Key: 16 bytes, Value: 32 bytes (48 bytes per KV pair)
- ~200K total KV pairs
- Restart interval: 4, Block size: 32KB
- 992 output data blocks, 4 output SST files

### Results (best of 5 runs)

```
CPU total: min=131.15 ms
GPU total: min= 47.83 ms
Speedup:   2.74x
Output identical: PASS
```

### Stage-by-Stage Comparison

| Stage | CPU (ms) | GPU Wall (ms) | GPU Kernel (ms) | Speedup |
|-------|----------|---------------|-----------------|---------|
| read+parse | 4.21 | 4.14 | -- | 1.0x |
| unpack | 20.28 | 3.93 | 1.98 | 5.2x |
| sort(merge) | 19.66 | 2.04 | 1.97 | 9.6x |
| plan | 2.64 | 1.68 | ~0.01 | 1.6x |
| bloom | 33.09 | 0.50 | 0.09 | 66.2x |
| pack+assemble | 44.31 | 27.44 | 1.29 | 1.6x |
| write | 6.15 | 6.06 | -- | 1.0x |

### Key Observations

1. **Bloom is the biggest GPU win:** 66x speedup from parallel hashing across all keys.
2. **Merge is highly parallel:** 9.6x speedup from O(N log N) parallel binary search.
3. **Unpack benefits from streams:** 5.2x from cross-SST stream parallelism.
4. **Pack is bottlenecked by D2H:** Only 1.6x speedup despite 34x kernel speedup (1.29ms vs 44.31ms) because 32MB D2H transfer dominates.
5. **Planning is mostly CPU:** Group sizes computed on GPU in ~0.01ms, but CPU bin-packing takes the rest.
6. **Overall 2.74x speedup** aligns with paper's reported 3.61x for 32-byte values (difference likely due to hardware differences: paper uses Quadro A6000, our system may differ).

### The Bottleneck

At this data scale, the GPU finishes kernel work in ~5.3ms total. The remaining ~42.5ms is:
- PCIe data transfer (~22ms for H2D + D2H)
- CPU SST assembly (~6ms)
- CPU planning (~1.7ms)
- Disk I/O (~10ms)

Further speedup requires either:
- GDS (eliminate CPU memory as intermediary for I/O)
- On-device SST assembly (eliminate D2H of packed blocks)
- Larger compaction jobs (amortize fixed costs over more data)
