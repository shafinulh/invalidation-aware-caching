// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

// CUDA translation unit — compiled by nvcc, not g++.
// Implements the variable-length RocksDB internal key N-way merge kernel
// (Algorithm 1 of the GPComp paper, extended to byte-string keys with the
// RocksDB InternalKeyComparator ordering).

#include "gpu_varlength_merge.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

namespace ROCKSDB_NAMESPACE {

// ─── Device helpers ──────────────────────────────────────────────────────────

// Decode a little-endian uint64 from p[0..7].  Used to read the 8-byte
// InternalKeyTrailer = (sequence_number << 8) | value_type.
__device__ __forceinline__ static uint64_t load_le64(const unsigned char* p) {
  return ((uint64_t)p[0])        | ((uint64_t)p[1] <<  8) |
         ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
         ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
         ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// Compare two RocksDB internal keys using BytewiseComparator ordering:
//   1. User key ascending (byte lexicographic).
//   2. On equal user key: InternalKeyTrailer descending (higher sequence first).
// Returns <0 / 0 / >0 like strcmp.
__device__ __forceinline__ static int cmp_ikeys(
    const char* a, uint32_t la,
    const char* b, uint32_t lb)
{
  // User key byte lengths (strip 8-byte trailer).
  const uint32_t ula    = la > 8u ? la - 8u : 0u;
  const uint32_t ulb    = lb > 8u ? lb - 8u : 0u;
  const uint32_t common = ula < ulb ? ula : ulb;

  // Byte-by-byte user key comparison.
  for (uint32_t i = 0; i < common; ++i) {
    const int diff = (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
    if (diff) return diff;
  }
  if (ula != ulb) return (int)ula - (int)ulb;

  // Equal user keys: compare 8-byte LE InternalKeyTrailer in DESCENDING order
  // so that the newer record (higher sequence) sorts before the older one.
  if (la >= 8u && lb >= 8u) {
    const uint64_t ta = load_le64((const unsigned char*)(a + ula));
    const uint64_t tb = load_le64((const unsigned char*)(b + ulb));
    if (ta > tb) return -1;  // a has higher seq → a sorts BEFORE b
    if (ta < tb) return +1;
  }
  return 0;
}

// Lower-bound search: count entries in run [run_start, run_start+run_size)
// whose internal key is strictly less than (qkey, qlen).
// Precondition: entries in the run are sorted by InternalKeyComparator.
__device__ __forceinline__ static int lower_bound_vl(
    const char*     __restrict__ d_keys,
    const uint32_t* __restrict__ d_key_offs,
    uint32_t        run_start,
    uint32_t        run_size,
    const char*     qkey,
    uint32_t        qlen)
{
  int lo = 0, hi = (int)run_size;
  while (lo < hi) {
    const int     mid      = lo + ((hi - lo) >> 1);
    const uint32_t eoff    = d_key_offs[run_start + mid];
    const uint32_t elen    = d_key_offs[run_start + mid + 1] - eoff;
    if (cmp_ikeys(d_keys + eoff, elen, qkey, qlen) < 0)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// ─── Per-SST run descriptor stored in device memory ─────────────────────────

struct DevSstRun {
  uint32_t global_start;   // first entry index in the flat key/offset arrays
  uint32_t num_entries;    // entries in this run
};

// ─── Merge kernel ─────────────────────────────────────────────────────────────
//
// One thread per global entry.  Thread gid:
//   1. Locates which run j it belongs to (linear scan; runs ≤ 64 in practice).
//   2. Computes output position = sum_i lower_bound(run_i, my_key).
//   3. Writes (j << 24 | local_idx) to out_mapping[output_pos].
//
// d_global_offsets[j] = prefix sum of run sizes = run[j].global_start.
// d_global_offsets[num_runs] = total entries.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void merge_kernel_vl(
    const char*     __restrict__ d_keys,
    const uint32_t* __restrict__ d_key_offs,
    const DevSstRun* __restrict__ d_runs,
    const int*       __restrict__ d_global_offsets,
    int              num_runs,
    uint32_t*        __restrict__ out_mapping)
{
  const int gid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = d_global_offsets[num_runs];
  if (gid >= total) return;

  // Find run j that owns this thread.
  int j = 0;
  while (j + 1 < num_runs && gid >= d_global_offsets[j + 1]) ++j;

  const int local_idx = gid - d_global_offsets[j];

  // Fetch this entry's key.
  const uint32_t my_koff = d_key_offs[d_runs[j].global_start + local_idx];
  const uint32_t my_klen = d_key_offs[d_runs[j].global_start + local_idx + 1] - my_koff;
  const char*    my_key  = d_keys + my_koff;

  // Accumulate output position via lower_bound in every run.
  int out_pos = 0;
  for (int i = 0; i < num_runs; ++i) {
    if (i == j) {
      out_pos += local_idx;
    } else {
      out_pos += lower_bound_vl(d_keys, d_key_offs,
                                d_runs[i].global_start, d_runs[i].num_entries,
                                my_key, my_klen);
    }
  }

  // packed: upper 8 bits = SST index, lower 24 bits = local entry index.
  out_mapping[out_pos] = ((uint32_t)j << 24) | (uint32_t)local_idx;
}

// ─── Host launcher ───────────────────────────────────────────────────────────

Status GpuVarlengthMerge(const char*     h_keys,
                          const uint32_t* h_key_offsets,
                          uint32_t        total_entries,
                          const uint32_t* h_run_offsets,
                          uint32_t        num_runs,
                          uint32_t*       h_out_mapping) {
  if (total_entries == 0 || num_runs == 0) {
    return Status::OK();
  }
  if (num_runs > 255) {
    return Status::InvalidArgument(
        "GpuVarlengthMerge: num_runs exceeds 255 (packed sst_id limit)");
  }

  // ── Device allocations ───────────────────────────────────────────────────
  const uint32_t keys_bytes =
      h_key_offsets[total_entries];  // total key bytes

  char*     d_keys      = nullptr;
  uint32_t* d_key_offs  = nullptr;
  DevSstRun* d_runs     = nullptr;
  int*       d_goffs    = nullptr;
  uint32_t*  d_mapping  = nullptr;

  cudaError_t err;

#define CUDA_RET(call) \
  do { err = (call); if (err != cudaSuccess) goto cuda_cleanup; } while (0)

  CUDA_RET(cudaMalloc(&d_keys,     keys_bytes));
  CUDA_RET(cudaMalloc(&d_key_offs, sizeof(uint32_t) * (total_entries + 1)));
  CUDA_RET(cudaMalloc(&d_runs,     sizeof(DevSstRun) * num_runs));
  CUDA_RET(cudaMalloc(&d_goffs,    sizeof(int) * (num_runs + 1)));
  CUDA_RET(cudaMalloc(&d_mapping,  sizeof(uint32_t) * total_entries));

  // ── H2D transfers ────────────────────────────────────────────────────────
  CUDA_RET(cudaMemcpy(d_keys, h_keys, keys_bytes, cudaMemcpyHostToDevice));
  CUDA_RET(cudaMemcpy(d_key_offs, h_key_offsets,
                      sizeof(uint32_t) * (total_entries + 1),
                      cudaMemcpyHostToDevice));

  {
    // Build DevSstRun + global_offsets arrays on host, then upload.
    DevSstRun* h_runs = new DevSstRun[num_runs];
    int*       h_goffs = new int[num_runs + 1];
    for (uint32_t j = 0; j < num_runs; ++j) {
      h_runs[j].global_start = h_run_offsets[j];
      h_runs[j].num_entries  = h_run_offsets[j + 1] - h_run_offsets[j];
      h_goffs[j]             = (int)h_run_offsets[j];
    }
    h_goffs[num_runs] = (int)total_entries;

    err = cudaMemcpy(d_runs,  h_runs,  sizeof(DevSstRun) * num_runs,
                     cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
      err = cudaMemcpy(d_goffs, h_goffs, sizeof(int) * (num_runs + 1),
                       cudaMemcpyHostToDevice);
    delete[] h_runs;
    delete[] h_goffs;
    if (err != cudaSuccess) goto cuda_cleanup;
  }

  // ── Kernel launch ─────────────────────────────────────────────────────────
  {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    const int block = prop.maxThreadsPerBlock;  // 1024 on modern HW
    const int grid  = ((int)total_entries + block - 1) / block;

    merge_kernel_vl<<<grid, block>>>(
        d_keys, d_key_offs, d_runs, d_goffs, (int)num_runs, d_mapping);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cuda_cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cuda_cleanup;
  }

  // ── D2H: download permutation ─────────────────────────────────────────────
  CUDA_RET(cudaMemcpy(h_out_mapping, d_mapping,
                      sizeof(uint32_t) * total_entries,
                      cudaMemcpyDeviceToHost));

#undef CUDA_RET

cuda_cleanup:
  if (d_keys)     cudaFree(d_keys);
  if (d_key_offs) cudaFree(d_key_offs);
  if (d_runs)     cudaFree(d_runs);
  if (d_goffs)    cudaFree(d_goffs);
  if (d_mapping)  cudaFree(d_mapping);

  if (err != cudaSuccess) {
    return Status::IOError(
        std::string("GpuVarlengthMerge CUDA error: ") +
        cudaGetErrorString(err));
  }
  return Status::OK();
}

}  // namespace ROCKSDB_NAMESPACE
