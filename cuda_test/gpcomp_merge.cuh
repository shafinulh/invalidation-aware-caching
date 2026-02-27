/*
 * gpcomp_merge.cuh
 *
 * Device kernel and host launcher for Algorithm 1 (GPComp):
 * Merge Multiple Ordered Key-Value Pair Arrays.
 *
 * Include this header in exactly one translation unit at a time.
 * Compile each .cu file independently; do NOT link two object files that
 * both include this header (the __global__ symbol would be duplicated).
 */
#pragma once

#include "gpcomp_common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------
 * Device helper: lower-bound binary search
 *
 * Returns the number of elements in arr[0..size-1] with key < query_key.
 * Precondition: arr is sorted in ascending key order.
 * ---------------------------------------------------------------------- */

__device__ static int binary_search_lower_bound(
        const KVPair * __restrict__ arr,
        int                         size,
        uint64_t                    query_key)
{
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid].key < query_key) lo = mid + 1;
        else                          hi = mid;
    }
    return lo;
}

/* -------------------------------------------------------------------------
 * Kernel 1 – Merge Multiple Ordered Key-Value Pair Arrays  (Algorithm 1)
 *
 * One thread per KV pair.  Thread global_id belongs to array j determined
 * by sst_offsets[].  It accumulates in I_Array:
 *   - its own local index in SST_j (line 8 of the algorithm), and
 *   - lower_bound(SST_i, my_key) for every other array i (line 10).
 * The pair is written to output[I_Array].
 * ---------------------------------------------------------------------- */

__global__ void merge_kernel(KVPair * const * __restrict__ sst_arrays,
                              const int     * __restrict__ sst_sizes,
                              const int     * __restrict__ sst_offsets,
                              int                          num_arrays,
                              KVPair       * __restrict__  output)
{
    int global_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total     = sst_offsets[num_arrays];
    if (global_id >= total) return;

    /* Identify which source array j this thread belongs to. */
    int j = 0;
    while (j + 1 < num_arrays && global_id >= sst_offsets[j + 1])
        ++j;

    int      idx    = global_id - sst_offsets[j];
    int      I_Array = 0;
    uint64_t my_key = sst_arrays[j][idx].key;

    for (int i = 0; i < num_arrays; ++i) {
        int I = (i == j) ? idx
                         : binary_search_lower_bound(sst_arrays[i],
                                                     sst_sizes[i],
                                                     my_key);
        I_Array += I;
    }

    output[I_Array] = sst_arrays[j][idx];
}

/* -------------------------------------------------------------------------
 * Host launch function for Kernel 1
 *
 * Manages all device memory, builds prefix-sum offsets, launches the
 * kernel, and copies the merged result back to h_output.
 *
 * Parameters:
 *   h_sst_arrays – array of num_arrays pointers to host sorted KVPair arrays.
 *   h_sst_sizes  – number of pairs in each SST array.
 *   num_arrays   – total number of SST arrays to merge.
 *   h_output     – caller-provided host buffer (>= sum(h_sst_sizes) elements).
 *
 * Returns 0 on success, non-zero on CUDA error.
 * ---------------------------------------------------------------------- */

int launch_merge(KVPair * const *h_sst_arrays,
                 const int      *h_sst_sizes,
                 int             num_arrays,
                 KVPair         *h_output)
{
    /* Build host-side prefix-sum offsets. */
    int *h_offsets = (int *)malloc((num_arrays + 1) * sizeof(int));
    h_offsets[0] = 0;
    for (int i = 0; i < num_arrays; ++i)
        h_offsets[i + 1] = h_offsets[i] + h_sst_sizes[i];
    int total = h_offsets[num_arrays];

    /* Upload each SST array to the device. */
    KVPair **d_sst_ptrs_host = (KVPair **)malloc(num_arrays * sizeof(KVPair *));
    for (int i = 0; i < num_arrays; ++i) {
        cudaMalloc(&d_sst_ptrs_host[i], h_sst_sizes[i] * sizeof(KVPair));
        cudaMemcpy(d_sst_ptrs_host[i], h_sst_arrays[i],
                   h_sst_sizes[i] * sizeof(KVPair), cudaMemcpyHostToDevice);
    }

    /* Device array-of-pointers (pointer to above per-array buffers). */
    KVPair **d_sst_arrays;
    cudaMalloc(&d_sst_arrays, num_arrays * sizeof(KVPair *));
    cudaMemcpy(d_sst_arrays, d_sst_ptrs_host,
               num_arrays * sizeof(KVPair *), cudaMemcpyHostToDevice);

    /* Device sizes and prefix-sum offsets. */
    int *d_sst_sizes, *d_sst_offsets;
    cudaMalloc(&d_sst_sizes,   num_arrays       * sizeof(int));
    cudaMalloc(&d_sst_offsets, (num_arrays + 1) * sizeof(int));
    cudaMemcpy(d_sst_sizes,   h_sst_sizes, num_arrays       * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sst_offsets, h_offsets,  (num_arrays + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    /* Output buffer. */
    KVPair *d_output;
    cudaMalloc(&d_output, total * sizeof(KVPair));

    /* Launch: one thread per pair. */
    const int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    merge_kernel<<<grid, BLOCK>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets,
                                  num_arrays, d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[merge_kernel] launch error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, total * sizeof(KVPair),
               cudaMemcpyDeviceToHost);

    /* Cleanup. */
    for (int i = 0; i < num_arrays; ++i) cudaFree(d_sst_ptrs_host[i]);
    free(d_sst_ptrs_host);
    cudaFree(d_sst_arrays);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_offsets);
    cudaFree(d_output);
    free(h_offsets);

    return (int)err;
}
