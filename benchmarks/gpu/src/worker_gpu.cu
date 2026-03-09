/*
 * worker_gpu.cu — Thin CUDA wrapper for gpu_compaction_worker.
 *
 * Exposes a plain C interface so the RocksDB-heavy worker_main.cpp
 * can be compiled by g++ (which supports C++20 features in RocksDB
 * headers) while this file is compiled by nvcc for the CUDA kernels.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../../cuda_test/gpcomp_common.cuh"
#include "../../cuda_test/gpcomp_merge.cuh"

extern "C" {

/*
 * gpu_merge_worker
 *
 * Merges num_ssts sorted KVPair arrays on the GPU.
 *
 * Parameters:
 *   sst_arrays    — array of `num_ssts` pointers, each pointing to a
 *                   host-side KVPair array for one SST
 *   sst_sizes     — array of `num_ssts` sizes (element counts)
 *   num_ssts      — number of input SSTs
 *   output        — pre-allocated host buffer of size `total_count` KVPairs
 *   total_count   — total number of input KVPairs across all SSTs
 *   h2d_us_out    — output: time spent staging data (microseconds)
 *   kernel_us_out — output: time for kernel + D2H (microseconds)
 *
 * Returns 0 on success, non-zero on CUDA error.
 */
int gpu_merge_worker(KVPair** sst_arrays, const int* sst_sizes, int num_ssts,
                     KVPair* output, long long total_count,
                     double* h2d_us_out, double* kernel_us_out)
{
    (void)total_count;
    (void)h2d_us_out;
    (void)kernel_us_out;

    int rc = launch_merge((KVPair * const *)sst_arrays, sst_sizes, num_ssts, output);
    return rc;
}

} /* extern "C" */
