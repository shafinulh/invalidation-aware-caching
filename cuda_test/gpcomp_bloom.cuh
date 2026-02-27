/*
 * gpcomp_bloom.cuh
 *
 * Device kernel and host launcher for Algorithm 2 (GPComp):
 * Generate Bloom Filter Block.
 *
 * Include this header in exactly one translation unit at a time.
 * Compile each .cu file independently; do NOT link two object files that
 * both include this header (the __global__ symbol would be duplicated).
 */
#pragma once

#include "gpcomp_common.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------
 * Device helper: per-seed MurmurHash3-style 64-bit finalizer
 *
 * Produces K independent hash values when called with seed = 1..K.
 * Must stay bit-for-bit identical to cpu_bloom_hash() in gpcomp_common.cuh.
 * ---------------------------------------------------------------------- */

__device__ static uint32_t bloom_hash(uint64_t key, int seed)
{
    uint64_t h = key ^ ((uint64_t)seed * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (uint32_t)(h & 0xFFFFFFFFULL);
}

/* -------------------------------------------------------------------------
 * Kernel 2 – Generate Bloom Filter Block  (Algorithm 2)
 *
 * Launch constraints (enforced by launch_bloom_filter):
 *   <<<1, blockDim>>> with blockDim >= max(array_size, BitVector_Len).
 *   Dynamic shared memory = byte_vector_len bytes.
 *
 * Phase 1 – Hashing  (lines 3-7):
 *   Thread idx < array_size marks K bit-slots in shared ByteVector.
 *   Write-write races are benign: all colliding threads write the same
 *   value (1), so no atomic is needed.
 *
 * Phase 2 – Compaction  (lines 9-18):
 *   After __syncthreads(), thread idx < BitVector_Len reads 8 consecutive
 *   bytes of ByteVector and packs them into a single BitVector byte.
 * ---------------------------------------------------------------------- */

__global__ void bloom_filter_kernel(const KVPair * __restrict__ array,
                                     int                         array_size,
                                     int                         K,
                                     int                         byte_vector_len,
                                     uint8_t      * __restrict__ bit_vector)
{
    extern __shared__ uint8_t ByteVector[];

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    /* Co-operatively zero the shared ByteVector. */
    for (int s = idx; s < byte_vector_len; s += blockDim.x)
        ByteVector[s] = 0;
    __syncthreads();

    /* ---- Phase 1: Hashing (Algorithm 2, lines 3-7) --------------------- */
    if (idx < array_size) {
        for (int i = 1; i <= K; ++i) {
            uint32_t h       = bloom_hash(array[idx].key, i);
            int      bytepos = (int)(h % (uint32_t)byte_vector_len);
            ByteVector[bytepos] = 1;          /* benign race */
        }
    }

    __syncthreads();    /* Algorithm 2, line 8 */

    /* ---- Phase 2: Compaction (Algorithm 2, lines 9-18) ----------------- */
    int BitVector_Len = (byte_vector_len + 7) / 8;

    if (idx < BitVector_Len) {
        int     base   = idx * 8;
        uint8_t packed = 0;
        for (int j = 0; j <= 7; ++j) {
            int bytepos = base + j;
            if (bytepos < byte_vector_len && ByteVector[bytepos] == 1)
                packed |= (uint8_t)(1 << j);
        }
        bit_vector[idx] = packed;
    }
}

/* -------------------------------------------------------------------------
 * Host launch function for Kernel 2
 *
 * Manages device memory, validates shared-memory requirements, launches
 * bloom_filter_kernel as a single block, and retrieves the BitVector.
 *
 * Parameters:
 *   h_array         – host KVPair input (any order).
 *   array_size      – number of KV pairs.
 *   K               – number of independent hash functions.
 *   byte_vector_len – Bloom filter width in bits; must fit in shared memory.
 *   h_bit_vector    – caller-provided host output buffer of size
 *                     (byte_vector_len + 7) / 8 bytes.
 *
 * Returns 0 on success, non-zero on error.
 * ---------------------------------------------------------------------- */

int launch_bloom_filter(const KVPair *h_array,
                         int           array_size,
                         int           K,
                         int           byte_vector_len,
                         uint8_t      *h_bit_vector)
{
    /* Validate shared memory availability. */
    {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        if ((size_t)byte_vector_len > prop.sharedMemPerBlock) {
            fprintf(stderr,
                "[bloom_filter] byte_vector_len (%d) exceeds shared memory "
                "per block (%zu B).\n", byte_vector_len, prop.sharedMemPerBlock);
            return -1;
        }
    }

    int BitVector_Len = (byte_vector_len + 7) / 8;

    /* Block size must cover both phases; round to warp boundary. */
    int block_dim = (array_size > BitVector_Len ? array_size : BitVector_Len);
    block_dim = ((block_dim + 31) / 32) * 32;
    if (block_dim == 0) block_dim = 32;   /* guard for zero-element input */

    {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        if (block_dim > prop.maxThreadsPerBlock) {
            fprintf(stderr,
                "[bloom_filter] block_dim (%d) exceeds device max (%d).\n",
                block_dim, prop.maxThreadsPerBlock);
            return -1;
        }
    }

    /* Device buffers. */
    KVPair  *d_array;
    uint8_t *d_bit_vector;
    cudaMalloc(&d_array,      (array_size ? array_size : 1) * sizeof(KVPair));
    cudaMalloc(&d_bit_vector, BitVector_Len * sizeof(uint8_t));
    if (array_size > 0)
        cudaMemcpy(d_array, h_array, array_size * sizeof(KVPair),
                   cudaMemcpyHostToDevice);

    /* Launch. */
    size_t shared_bytes = (size_t)byte_vector_len * sizeof(uint8_t);
    bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
        d_array, array_size, K, byte_vector_len, d_bit_vector);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[bloom_filter_kernel] launch error: %s\n",
                cudaGetErrorString(err));

    cudaDeviceSynchronize();
    cudaMemcpy(h_bit_vector, d_bit_vector, BitVector_Len * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_bit_vector);
    return (int)err;
}
