/*
 * gpcomp_common.cuh
 *
 * Shared data types and CPU-side helpers used by both the merge and bloom
 * filter kernels (and their unit tests).
 */
#pragma once

#include <cstdint>

/* -------------------------------------------------------------------------
 * Core key-value pair type
 * ---------------------------------------------------------------------- */

struct KVPair {
    uint64_t key;
    uint64_t value;
};

/* -------------------------------------------------------------------------
 * CPU-side Bloom hash (mirrors the __device__ bloom_hash in gpcomp_bloom.cuh)
 *
 * Must stay bit-for-bit identical to the GPU version so test oracles can
 * replicate the GPU output without launching a kernel.
 * ---------------------------------------------------------------------- */

static inline uint32_t cpu_bloom_hash(uint64_t key, int seed)
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
 * CPU oracle: build the expected ByteVector for a given key set
 * ---------------------------------------------------------------------- */

static inline void cpu_build_byte_vector(const KVPair *array, int n,
                                          int K, int byte_vector_len,
                                          uint8_t *bv_out)
{
    for (int s = 0; s < byte_vector_len; ++s) bv_out[s] = 0;
    for (int i = 0; i < n; ++i)
        for (int k = 1; k <= K; ++k) {
            uint32_t h   = cpu_bloom_hash(array[i].key, k);
            int      pos = (int)(h % (uint32_t)byte_vector_len);
            bv_out[pos]  = 1;
        }
}

/* -------------------------------------------------------------------------
 * CPU oracle: pack ByteVector -> BitVector (mirrors Phase 2 of Algorithm 2)
 * ---------------------------------------------------------------------- */

static inline void cpu_pack_bit_vector(const uint8_t *bv, int byte_vector_len,
                                        uint8_t *bitvec_out)
{
    int bitvec_len = (byte_vector_len + 7) / 8;
    for (int i = 0; i < bitvec_len; ++i) {
        uint8_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            int pos = i * 8 + j;
            if (pos < byte_vector_len && bv[pos])
                packed |= (uint8_t)(1 << j);
        }
        bitvec_out[i] = packed;
    }
}

/* -------------------------------------------------------------------------
 * CPU oracle: query BitVector membership (used in test assertions)
 * ---------------------------------------------------------------------- */

static inline int cpu_bloom_check_bit(const uint8_t *bit_vector,
                                       int            byte_vector_len,
                                       int            K,
                                       uint64_t       key)
{
    for (int i = 1; i <= K; ++i) {
        uint32_t h        = cpu_bloom_hash(key, i);
        int      bytepos  = (int)(h % (uint32_t)byte_vector_len);
        int      byte_idx = bytepos / 8;
        int      bit_idx  = bytepos % 8;
        if (!((bit_vector[byte_idx] >> bit_idx) & 1))
            return 0;
    }
    return 1;
}
