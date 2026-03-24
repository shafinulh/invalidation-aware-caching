#pragma once

#include "gpcomp_common.cuh"

#include <chrono>
#include <cuda_runtime.h>
#include <vector>

__device__ static uint32_t device_bloom_hash(const Key128& key, int seed)
{
    uint64_t lo = load_be64(key.bytes);
    uint64_t hi = load_be64(key.bytes + 8);
    uint64_t x = lo ^ (hi + 0x9e3779b97f4a7c15ULL * (uint64_t)seed);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)(x & 0xffffffffu);
}

/* Algorithm 2 exactly: hash into ByteVector, synchronize, pack into BitVector. */
__global__ void bloom_filter_kernel(const KVPair* __restrict__ array,
                                    int                        array_size,
                                    int                        K,
                                    int                        byte_vector_len,
                                    uint8_t* __restrict__      bit_vector)
{
    extern __shared__ uint8_t byte_vector[];

    int idx = (int)threadIdx.x;
    for (int s = idx; s < byte_vector_len; s += blockDim.x) byte_vector[s] = 0;
    __syncthreads();

    if (idx < array_size) {
        for (int i = 1; i <= K; ++i) {
            uint32_t h = device_bloom_hash(array[idx].key, i);
            int pos = (int)(h % (uint32_t)byte_vector_len);
            byte_vector[pos] = 1;
        }
    }
    __syncthreads();

    int bitvec_len = (byte_vector_len + 7) / 8;
    if (idx < bitvec_len) {
        int base = idx * 8;
        uint8_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            int pos = base + j;
            if (pos < byte_vector_len && byte_vector[pos]) packed |= (uint8_t)(1u << j);
        }
        bit_vector[idx] = packed;
    }
}

__global__ void bloom_filter_kernel_batched(const KVPair*  __restrict__ all_keys,
                                            const uint32_t* __restrict__ block_first_kv,
                                            const uint32_t* __restrict__ block_num_kv,
                                            const uint32_t* __restrict__ block_bitvec_offsets,
                                            int                           bits_per_key,
                                            int                           K,
                                            int                           max_byte_vector_len,
                                            uint8_t*       __restrict__   all_bitvecs)
{
    extern __shared__ uint8_t byte_vector[];

    int block_id = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int num_keys = (int)block_num_kv[block_id];
    int first_kv = (int)block_first_kv[block_id];
    int byte_vector_len = num_keys * bits_per_key;
    int bitvec_len = (byte_vector_len + 7) / 8;

    for (int s = tid; s < max_byte_vector_len; s += blockDim.x) {
        if (s < byte_vector_len) byte_vector[s] = 0;
    }
    __syncthreads();

    if (tid < num_keys) {
        const Key128& key = all_keys[first_kv + tid].key;
        for (int i = 1; i <= K; ++i) {
            uint32_t h = device_bloom_hash(key, i);
            int pos = (int)(h % (uint32_t)byte_vector_len);
            byte_vector[pos] = 1;
        }
    }
    __syncthreads();

    uint8_t* out = all_bitvecs + block_bitvec_offsets[block_id];
    if (tid < bitvec_len) {
        int base = tid * 8;
        uint8_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            int pos = base + j;
            if (pos < byte_vector_len && byte_vector[pos]) packed |= (uint8_t)(1u << j);
        }
        out[tid] = packed;
    }
}

__global__ void bloom_filter_layout_kernel(const uint32_t* __restrict__ block_num_kv,
                                           int                          num_blocks,
                                           uint32_t* __restrict__       block_bitvec_offsets,
                                           uint32_t* __restrict__       block_bitvec_lengths,
                                           uint32_t* __restrict__       total_bytes)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t offset = 0;
    for (int i = 0; i < num_blocks; ++i) {
        uint32_t byte_vector_len = block_num_kv[i] * GP_BLOOM_BITS_PER_KEY;
        uint32_t bitvec_len = (byte_vector_len + 7) / 8;
        block_bitvec_offsets[i] = offset;
        block_bitvec_lengths[i] = bitvec_len;
        offset += bitvec_len;
    }
    *total_bytes = offset;
}

static inline int launch_bloom_filter(const KVPair* h_array,
                                      int           array_size,
                                      int           K,
                                      int           byte_vector_len,
                                      uint8_t*      h_bit_vector)
{
    int bitvec_len = (byte_vector_len + 7) / 8;
    int block_dim = ((std::max(array_size, bitvec_len) + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    KVPair*  d_array = nullptr;
    uint8_t* d_bit_vector = nullptr;
    cudaMalloc(&d_array, (size_t)std::max(array_size, 1) * sizeof(KVPair));
    cudaMalloc(&d_bit_vector, (size_t)std::max(bitvec_len, 1));
    if (array_size > 0) {
        cudaMemcpy(d_array, h_array, (size_t)array_size * sizeof(KVPair), cudaMemcpyHostToDevice);
    }

    bloom_filter_kernel<<<1, block_dim, (size_t)byte_vector_len>>>(d_array, array_size, K,
                                                                   byte_vector_len, d_bit_vector);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();

    if (bitvec_len > 0) {
        cudaMemcpy(h_bit_vector, d_bit_vector, (size_t)bitvec_len, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_bit_vector);
    cudaFree(d_array);
    return (int)err;
}

struct BloomBatchResult {
    std::vector<uint8_t>  filter_bytes;
    std::vector<uint32_t> bitvec_offsets;
    std::vector<uint32_t> bitvec_lengths;
    float                 kernel_ms = 0.0f;
    float                 wall_ms = 0.0f;
    float                 h2d_ms = 0.0f;
    float                 d2h_ms = 0.0f;
    size_t                h2d_bytes = 0;
    size_t                d2h_bytes = 0;
};

static inline BloomBatchResult launch_bloom_filter_batched(const std::vector<KVPair>&              kv_array,
                                                           const std::vector<DataBlockPlanEntry>&   plans)
{
    BloomBatchResult result;
    if (plans.empty()) return result;

    uint32_t max_num_kv = 0;
    result.bitvec_offsets.resize(plans.size());
    result.bitvec_lengths.resize(plans.size());

    uint32_t total_bytes = 0;
    std::vector<uint32_t> first_kv(plans.size()), num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
        max_num_kv = std::max(max_num_kv, plans[i].num_kv);
        result.bitvec_offsets[i] = total_bytes;
        uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        uint32_t bitvec_len = (byte_vector_len + 7) / 8;
        result.bitvec_lengths[i] = bitvec_len;
        total_bytes += bitvec_len;
    }
    result.filter_bytes.resize(total_bytes);

    KVPair*   d_kv = nullptr;
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    uint32_t* d_offsets = nullptr;
    uint8_t*  d_filter = nullptr;
    cudaMalloc(&d_kv, kv_array.size() * sizeof(KVPair));
    cudaMalloc(&d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_offsets, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_filter, total_bytes);

    auto wall_start = std::chrono::steady_clock::now();
    auto h2d_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_kv, kv_array.data(), kv_array.size() * sizeof(KVPair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_first_kv, first_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_kv, num_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, result.bitvec_offsets.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = kv_array.size() * sizeof(KVPair) + plans.size() * sizeof(uint32_t) * 3;

    int max_byte_vector_len = (int)(max_num_kv * GP_BLOOM_BITS_PER_KEY);
    int max_bitvec_len = (max_byte_vector_len + 7) / 8;
    int block_dim = ((std::max((int)max_num_kv, max_bitvec_len) + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    bloom_filter_kernel_batched<<<(int)plans.size(), block_dim, (size_t)max_byte_vector_len>>>(
        d_kv, d_first_kv, d_num_kv, d_offsets, GP_BLOOM_BITS_PER_KEY, GP_BLOOM_HASHES,
        max_byte_vector_len, d_filter);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    auto d2h_start = std::chrono::steady_clock::now();
    cudaMemcpy(result.filter_bytes.data(), d_filter, total_bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::steady_clock::now();
    result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
    result.d2h_bytes = total_bytes;
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_filter);
    cudaFree(d_offsets);
    cudaFree(d_num_kv);
    cudaFree(d_first_kv);
    cudaFree(d_kv);
    return result;
}

static inline BloomBatchResult
launch_bloom_filter_batched_from_device(const KVPair*                         d_kv,
                                        const std::vector<DataBlockPlanEntry>& plans)
{
    BloomBatchResult result;
    if (plans.empty()) return result;

    uint32_t max_num_kv = 0;
    result.bitvec_offsets.resize(plans.size());
    result.bitvec_lengths.resize(plans.size());

    uint32_t total_bytes = 0;
    std::vector<uint32_t> first_kv(plans.size()), num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
        max_num_kv = std::max(max_num_kv, plans[i].num_kv);
        result.bitvec_offsets[i] = total_bytes;
        uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        uint32_t bitvec_len = (byte_vector_len + 7) / 8;
        result.bitvec_lengths[i] = bitvec_len;
        total_bytes += bitvec_len;
    }
    result.filter_bytes.resize(total_bytes);

    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    uint32_t* d_offsets = nullptr;
    uint8_t*  d_filter = nullptr;
    cudaMalloc(&d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_offsets, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_filter, total_bytes);

    auto wall_start = std::chrono::steady_clock::now();
    auto h2d_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_first_kv, first_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_kv, num_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, result.bitvec_offsets.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = plans.size() * sizeof(uint32_t) * 3;

    int max_byte_vector_len = (int)(max_num_kv * GP_BLOOM_BITS_PER_KEY);
    int max_bitvec_len = (max_byte_vector_len + 7) / 8;
    int block_dim = ((std::max((int)max_num_kv, max_bitvec_len) + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    bloom_filter_kernel_batched<<<(int)plans.size(), block_dim, (size_t)max_byte_vector_len>>>(
        d_kv, d_first_kv, d_num_kv, d_offsets, GP_BLOOM_BITS_PER_KEY, GP_BLOOM_HASHES,
        max_byte_vector_len, d_filter);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    auto d2h_start = std::chrono::steady_clock::now();
    cudaMemcpy(result.filter_bytes.data(), d_filter, total_bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::steady_clock::now();
    result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
    result.d2h_bytes = total_bytes;
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_filter);
    cudaFree(d_offsets);
    cudaFree(d_num_kv);
    cudaFree(d_first_kv);
    return result;
}

static inline BloomBatchResult
launch_bloom_filter_batched_from_device_plans(const KVPair*      d_kv,
                                              const uint32_t*    d_first_kv,
                                              const uint32_t*    d_num_kv,
                                              int                num_blocks)
{
    BloomBatchResult result;
    if (num_blocks <= 0) return result;

    std::vector<uint32_t> host_num_kv((size_t)num_blocks);
    auto d2h_plan_start = std::chrono::steady_clock::now();
    cudaMemcpy(host_num_kv.data(), d_num_kv, (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    auto d2h_plan_end = std::chrono::steady_clock::now();
    result.d2h_ms += (float)std::chrono::duration<double, std::milli>(d2h_plan_end - d2h_plan_start).count();
    result.d2h_bytes += (size_t)num_blocks * sizeof(uint32_t);

    uint32_t max_num_kv = 0;
    for (uint32_t n : host_num_kv) max_num_kv = std::max(max_num_kv, n);

    uint32_t* d_offsets = nullptr;
    uint32_t* d_lengths = nullptr;
    uint32_t* d_total_bytes = nullptr;
    cudaMalloc(&d_offsets, (size_t)num_blocks * sizeof(uint32_t));
    cudaMalloc(&d_lengths, (size_t)num_blocks * sizeof(uint32_t));
    cudaMalloc(&d_total_bytes, sizeof(uint32_t));

    bloom_filter_layout_kernel<<<1, 1>>>(d_num_kv, num_blocks, d_offsets, d_lengths, d_total_bytes);

    uint32_t total_bytes = 0;
    auto d2h_total_start = std::chrono::steady_clock::now();
    cudaMemcpy(&total_bytes, d_total_bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    auto d2h_total_end = std::chrono::steady_clock::now();
    result.d2h_ms += (float)std::chrono::duration<double, std::milli>(d2h_total_end - d2h_total_start).count();
    result.d2h_bytes += sizeof(uint32_t);
    result.filter_bytes.resize(total_bytes);
    result.bitvec_offsets.resize((size_t)num_blocks);
    result.bitvec_lengths.resize((size_t)num_blocks);

    uint8_t* d_filter = nullptr;
    cudaMalloc(&d_filter, (size_t)std::max(total_bytes, 1u));

    int max_byte_vector_len = (int)(max_num_kv * GP_BLOOM_BITS_PER_KEY);
    int max_bitvec_len = (max_byte_vector_len + 7) / 8;
    int block_dim = ((std::max((int)max_num_kv, max_bitvec_len) + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    auto wall_start = std::chrono::steady_clock::now();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    bloom_filter_kernel_batched<<<num_blocks, block_dim, (size_t)max_byte_vector_len>>>(
        d_kv, d_first_kv, d_num_kv, d_offsets, GP_BLOOM_BITS_PER_KEY, GP_BLOOM_HASHES,
        max_byte_vector_len, d_filter);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    auto d2h_data_start = std::chrono::steady_clock::now();
    cudaMemcpy(result.filter_bytes.data(), d_filter, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.bitvec_offsets.data(), d_offsets,
               (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.bitvec_lengths.data(), d_lengths,
               (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    auto d2h_data_end = std::chrono::steady_clock::now();
    result.d2h_ms += (float)std::chrono::duration<double, std::milli>(d2h_data_end - d2h_data_start).count();
    result.d2h_bytes += total_bytes + (size_t)num_blocks * sizeof(uint32_t) * 2;
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_filter);
    cudaFree(d_total_bytes);
    cudaFree(d_lengths);
    cudaFree(d_offsets);
    return result;
}
