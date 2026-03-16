#pragma once

#include "gpcomp_common.cuh"

#include <chrono>
#include <cuda_runtime.h>
#include <vector>

struct __attribute__((packed)) PackBlockHeader {
    uint32_t num_entries;
    uint32_t restart_interval;
    uint32_t num_restarts;
    uint32_t data_size;
};

static constexpr int PACK_HEADER_BYTES    = (int)sizeof(PackBlockHeader);
static constexpr int PACK_MAX_ENTRY_BYTES = 2 + GP_KEY_BYTES + GP_VALUE_BYTES;

static inline uint32_t packed_entry_size(const KVPair& current,
                                         const KVPair* previous_in_group)
{
    int shared = previous_in_group ? key_shared_prefix(previous_in_group->key, current.key) : 0;
    return (uint32_t)(2 + (GP_KEY_BYTES - shared) + GP_VALUE_BYTES);
}

static inline std::vector<DataBlockPlanEntry> plan_data_blocks(const std::vector<KVPair>& kv_array)
{
    std::vector<DataBlockPlanEntry> plans;
    if (kv_array.empty()) return plans;

    uint32_t first = 0;
    while (first < kv_array.size()) {
        uint32_t num = 0;
        uint32_t data_size = 0;
        uint32_t num_restarts = 0;
        while (first + num < kv_array.size()) {
            uint32_t group_pos = num % GP_RESTART_INTERVAL;
            const KVPair* prev = (group_pos == 0) ? nullptr : &kv_array[first + num - 1];
            uint32_t entry_size = packed_entry_size(kv_array[first + num], prev);
            uint32_t candidate_data_size = data_size + entry_size;
            uint32_t candidate_restarts = num_restarts + (group_pos == 0 ? 1u : 0u);
            uint32_t candidate_total = PACK_HEADER_BYTES + candidate_data_size
                                     + candidate_restarts * sizeof(uint32_t);
            if (num > 0 && candidate_total > GP_DATA_BLOCK_BYTES) break;
            data_size = candidate_data_size;
            num_restarts = candidate_restarts;
            ++num;
        }

        DataBlockPlanEntry entry{};
        entry.first_kv = first;
        entry.num_kv = num;
        entry.serialized_size = PACK_HEADER_BYTES + data_size + num_restarts * sizeof(uint32_t);
        plans.push_back(entry);
        first += num;
    }
    return plans;
}

static inline std::vector<uint8_t> cpu_pack_block(const KVPair* kv_array, uint32_t num_kv)
{
    if (num_kv == 0) return {};

    std::vector<uint8_t> entry_data;
    entry_data.reserve((size_t)num_kv * PACK_MAX_ENTRY_BYTES);
    std::vector<uint32_t> restart_offsets;
    restart_offsets.reserve((num_kv + GP_RESTART_INTERVAL - 1) / GP_RESTART_INTERVAL);

    for (uint32_t i = 0; i < num_kv; ++i) {
        int shared = 0;
        if ((i % GP_RESTART_INTERVAL) != 0) shared = key_shared_prefix(kv_array[i - 1].key, kv_array[i].key);
        else restart_offsets.push_back((uint32_t)entry_data.size());

        entry_data.push_back((uint8_t)shared);
        entry_data.push_back((uint8_t)(GP_KEY_BYTES - shared));
        entry_data.insert(entry_data.end(),
                          kv_array[i].key.bytes + shared,
                          kv_array[i].key.bytes + GP_KEY_BYTES);
        entry_data.insert(entry_data.end(),
                          kv_array[i].value.bytes,
                          kv_array[i].value.bytes + GP_VALUE_BYTES);
    }

    PackBlockHeader header{};
    header.num_entries = num_kv;
    header.restart_interval = GP_RESTART_INTERVAL;
    header.num_restarts = (uint32_t)restart_offsets.size();
    header.data_size = (uint32_t)entry_data.size();

    std::vector<uint8_t> out(PACK_HEADER_BYTES + header.data_size
                           + restart_offsets.size() * sizeof(uint32_t));
    std::memcpy(out.data(), &header, PACK_HEADER_BYTES);
    std::memcpy(out.data() + PACK_HEADER_BYTES, entry_data.data(), header.data_size);
    std::memcpy(out.data() + PACK_HEADER_BYTES + header.data_size,
                restart_offsets.data(),
                restart_offsets.size() * sizeof(uint32_t));
    return out;
}

static inline int cpu_unpack_block(const uint8_t* block, size_t block_len, KVPair* out)
{
    if (block_len < (size_t)PACK_HEADER_BYTES) return 0;
    PackBlockHeader header{};
    std::memcpy(&header, block, PACK_HEADER_BYTES);
    const uint8_t* data = block + PACK_HEADER_BYTES;
    const uint8_t* data_end = data + header.data_size;
    Key128 prev_key{};

    int count = 0;
    const uint8_t* p = data;
    while (count < (int)header.num_entries && p + 2 <= data_end) {
        int shared = p[0];
        int unshared = p[1];
        p += 2;
        if (p + unshared + GP_VALUE_BYTES > data_end) break;

        Key128 current{};
        if ((count % GP_RESTART_INTERVAL) != 0) {
            std::memcpy(current.bytes, prev_key.bytes, (size_t)shared);
        }
        std::memcpy(current.bytes + shared, p, (size_t)unshared);
        p += unshared;
        std::memcpy(out[count].key.bytes, current.bytes, GP_KEY_BYTES);
        std::memcpy(out[count].value.bytes, p, GP_VALUE_BYTES);
        p += GP_VALUE_BYTES;
        prev_key = current;
        ++count;
    }
    return count;
}

__global__ void pack_kernel(const KVPair* __restrict__ kv_array,
                            const uint32_t* __restrict__ block_first_kv,
                            const uint32_t* __restrict__ block_num_kv,
                            int                          num_blocks,
                            uint8_t* __restrict__       block_out,
                            uint32_t* __restrict__      block_sizes)
{
    int block_id = (int)blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = (int)threadIdx.x;
    int first_kv = (int)block_first_kv[block_id];
    int block_entries = (int)block_num_kv[block_id];
    int num_restarts = (block_entries + GP_RESTART_INTERVAL - 1) / GP_RESTART_INTERVAL;

    extern __shared__ uint32_t smem[];
    uint32_t* interval_bytes = smem;
    uint32_t* interval_offsets = smem + num_restarts;
    uint32_t* restart_offsets = smem + 2 * num_restarts;

    const KVPair* block_kv = kv_array + first_kv;

    if (tid < num_restarts) {
        uint32_t bytes = 0;
        int start = tid * GP_RESTART_INTERVAL;
        int end = (start + GP_RESTART_INTERVAL < block_entries)
                ? (start + GP_RESTART_INTERVAL)
                : block_entries;
        for (int i = start; i < end; ++i) {
            int shared = 0;
            if (i > start) shared = key_shared_prefix(block_kv[i - 1].key, block_kv[i].key);
            bytes += (uint32_t)(2 + (GP_KEY_BYTES - shared) + GP_VALUE_BYTES);
        }
        interval_bytes[tid] = bytes;
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t prefix = 0;
        for (int i = 0; i < num_restarts; ++i) {
            restart_offsets[i] = prefix;
            interval_offsets[i] = prefix;
            prefix += interval_bytes[i];
        }

        PackBlockHeader header{};
        header.num_entries = (uint32_t)block_entries;
        header.restart_interval = GP_RESTART_INTERVAL;
        header.num_restarts = (uint32_t)num_restarts;
        header.data_size = prefix;
        uint8_t* dst = block_out + (size_t)block_id * GP_DATA_BLOCK_BYTES;
        const uint8_t* hdr_bytes = reinterpret_cast<const uint8_t*>(&header);
        for (int i = 0; i < PACK_HEADER_BYTES; ++i) dst[i] = hdr_bytes[i];

        uint8_t* restart_dst = dst + PACK_HEADER_BYTES + header.data_size;
        for (int i = 0; i < num_restarts; ++i) {
            uint32_t value = restart_offsets[i];
            const uint8_t* src = reinterpret_cast<const uint8_t*>(&value);
            for (int b = 0; b < (int)sizeof(uint32_t); ++b) {
                restart_dst[i * sizeof(uint32_t) + b] = src[b];
            }
        }
        block_sizes[block_id] = PACK_HEADER_BYTES + header.data_size
                              + (uint32_t)num_restarts * sizeof(uint32_t);
    }
    __syncthreads();

    if (tid < num_restarts) {
        uint8_t* dst = block_out + (size_t)block_id * GP_DATA_BLOCK_BYTES
                     + PACK_HEADER_BYTES + interval_offsets[tid];
        int start = tid * GP_RESTART_INTERVAL;
        int end = (start + GP_RESTART_INTERVAL < block_entries)
                ? (start + GP_RESTART_INTERVAL)
                : block_entries;
        for (int i = start; i < end; ++i) {
            int shared = 0;
            if (i > start) shared = key_shared_prefix(block_kv[i - 1].key, block_kv[i].key);
            int unshared = GP_KEY_BYTES - shared;
            *dst++ = (uint8_t)shared;
            *dst++ = (uint8_t)unshared;
            for (int b = 0; b < unshared; ++b) *dst++ = block_kv[i].key.bytes[shared + b];
            for (int b = 0; b < GP_VALUE_BYTES; ++b) *dst++ = block_kv[i].value.bytes[b];
        }
    }
}

__global__ void unpack_kernel(const uint8_t*  __restrict__ block_buf,
                              const uint32_t* __restrict__ block_offsets,
                              const uint32_t* __restrict__ block_first_kv,
                              const uint32_t* __restrict__ block_num_kv,
                              int                          num_blocks,
                              KVPair*        __restrict__ kv_out)
{
    int block_id = (int)blockIdx.x;
    if (block_id >= num_blocks) return;

    const uint8_t* block = block_buf + block_offsets[block_id];
    PackBlockHeader header{};
    uint8_t* header_bytes = reinterpret_cast<uint8_t*>(&header);
    for (int i = 0; i < PACK_HEADER_BYTES; ++i) header_bytes[i] = block[i];

    int tid = (int)threadIdx.x;
    int block_entries = (int)header.num_entries;
    int num_restarts = (int)header.num_restarts;
    if (tid >= num_restarts) return;

    const uint8_t* restart_src = block + PACK_HEADER_BYTES + header.data_size;
    uint32_t restart_offset = 0;
    uint8_t* restart_bytes = reinterpret_cast<uint8_t*>(&restart_offset);
    for (int i = 0; i < (int)sizeof(uint32_t); ++i) {
        restart_bytes[i] = restart_src[tid * sizeof(uint32_t) + i];
    }

    int start = tid * GP_RESTART_INTERVAL;
    int end = (start + GP_RESTART_INTERVAL < block_entries)
            ? (start + GP_RESTART_INTERVAL)
            : block_entries;
    const uint8_t* p = block + PACK_HEADER_BYTES + restart_offset;
    Key128 prev_key{};
    int kv_base = (int)block_first_kv[block_id];

    for (int i = start; i < end; ++i) {
        int shared = p[0];
        int unshared = p[1];
        p += 2;
        Key128 current{};
        if (i > start) std::memcpy(current.bytes, prev_key.bytes, (size_t)shared);
        std::memcpy(current.bytes + shared, p, (size_t)unshared);
        p += unshared;

        KVPair out{};
        std::memcpy(out.key.bytes, current.bytes, GP_KEY_BYTES);
        std::memcpy(out.value.bytes, p, GP_VALUE_BYTES);
        p += GP_VALUE_BYTES;
        kv_out[kv_base + i] = out;
        prev_key = current;
    }
}

struct PackResult {
    std::vector<uint8_t>            block_buf;
    std::vector<uint32_t>           block_offsets;
    std::vector<uint32_t>           block_sizes;
    std::vector<DataBlockPlanEntry> plans;
};

struct PackTimedResult {
    PackResult result;
    float      kernel_ms = 0.0f;
    float      wall_ms = 0.0f;
};

static inline PackTimedResult launch_pack_timed(const std::vector<KVPair>& kv_array,
                                                const std::vector<DataBlockPlanEntry>& plans)
{
    PackTimedResult timed;
    timed.result.plans = plans;
    if (plans.empty()) return timed;

    std::vector<uint32_t> first_kv(plans.size()), num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
    }

    KVPair*   d_kv = nullptr;
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    uint8_t*  d_blocks = nullptr;
    uint32_t* d_sizes = nullptr;
    cudaMalloc(&d_kv, kv_array.size() * sizeof(KVPair));
    cudaMalloc(&d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_blocks, plans.size() * GP_DATA_BLOCK_BYTES);
    cudaMalloc(&d_sizes, plans.size() * sizeof(uint32_t));

    auto wall_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_kv, kv_array.data(), kv_array.size() * sizeof(KVPair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_first_kv, first_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_kv, num_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t max_restarts = 0;
    for (const auto& p : plans) {
        max_restarts = std::max(max_restarts,
                                (p.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                              / (uint32_t)GP_RESTART_INTERVAL);
    }
    int block_dim = ((int)max_restarts + 31) / 32 * 32;
    if (block_dim < 32) block_dim = 32;
    int shared_bytes = (int)(max_restarts * 3 * sizeof(uint32_t));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    pack_kernel<<<(int)plans.size(), block_dim, shared_bytes>>>(
        d_kv, d_first_kv, d_num_kv, (int)plans.size(), d_blocks, d_sizes);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&timed.kernel_ms, ev0, ev1);

    timed.result.block_sizes.resize(plans.size());
    cudaMemcpy(timed.result.block_sizes.data(), d_sizes,
               plans.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> raw(plans.size() * GP_DATA_BLOCK_BYTES);
    cudaMemcpy(raw.data(), d_blocks, raw.size(), cudaMemcpyDeviceToHost);

    auto wall_end = std::chrono::steady_clock::now();
    timed.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    timed.result.block_offsets.resize(plans.size());
    size_t offset = 0;
    timed.result.block_buf.resize(0);
    for (size_t i = 0; i < plans.size(); ++i) {
        timed.result.block_offsets[i] = (uint32_t)offset;
        timed.result.block_buf.insert(timed.result.block_buf.end(),
                                      raw.begin() + (ptrdiff_t)(i * GP_DATA_BLOCK_BYTES),
                                      raw.begin() + (ptrdiff_t)(i * GP_DATA_BLOCK_BYTES
                                                              + timed.result.block_sizes[i]));
        offset += timed.result.block_sizes[i];
    }

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_sizes);
    cudaFree(d_blocks);
    cudaFree(d_num_kv);
    cudaFree(d_first_kv);
    cudaFree(d_kv);
    return timed;
}

static inline PackResult launch_pack(const std::vector<KVPair>& kv_array,
                                     const std::vector<DataBlockPlanEntry>& plans)
{
    return launch_pack_timed(kv_array, plans).result;
}

static inline PackResult cpu_pack_all(const std::vector<KVPair>& kv_array,
                                      const std::vector<DataBlockPlanEntry>& plans)
{
    PackResult result;
    result.plans = plans;
    size_t offset = 0;
    for (const auto& plan : plans) {
        std::vector<uint8_t> block = cpu_pack_block(kv_array.data() + plan.first_kv, plan.num_kv);
        result.block_offsets.push_back((uint32_t)offset);
        result.block_sizes.push_back((uint32_t)block.size());
        result.block_buf.insert(result.block_buf.end(), block.begin(), block.end());
        offset += block.size();
    }
    return result;
}

struct UnpackTimedResult {
    std::vector<KVPair> kv_array;
    float               kernel_ms = 0.0f;
    float               wall_ms = 0.0f;
};

static inline UnpackTimedResult launch_unpack_timed(const std::vector<uint8_t>&            block_buf,
                                                    const std::vector<uint32_t>&           block_offsets,
                                                    const std::vector<DataBlockPlanEntry>& plans,
                                                    uint32_t                               total_kv)
{
    UnpackTimedResult result;
    result.kv_array.resize(total_kv);
    if (plans.empty()) return result;

    std::vector<uint32_t> first_kv(plans.size()), num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
    }

    uint8_t*  d_buf = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    KVPair*   d_out = nullptr;
    cudaMalloc(&d_buf, block_buf.size());
    cudaMalloc(&d_offsets, block_offsets.size() * sizeof(uint32_t));
    cudaMalloc(&d_first_kv, first_kv.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, num_kv.size() * sizeof(uint32_t));
    cudaMalloc(&d_out, (size_t)total_kv * sizeof(KVPair));

    auto wall_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_buf, block_buf.data(), block_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, block_offsets.data(), block_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_first_kv, first_kv.data(), first_kv.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_kv, num_kv.data(), num_kv.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t max_restarts = 0;
    for (const auto& p : plans) {
        max_restarts = std::max(max_restarts,
                                (p.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                              / (uint32_t)GP_RESTART_INTERVAL);
    }
    int block_dim = ((int)max_restarts + 31) / 32 * 32;
    if (block_dim < 32) block_dim = 32;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    unpack_kernel<<<(int)plans.size(), block_dim>>>(d_buf, d_offsets, d_first_kv, d_num_kv,
                                                    (int)plans.size(), d_out);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    cudaMemcpy(result.kv_array.data(), d_out, (size_t)total_kv * sizeof(KVPair), cudaMemcpyDeviceToHost);
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_out);
    cudaFree(d_num_kv);
    cudaFree(d_first_kv);
    cudaFree(d_offsets);
    cudaFree(d_buf);
    return result;
}

static inline std::vector<KVPair> cpu_unpack_all(const std::vector<uint8_t>&            block_buf,
                                                 const std::vector<uint32_t>&           block_offsets,
                                                 const std::vector<DataBlockPlanEntry>& plans,
                                                 uint32_t                               total_kv)
{
    std::vector<KVPair> out(total_kv);
    for (size_t i = 0; i < plans.size(); ++i) {
        cpu_unpack_block(block_buf.data() + block_offsets[i], plans[i].serialized_size,
                         out.data() + plans[i].first_kv);
    }
    return out;
}
