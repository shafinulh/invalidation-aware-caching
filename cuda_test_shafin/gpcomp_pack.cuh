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
static constexpr int PACK_GROUP_MAX_BYTES = GP_RESTART_INTERVAL * PACK_MAX_ENTRY_BYTES;

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

static inline std::vector<uint32_t> compute_restart_group_sizes_cpu(const std::vector<KVPair>& kv_array)
{
    std::vector<uint32_t> group_sizes;
    if (kv_array.empty()) return group_sizes;

    size_t num_groups = (kv_array.size() + (size_t)GP_RESTART_INTERVAL - 1) / (size_t)GP_RESTART_INTERVAL;
    group_sizes.resize(num_groups);
    for (size_t group = 0; group < num_groups; ++group) {
        size_t start = group * (size_t)GP_RESTART_INTERVAL;
        size_t end = std::min(start + (size_t)GP_RESTART_INTERVAL, kv_array.size());
        uint32_t group_bytes = 0;
        for (size_t i = start; i < end; ++i) {
            int shared = 0;
            if (i > start) shared = key_shared_prefix(kv_array[i - 1].key, kv_array[i].key);
            group_bytes += (uint32_t)(2 + (GP_KEY_BYTES - shared) + GP_VALUE_BYTES);
        }
        group_sizes[group] = group_bytes;
    }
    return group_sizes;
}

static inline std::vector<DataBlockPlanEntry> plan_data_blocks_static(uint32_t total_kv)
{
    std::vector<DataBlockPlanEntry> plans;
    if (total_kv == 0) return plans;

    // Compute a static, group-aligned cap that is always safe in the worst case
    // (no prefix sharing), so we can pack denser than the old 75% heuristic.
    uint32_t worst_entry_size = (uint32_t)(2 + GP_KEY_BYTES + GP_VALUE_BYTES);
    uint32_t fixed_kvs_per_block = 0;
    uint32_t data_size = 0;
    uint32_t num_restarts = 0;
    while (true) {
        uint32_t group_pos = fixed_kvs_per_block % (uint32_t)GP_RESTART_INTERVAL;
        uint32_t candidate_data_size = data_size + worst_entry_size;
        uint32_t candidate_restarts = num_restarts + (group_pos == 0 ? 1u : 0u);
        uint32_t candidate_total = PACK_HEADER_BYTES + candidate_data_size
                                 + candidate_restarts * (uint32_t)sizeof(uint32_t);
        if (candidate_total > GP_DATA_BLOCK_BYTES) break;
        data_size = candidate_data_size;
        num_restarts = candidate_restarts;
        ++fixed_kvs_per_block;
    }
    fixed_kvs_per_block =
        (fixed_kvs_per_block / (uint32_t)GP_RESTART_INTERVAL) * (uint32_t)GP_RESTART_INTERVAL;
    if (fixed_kvs_per_block == 0) fixed_kvs_per_block = (uint32_t)GP_RESTART_INTERVAL;

    uint32_t first_kv = 0;
    while (first_kv < total_kv) {
        uint32_t num_kv = std::min(fixed_kvs_per_block, total_kv - first_kv);
        DataBlockPlanEntry entry;
        entry.first_kv = first_kv;
        entry.num_kv = num_kv;
        // Conservative worst-case estimate for metadata/planning use only.
        uint32_t num_restarts = (num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u) / (uint32_t)GP_RESTART_INTERVAL;
        entry.serialized_size = PACK_HEADER_BYTES + num_kv * worst_entry_size
                              + num_restarts * (uint32_t)sizeof(uint32_t);
        plans.push_back(entry);
        first_kv += num_kv;
    }
    return plans;
}

static inline std::vector<DataBlockPlanEntry>
plan_data_blocks_group_aligned_from_group_sizes(const std::vector<uint32_t>& group_sizes,
                                                uint32_t                     total_kv)
{
    std::vector<DataBlockPlanEntry> plans;
    if (total_kv == 0) return plans;

    uint32_t first_kv = 0;
    size_t group = 0;
    while (first_kv < total_kv && group < group_sizes.size()) {
        uint32_t num_kv = 0;
        uint32_t data_size = 0;
        uint32_t num_restarts = 0;
        while (group < group_sizes.size()) {
            uint32_t remaining_kv = total_kv - (uint32_t)(group * (size_t)GP_RESTART_INTERVAL);
            uint32_t group_num_kv = std::min((uint32_t)GP_RESTART_INTERVAL, remaining_kv);
            uint32_t candidate_data_size = data_size + group_sizes[group];
            uint32_t candidate_restarts = num_restarts + 1u;
            uint32_t candidate_total = PACK_HEADER_BYTES + candidate_data_size
                                     + candidate_restarts * (uint32_t)sizeof(uint32_t);
            if (num_kv > 0 && candidate_total > GP_DATA_BLOCK_BYTES) break;
            data_size = candidate_data_size;
            num_restarts = candidate_restarts;
            num_kv += group_num_kv;
            ++group;
        }

        DataBlockPlanEntry entry{};
        entry.first_kv = first_kv;
        entry.num_kv = num_kv;
        entry.serialized_size = PACK_HEADER_BYTES + data_size + num_restarts * (uint32_t)sizeof(uint32_t);
        plans.push_back(entry);
        first_kv += num_kv;
    }
    return plans;
}

static inline std::vector<DataBlockPlanEntry> plan_data_blocks_group_aligned(const std::vector<KVPair>& kv_array)
{
    return plan_data_blocks_group_aligned_from_group_sizes(
        compute_restart_group_sizes_cpu(kv_array), (uint32_t)kv_array.size());
}

struct RestartGroupSizeTimedResult {
    std::vector<uint32_t> group_sizes;
    float                 kernel_ms = 0.0f;
    float                 wall_ms = 0.0f;
};

struct GroupPrecompressState {
    uint8_t*     d_group_payloads = nullptr;
    uint32_t*    d_group_sizes = nullptr;
    int          num_groups = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t  kernel_start = nullptr;
    cudaEvent_t  kernel_stop = nullptr;
};

struct DevicePlanResult {
    std::vector<DataBlockPlanEntry> plans;
    uint32_t*                       d_first_kv = nullptr;
    uint32_t*                       d_num_kv = nullptr;
    uint32_t*                       d_serialized_size = nullptr;
    int*                            d_num_blocks = nullptr;
    int                             capacity = 0;
    int                             num_blocks = 0;
    float                           kernel_ms = 0.0f;
    float                           wall_ms = 0.0f;
    bool                            owns_device_buffers = true;
};

struct DevicePlanWorkspace {
    uint32_t* d_adjacent_shared = nullptr;
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    uint32_t* d_serialized_size = nullptr;
    int*      d_num_blocks = nullptr;
    int       entry_capacity = 0;
};

static inline DevicePlanWorkspace& device_plan_workspace()
{
    static DevicePlanWorkspace ws{};
    return ws;
}

static inline void ensure_device_plan_workspace(int entry_capacity)
{
    DevicePlanWorkspace& ws = device_plan_workspace();
    if (ws.entry_capacity >= entry_capacity && ws.d_num_blocks != nullptr) return;

    if (ws.d_adjacent_shared) cudaFree(ws.d_adjacent_shared);
    if (ws.d_first_kv) cudaFree(ws.d_first_kv);
    if (ws.d_num_kv) cudaFree(ws.d_num_kv);
    if (ws.d_serialized_size) cudaFree(ws.d_serialized_size);
    if (ws.d_num_blocks) cudaFree(ws.d_num_blocks);

    cudaMalloc(&ws.d_adjacent_shared, (size_t)entry_capacity * sizeof(uint32_t));
    cudaMalloc(&ws.d_first_kv, (size_t)entry_capacity * sizeof(uint32_t));
    cudaMalloc(&ws.d_num_kv, (size_t)entry_capacity * sizeof(uint32_t));
    cudaMalloc(&ws.d_serialized_size, (size_t)entry_capacity * sizeof(uint32_t));
    cudaMalloc(&ws.d_num_blocks, sizeof(int));
    ws.entry_capacity = entry_capacity;
}

__global__ void compute_adjacent_shared_prefix_kernel(const KVPair* __restrict__ kv_array,
                                                      int                        total_kv,
                                                      uint32_t* __restrict__     adjacent_shared)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_kv) return;
    if (idx == 0) {
        adjacent_shared[idx] = 0;
        return;
    }
    adjacent_shared[idx] = (uint32_t)key_shared_prefix(kv_array[idx - 1].key, kv_array[idx].key);
}

__global__ void compute_restart_group_sizes_kernel(const KVPair* __restrict__ kv_array,
                                                   int                        total_kv,
                                                   uint32_t* __restrict__     group_sizes)
{
    int group_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int start = group_id * GP_RESTART_INTERVAL;
    if (start >= total_kv) return;

    int end = start + GP_RESTART_INTERVAL;
    if (end > total_kv) end = total_kv;

    uint32_t group_bytes = 0;
    for (int i = start; i < end; ++i) {
        int shared = 0;
        if (i > start) shared = key_shared_prefix(kv_array[i - 1].key, kv_array[i].key);
        group_bytes += (uint32_t)(2 + (GP_KEY_BYTES - shared) + GP_VALUE_BYTES);
    }
    group_sizes[group_id] = group_bytes;
}

__global__ void precompress_restart_groups_kernel(const KVPair* __restrict__ kv_array,
                                                  int                        total_kv,
                                                  uint8_t* __restrict__      group_payloads,
                                                  uint32_t* __restrict__     group_sizes)
{
    int group_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int start = group_id * GP_RESTART_INTERVAL;
    if (start >= total_kv) return;

    int end = start + GP_RESTART_INTERVAL;
    if (end > total_kv) end = total_kv;

    uint8_t* dst = group_payloads + (size_t)group_id * PACK_GROUP_MAX_BYTES;
    uint8_t* out = dst;
    for (int i = start; i < end; ++i) {
        int shared = 0;
        if (i > start) shared = key_shared_prefix(kv_array[i - 1].key, kv_array[i].key);
        int unshared = GP_KEY_BYTES - shared;
        *out++ = (uint8_t)shared;
        *out++ = (uint8_t)unshared;
        for (int b = 0; b < unshared; ++b) *out++ = kv_array[i].key.bytes[shared + b];
        for (int b = 0; b < GP_VALUE_BYTES; ++b) *out++ = kv_array[i].value.bytes[b];
    }
    group_sizes[group_id] = (uint32_t)(out - dst);
}

__global__ void plan_data_blocks_exact_kernel(const uint32_t* __restrict__ adjacent_shared,
                                              int                          total_kv,
                                              uint32_t* __restrict__       block_first_kv,
                                              uint32_t* __restrict__       block_num_kv,
                                              uint32_t* __restrict__       block_serialized_size,
                                              int* __restrict__            out_num_blocks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int block_id = 0;
    uint32_t first_kv = 0;
    while (first_kv < (uint32_t)total_kv) {
        uint32_t num_kv = 0;
        uint32_t data_size = 0;
        uint32_t num_restarts = 0;
        while (first_kv + num_kv < (uint32_t)total_kv) {
            uint32_t group_pos = num_kv % (uint32_t)GP_RESTART_INTERVAL;
            uint32_t shared = (group_pos == 0) ? 0u : adjacent_shared[first_kv + num_kv];
            uint32_t entry_size = (uint32_t)(2 + (GP_KEY_BYTES - (int)shared) + GP_VALUE_BYTES);
            uint32_t candidate_data_size = data_size + entry_size;
            uint32_t candidate_restarts = num_restarts + (group_pos == 0 ? 1u : 0u);
            uint32_t candidate_total = PACK_HEADER_BYTES + candidate_data_size
                                     + candidate_restarts * sizeof(uint32_t);
            if (num_kv > 0 && candidate_total > GP_DATA_BLOCK_BYTES) break;
            data_size = candidate_data_size;
            num_restarts = candidate_restarts;
            ++num_kv;
        }

        block_first_kv[block_id] = first_kv;
        block_num_kv[block_id] = num_kv;
        block_serialized_size[block_id] =
            PACK_HEADER_BYTES + data_size + num_restarts * (uint32_t)sizeof(uint32_t);
        ++block_id;
        first_kv += num_kv;
    }
    *out_num_blocks = block_id;
}

static inline void destroy_device_plan_result(DevicePlanResult& plan)
{
    if (plan.owns_device_buffers) {
        if (plan.d_num_blocks) cudaFree(plan.d_num_blocks);
        if (plan.d_serialized_size) cudaFree(plan.d_serialized_size);
        if (plan.d_num_kv) cudaFree(plan.d_num_kv);
        if (plan.d_first_kv) cudaFree(plan.d_first_kv);
    }
    plan = DevicePlanResult{};
}

static inline DevicePlanResult launch_plan_data_blocks_timed_from_device(const KVPair* d_kv,
                                                                         int           total_kv)
{
    DevicePlanResult result;
    if (total_kv <= 0) return result;

    ensure_device_plan_workspace(total_kv);
    DevicePlanWorkspace& ws = device_plan_workspace();
    result.capacity = total_kv;
    result.d_first_kv = ws.d_first_kv;
    result.d_num_kv = ws.d_num_kv;
    result.d_serialized_size = ws.d_serialized_size;
    result.d_num_blocks = ws.d_num_blocks;
    result.owns_device_buffers = false;

    auto wall_start = std::chrono::steady_clock::now();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    int block = 256;
    int grid = (total_kv + block - 1) / block;
    compute_adjacent_shared_prefix_kernel<<<grid, block>>>(d_kv, total_kv, ws.d_adjacent_shared);
    plan_data_blocks_exact_kernel<<<1, 1>>>(ws.d_adjacent_shared, total_kv,
                                            result.d_first_kv, result.d_num_kv,
                                            result.d_serialized_size, result.d_num_blocks);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);
    cudaMemcpy(&result.num_blocks, result.d_num_blocks, sizeof(int), cudaMemcpyDeviceToHost);
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    return result;
}

static inline RestartGroupSizeTimedResult launch_restart_group_sizes_timed_from_device(const KVPair* d_kv,
                                                                                        int           total_kv)
{
    RestartGroupSizeTimedResult result;
    if (total_kv <= 0) return result;

    int num_groups = (total_kv + GP_RESTART_INTERVAL - 1) / GP_RESTART_INTERVAL;
    uint32_t* d_group_sizes = nullptr;
    cudaMalloc(&d_group_sizes, (size_t)num_groups * sizeof(uint32_t));
    uint32_t* h_group_sizes = nullptr;
    cudaHostAlloc(&h_group_sizes, (size_t)num_groups * sizeof(uint32_t), cudaHostAllocDefault);

    auto wall_start = std::chrono::steady_clock::now();
    cudaEvent_t ev0, ev1;
    cudaStream_t stream = nullptr;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    int block = 256;
    int grid = (num_groups + block - 1) / block;
    cudaEventRecord(ev0, stream);
    compute_restart_group_sizes_kernel<<<grid, block, 0, stream>>>(d_kv, total_kv, d_group_sizes);
    cudaEventRecord(ev1, stream);
    cudaMemcpyAsync(h_group_sizes, d_group_sizes,
                    (size_t)num_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);
    result.group_sizes.assign(h_group_sizes, h_group_sizes + num_groups);
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_group_sizes);
    cudaFree(d_group_sizes);
    return result;
}

static inline void copy_device_plans_to_host(DevicePlanResult& result)
{
    result.plans.resize((size_t)result.num_blocks);
    if (result.num_blocks == 0) return;

    std::vector<uint32_t> first_kv((size_t)result.num_blocks);
    std::vector<uint32_t> num_kv((size_t)result.num_blocks);
    std::vector<uint32_t> serialized_size((size_t)result.num_blocks);
    cudaMemcpy(first_kv.data(), result.d_first_kv,
               (size_t)result.num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_kv.data(), result.d_num_kv,
               (size_t)result.num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(serialized_size.data(), result.d_serialized_size,
               (size_t)result.num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < result.num_blocks; ++i) {
        result.plans[(size_t)i].first_kv = first_kv[(size_t)i];
        result.plans[(size_t)i].num_kv = num_kv[(size_t)i];
        result.plans[(size_t)i].serialized_size = serialized_size[(size_t)i];
    }
}

struct DevicePlanArrays {
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    int       num_blocks = 0;
};

static inline DevicePlanArrays upload_plans_to_device(const std::vector<DataBlockPlanEntry>& plans)
{
    DevicePlanArrays arrays;
    arrays.num_blocks = (int)plans.size();
    if (plans.empty()) return arrays;

    std::vector<uint32_t> first_kv(plans.size());
    std::vector<uint32_t> num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
    }

    cudaMalloc(&arrays.d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&arrays.d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMemcpy(arrays.d_first_kv, first_kv.data(),
               plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.d_num_kv, num_kv.data(),
               plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    return arrays;
}

static inline void destroy_device_plan_arrays(DevicePlanArrays& arrays)
{
    if (arrays.d_num_kv) cudaFree(arrays.d_num_kv);
    if (arrays.d_first_kv) cudaFree(arrays.d_first_kv);
    arrays = DevicePlanArrays{};
}

static inline void destroy_group_precompress_state(GroupPrecompressState& state)
{
    if (state.kernel_stop) cudaEventDestroy(state.kernel_stop);
    if (state.kernel_start) cudaEventDestroy(state.kernel_start);
    if (state.stream) cudaStreamDestroy(state.stream);
    if (state.d_group_sizes) cudaFree(state.d_group_sizes);
    if (state.d_group_payloads) cudaFree(state.d_group_payloads);
    state = GroupPrecompressState{};
}

static inline GroupPrecompressState launch_group_precompress_async(const KVPair* d_kv,
                                                                   int           total_kv)
{
    GroupPrecompressState state;
    if (total_kv <= 0) return state;

    state.num_groups = (total_kv + GP_RESTART_INTERVAL - 1) / GP_RESTART_INTERVAL;
    cudaMalloc(&state.d_group_payloads, (size_t)state.num_groups * PACK_GROUP_MAX_BYTES);
    cudaMalloc(&state.d_group_sizes, (size_t)state.num_groups * sizeof(uint32_t));
    cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking);
    cudaEventCreate(&state.kernel_start);
    cudaEventCreate(&state.kernel_stop);

    int block = 256;
    int grid = (state.num_groups + block - 1) / block;
    cudaEventRecord(state.kernel_start, state.stream);
    precompress_restart_groups_kernel<<<grid, block, 0, state.stream>>>(
        d_kv, total_kv, state.d_group_payloads, state.d_group_sizes);
    cudaEventRecord(state.kernel_stop, state.stream);
    return state;
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

__global__ void assemble_precompressed_blocks_kernel(const uint8_t* __restrict__  group_payloads,
                                                     const uint32_t* __restrict__ group_sizes,
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
    int first_group = first_kv / GP_RESTART_INTERVAL;
    int num_restarts = (block_entries + GP_RESTART_INTERVAL - 1) / GP_RESTART_INTERVAL;

    extern __shared__ uint32_t smem[];
    uint32_t* interval_bytes = smem;
    uint32_t* interval_offsets = smem + num_restarts;
    uint32_t* restart_offsets = smem + 2 * num_restarts;

    if (tid < num_restarts) interval_bytes[tid] = group_sizes[first_group + tid];
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
        const uint8_t* src = group_payloads + (size_t)(first_group + tid) * PACK_GROUP_MAX_BYTES;
        uint8_t* dst = block_out + (size_t)block_id * GP_DATA_BLOCK_BYTES
                     + PACK_HEADER_BYTES + interval_offsets[tid];
        uint32_t bytes = interval_bytes[tid];
        for (uint32_t i = 0; i < bytes; ++i) dst[i] = src[i];
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

static inline PackTimedResult launch_pack_timed_from_device(const KVPair*                         d_kv,
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

    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    uint8_t*  d_blocks = nullptr;
    uint32_t* d_sizes = nullptr;
    cudaMalloc(&d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_blocks, plans.size() * GP_DATA_BLOCK_BYTES);
    cudaMalloc(&d_sizes, plans.size() * sizeof(uint32_t));

    auto wall_start = std::chrono::steady_clock::now();
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
    timed.result.block_buf.clear();
    timed.result.block_buf.reserve(raw.size());
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
    return timed;
}

static inline PackTimedResult launch_pack_timed_from_device_plans(const KVPair*      d_kv,
                                                                  const uint32_t*    d_first_kv,
                                                                  const uint32_t*    d_num_kv,
                                                                  int                num_blocks)
{
    PackTimedResult timed;
    if (num_blocks <= 0) return timed;

    uint8_t*  d_blocks = nullptr;
    uint32_t* d_sizes = nullptr;
    cudaMalloc(&d_blocks, (size_t)num_blocks * GP_DATA_BLOCK_BYTES);
    cudaMalloc(&d_sizes, (size_t)num_blocks * sizeof(uint32_t));

    std::vector<uint32_t> host_num_kv((size_t)num_blocks);
    cudaMemcpy(host_num_kv.data(), d_num_kv, (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t max_restarts = 0;
    for (uint32_t n : host_num_kv) {
        max_restarts = std::max(max_restarts,
                                (n + (uint32_t)GP_RESTART_INTERVAL - 1u)
                              / (uint32_t)GP_RESTART_INTERVAL);
    }
    int block_dim = ((int)max_restarts + 31) / 32 * 32;
    if (block_dim < 32) block_dim = 32;
    int shared_bytes = (int)(max_restarts * 3 * sizeof(uint32_t));

    auto wall_start = std::chrono::steady_clock::now();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    pack_kernel<<<num_blocks, block_dim, shared_bytes>>>(
        d_kv, d_first_kv, d_num_kv, num_blocks, d_blocks, d_sizes);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&timed.kernel_ms, ev0, ev1);

    timed.result.block_sizes.resize((size_t)num_blocks);
    cudaMemcpy(timed.result.block_sizes.data(), d_sizes,
               (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> raw((size_t)num_blocks * GP_DATA_BLOCK_BYTES);
    cudaMemcpy(raw.data(), d_blocks, raw.size(), cudaMemcpyDeviceToHost);
    auto wall_end = std::chrono::steady_clock::now();
    timed.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    timed.result.block_offsets.resize((size_t)num_blocks);
    size_t offset = 0;
    timed.result.block_buf.clear();
    timed.result.block_buf.reserve(raw.size());
    for (int i = 0; i < num_blocks; ++i) {
        timed.result.block_offsets[(size_t)i] = (uint32_t)offset;
        timed.result.block_buf.insert(timed.result.block_buf.end(),
                                      raw.begin() + (ptrdiff_t)((size_t)i * GP_DATA_BLOCK_BYTES),
                                      raw.begin() + (ptrdiff_t)((size_t)i * GP_DATA_BLOCK_BYTES
                                                              + timed.result.block_sizes[(size_t)i]));
        offset += timed.result.block_sizes[(size_t)i];
    }

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_sizes);
    cudaFree(d_blocks);
    return timed;
}

struct PackStreamState {
    size_t       block_begin = 0;
    size_t       block_end = 0;
    int          block_count = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t  start = nullptr;
    cudaEvent_t  stop = nullptr;
    uint8_t*     d_blocks = nullptr;
    uint32_t*    d_sizes = nullptr;
    uint8_t*     h_raw = nullptr;
    uint32_t*    h_sizes = nullptr;
};

static inline void destroy_pack_stream_state(PackStreamState& state)
{
    if (state.h_sizes) cudaFreeHost(state.h_sizes);
    if (state.h_raw) cudaFreeHost(state.h_raw);
    if (state.d_sizes) cudaFree(state.d_sizes);
    if (state.d_blocks) cudaFree(state.d_blocks);
    if (state.stop) cudaEventDestroy(state.stop);
    if (state.start) cudaEventDestroy(state.start);
    if (state.stream) cudaStreamDestroy(state.stream);
    state = PackStreamState{};
}

static inline PackTimedResult
launch_pack_timed_from_device_plan_spans(const KVPair*                               d_kv,
                                         const uint32_t*                             d_first_kv,
                                         const uint32_t*                             d_num_kv,
                                         const std::vector<DataBlockPlanEntry>&      plans,
                                         const std::vector<std::pair<size_t, size_t>>& spans)
{
    PackTimedResult timed;
    timed.result.plans = plans;
    if (plans.empty()) return timed;

    std::vector<PackStreamState> states(spans.size());
    auto wall_start = std::chrono::steady_clock::now();

    for (size_t s = 0; s < spans.size(); ++s) {
        PackStreamState& state = states[s];
        state.block_begin = spans[s].first;
        state.block_end = spans[s].second;
        state.block_count = (int)(state.block_end - state.block_begin);
        if (state.block_count <= 0) continue;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.start);
        cudaEventCreate(&state.stop);
        cudaMalloc(&state.d_blocks, (size_t)state.block_count * GP_DATA_BLOCK_BYTES);
        cudaMalloc(&state.d_sizes, (size_t)state.block_count * sizeof(uint32_t));
        cudaHostAlloc(&state.h_raw, (size_t)state.block_count * GP_DATA_BLOCK_BYTES, cudaHostAllocDefault);
        cudaHostAlloc(&state.h_sizes, (size_t)state.block_count * sizeof(uint32_t), cudaHostAllocDefault);

        uint32_t max_restarts = 0;
        for (size_t i = state.block_begin; i < state.block_end; ++i) {
            max_restarts = std::max(max_restarts,
                                    (plans[i].num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;
        int shared_bytes = (int)(max_restarts * 3 * sizeof(uint32_t));

        cudaEventRecord(state.start, state.stream);
        pack_kernel<<<state.block_count, block_dim, shared_bytes, state.stream>>>(
            d_kv,
            d_first_kv + state.block_begin,
            d_num_kv + state.block_begin,
            state.block_count,
            state.d_blocks,
            state.d_sizes);
        cudaMemcpyAsync(state.h_sizes, state.d_sizes,
                        (size_t)state.block_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        state.stream);
        cudaMemcpyAsync(state.h_raw, state.d_blocks,
                        (size_t)state.block_count * GP_DATA_BLOCK_BYTES,
                        cudaMemcpyDeviceToHost,
                        state.stream);
        cudaEventRecord(state.stop, state.stream);
    }

    timed.result.block_sizes.reserve(plans.size());
    timed.result.block_offsets.reserve(plans.size());
    timed.result.block_buf.reserve(plans.size() * GP_DATA_BLOCK_BYTES);
    size_t global_offset = 0;

    for (PackStreamState& state : states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float stream_kernel_ms = 0.0f;
        cudaEventElapsedTime(&stream_kernel_ms, state.start, state.stop);
        timed.kernel_ms = std::max(timed.kernel_ms, stream_kernel_ms);

        for (int i = 0; i < state.block_count; ++i) {
            uint32_t block_size = state.h_sizes[i];
            timed.result.block_sizes.push_back(block_size);
            timed.result.block_offsets.push_back((uint32_t)global_offset);
            timed.result.block_buf.insert(timed.result.block_buf.end(),
                                          state.h_raw + (size_t)i * GP_DATA_BLOCK_BYTES,
                                          state.h_raw + (size_t)i * GP_DATA_BLOCK_BYTES + block_size);
            global_offset += block_size;
        }
        destroy_pack_stream_state(state);
    }

    auto wall_end = std::chrono::steady_clock::now();
    timed.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    return timed;
}

static inline PackTimedResult
launch_pack_timed_from_precompressed_groups_plan_spans(const uint8_t*                             d_group_payloads,
                                                       const uint32_t*                            d_group_sizes,
                                                       const uint32_t*                            d_first_kv,
                                                       const uint32_t*                            d_num_kv,
                                                       const std::vector<DataBlockPlanEntry>&     plans,
                                                       const std::vector<std::pair<size_t, size_t>>& spans)
{
    PackTimedResult timed;
    timed.result.plans = plans;
    if (plans.empty()) return timed;

    std::vector<PackStreamState> states(spans.size());
    auto wall_start = std::chrono::steady_clock::now();

    for (size_t s = 0; s < spans.size(); ++s) {
        PackStreamState& state = states[s];
        state.block_begin = spans[s].first;
        state.block_end = spans[s].second;
        state.block_count = (int)(state.block_end - state.block_begin);
        if (state.block_count <= 0) continue;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.start);
        cudaEventCreate(&state.stop);
        cudaMalloc(&state.d_blocks, (size_t)state.block_count * GP_DATA_BLOCK_BYTES);
        cudaMalloc(&state.d_sizes, (size_t)state.block_count * sizeof(uint32_t));
        cudaHostAlloc(&state.h_raw, (size_t)state.block_count * GP_DATA_BLOCK_BYTES, cudaHostAllocDefault);
        cudaHostAlloc(&state.h_sizes, (size_t)state.block_count * sizeof(uint32_t), cudaHostAllocDefault);

        uint32_t max_restarts = 0;
        for (size_t i = state.block_begin; i < state.block_end; ++i) {
            max_restarts = std::max(max_restarts,
                                    (plans[i].num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;
        int shared_bytes = (int)(max_restarts * 3 * sizeof(uint32_t));

        cudaEventRecord(state.start, state.stream);
        assemble_precompressed_blocks_kernel<<<state.block_count, block_dim, shared_bytes, state.stream>>>(
            d_group_payloads,
            d_group_sizes,
            d_first_kv + state.block_begin,
            d_num_kv + state.block_begin,
            state.block_count,
            state.d_blocks,
            state.d_sizes);
        cudaMemcpyAsync(state.h_sizes, state.d_sizes,
                        (size_t)state.block_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        state.stream);
        cudaMemcpyAsync(state.h_raw, state.d_blocks,
                        (size_t)state.block_count * GP_DATA_BLOCK_BYTES,
                        cudaMemcpyDeviceToHost,
                        state.stream);
        cudaEventRecord(state.stop, state.stream);
    }

    timed.result.block_sizes.reserve(plans.size());
    timed.result.block_offsets.reserve(plans.size());
    timed.result.block_buf.reserve(plans.size() * GP_DATA_BLOCK_BYTES);
    size_t global_offset = 0;

    for (PackStreamState& state : states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float stream_kernel_ms = 0.0f;
        cudaEventElapsedTime(&stream_kernel_ms, state.start, state.stop);
        timed.kernel_ms = std::max(timed.kernel_ms, stream_kernel_ms);

        for (int i = 0; i < state.block_count; ++i) {
            uint32_t block_size = state.h_sizes[i];
            timed.result.block_sizes.push_back(block_size);
            timed.result.block_offsets.push_back((uint32_t)global_offset);
            timed.result.block_buf.insert(timed.result.block_buf.end(),
                                          state.h_raw + (size_t)i * GP_DATA_BLOCK_BYTES,
                                          state.h_raw + (size_t)i * GP_DATA_BLOCK_BYTES + block_size);
            global_offset += block_size;
        }
        destroy_pack_stream_state(state);
    }

    auto wall_end = std::chrono::steady_clock::now();
    timed.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    return timed;
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
