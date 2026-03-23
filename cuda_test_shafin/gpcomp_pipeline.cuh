#pragma once

#include "gpcomp_sstable.cuh"
#include "gpcomp_merge.cuh"

#include <chrono>

struct CompactionStageTimes {
    double unpack_ms = 0.0;
    double merge_ms = 0.0;
    double planning_ms = 0.0;
    double bloom_ms = 0.0;
    double pack_ms = 0.0;
};

struct CPUCompactionResult {
    std::vector<std::vector<KVPair>> unpacked;
    std::vector<KVPair>              merged;
    SSTBuildSet                      output;
    CompactionStageTimes             stage;
};

struct GPUCompactionResult {
    std::vector<std::vector<KVPair>> unpacked;
    std::vector<KVPair>              merged;
    SSTBuildSet                      output;
    CompactionStageTimes             stage;
    float                            unpack_kernel_ms = 0.0f;
    float                            merge_kernel_ms = 0.0f;
    float                            bloom_kernel_ms = 0.0f;
    float                            pack_kernel_ms = 0.0f;
};

struct GPUUnpackStreamState {
    std::vector<uint32_t> block_offsets;
    std::vector<uint32_t> first_kv;
    std::vector<uint32_t> num_kv;
    uint8_t*              d_buf = nullptr;
    uint32_t*             d_offsets = nullptr;
    uint32_t*             d_first_kv = nullptr;
    uint32_t*             d_num_kv = nullptr;
    KVPair*               d_out = nullptr;
    uint32_t              total_kv = 0;
    int                   num_blocks = 0;
    cudaStream_t          stream = nullptr;
    cudaEvent_t           kernel_start = nullptr;
    cudaEvent_t           kernel_stop = nullptr;
};

static inline void destroy_unpack_stream_state(GPUUnpackStreamState& state)
{
    if (state.kernel_stop) cudaEventDestroy(state.kernel_stop);
    if (state.kernel_start) cudaEventDestroy(state.kernel_start);
    if (state.stream) cudaStreamDestroy(state.stream);
    if (state.d_out) cudaFree(state.d_out);
    if (state.d_num_kv) cudaFree(state.d_num_kv);
    if (state.d_first_kv) cudaFree(state.d_first_kv);
    if (state.d_offsets) cudaFree(state.d_offsets);
    if (state.d_buf) cudaFree(state.d_buf);
    state = GPUUnpackStreamState{};
}

__global__ static void gather_largest_keys_kernel(const KVPair* __restrict__   kv_array,
                                                  const uint32_t* __restrict__ block_first_kv,
                                                  const uint32_t* __restrict__ block_num_kv,
                                                  int                          num_blocks,
                                                  Key128* __restrict__         largest_keys)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= num_blocks) return;
    int last = (int)block_first_kv[idx] + (int)block_num_kv[idx] - 1;
    largest_keys[idx] = kv_array[last].key;
}

static inline std::vector<Key128> copy_largest_keys_from_device(const KVPair*   d_kv,
                                                                const uint32_t* d_first_kv,
                                                                const uint32_t* d_num_kv,
                                                                int             num_blocks)
{
    std::vector<Key128> largest_keys((size_t)num_blocks);
    if (num_blocks <= 0) return largest_keys;

    Key128* d_largest_keys = nullptr;
    cudaMalloc(&d_largest_keys, (size_t)num_blocks * sizeof(Key128));
    int block = 256;
    int grid = (num_blocks + block - 1) / block;
    gather_largest_keys_kernel<<<grid, block>>>(d_kv, d_first_kv, d_num_kv, num_blocks, d_largest_keys);
    cudaMemcpy(largest_keys.data(), d_largest_keys,
               (size_t)num_blocks * sizeof(Key128), cudaMemcpyDeviceToHost);
    cudaFree(d_largest_keys);
    return largest_keys;
}

static inline CPUCompactionResult cpu_q_compaction_from_parsed(const std::vector<ParsedSST>& inputs)
{
    CPUCompactionResult result;
    result.unpacked.resize(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) result.unpacked[i] = cpu_unpack_sst(inputs[i]);
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    result.merged = cpu_merge_reference(result.unpacked);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(result.merged);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(result.merged, plans, filter_offsets, filter_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    PackResult pack = cpu_pack_all(result.merged, plans);
    result.output = assemble_sst_files_targeted(result.merged, pack, filter_bytes, filter_offsets, filter_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

static inline CPUCompactionResult cpu_q_compaction_paper_from_parsed(const std::vector<ParsedSST>& inputs)
{
    CPUCompactionResult result;
    result.unpacked.resize(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) result.unpacked[i] = cpu_unpack_sst(inputs[i]);
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    result.merged = cpu_merge_reference(result.unpacked);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks_group_aligned(result.merged);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(result.merged, plans, filter_offsets, filter_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    PackResult pack = cpu_pack_all(result.merged, plans);
    result.output = assemble_sst_files_targeted(result.merged, pack, filter_bytes, filter_offsets, filter_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

static inline GPUCompactionResult gpu_q_compaction_from_parsed(const std::vector<ParsedSST>& inputs)
{
    GPUCompactionResult result;
    std::vector<GPUUnpackStreamState> unpack_states(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        GPUUnpackStreamState& state = unpack_states[i];
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(inputs[i]);
        state.block_offsets = block_offsets_from_parsed(inputs[i]);
        state.first_kv.resize(plans.size());
        state.num_kv.resize(plans.size());
        state.total_kv = inputs[i].footer.total_kv;
        state.num_blocks = (int)plans.size();
        for (size_t p = 0; p < plans.size(); ++p) {
            state.first_kv[p] = plans[p].first_kv;
            state.num_kv[p] = plans[p].num_kv;
        }
        if (state.num_blocks == 0) continue;

        uint32_t max_restarts = 0;
        for (const auto& plan : plans) {
            max_restarts = std::max(max_restarts,
                                    (plan.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.kernel_start);
        cudaEventCreate(&state.kernel_stop);
        cudaMalloc(&state.d_buf, inputs[i].file_bytes.size());
        cudaMalloc(&state.d_offsets, state.block_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_first_kv, state.first_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_num_kv, state.num_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_out, (size_t)state.total_kv * sizeof(KVPair));

        cudaMemcpyAsync(state.d_buf, inputs[i].file_bytes.data(), inputs[i].file_bytes.size(),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_offsets, state.block_offsets.data(),
                        state.block_offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_first_kv, state.first_kv.data(),
                        state.first_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_num_kv, state.num_kv.data(),
                        state.num_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);

        cudaEventRecord(state.kernel_start, state.stream);
        unpack_kernel<<<state.num_blocks, block_dim, 0, state.stream>>>(
            state.d_buf, state.d_offsets, state.d_first_kv, state.d_num_kv, state.num_blocks, state.d_out);
        cudaEventRecord(state.kernel_stop, state.stream);
    }
    for (auto& state : unpack_states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, state.kernel_start, state.kernel_stop);
        result.unpack_kernel_ms += kernel_ms;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<KVPair*> d_unpacked;
    std::vector<int> unpack_sizes;
    d_unpacked.reserve(unpack_states.size());
    unpack_sizes.reserve(unpack_states.size());
    for (const auto& state : unpack_states) {
        d_unpacked.push_back(state.d_out);
        unpack_sizes.push_back((int)state.total_kv);
    }

    t0 = std::chrono::steady_clock::now();
    DeviceMergeTimedResult merged = launch_merge_timed_from_device(d_unpacked, unpack_sizes);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.merge_kernel_ms = merged.kernel_ms;
    result.merged = std::move(merged.merged);
    for (auto& state : unpack_states) destroy_unpack_stream_state(state);

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(result.merged);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    BloomBatchResult bloom = launch_bloom_filter_batched_from_device(merged.d_output, plans);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.bloom_kernel_ms = bloom.kernel_ms;

    t0 = std::chrono::steady_clock::now();
    PackTimedResult pack = launch_pack_timed_from_device(merged.d_output, plans);
    result.output = assemble_sst_files_targeted_gpu_fast(result.merged,
                                                         pack.result,
                                                         bloom.filter_bytes,
                                                         bloom.bitvec_offsets,
                                                         bloom.bitvec_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.pack_kernel_ms = pack.kernel_ms;
    cudaFree(merged.d_output);
    return result;
}

static inline GPUCompactionResult gpu_q_compaction_pipeline_from_parsed(const std::vector<ParsedSST>& inputs)
{
    GPUCompactionResult result;
    std::vector<GPUUnpackStreamState> unpack_states(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        GPUUnpackStreamState& state = unpack_states[i];
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(inputs[i]);
        state.block_offsets = block_offsets_from_parsed(inputs[i]);
        state.first_kv.resize(plans.size());
        state.num_kv.resize(plans.size());
        state.total_kv = inputs[i].footer.total_kv;
        state.num_blocks = (int)plans.size();
        for (size_t p = 0; p < plans.size(); ++p) {
            state.first_kv[p] = plans[p].first_kv;
            state.num_kv[p] = plans[p].num_kv;
        }
        if (state.num_blocks == 0) continue;

        uint32_t max_restarts = 0;
        for (const auto& plan : plans) {
            max_restarts = std::max(max_restarts,
                                    (plan.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.kernel_start);
        cudaEventCreate(&state.kernel_stop);
        cudaMalloc(&state.d_buf, inputs[i].file_bytes.size());
        cudaMalloc(&state.d_offsets, state.block_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_first_kv, state.first_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_num_kv, state.num_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_out, (size_t)state.total_kv * sizeof(KVPair));

        cudaMemcpyAsync(state.d_buf, inputs[i].file_bytes.data(), inputs[i].file_bytes.size(),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_offsets, state.block_offsets.data(),
                        state.block_offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_first_kv, state.first_kv.data(),
                        state.first_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_num_kv, state.num_kv.data(),
                        state.num_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);

        cudaEventRecord(state.kernel_start, state.stream);
        unpack_kernel<<<state.num_blocks, block_dim, 0, state.stream>>>(
            state.d_buf, state.d_offsets, state.d_first_kv, state.d_num_kv, state.num_blocks, state.d_out);
        cudaEventRecord(state.kernel_stop, state.stream);
    }
    for (auto& state : unpack_states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, state.kernel_start, state.kernel_stop);
        result.unpack_kernel_ms += kernel_ms;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<KVPair*> d_unpacked;
    std::vector<int> unpack_sizes;
    d_unpacked.reserve(unpack_states.size());
    unpack_sizes.reserve(unpack_states.size());
    for (const auto& state : unpack_states) {
        d_unpacked.push_back(state.d_out);
        unpack_sizes.push_back((int)state.total_kv);
    }

    t0 = std::chrono::steady_clock::now();
    DeviceMergeTimedResult merged = launch_merge_timed_from_device(d_unpacked, unpack_sizes, false);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.merge_kernel_ms = merged.kernel_ms;
    for (auto& state : unpack_states) destroy_unpack_stream_state(state);

    t0 = std::chrono::steady_clock::now();
    DevicePlanResult plan = launch_plan_data_blocks_timed_from_device(merged.d_output, merged.total);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    copy_device_plans_to_host(plan);

    t0 = std::chrono::steady_clock::now();
    BloomBatchResult bloom = launch_bloom_filter_batched_from_device_plans(
        merged.d_output, plan.d_first_kv, plan.d_num_kv, plan.num_blocks);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.bloom_kernel_ms = bloom.kernel_ms;

    PackResult planned_layout;
    planned_layout.plans = plan.plans;
    planned_layout.block_sizes.resize(plan.plans.size());
    std::vector<uint32_t> predicted_filter_lengths(plan.plans.size());
    for (size_t i = 0; i < plan.plans.size(); ++i) {
        planned_layout.block_sizes[i] = plan.plans[i].serialized_size;
        uint32_t byte_vector_len = plan.plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        predicted_filter_lengths[i] = (byte_vector_len + 7u) / 8u;
    }
    // Match the paper's pack structure: one grid per output SST file, one CUDA stream per grid.
    std::vector<std::pair<size_t, size_t>> pack_spans =
        partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

    t0 = std::chrono::steady_clock::now();
    PackTimedResult pack = launch_pack_timed_from_device_plan_spans(
        merged.d_output, plan.d_first_kv, plan.d_num_kv, plan.plans, pack_spans);
    std::vector<Key128> largest_keys =
        copy_largest_keys_from_device(merged.d_output, plan.d_first_kv, plan.d_num_kv, plan.num_blocks);
    result.output = assemble_sst_files_targeted_from_largest_keys(largest_keys,
                                                                  pack.result,
                                                                  bloom.filter_bytes,
                                                                  bloom.bitvec_offsets,
                                                                  bloom.bitvec_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.pack_kernel_ms = pack.kernel_ms;

    destroy_device_plan_result(plan);
    cudaFree(merged.d_output);
    return result;
}

static inline GPUCompactionResult gpu_q_compaction_paper_from_parsed(const std::vector<ParsedSST>& inputs)
{
    GPUCompactionResult result;
    std::vector<GPUUnpackStreamState> unpack_states(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        GPUUnpackStreamState& state = unpack_states[i];
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(inputs[i]);
        state.block_offsets = block_offsets_from_parsed(inputs[i]);
        state.first_kv.resize(plans.size());
        state.num_kv.resize(plans.size());
        state.total_kv = inputs[i].footer.total_kv;
        state.num_blocks = (int)plans.size();
        for (size_t p = 0; p < plans.size(); ++p) {
            state.first_kv[p] = plans[p].first_kv;
            state.num_kv[p] = plans[p].num_kv;
        }
        if (state.num_blocks == 0) continue;

        uint32_t max_restarts = 0;
        for (const auto& plan : plans) {
            max_restarts = std::max(max_restarts,
                                    (plan.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.kernel_start);
        cudaEventCreate(&state.kernel_stop);
        cudaMalloc(&state.d_buf, inputs[i].file_bytes.size());
        cudaMalloc(&state.d_offsets, state.block_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_first_kv, state.first_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_num_kv, state.num_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_out, (size_t)state.total_kv * sizeof(KVPair));

        cudaMemcpyAsync(state.d_buf, inputs[i].file_bytes.data(), inputs[i].file_bytes.size(),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_offsets, state.block_offsets.data(),
                        state.block_offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_first_kv, state.first_kv.data(),
                        state.first_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_num_kv, state.num_kv.data(),
                        state.num_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);

        cudaEventRecord(state.kernel_start, state.stream);
        unpack_kernel<<<state.num_blocks, block_dim, 0, state.stream>>>(
            state.d_buf, state.d_offsets, state.d_first_kv, state.d_num_kv, state.num_blocks, state.d_out);
        cudaEventRecord(state.kernel_stop, state.stream);
    }
    for (auto& state : unpack_states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, state.kernel_start, state.kernel_stop);
        result.unpack_kernel_ms += kernel_ms;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<KVPair*> d_unpacked;
    std::vector<int> unpack_sizes;
    d_unpacked.reserve(unpack_states.size());
    unpack_sizes.reserve(unpack_states.size());
    for (const auto& state : unpack_states) {
        d_unpacked.push_back(state.d_out);
        unpack_sizes.push_back((int)state.total_kv);
    }

    t0 = std::chrono::steady_clock::now();
    DeviceMergeTimedResult merged = launch_merge_timed_from_device(d_unpacked, unpack_sizes, false);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.merge_kernel_ms = merged.kernel_ms;
    for (auto& state : unpack_states) destroy_unpack_stream_state(state);

    t0 = std::chrono::steady_clock::now();
    RestartGroupSizeTimedResult group_sizes =
        launch_restart_group_sizes_timed_from_device(merged.d_output, merged.total);
    std::vector<DataBlockPlanEntry> plans =
        plan_data_blocks_group_aligned_from_group_sizes(group_sizes.group_sizes, (uint32_t)merged.total);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    DevicePlanArrays device_plans = upload_plans_to_device(plans);

    t0 = std::chrono::steady_clock::now();
    BloomBatchResult bloom = launch_bloom_filter_batched_from_device_plans(
        merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, device_plans.num_blocks);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.bloom_kernel_ms = bloom.kernel_ms;

    PackResult planned_layout;
    planned_layout.plans = plans;
    planned_layout.block_sizes.resize(plans.size());
    std::vector<uint32_t> predicted_filter_lengths(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        planned_layout.block_sizes[i] = plans[i].serialized_size;
        uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        predicted_filter_lengths[i] = (byte_vector_len + 7u) / 8u;
    }
    std::vector<std::pair<size_t, size_t>> pack_spans =
        partition_output_blocks(planned_layout, predicted_filter_lengths, GP_TARGET_FILE_BYTES);

    t0 = std::chrono::steady_clock::now();
    PackTimedResult pack = launch_pack_timed_from_device_plan_spans(
        merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, plans, pack_spans);
    std::vector<Key128> largest_keys =
        copy_largest_keys_from_device(merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv,
                                      device_plans.num_blocks);
    result.output = assemble_sst_files_targeted_from_largest_keys(largest_keys,
                                                                  pack.result,
                                                                  bloom.filter_bytes,
                                                                  bloom.bitvec_offsets,
                                                                  bloom.bitvec_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.pack_kernel_ms = pack.kernel_ms;

    destroy_device_plan_arrays(device_plans);
    cudaFree(merged.d_output);
    return result;
}

static inline CPUCompactionResult cpu_q_compaction_without_plan_from_parsed(const std::vector<ParsedSST>& inputs)
{
    CPUCompactionResult result;
    result.unpacked.resize(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) result.unpacked[i] = cpu_unpack_sst(inputs[i]);
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    result.merged = cpu_merge_reference(result.unpacked);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks_static((uint32_t)result.merged.size());
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(result.merged, plans, filter_offsets, filter_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    PackResult pack = cpu_pack_all(result.merged, plans);
    result.output = assemble_sst_files_targeted(result.merged, pack, filter_bytes, filter_offsets, filter_lengths, GP_TARGET_FILE_BYTES);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

static inline GPUCompactionResult gpu_q_compaction_without_plan_from_parsed(const std::vector<ParsedSST>& inputs)
{
    GPUCompactionResult result;
    std::vector<GPUUnpackStreamState> unpack_states(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        GPUUnpackStreamState& state = unpack_states[i];
        std::vector<DataBlockPlanEntry> plans = plans_from_parsed(inputs[i]);
        state.block_offsets = block_offsets_from_parsed(inputs[i]);
        state.first_kv.resize(plans.size());
        state.num_kv.resize(plans.size());
        state.total_kv = inputs[i].footer.total_kv;
        state.num_blocks = (int)plans.size();
        for (size_t p = 0; p < plans.size(); ++p) {
            state.first_kv[p] = plans[p].first_kv;
            state.num_kv[p] = plans[p].num_kv;
        }
        if (state.num_blocks == 0) continue;

        uint32_t max_restarts = 0;
        for (const auto& plan : plans) {
            max_restarts = std::max(max_restarts,
                                    (plan.num_kv + (uint32_t)GP_RESTART_INTERVAL - 1u)
                                  / (uint32_t)GP_RESTART_INTERVAL);
        }
        int block_dim = ((int)max_restarts + 31) / 32 * 32;
        if (block_dim < 32) block_dim = 32;

        cudaStreamCreate(&state.stream);
        cudaEventCreate(&state.kernel_start);
        cudaEventCreate(&state.kernel_stop);
        cudaMalloc(&state.d_buf, inputs[i].file_bytes.size());
        cudaMalloc(&state.d_offsets, state.block_offsets.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_first_kv, state.first_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_num_kv, state.num_kv.size() * sizeof(uint32_t));
        cudaMalloc(&state.d_out, (size_t)state.total_kv * sizeof(KVPair));

        cudaMemcpyAsync(state.d_buf, inputs[i].file_bytes.data(), inputs[i].file_bytes.size(),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_offsets, state.block_offsets.data(),
                        state.block_offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_first_kv, state.first_kv.data(),
                        state.first_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);
        cudaMemcpyAsync(state.d_num_kv, state.num_kv.data(),
                        state.num_kv.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream);

        cudaEventRecord(state.kernel_start, state.stream);
        unpack_kernel<<<state.num_blocks, block_dim, 0, state.stream>>>(
            state.d_buf, state.d_offsets, state.d_first_kv, state.d_num_kv, state.num_blocks, state.d_out);
        cudaEventRecord(state.kernel_stop, state.stream);
    }
    for (auto& state : unpack_states) {
        if (!state.stream) continue;
        cudaStreamSynchronize(state.stream);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, state.kernel_start, state.kernel_stop);
        result.unpack_kernel_ms += kernel_ms;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<KVPair*> d_unpacked;
    std::vector<int> unpack_sizes;
    d_unpacked.reserve(unpack_states.size());
    unpack_sizes.reserve(unpack_states.size());
    for (const auto& state : unpack_states) {
        d_unpacked.push_back(state.d_out);
        unpack_sizes.push_back((int)state.total_kv);
    }

    t0 = std::chrono::steady_clock::now();
    DeviceMergeTimedResult merged = launch_merge_timed_from_device(d_unpacked, unpack_sizes, false);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.merge_kernel_ms = merged.kernel_ms;
    for (auto& state : unpack_states) destroy_unpack_stream_state(state);

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks_static((uint32_t)merged.total);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    DevicePlanArrays device_plans = upload_plans_to_device(plans);

    t0 = std::chrono::steady_clock::now();
    BloomBatchResult bloom = launch_bloom_filter_batched_from_device_plans(
        merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv, device_plans.num_blocks);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.bloom_kernel_ms = bloom.kernel_ms;

    t0 = std::chrono::steady_clock::now();
    PackTimedResult pack =
        launch_pack_timed_from_device_plans(merged.d_output,
                                            device_plans.d_first_kv,
                                            device_plans.d_num_kv,
                                            device_plans.num_blocks);
    pack.result.plans = plans;
    std::vector<Key128> largest_keys =
        copy_largest_keys_from_device(merged.d_output, device_plans.d_first_kv, device_plans.d_num_kv,
                                      device_plans.num_blocks);
    result.output = assemble_sst_files_targeted_from_largest_keys(largest_keys,
                                                                  pack.result,
                                                                  bloom.filter_bytes,
                                                                  bloom.bitvec_offsets,
                                                                  bloom.bitvec_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.pack_kernel_ms = pack.kernel_ms;

    destroy_device_plan_arrays(device_plans);
    cudaFree(merged.d_output);
    return result;
}


