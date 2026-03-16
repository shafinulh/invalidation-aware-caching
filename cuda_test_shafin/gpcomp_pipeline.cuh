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

static inline GPUCompactionResult gpu_q_compaction_from_parsed(const std::vector<ParsedSST>& inputs)
{
    GPUCompactionResult result;
    result.unpacked.resize(inputs.size());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        UnpackTimedResult unpack = launch_unpack_timed(inputs[i].file_bytes,
                                                       block_offsets_from_parsed(inputs[i]),
                                                       plans_from_parsed(inputs[i]),
                                                       inputs[i].footer.total_kv);
        result.unpacked[i] = std::move(unpack.kv_array);
        result.unpack_kernel_ms += unpack.kernel_ms;
    }
    auto t1 = std::chrono::steady_clock::now();
    result.stage.unpack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    MergeTimedResult merged = launch_merge_timed(result.unpacked);
    t1 = std::chrono::steady_clock::now();
    result.stage.merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.merge_kernel_ms = merged.kernel_ms;
    result.merged = std::move(merged.merged);

    t0 = std::chrono::steady_clock::now();
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(result.merged);
    t1 = std::chrono::steady_clock::now();
    result.stage.planning_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    BloomBatchResult bloom = launch_bloom_filter_batched(result.merged, plans);
    t1 = std::chrono::steady_clock::now();
    result.stage.bloom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.bloom_kernel_ms = bloom.kernel_ms;

    t0 = std::chrono::steady_clock::now();
    PackTimedResult pack = launch_pack_timed(result.merged, plans);
    result.output = assemble_sst_files_targeted(result.merged,
                                                pack.result,
                                                bloom.filter_bytes,
                                                bloom.bitvec_offsets,
                                                bloom.bitvec_lengths);
    t1 = std::chrono::steady_clock::now();
    result.stage.pack_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.pack_kernel_ms = pack.kernel_ms;
    return result;
}
