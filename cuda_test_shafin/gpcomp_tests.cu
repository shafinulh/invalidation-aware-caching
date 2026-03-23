#include "gpcomp_pipeline.cuh"
#include "gpcomp_dummy_data.cuh"

#include <cstdio>

static int g_passed = 0;
static int g_failed = 0;

static bool check(bool cond, const char* msg)
{
    if (cond) {
        std::printf("  [PASS] %s\n", msg);
        ++g_passed;
    } else {
        std::printf("  [FAIL] %s\n", msg);
        ++g_failed;
    }
    return cond;
}

static std::vector<KVPair> make_test_array(uint32_t count, uint32_t sst_id, uint64_t base)
{
    std::vector<KVPair> out(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t user_key = base + (uint64_t)i * 3;
        out[i].key = make_internal_key(user_key, sst_id, i);
        out[i].value = make_value_payload(user_key, sst_id, i);
    }
    return out;
}

static bool test_algorithm1_merge()
{
    std::vector<std::vector<KVPair>> arrays;
    arrays.push_back(make_test_array(32, 0, 10));
    arrays.push_back(make_test_array(32, 1, 11));
    arrays.push_back(make_test_array(32, 2, 12));
    arrays.push_back(make_test_array(32, 3, 13));

    std::vector<KVPair> cpu = cpu_merge_reference(arrays);
    MergeTimedResult gpu = launch_merge_timed(arrays);
    bool ok = check(cpu.size() == gpu.merged.size(), "Algorithm 1 output size matches");
    for (size_t i = 0; i < cpu.size() && ok; ++i) ok &= check(kv_equal(cpu[i], gpu.merged[i]), "Algorithm 1 ordering matches CPU");
    return ok;
}

static bool test_algorithm2_bloom()
{
    std::vector<KVPair> kv = make_test_array(64, 0, 1000);
    int byte_vector_len = (int)kv.size() * GP_BLOOM_BITS_PER_KEY;
    int bitvec_len = (byte_vector_len + 7) / 8;
    std::vector<uint8_t> gpu(bitvec_len), cpu(bitvec_len), bytevec(byte_vector_len);
    launch_bloom_filter(kv.data(), (int)kv.size(), GP_BLOOM_HASHES, byte_vector_len, gpu.data());
    cpu_build_byte_vector(kv.data(), (int)kv.size(), GP_BLOOM_HASHES, byte_vector_len, bytevec.data());
    cpu_pack_bit_vector(bytevec.data(), byte_vector_len, cpu.data());
    bool ok = check(std::memcmp(cpu.data(), gpu.data(), cpu.size()) == 0,
                    "Algorithm 2 bitvector matches CPU oracle");
    ok &= check(cpu_bloom_may_contain(gpu.data(), byte_vector_len, GP_BLOOM_HASHES, kv[9].key),
                "Algorithm 2 has no false negative on inserted key");
    return ok;
}

static bool test_pack_roundtrip()
{
    std::vector<KVPair> kv = make_test_array(256, 0, 2000);
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(kv);
    PackResult cpu_pack = cpu_pack_all(kv, plans);
    std::vector<KVPair> cpu_unpacked = cpu_unpack_all(cpu_pack.block_buf, cpu_pack.block_offsets, plans, (uint32_t)kv.size());
    bool ok = check(cpu_unpacked.size() == kv.size(), "CPU unpack count matches input");
    for (size_t i = 0; i < kv.size() && ok; ++i) ok &= check(kv_equal(kv[i], cpu_unpacked[i]), "CPU pack/unpack round-trip");

    PackTimedResult gpu_pack = launch_pack_timed(kv, plans);
    UnpackTimedResult gpu_unpacked = launch_unpack_timed(gpu_pack.result.block_buf,
                                                         gpu_pack.result.block_offsets,
                                                         plans,
                                                         (uint32_t)kv.size());
    ok &= check(cpu_pack.block_buf == gpu_pack.result.block_buf, "GPU pack bytes match CPU pack bytes");
    for (size_t i = 0; i < kv.size() && ok; ++i) ok &= check(kv_equal(kv[i], gpu_unpacked.kv_array[i]), "GPU unpack round-trip");
    return ok;
}

static bool test_sst_roundtrip()
{
    SSTBuildArtifacts build = build_cpu_sst(2048, 0, 23);
    ParsedSST parsed = parse_sst_bytes(build.file_bytes);
    std::vector<KVPair> unpacked = cpu_unpack_sst(parsed);
    bool ok = check(parsed.footer.restart_interval == GP_RESTART_INTERVAL, "SST footer uses restart interval 4");
    ok &= check(parsed.footer.data_block_size == GP_DATA_BLOCK_BYTES, "SST footer uses 32KB data blocks");
    ok &= check(unpacked.size() == 2048, "SST round-trip preserves KV count");
    return ok;
}

static bool test_target_file_partitioning()
{
    std::vector<KVPair> kv = generate_sorted_kv(12000, 0, 99);
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(kv);
    PackResult pack = cpu_pack_all(kv, plans);
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(kv, plans, filter_offsets, filter_lengths);
    SSTBuildSet files = assemble_sst_files_targeted(kv, pack, filter_bytes, filter_offsets, filter_lengths, 64 * 1024);

    bool ok = check(files.files.size() > 1, "Target file size partitions output into multiple SSTs");
    for (const auto& file : files.files) {
        ok &= check(file.file_bytes.size() <= 64 * 1024, "Each partitioned SST respects the target size");
    }
    size_t actual_total_kv = 0;
    for (const auto& file : files.files) {
        ParsedSST parsed = parse_sst_bytes(file.file_bytes);
        actual_total_kv += parsed.footer.total_kv;
    }
    ok &= check(actual_total_kv == kv.size(), "Partitioned SSTs preserve total KV count");
    return ok;
}

static bool test_device_plan_matches_cpu()
{
    std::vector<KVPair> kv = generate_sorted_kv(12000, 0, 111);
    std::vector<DataBlockPlanEntry> cpu = plan_data_blocks(kv);

    KVPair* d_kv = nullptr;
    cudaMalloc(&d_kv, kv.size() * sizeof(KVPair));
    cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(KVPair), cudaMemcpyHostToDevice);
    DevicePlanResult gpu = launch_plan_data_blocks_timed_from_device(d_kv, (int)kv.size());
    copy_device_plans_to_host(gpu);
    cudaFree(d_kv);

    bool same = cpu.size() == gpu.plans.size();
    for (size_t i = 0; i < cpu.size() && same; ++i) {
        same &= cpu[i].first_kv == gpu.plans[i].first_kv;
        same &= cpu[i].num_kv == gpu.plans[i].num_kv;
        same &= cpu[i].serialized_size == gpu.plans[i].serialized_size;
    }
    if (!same) {
        std::printf("  [DEBUG] cpu blocks=%zu gpu blocks=%zu\n", cpu.size(), gpu.plans.size());
        size_t mismatch = 0;
        size_t limit = std::min(cpu.size(), gpu.plans.size());
        while (mismatch < limit &&
               cpu[mismatch].first_kv == gpu.plans[mismatch].first_kv &&
               cpu[mismatch].num_kv == gpu.plans[mismatch].num_kv &&
               cpu[mismatch].serialized_size == gpu.plans[mismatch].serialized_size) {
            ++mismatch;
        }
        if (mismatch < limit) {
            std::printf("  [DEBUG] mismatch at block %zu: cpu=(first=%u num=%u size=%u) gpu=(first=%u num=%u size=%u)\n",
                        mismatch,
                        cpu[mismatch].first_kv, cpu[mismatch].num_kv, cpu[mismatch].serialized_size,
                        gpu.plans[mismatch].first_kv, gpu.plans[mismatch].num_kv, gpu.plans[mismatch].serialized_size);
        }
    }
    bool ok = check(same, "Device plan matches CPU plan exactly");
    destroy_device_plan_result(gpu);
    return ok;
}

static bool test_group_aligned_plan_matches_gpu_group_sizes()
{
    std::vector<KVPair> kv = generate_sorted_kv(12000, 0, 211);
    std::vector<DataBlockPlanEntry> cpu = plan_data_blocks_group_aligned(kv);

    KVPair* d_kv = nullptr;
    cudaMalloc(&d_kv, kv.size() * sizeof(KVPair));
    cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(KVPair), cudaMemcpyHostToDevice);
    RestartGroupSizeTimedResult group_sizes = launch_restart_group_sizes_timed_from_device(d_kv, (int)kv.size());
    cudaFree(d_kv);

    std::vector<DataBlockPlanEntry> gpu =
        plan_data_blocks_group_aligned_from_group_sizes(group_sizes.group_sizes, (uint32_t)kv.size());
    bool same = cpu.size() == gpu.size();
    for (size_t i = 0; i < cpu.size() && same; ++i) {
        same &= cpu[i].first_kv == gpu[i].first_kv;
        same &= cpu[i].num_kv == gpu[i].num_kv;
        same &= cpu[i].serialized_size == gpu[i].serialized_size;
    }
    return check(same, "Group-aligned plan matches GPU restart-group sizing");
}

static bool test_streamed_pack_matches_single_grid_pack()
{
    std::vector<KVPair> kv = generate_sorted_kv(12000, 0, 123);
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(kv);

    PackResult planned_layout;
    planned_layout.plans = plans;
    planned_layout.block_sizes.resize(plans.size());
    std::vector<uint32_t> filter_lengths(plans.size());
    std::vector<uint32_t> first_kv(plans.size());
    std::vector<uint32_t> num_kv(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        planned_layout.block_sizes[i] = plans[i].serialized_size;
        first_kv[i] = plans[i].first_kv;
        num_kv[i] = plans[i].num_kv;
        uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        filter_lengths[i] = (byte_vector_len + 7u) / 8u;
    }
    std::vector<std::pair<size_t, size_t>> spans =
        partition_output_blocks(planned_layout, filter_lengths, 64 * 1024);

    KVPair* d_kv = nullptr;
    uint32_t* d_first_kv = nullptr;
    uint32_t* d_num_kv = nullptr;
    cudaMalloc(&d_kv, kv.size() * sizeof(KVPair));
    cudaMalloc(&d_first_kv, plans.size() * sizeof(uint32_t));
    cudaMalloc(&d_num_kv, plans.size() * sizeof(uint32_t));
    cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(KVPair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_first_kv, first_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_kv, num_kv.data(), plans.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    PackTimedResult single = launch_pack_timed_from_device_plans(d_kv, d_first_kv, d_num_kv, (int)plans.size());
    PackTimedResult streamed =
        launch_pack_timed_from_device_plan_spans(d_kv, d_first_kv, d_num_kv, plans, spans);

    cudaFree(d_num_kv);
    cudaFree(d_first_kv);
    cudaFree(d_kv);

    bool ok = check(single.result.block_sizes == streamed.result.block_sizes,
                    "Streamed pack block sizes match single-grid pack");
    ok &= check(single.result.block_offsets == streamed.result.block_offsets,
                "Streamed pack block offsets match single-grid pack");
    ok &= check(single.result.block_buf == streamed.result.block_buf,
                "Streamed pack bytes match single-grid pack");
    return ok;
}

static bool test_gpu_q_compaction_matches_cpu()
{
    std::vector<ParsedSST> inputs;
    for (uint32_t sst = 0; sst < GP_NUM_INPUT_SSTS; ++sst) {
        SSTBuildArtifacts build = build_cpu_sst(1024, sst, 77);
        inputs.push_back(parse_sst_bytes(build.file_bytes));
    }

    CPUCompactionResult cpu = cpu_q_compaction_from_parsed(inputs);
    GPUCompactionResult gpu = gpu_q_compaction_from_parsed(inputs);

    bool ok = check(cpu.output.files.size() == gpu.output.files.size(),
                    "GPU Q-compaction emits the same number of SST files as CPU");
    for (size_t i = 0; i < cpu.output.files.size() && ok; ++i) {
        ok &= check(cpu.output.files[i].file_bytes == gpu.output.files[i].file_bytes,
                    "GPU Q-compaction output SST matches CPU output exactly");
    }
    ok &= check(cpu.merged.size() == gpu.merged.size(), "Merged KV count matches");
    for (size_t i = 0; i < cpu.merged.size() && ok; ++i) ok &= check(kv_equal(cpu.merged[i], gpu.merged[i]), "Merged KV payload matches");
    return ok;
}

static bool test_gpu_q_compaction_pipeline_matches_cpu()
{
    std::vector<ParsedSST> inputs;
    for (uint32_t sst = 0; sst < GP_NUM_INPUT_SSTS; ++sst) {
        SSTBuildArtifacts build = build_cpu_sst(1024, sst, 91);
        inputs.push_back(parse_sst_bytes(build.file_bytes));
    }

    CPUCompactionResult cpu = cpu_q_compaction_from_parsed(inputs);
    GPUCompactionResult gpu = gpu_q_compaction_pipeline_from_parsed(inputs);

    bool ok = check(cpu.output.files.size() == gpu.output.files.size(),
                    "GPU Q-compaction pipeline emits the same number of SST files as CPU");
    for (size_t i = 0; i < cpu.output.files.size() && ok; ++i) {
        ok &= check(cpu.output.files[i].file_bytes == gpu.output.files[i].file_bytes,
                    "GPU Q-compaction pipeline output SST matches CPU output exactly");
    }
    return ok;
}

static bool test_gpu_q_compaction_paper_matches_cpu()
{
    std::vector<ParsedSST> inputs;
    for (uint32_t sst = 0; sst < GP_NUM_INPUT_SSTS; ++sst) {
        SSTBuildArtifacts build = build_cpu_sst(1024, sst, 141);
        inputs.push_back(parse_sst_bytes(build.file_bytes));
    }

    CPUCompactionResult cpu = cpu_q_compaction_paper_from_parsed(inputs);
    GPUCompactionResult gpu = gpu_q_compaction_paper_from_parsed(inputs);

    bool ok = check(cpu.output.files.size() == gpu.output.files.size(),
                    "GPU Q-paper compaction emits the same number of SST files as CPU");
    for (size_t i = 0; i < cpu.output.files.size() && ok; ++i) {
        ok &= check(cpu.output.files[i].file_bytes == gpu.output.files[i].file_bytes,
                    "GPU Q-paper compaction output SST matches CPU output exactly");
    }
    return ok;
}

static bool test_gpu_q_compaction_without_plan_matches_cpu()
{
    std::vector<ParsedSST> inputs;
    for (uint32_t sst = 0; sst < GP_NUM_INPUT_SSTS; ++sst) {
        SSTBuildArtifacts build = build_cpu_sst(1024, sst, 151);
        inputs.push_back(parse_sst_bytes(build.file_bytes));
    }

    CPUCompactionResult cpu = cpu_q_compaction_without_plan_from_parsed(inputs);
    GPUCompactionResult gpu = gpu_q_compaction_without_plan_from_parsed(inputs);

    bool ok = check(cpu.output.files.size() == gpu.output.files.size(),
                    "GPU Q-compaction without plan emits the same number of SST files as CPU");
    for (size_t i = 0; i < cpu.output.files.size() && ok; ++i) {
        ok &= check(cpu.output.files[i].file_bytes == gpu.output.files[i].file_bytes,
                    "GPU Q-compaction without plan output SST matches CPU output exactly");
    }
    return ok;
}

int main()
{
    cudaFree(0);
    test_algorithm1_merge();
    test_algorithm2_bloom();
    test_pack_roundtrip();
    test_sst_roundtrip();
    test_target_file_partitioning();
    test_device_plan_matches_cpu();
    test_group_aligned_plan_matches_gpu_group_sizes();
    test_streamed_pack_matches_single_grid_pack();
    test_gpu_q_compaction_matches_cpu();
    test_gpu_q_compaction_pipeline_matches_cpu();
    test_gpu_q_compaction_paper_matches_cpu();
    test_gpu_q_compaction_without_plan_matches_cpu();
    std::printf("\nPassed: %d  Failed: %d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
