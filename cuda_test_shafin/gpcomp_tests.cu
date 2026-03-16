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

int main()
{
    cudaFree(0);
    test_algorithm1_merge();
    test_algorithm2_bloom();
    test_pack_roundtrip();
    test_sst_roundtrip();
    test_target_file_partitioning();
    test_gpu_q_compaction_matches_cpu();
    std::printf("\nPassed: %d  Failed: %d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
