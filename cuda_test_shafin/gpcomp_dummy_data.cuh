#pragma once

#include "gpcomp_sstable.cuh"

#include <algorithm>
#include <vector>

static inline uint64_t gp_splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline std::vector<KVPair> generate_sorted_kv(uint32_t entries, uint32_t sst_id, uint64_t seed)
{
    std::vector<uint64_t> user_keys(entries);
    const uint64_t key_space = (1ULL << 40);
    for (uint32_t i = 0; i < entries; ++i) {
        uint64_t x = gp_splitmix64(seed ^ ((uint64_t)sst_id << 32) ^ (uint64_t)i);
        user_keys[i] = x % key_space;
    }
    std::sort(user_keys.begin(), user_keys.end());

    std::vector<KVPair> kv(entries);
    for (uint32_t i = 0; i < entries; ++i) {
        kv[i].key = make_internal_key(user_keys[i], sst_id, i);
        kv[i].value = make_value_payload(user_keys[i], sst_id, i);
    }
    return kv;
}

static inline SSTBuildArtifacts build_cpu_sst(uint32_t entries, uint32_t sst_id, uint64_t seed)
{
    std::vector<KVPair> kv = generate_sorted_kv(entries, sst_id, seed);
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(kv);
    PackResult pack = cpu_pack_all(kv, plans);
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(kv, plans, filter_offsets, filter_lengths);
    return assemble_sst_file(kv, pack, filter_bytes, filter_offsets, filter_lengths);
}

