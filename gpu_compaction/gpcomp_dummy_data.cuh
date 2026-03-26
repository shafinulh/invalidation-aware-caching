#pragma once

#include "gpcomp_sstable.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr double GP_DUMMY_ZIPF_ALPHA = 0.6;
static constexpr uint64_t GP_DUMMY_DATASET_USER_KEY_SPACE = 20000000ULL;
static constexpr uint64_t GP_DUMMY_KEY_MASK_40 = (1ULL << 40) - 1ULL;
static constexpr uint64_t GP_DUMMY_KEY_PERMUTE_MULT = 0x9e3779b97fULL;

struct DummyDataConfig {
    double zipf_alpha;
    uint64_t user_key_space;
};

static inline uint64_t gp_splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double gp_uniform01_from_u64(uint64_t x)
{
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

static inline DummyDataConfig make_dummy_data_config(double zipf_alpha = GP_DUMMY_ZIPF_ALPHA,
                                                     uint64_t user_key_space = 0)
{
    DummyDataConfig cfg{};
    cfg.zipf_alpha = zipf_alpha;
    cfg.user_key_space = user_key_space;
    return cfg;
}

static inline DummyDataConfig make_default_gpcomp_dataset_config()
{
    return make_dummy_data_config(GP_DUMMY_ZIPF_ALPHA, GP_DUMMY_DATASET_USER_KEY_SPACE);
}

static inline bool gp_dummy_distribution_is_uniform(double alpha)
{
    return std::abs(alpha) < 1e-12;
}

static inline uint64_t clamp_user_key_space(uint64_t space)
{
    if (space < 1) space = 1;
    if (space > (GP_DUMMY_KEY_MASK_40 + 1ULL)) space = GP_DUMMY_KEY_MASK_40 + 1ULL;
    return space;
}

static inline uint64_t zipf_user_key_space(uint32_t entries)
{
    uint64_t space = (uint64_t)std::max(entries, 1u) * (uint64_t)GP_NUM_INPUT_SSTS;
    if (space < 1024) space = 1024;
    return clamp_user_key_space(space);
}

static inline uint64_t resolve_user_key_space(uint32_t entries, const DummyDataConfig& config)
{
    return config.user_key_space ? clamp_user_key_space(config.user_key_space)
                                 : zipf_user_key_space(entries);
}

static inline std::vector<double> build_zipf_cdf(uint64_t key_space, double alpha)
{
    std::vector<double> cdf((size_t)key_space);
    double running = 0.0;
    for (uint64_t rank = 0; rank < key_space; ++rank) {
        running += 1.0 / std::pow((double)(rank + 1ULL), alpha);
        cdf[(size_t)rank] = running;
    }

    double inv_normalizer = (running > 0.0) ? (1.0 / running) : 0.0;
    for (uint64_t rank = 0; rank < key_space; ++rank) {
        cdf[(size_t)rank] *= inv_normalizer;
    }
    if (!cdf.empty()) cdf.back() = 1.0;
    return cdf;
}

static inline const std::vector<double>& get_zipf_cdf_cached(uint64_t key_space, double alpha)
{
    struct ZipfCdfCache {
        uint64_t key_space = 0;
        double alpha = 0.0;
        std::vector<double> cdf;
    };

    static ZipfCdfCache cache{};
    if (cache.key_space != key_space || cache.alpha != alpha) {
        cache.key_space = key_space;
        cache.alpha = alpha;
        cache.cdf = build_zipf_cdf(key_space, alpha);
    }
    return cache.cdf;
}

static inline uint64_t sample_zipf_rank(const std::vector<double>& cdf, uint64_t sample_bits)
{
    double u = gp_uniform01_from_u64(sample_bits);
    return (uint64_t)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
}

static inline uint64_t permute_zipf_rank_to_user_key(uint64_t rank, uint64_t seed)
{
    uint64_t offset = gp_splitmix64(seed) & GP_DUMMY_KEY_MASK_40;
    return (rank * GP_DUMMY_KEY_PERMUTE_MULT + offset) & GP_DUMMY_KEY_MASK_40;
}

static inline std::vector<KVPair> generate_sorted_kv(uint32_t entries,
                                                     uint32_t sst_id,
                                                     uint64_t seed,
                                                     const DummyDataConfig& config = make_dummy_data_config())
{
    std::vector<KVPair> kv(entries);
    uint64_t key_space = resolve_user_key_space(entries, config);
    const bool uniform = gp_dummy_distribution_is_uniform(config.zipf_alpha);
    const std::vector<double>* zipf_cdf = uniform ? nullptr : &get_zipf_cdf_cached(key_space, config.zipf_alpha);
    uint64_t state = seed ^ (((uint64_t)sst_id + 1ULL) * 0x6a09e667f3bcc909ULL);
    for (uint32_t i = 0; i < entries; ++i) {
        state = gp_splitmix64(state ^ (uint64_t)i);
        uint64_t rank = uniform ? (state % key_space) : sample_zipf_rank(*zipf_cdf, state);
        uint64_t user_key = permute_zipf_rank_to_user_key(rank, seed);
        kv[i].key = make_internal_key(user_key, sst_id, i);
        kv[i].value = make_value_payload(user_key, sst_id, i);
    }
    std::sort(kv.begin(), kv.end(), kv_key_less);
    return kv;
}

static inline SSTBuildArtifacts build_cpu_sst(uint32_t entries,
                                              uint32_t sst_id,
                                              uint64_t seed,
                                              const DummyDataConfig& config = make_dummy_data_config())
{
    std::vector<KVPair> kv = generate_sorted_kv(entries, sst_id, seed, config);
    std::vector<DataBlockPlanEntry> plans = plan_data_blocks(kv);
    PackResult pack = cpu_pack_all(kv, plans);
    std::vector<uint32_t> filter_offsets, filter_lengths;
    std::vector<uint8_t> filter_bytes = build_cpu_filter_bytes(kv, plans, filter_offsets, filter_lengths);
    return assemble_sst_file(kv, pack, filter_bytes, filter_offsets, filter_lengths);
}
