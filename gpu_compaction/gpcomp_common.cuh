#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int GP_KEY_BYTES          = 16;
static constexpr int GP_VALUE_BYTES        = 32;
static constexpr int GP_RESTART_INTERVAL   = 4;
static constexpr int GP_DATA_BLOCK_BYTES   = 32 * 1024;
static constexpr int GP_TARGET_FILE_BYTES  = 8 * 1024 * 1024;
static constexpr int GP_NUM_INPUT_SSTS     = 4;
static constexpr int GP_BLOOM_BITS_PER_KEY = 10;
static constexpr int GP_BLOOM_HASHES       = 7;
static constexpr uint32_t GP_SST_MAGIC     = 0x47505351u; /* "GPSQ" */
static constexpr uint32_t GP_SST_VERSION   = 1u;

struct Key128 {
    uint8_t bytes[GP_KEY_BYTES];
};

struct Value32 {
    uint8_t bytes[GP_VALUE_BYTES];
};

struct KVPair {
    Key128  key;
    Value32 value;
};

struct DataBlockPlanEntry {
    uint32_t first_kv;
    uint32_t num_kv;
    uint32_t serialized_size;
};

struct __attribute__((packed)) DataBlockMeta {
    uint64_t offset;
    uint32_t size;
    uint32_t first_kv;
    uint32_t num_kv;
};

struct __attribute__((packed)) FilterBlockMeta {
    uint64_t offset;
    uint32_t size;
    uint32_t byte_vector_len;
    uint32_t num_keys;
};

struct __attribute__((packed)) IndexEntry {
    Key128   largest_key;
    uint64_t data_offset;
    uint32_t data_size;
    uint32_t num_kv;
};

struct SSTFooter {
    uint32_t magic;
    uint32_t version;
    uint32_t key_bytes;
    uint32_t value_bytes;
    uint32_t restart_interval;
    uint32_t data_block_size;
    uint32_t bloom_bits_per_key;
    uint32_t bloom_hashes;
    uint32_t num_data_blocks;
    uint32_t total_kv;
    uint64_t filter_meta_offset;
    uint32_t filter_meta_size;
    uint64_t data_meta_offset;
    uint32_t data_meta_size;
    uint64_t filter_region_offset;
    uint32_t filter_region_size;
    uint64_t index_region_offset;
    uint32_t index_region_size;
};

struct ParsedSST {
    std::vector<uint8_t>         file_bytes;
    SSTFooter                    footer{};
    std::vector<DataBlockMeta>   data_blocks;
    std::vector<FilterBlockMeta> filter_blocks;
};

static inline void gp_fail_if(bool cond, const std::string& message)
{
    if (cond) throw std::runtime_error(message);
}

__host__ __device__ static inline void store_be64(uint64_t v, uint8_t out[8])
{
    out[0] = (uint8_t)(v >> 56);
    out[1] = (uint8_t)(v >> 48);
    out[2] = (uint8_t)(v >> 40);
    out[3] = (uint8_t)(v >> 32);
    out[4] = (uint8_t)(v >> 24);
    out[5] = (uint8_t)(v >> 16);
    out[6] = (uint8_t)(v >> 8);
    out[7] = (uint8_t)(v);
}

__host__ __device__ static inline uint64_t load_be64(const uint8_t in[8])
{
    return ((uint64_t)in[0] << 56) |
           ((uint64_t)in[1] << 48) |
           ((uint64_t)in[2] << 40) |
           ((uint64_t)in[3] << 32) |
           ((uint64_t)in[4] << 24) |
           ((uint64_t)in[5] << 16) |
           ((uint64_t)in[6] << 8) |
           ((uint64_t)in[7]);
}

__host__ __device__ static inline int key_compare(const Key128& a, const Key128& b)
{
    for (int i = 0; i < GP_KEY_BYTES; ++i) {
        if (a.bytes[i] < b.bytes[i]) return -1;
        if (a.bytes[i] > b.bytes[i]) return 1;
    }
    return 0;
}

static inline bool key_equal(const Key128& a, const Key128& b)
{
    return std::memcmp(a.bytes, b.bytes, GP_KEY_BYTES) == 0;
}

static inline bool kv_key_less(const KVPair& a, const KVPair& b)
{
    return key_compare(a.key, b.key) < 0;
}

static inline bool kv_user_key_equal(const KVPair& a, const KVPair& b)
{
    return load_be64(a.key.bytes) == load_be64(b.key.bytes);
}

__host__ __device__ static inline int key_shared_prefix(const Key128& a, const Key128& b)
{
    int shared = 0;
    for (int i = 0; i < GP_KEY_BYTES; ++i) {
        if (a.bytes[i] != b.bytes[i]) break;
        ++shared;
    }
    return shared;
}

static inline Key128 make_internal_key(uint64_t user_key, uint32_t sst_id, uint32_t ordinal)
{
    Key128 key{};
    store_be64(user_key, key.bytes);
    uint64_t tag = ((uint64_t)sst_id << 32) | (uint64_t)ordinal;
    store_be64(tag, key.bytes + 8);
    return key;
}

static inline Value32 make_value_payload(uint64_t user_key, uint32_t sst_id, uint32_t ordinal)
{
    Value32 value{};
    uint64_t words[4];
    words[0] = user_key;
    words[1] = ((uint64_t)sst_id << 32) | (uint64_t)ordinal;
    words[2] = user_key ^ 0x9e3779b97f4a7c15ULL;
    words[3] = words[1] ^ 0xc2b2ae3d27d4eb4fULL;
    for (int i = 0; i < 4; ++i) store_be64(words[i], value.bytes + i * 8);
    return value;
}

static inline uint64_t key_user_component(const Key128& key)
{
    return load_be64(key.bytes);
}

static inline uint64_t key_tag_component(const Key128& key)
{
    return load_be64(key.bytes + 8);
}

static inline bool key_user_equal(const Key128& a, const Key128& b)
{
    return key_user_component(a) == key_user_component(b);
}

static inline std::string format_key(const Key128& key)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%016llx:%016llx",
                  (unsigned long long)key_user_component(key),
                  (unsigned long long)key_tag_component(key));
    return std::string(buf);
}

static inline uint32_t bloom_hash_key(const Key128& key, int seed)
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

static inline void cpu_build_byte_vector(const KVPair* array,
                                         int           n,
                                         int           K,
                                         int           byte_vector_len,
                                         uint8_t*      out)
{
    std::fill(out, out + byte_vector_len, 0);
    for (int i = 0; i < n; ++i) {
        for (int k = 1; k <= K; ++k) {
            uint32_t h = bloom_hash_key(array[i].key, k);
            out[h % (uint32_t)byte_vector_len] = 1;
        }
    }
}

static inline void cpu_pack_bit_vector(const uint8_t* byte_vector,
                                       int            byte_vector_len,
                                       uint8_t*       bit_vector)
{
    int bitvec_len = (byte_vector_len + 7) / 8;
    for (int i = 0; i < bitvec_len; ++i) {
        uint8_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            int pos = i * 8 + j;
            if (pos < byte_vector_len && byte_vector[pos]) packed |= (uint8_t)(1u << j);
        }
        bit_vector[i] = packed;
    }
}

static inline bool cpu_bloom_may_contain(const uint8_t* bit_vector,
                                         int            byte_vector_len,
                                         int            K,
                                         const Key128&  key)
{
    for (int i = 1; i <= K; ++i) {
        uint32_t h = bloom_hash_key(key, i);
        int pos = (int)(h % (uint32_t)byte_vector_len);
        int byte_idx = pos / 8;
        int bit_idx = pos % 8;
        if (((bit_vector[byte_idx] >> bit_idx) & 1u) == 0) return false;
    }
    return true;
}

static inline std::vector<KVPair> cpu_merge_reference(const std::vector<std::vector<KVPair>>& arrays)
{
    std::vector<KVPair> merged;
    size_t total = 0;
    for (const auto& arr : arrays) total += arr.size();
    merged.reserve(total);

    std::vector<size_t> indices(arrays.size(), 0);
    for (;;) {
        int best = -1;
        for (int i = 0; i < (int)arrays.size(); ++i) {
            if (indices[i] >= arrays[i].size()) continue;
            if (best < 0 || kv_key_less(arrays[i][indices[i]], arrays[best][indices[best]])) {
                best = i;
            }
        }
        if (best < 0) break;
        merged.push_back(arrays[best][indices[best]++]);
    }
    return merged;
}

static inline std::vector<KVPair> garbage_collect_sorted_kv(const std::vector<KVPair>& sorted_kv)
{
    std::vector<KVPair> survivors;
    survivors.reserve(sorted_kv.size());

    size_t i = 0;
    while (i < sorted_kv.size()) {
        size_t j = i + 1;
        while (j < sorted_kv.size() && kv_user_key_equal(sorted_kv[i], sorted_kv[j])) ++j;

        // Internal keys are encoded big-endian as {user_key, tag}, so the
        // last KV in a run is the newest surviving version for that user key.
        survivors.push_back(sorted_kv[j - 1]);
        i = j;
    }
    return survivors;
}

static inline std::vector<KVPair> garbage_collect_sorted_kv(const KVPair* sorted_kv, size_t count)
{
    std::vector<KVPair> survivors;
    survivors.reserve(count);

    size_t i = 0;
    while (i < count) {
        size_t j = i + 1;
        while (j < count && kv_user_key_equal(sorted_kv[i], sorted_kv[j])) ++j;

        survivors.push_back(sorted_kv[j - 1]);
        i = j;
    }
    return survivors;
}

static inline std::vector<uint32_t> garbage_collect_sorted_keys_to_indices(const std::vector<Key128>& sorted_keys)
{
    std::vector<uint32_t> survivor_indices;
    survivor_indices.reserve(sorted_keys.size());

    size_t i = 0;
    while (i < sorted_keys.size()) {
        size_t j = i + 1;
        while (j < sorted_keys.size() && key_user_equal(sorted_keys[i], sorted_keys[j])) ++j;
        survivor_indices.push_back((uint32_t)(j - 1));
        i = j;
    }
    return survivor_indices;
}

static inline bool kv_equal(const KVPair& a, const KVPair& b)
{
    return key_equal(a.key, b.key) &&
           std::memcmp(a.value.bytes, b.value.bytes, GP_VALUE_BYTES) == 0;
}
