#pragma once

#include "gpcomp_bloom.cuh"
#include "gpcomp_pack.cuh"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

struct __attribute__((packed)) FilterBlockHeader {
    uint32_t num_keys;
    uint32_t byte_vector_len;
    uint32_t bitvec_len;
    uint32_t reserved;
};

struct SSTBuildArtifacts {
    PackResult                  pack_result;
    BloomBatchResult            bloom_result;
    std::vector<DataBlockMeta>  data_meta;
    std::vector<FilterBlockMeta> filter_meta;
    std::vector<IndexEntry>     index_entries;
    std::vector<uint8_t>        file_bytes;
};

struct SSTBuildSet {
    std::vector<SSTBuildArtifacts> files;
};

struct PinnedHostBytes {
    uint8_t* data = nullptr;
    size_t   size = 0;

    PinnedHostBytes() = default;
    PinnedHostBytes(const PinnedHostBytes&) = delete;
    PinnedHostBytes& operator=(const PinnedHostBytes&) = delete;

    PinnedHostBytes(PinnedHostBytes&& other) noexcept : data(other.data), size(other.size)
    {
        other.data = nullptr;
        other.size = 0;
    }

    PinnedHostBytes& operator=(PinnedHostBytes&& other) noexcept
    {
        if (this != &other) {
            reset();
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ~PinnedHostBytes() { reset(); }

    void reset()
    {
        if (data) cudaFreeHost(data);
        data = nullptr;
        size = 0;
    }
};

struct SerializedSSTHostSet {
    PinnedHostBytes       all_file_bytes;
    std::vector<uint64_t> file_offsets;
    std::vector<uint64_t> file_sizes;
    std::vector<uint32_t> file_blocks;

    bool empty() const { return file_sizes.empty(); }

    size_t total_bytes() const { return all_file_bytes.size; }

    size_t total_blocks() const
    {
        size_t total = 0;
        for (uint32_t blocks : file_blocks) total += blocks;
        return total;
    }
};

static inline std::vector<uint8_t> build_cpu_filter_bytes(const std::vector<KVPair>& kv_array,
                                                          const std::vector<DataBlockPlanEntry>& plans,
                                                          std::vector<uint32_t>& bitvec_offsets,
                                                          std::vector<uint32_t>& bitvec_lengths)
{
    bitvec_offsets.resize(plans.size());
    bitvec_lengths.resize(plans.size());
    std::vector<uint8_t> out;
    uint32_t offset = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        int byte_vector_len = (int)plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        int bitvec_len = (byte_vector_len + 7) / 8;
        std::vector<uint8_t> byte_vector((size_t)byte_vector_len);
        std::vector<uint8_t> bit_vector((size_t)bitvec_len);
        cpu_build_byte_vector(kv_array.data() + plans[i].first_kv,
                              (int)plans[i].num_kv,
                              GP_BLOOM_HASHES,
                              byte_vector_len,
                              byte_vector.data());
        cpu_pack_bit_vector(byte_vector.data(), byte_vector_len, bit_vector.data());
        bitvec_offsets[i] = offset;
        bitvec_lengths[i] = (uint32_t)bitvec_len;
        out.insert(out.end(), bit_vector.begin(), bit_vector.end());
        offset += (uint32_t)bitvec_len;
    }
    return out;
}

static inline std::vector<IndexEntry> build_index_entries(const std::vector<KVPair>& kv_array,
                                                          const std::vector<DataBlockMeta>& data_meta)
{
    std::vector<IndexEntry> index_entries(data_meta.size());
    for (size_t i = 0; i < data_meta.size(); ++i) {
        const DataBlockMeta& meta = data_meta[i];
        index_entries[i].largest_key = kv_array[meta.first_kv + meta.num_kv - 1].key;
        index_entries[i].data_offset = meta.offset;
        index_entries[i].data_size = meta.size;
        index_entries[i].num_kv = meta.num_kv;
    }
    return index_entries;
}

static inline std::vector<uint8_t> serialize_index_region(const std::vector<IndexEntry>& index_entries)
{
    std::vector<uint8_t> out(index_entries.size() * sizeof(IndexEntry));
    if (!out.empty()) std::memcpy(out.data(), index_entries.data(), out.size());
    return out;
}

static inline SSTBuildArtifacts assemble_sst_file(const std::vector<KVPair>&            kv_array,
                                                  const PackResult&                      pack_result,
                                                  const std::vector<uint8_t>&           filter_bytes,
                                                  const std::vector<uint32_t>&          filter_offsets,
                                                  const std::vector<uint32_t>&          filter_lengths)
{
    SSTBuildArtifacts artifacts;
    artifacts.pack_result = pack_result;
    artifacts.bloom_result.filter_bytes = filter_bytes;
    artifacts.bloom_result.bitvec_offsets = filter_offsets;
    artifacts.bloom_result.bitvec_lengths = filter_lengths;

    std::vector<uint8_t> file_bytes;
    file_bytes.reserve(pack_result.block_buf.size() + filter_bytes.size()
                     + pack_result.plans.size() * (sizeof(DataBlockMeta) + sizeof(FilterBlockMeta) + sizeof(IndexEntry))
                     + sizeof(SSTFooter) + 4096);

    artifacts.data_meta.resize(pack_result.plans.size());
    size_t running_offset = 0;
    for (size_t i = 0; i < pack_result.plans.size(); ++i) {
        artifacts.data_meta[i].offset = running_offset;
        artifacts.data_meta[i].size = pack_result.block_sizes[i];
        artifacts.data_meta[i].first_kv = pack_result.plans[i].first_kv;
        artifacts.data_meta[i].num_kv = pack_result.plans[i].num_kv;
        file_bytes.insert(file_bytes.end(),
                          pack_result.block_buf.begin() + (ptrdiff_t)pack_result.block_offsets[i],
                          pack_result.block_buf.begin() + (ptrdiff_t)(pack_result.block_offsets[i]
                                                                   + pack_result.block_sizes[i]));
        running_offset += pack_result.block_sizes[i];
    }

    uint64_t filter_region_offset = running_offset;
    artifacts.filter_meta.resize(pack_result.plans.size());
    for (size_t i = 0; i < pack_result.plans.size(); ++i) {
        FilterBlockHeader hdr{};
        hdr.num_keys = pack_result.plans[i].num_kv;
        hdr.byte_vector_len = pack_result.plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        hdr.bitvec_len = filter_lengths[i];

        artifacts.filter_meta[i].offset = running_offset;
        artifacts.filter_meta[i].size = (uint32_t)(sizeof(FilterBlockHeader) + filter_lengths[i]);
        artifacts.filter_meta[i].byte_vector_len = hdr.byte_vector_len;
        artifacts.filter_meta[i].num_keys = hdr.num_keys;

        file_bytes.insert(file_bytes.end(),
                          reinterpret_cast<const uint8_t*>(&hdr),
                          reinterpret_cast<const uint8_t*>(&hdr) + sizeof(FilterBlockHeader));
        file_bytes.insert(file_bytes.end(),
                          filter_bytes.begin() + (ptrdiff_t)filter_offsets[i],
                          filter_bytes.begin() + (ptrdiff_t)(filter_offsets[i] + filter_lengths[i]));
        running_offset += sizeof(FilterBlockHeader) + filter_lengths[i];
    }

    uint32_t filter_region_size = (uint32_t)(running_offset - filter_region_offset);
    artifacts.index_entries = build_index_entries(kv_array, artifacts.data_meta);
    std::vector<uint8_t> index_region = serialize_index_region(artifacts.index_entries);
    uint64_t index_region_offset = running_offset;
    file_bytes.insert(file_bytes.end(), index_region.begin(), index_region.end());
    running_offset += index_region.size();

    uint64_t filter_meta_offset = running_offset;
    if (!artifacts.filter_meta.empty()) {
        const uint8_t* begin = reinterpret_cast<const uint8_t*>(artifacts.filter_meta.data());
        file_bytes.insert(file_bytes.end(), begin,
                          begin + artifacts.filter_meta.size() * sizeof(FilterBlockMeta));
        running_offset += artifacts.filter_meta.size() * sizeof(FilterBlockMeta);
    }

    uint64_t data_meta_offset = running_offset;
    if (!artifacts.data_meta.empty()) {
        const uint8_t* begin = reinterpret_cast<const uint8_t*>(artifacts.data_meta.data());
        file_bytes.insert(file_bytes.end(), begin,
                          begin + artifacts.data_meta.size() * sizeof(DataBlockMeta));
        running_offset += artifacts.data_meta.size() * sizeof(DataBlockMeta);
    }

    SSTFooter footer{};
    footer.magic = GP_SST_MAGIC;
    footer.version = GP_SST_VERSION;
    footer.key_bytes = GP_KEY_BYTES;
    footer.value_bytes = GP_VALUE_BYTES;
    footer.restart_interval = GP_RESTART_INTERVAL;
    footer.data_block_size = GP_DATA_BLOCK_BYTES;
    footer.bloom_bits_per_key = GP_BLOOM_BITS_PER_KEY;
    footer.bloom_hashes = GP_BLOOM_HASHES;
    footer.num_data_blocks = (uint32_t)pack_result.plans.size();
    footer.total_kv = (uint32_t)kv_array.size();
    footer.filter_meta_offset = filter_meta_offset;
    footer.filter_meta_size = (uint32_t)(artifacts.filter_meta.size() * sizeof(FilterBlockMeta));
    footer.data_meta_offset = data_meta_offset;
    footer.data_meta_size = (uint32_t)(artifacts.data_meta.size() * sizeof(DataBlockMeta));
    footer.filter_region_offset = filter_region_offset;
    footer.filter_region_size = filter_region_size;
    footer.index_region_offset = index_region_offset;
    footer.index_region_size = (uint32_t)index_region.size();
    file_bytes.insert(file_bytes.end(),
                      reinterpret_cast<const uint8_t*>(&footer),
                      reinterpret_cast<const uint8_t*>(&footer) + sizeof(SSTFooter));

    artifacts.file_bytes = std::move(file_bytes);
    return artifacts;
}

static inline size_t estimate_sst_subset_size(const PackResult&             pack_result,
                                              const std::vector<uint32_t>&  filter_lengths,
                                              size_t                        block_begin,
                                              size_t                        block_end)
{
    size_t count = block_end - block_begin;
    size_t total = sizeof(SSTFooter)
                 + count * (sizeof(DataBlockMeta) + sizeof(FilterBlockMeta) + sizeof(IndexEntry));
    for (size_t i = block_begin; i < block_end; ++i) {
        total += pack_result.block_sizes[i];
        total += sizeof(FilterBlockHeader) + filter_lengths[i];
    }
    return total;
}

static inline std::vector<std::pair<size_t, size_t>>
partition_output_blocks(const PackResult&            pack_result,
                        const std::vector<uint32_t>& filter_lengths,
                        size_t                       target_file_bytes)
{
    std::vector<std::pair<size_t, size_t>> spans;
    size_t begin = 0;
    while (begin < pack_result.plans.size()) {
        size_t end = begin + 1;
        size_t best_end = end;
        while (end <= pack_result.plans.size()) {
            size_t candidate = estimate_sst_subset_size(pack_result, filter_lengths, begin, end);
            if (candidate > target_file_bytes && end > begin + 1) break;
            best_end = end;
            ++end;
        }
        spans.push_back({begin, best_end});
        begin = best_end;
    }
    return spans;
}

static inline std::vector<uint32_t>
predict_filter_lengths_from_plans(const std::vector<DataBlockPlanEntry>& plans)
{
    std::vector<uint32_t> filter_lengths(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        uint32_t byte_vector_len = plans[i].num_kv * GP_BLOOM_BITS_PER_KEY;
        filter_lengths[i] = (byte_vector_len + 7u) / 8u;
    }
    return filter_lengths;
}

static inline std::vector<std::pair<size_t, size_t>>
partition_output_blocks_from_plan_estimates(const std::vector<DataBlockPlanEntry>& plans,
                                            size_t                                 target_file_bytes = GP_TARGET_FILE_BYTES)
{
    PackResult planned_layout;
    planned_layout.plans = plans;
    planned_layout.block_sizes.resize(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        planned_layout.block_sizes[i] = plans[i].serialized_size;
    }
    std::vector<uint32_t> filter_lengths = predict_filter_lengths_from_plans(plans);
    return partition_output_blocks(planned_layout, filter_lengths, target_file_bytes);
}

static inline SSTBuildArtifacts assemble_sst_file_range(const std::vector<KVPair>&            kv_array,
                                                        const PackResult&                      pack_result,
                                                        const std::vector<uint8_t>&           filter_bytes,
                                                        const std::vector<uint32_t>&          filter_offsets,
                                                        const std::vector<uint32_t>&          filter_lengths,
                                                        size_t                                 block_begin,
                                                        size_t                                 block_end)
{
    gp_fail_if(block_begin >= block_end, "invalid SST block range");

    size_t kv_begin = pack_result.plans[block_begin].first_kv;
    size_t kv_end = pack_result.plans[block_end - 1].first_kv + pack_result.plans[block_end - 1].num_kv;

    std::vector<KVPair> local_kv(kv_array.begin() + (ptrdiff_t)kv_begin,
                                 kv_array.begin() + (ptrdiff_t)kv_end);

    PackResult local_pack;
    uint32_t kv_offset = 0;
    uint32_t block_buf_offset = 0;
    for (size_t i = block_begin; i < block_end; ++i) {
        DataBlockPlanEntry plan{};
        plan.first_kv = kv_offset;
        plan.num_kv = pack_result.plans[i].num_kv;
        plan.serialized_size = pack_result.plans[i].serialized_size;
        local_pack.plans.push_back(plan);
        local_pack.block_offsets.push_back(block_buf_offset);
        local_pack.block_sizes.push_back(pack_result.block_sizes[i]);
        local_pack.block_buf.insert(local_pack.block_buf.end(),
                                    pack_result.block_buf.begin() + (ptrdiff_t)pack_result.block_offsets[i],
                                    pack_result.block_buf.begin() + (ptrdiff_t)(pack_result.block_offsets[i]
                                                                             + pack_result.block_sizes[i]));
        kv_offset += plan.num_kv;
        block_buf_offset += pack_result.block_sizes[i];
    }

    std::vector<uint8_t> local_filter_bytes;
    std::vector<uint32_t> local_filter_offsets;
    std::vector<uint32_t> local_filter_lengths;
    uint32_t filter_offset = 0;
    for (size_t i = block_begin; i < block_end; ++i) {
        local_filter_offsets.push_back(filter_offset);
        local_filter_lengths.push_back(filter_lengths[i]);
        local_filter_bytes.insert(local_filter_bytes.end(),
                                  filter_bytes.begin() + (ptrdiff_t)filter_offsets[i],
                                  filter_bytes.begin() + (ptrdiff_t)(filter_offsets[i] + filter_lengths[i]));
        filter_offset += filter_lengths[i];
    }

    return assemble_sst_file(local_kv, local_pack, local_filter_bytes, local_filter_offsets, local_filter_lengths);
}

static inline SSTBuildArtifacts assemble_sst_file_range_fast(const std::vector<KVPair>&            kv_array,
                                                             const PackResult&                      pack_result,
                                                             const std::vector<uint8_t>&           filter_bytes,
                                                             const std::vector<uint32_t>&          filter_offsets,
                                                             const std::vector<uint32_t>&          filter_lengths,
                                                             size_t                                 block_begin,
                                                             size_t                                 block_end)
{
    gp_fail_if(block_begin >= block_end, "invalid SST block range");

    SSTBuildArtifacts artifacts;
    size_t block_count = block_end - block_begin;
    size_t kv_begin = pack_result.plans[block_begin].first_kv;
    size_t kv_end = pack_result.plans[block_end - 1].first_kv + pack_result.plans[block_end - 1].num_kv;

    size_t data_region_size = 0;
    size_t filter_region_size = 0;
    for (size_t i = block_begin; i < block_end; ++i) {
        data_region_size += pack_result.block_sizes[i];
        filter_region_size += sizeof(FilterBlockHeader) + filter_lengths[i];
    }

    size_t index_region_size = block_count * sizeof(IndexEntry);
    size_t filter_meta_size = block_count * sizeof(FilterBlockMeta);
    size_t data_meta_size = block_count * sizeof(DataBlockMeta);
    size_t total_size = data_region_size + filter_region_size + index_region_size
                      + filter_meta_size + data_meta_size + sizeof(SSTFooter);

    artifacts.data_meta.resize(block_count);
    artifacts.filter_meta.resize(block_count);
    artifacts.index_entries.resize(block_count);
    artifacts.file_bytes.resize(total_size);

    size_t running_offset = 0;
    uint8_t* file_bytes = artifacts.file_bytes.data();

    for (size_t local = 0; local < block_count; ++local) {
        size_t global = block_begin + local;
        const DataBlockPlanEntry& plan = pack_result.plans[global];
        uint32_t local_first_kv = (uint32_t)(plan.first_kv - kv_begin);
        uint32_t block_size = pack_result.block_sizes[global];

        artifacts.data_meta[local].offset = running_offset;
        artifacts.data_meta[local].size = block_size;
        artifacts.data_meta[local].first_kv = local_first_kv;
        artifacts.data_meta[local].num_kv = plan.num_kv;

        std::memcpy(file_bytes + running_offset,
                    pack_result.block_buf.data() + pack_result.block_offsets[global],
                    block_size);
        running_offset += block_size;

        artifacts.index_entries[local].largest_key = kv_array[plan.first_kv + plan.num_kv - 1].key;
        artifacts.index_entries[local].data_offset = artifacts.data_meta[local].offset;
        artifacts.index_entries[local].data_size = block_size;
        artifacts.index_entries[local].num_kv = plan.num_kv;
    }

    uint64_t filter_region_offset = running_offset;
    for (size_t local = 0; local < block_count; ++local) {
        size_t global = block_begin + local;
        const DataBlockPlanEntry& plan = pack_result.plans[global];
        FilterBlockHeader hdr{};
        hdr.num_keys = plan.num_kv;
        hdr.byte_vector_len = plan.num_kv * GP_BLOOM_BITS_PER_KEY;
        hdr.bitvec_len = filter_lengths[global];

        artifacts.filter_meta[local].offset = running_offset;
        artifacts.filter_meta[local].size = (uint32_t)(sizeof(FilterBlockHeader) + filter_lengths[global]);
        artifacts.filter_meta[local].byte_vector_len = hdr.byte_vector_len;
        artifacts.filter_meta[local].num_keys = hdr.num_keys;

        std::memcpy(file_bytes + running_offset, &hdr, sizeof(FilterBlockHeader));
        running_offset += sizeof(FilterBlockHeader);
        std::memcpy(file_bytes + running_offset,
                    filter_bytes.data() + filter_offsets[global],
                    filter_lengths[global]);
        running_offset += filter_lengths[global];
    }

    uint32_t computed_filter_region_size = (uint32_t)(running_offset - filter_region_offset);
    uint64_t index_region_offset = running_offset;
    if (!artifacts.index_entries.empty()) {
        std::memcpy(file_bytes + running_offset, artifacts.index_entries.data(), index_region_size);
        running_offset += index_region_size;
    }

    uint64_t filter_meta_offset = running_offset;
    if (!artifacts.filter_meta.empty()) {
        std::memcpy(file_bytes + running_offset, artifacts.filter_meta.data(), filter_meta_size);
        running_offset += filter_meta_size;
    }

    uint64_t data_meta_offset = running_offset;
    if (!artifacts.data_meta.empty()) {
        std::memcpy(file_bytes + running_offset, artifacts.data_meta.data(), data_meta_size);
        running_offset += data_meta_size;
    }

    SSTFooter footer{};
    footer.magic = GP_SST_MAGIC;
    footer.version = GP_SST_VERSION;
    footer.key_bytes = GP_KEY_BYTES;
    footer.value_bytes = GP_VALUE_BYTES;
    footer.restart_interval = GP_RESTART_INTERVAL;
    footer.data_block_size = GP_DATA_BLOCK_BYTES;
    footer.bloom_bits_per_key = GP_BLOOM_BITS_PER_KEY;
    footer.bloom_hashes = GP_BLOOM_HASHES;
    footer.num_data_blocks = (uint32_t)block_count;
    footer.total_kv = (uint32_t)(kv_end - kv_begin);
    footer.filter_meta_offset = filter_meta_offset;
    footer.filter_meta_size = (uint32_t)filter_meta_size;
    footer.data_meta_offset = data_meta_offset;
    footer.data_meta_size = (uint32_t)data_meta_size;
    footer.filter_region_offset = filter_region_offset;
    footer.filter_region_size = computed_filter_region_size;
    footer.index_region_offset = index_region_offset;
    footer.index_region_size = (uint32_t)index_region_size;
    std::memcpy(file_bytes + running_offset, &footer, sizeof(SSTFooter));

    return artifacts;
}

static inline SSTBuildSet assemble_sst_files_targeted(const std::vector<KVPair>&            kv_array,
                                                      const PackResult&                      pack_result,
                                                      const std::vector<uint8_t>&           filter_bytes,
                                                      const std::vector<uint32_t>&          filter_offsets,
                                                      const std::vector<uint32_t>&          filter_lengths,
                                                      size_t                                 target_file_bytes = GP_TARGET_FILE_BYTES)
{
    std::vector<std::pair<size_t, size_t>> spans =
        partition_output_blocks(pack_result, filter_lengths, target_file_bytes);
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range(kv_array, pack_result, filter_bytes,
                                                    filter_offsets, filter_lengths,
                                                    span.first, span.second));
    }
    return out;
}

static inline SSTBuildSet assemble_sst_files_from_spans(const std::vector<KVPair>&                  kv_array,
                                                        const PackResult&                            pack_result,
                                                        const std::vector<uint8_t>&                 filter_bytes,
                                                        const std::vector<uint32_t>&                filter_offsets,
                                                        const std::vector<uint32_t>&                filter_lengths,
                                                        const std::vector<std::pair<size_t, size_t>>& spans)
{
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range(kv_array, pack_result, filter_bytes,
                                                    filter_offsets, filter_lengths,
                                                    span.first, span.second));
    }
    return out;
}

static inline SSTBuildSet assemble_sst_files_targeted_gpu_fast(const std::vector<KVPair>&            kv_array,
                                                               const PackResult&                      pack_result,
                                                               const std::vector<uint8_t>&           filter_bytes,
                                                               const std::vector<uint32_t>&          filter_offsets,
                                                               const std::vector<uint32_t>&          filter_lengths,
                                                               size_t                                 target_file_bytes = GP_TARGET_FILE_BYTES)
{
    std::vector<std::pair<size_t, size_t>> spans =
        partition_output_blocks(pack_result, filter_lengths, target_file_bytes);
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range_fast(kv_array, pack_result, filter_bytes,
                                                         filter_offsets, filter_lengths,
                                                         span.first, span.second));
    }
    return out;
}

static inline SSTBuildSet assemble_sst_files_from_spans_fast(
    const std::vector<KVPair>&                  kv_array,
    const PackResult&                            pack_result,
    const std::vector<uint8_t>&                 filter_bytes,
    const std::vector<uint32_t>&                filter_offsets,
    const std::vector<uint32_t>&                filter_lengths,
    const std::vector<std::pair<size_t, size_t>>& spans)
{
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range_fast(kv_array, pack_result, filter_bytes,
                                                         filter_offsets, filter_lengths,
                                                         span.first, span.second));
    }
    return out;
}

static inline SSTBuildArtifacts assemble_sst_file_range_from_largest_keys(
    const std::vector<Key128>&            largest_keys,
    const PackResult&                     pack_result,
    const std::vector<uint8_t>&           filter_bytes,
    const std::vector<uint32_t>&          filter_offsets,
    const std::vector<uint32_t>&          filter_lengths,
    size_t                                block_begin,
    size_t                                block_end)
{
    gp_fail_if(block_begin >= block_end, "invalid SST block range");

    SSTBuildArtifacts artifacts;
    size_t block_count = block_end - block_begin;
    size_t kv_begin = pack_result.plans[block_begin].first_kv;
    size_t kv_end = pack_result.plans[block_end - 1].first_kv + pack_result.plans[block_end - 1].num_kv;

    size_t data_region_size = 0;
    size_t filter_region_size = 0;
    for (size_t i = block_begin; i < block_end; ++i) {
        data_region_size += pack_result.block_sizes[i];
        filter_region_size += sizeof(FilterBlockHeader) + filter_lengths[i];
    }

    size_t index_region_size = block_count * sizeof(IndexEntry);
    size_t filter_meta_size = block_count * sizeof(FilterBlockMeta);
    size_t data_meta_size = block_count * sizeof(DataBlockMeta);
    size_t total_size = data_region_size + filter_region_size + index_region_size
                      + filter_meta_size + data_meta_size + sizeof(SSTFooter);

    artifacts.data_meta.resize(block_count);
    artifacts.filter_meta.resize(block_count);
    artifacts.index_entries.resize(block_count);
    artifacts.file_bytes.resize(total_size);

    size_t running_offset = 0;
    uint8_t* file_bytes_out = artifacts.file_bytes.data();

    for (size_t local = 0; local < block_count; ++local) {
        size_t global = block_begin + local;
        const DataBlockPlanEntry& plan = pack_result.plans[global];
        uint32_t local_first_kv = (uint32_t)(plan.first_kv - kv_begin);
        uint32_t block_size = pack_result.block_sizes[global];

        artifacts.data_meta[local].offset = running_offset;
        artifacts.data_meta[local].size = block_size;
        artifacts.data_meta[local].first_kv = local_first_kv;
        artifacts.data_meta[local].num_kv = plan.num_kv;

        std::memcpy(file_bytes_out + running_offset,
                    pack_result.block_buf.data() + pack_result.block_offsets[global],
                    block_size);
        running_offset += block_size;

        artifacts.index_entries[local].largest_key = largest_keys[global];
        artifacts.index_entries[local].data_offset = artifacts.data_meta[local].offset;
        artifacts.index_entries[local].data_size = block_size;
        artifacts.index_entries[local].num_kv = plan.num_kv;
    }

    uint64_t filter_region_offset = running_offset;
    for (size_t local = 0; local < block_count; ++local) {
        size_t global = block_begin + local;
        const DataBlockPlanEntry& plan = pack_result.plans[global];
        FilterBlockHeader hdr{};
        hdr.num_keys = plan.num_kv;
        hdr.byte_vector_len = plan.num_kv * GP_BLOOM_BITS_PER_KEY;
        hdr.bitvec_len = filter_lengths[global];

        artifacts.filter_meta[local].offset = running_offset;
        artifacts.filter_meta[local].size = (uint32_t)(sizeof(FilterBlockHeader) + filter_lengths[global]);
        artifacts.filter_meta[local].byte_vector_len = hdr.byte_vector_len;
        artifacts.filter_meta[local].num_keys = hdr.num_keys;

        std::memcpy(file_bytes_out + running_offset, &hdr, sizeof(FilterBlockHeader));
        running_offset += sizeof(FilterBlockHeader);
        std::memcpy(file_bytes_out + running_offset,
                    filter_bytes.data() + filter_offsets[global],
                    filter_lengths[global]);
        running_offset += filter_lengths[global];
    }

    uint32_t computed_filter_region_size = (uint32_t)(running_offset - filter_region_offset);
    uint64_t index_region_offset = running_offset;
    if (!artifacts.index_entries.empty()) {
        std::memcpy(file_bytes_out + running_offset, artifacts.index_entries.data(), index_region_size);
        running_offset += index_region_size;
    }

    uint64_t filter_meta_offset = running_offset;
    if (!artifacts.filter_meta.empty()) {
        std::memcpy(file_bytes_out + running_offset, artifacts.filter_meta.data(), filter_meta_size);
        running_offset += filter_meta_size;
    }

    uint64_t data_meta_offset = running_offset;
    if (!artifacts.data_meta.empty()) {
        std::memcpy(file_bytes_out + running_offset, artifacts.data_meta.data(), data_meta_size);
        running_offset += data_meta_size;
    }

    SSTFooter footer{};
    footer.magic = GP_SST_MAGIC;
    footer.version = GP_SST_VERSION;
    footer.key_bytes = GP_KEY_BYTES;
    footer.value_bytes = GP_VALUE_BYTES;
    footer.restart_interval = GP_RESTART_INTERVAL;
    footer.data_block_size = GP_DATA_BLOCK_BYTES;
    footer.bloom_bits_per_key = GP_BLOOM_BITS_PER_KEY;
    footer.bloom_hashes = GP_BLOOM_HASHES;
    footer.num_data_blocks = (uint32_t)block_count;
    footer.total_kv = (uint32_t)(kv_end - kv_begin);
    footer.filter_meta_offset = filter_meta_offset;
    footer.filter_meta_size = (uint32_t)filter_meta_size;
    footer.data_meta_offset = data_meta_offset;
    footer.data_meta_size = (uint32_t)data_meta_size;
    footer.filter_region_offset = filter_region_offset;
    footer.filter_region_size = computed_filter_region_size;
    footer.index_region_offset = index_region_offset;
    footer.index_region_size = (uint32_t)index_region_size;
    std::memcpy(file_bytes_out + running_offset, &footer, sizeof(SSTFooter));

    return artifacts;
}

static inline SSTBuildSet assemble_sst_files_targeted_from_largest_keys(
    const std::vector<Key128>&            largest_keys,
    const PackResult&                     pack_result,
    const std::vector<uint8_t>&           filter_bytes,
    const std::vector<uint32_t>&          filter_offsets,
    const std::vector<uint32_t>&          filter_lengths,
    size_t                                target_file_bytes = GP_TARGET_FILE_BYTES)
{
    std::vector<std::pair<size_t, size_t>> spans =
        partition_output_blocks(pack_result, filter_lengths, target_file_bytes);
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range_from_largest_keys(
            largest_keys, pack_result, filter_bytes, filter_offsets, filter_lengths, span.first, span.second));
    }
    return out;
}

static inline SSTBuildSet assemble_sst_files_from_spans_from_largest_keys(
    const std::vector<Key128>&              largest_keys,
    const PackResult&                       pack_result,
    const std::vector<uint8_t>&             filter_bytes,
    const std::vector<uint32_t>&            filter_offsets,
    const std::vector<uint32_t>&            filter_lengths,
    const std::vector<std::pair<size_t, size_t>>& spans)
{
    SSTBuildSet out;
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range_from_largest_keys(
            largest_keys, pack_result, filter_bytes, filter_offsets, filter_lengths, span.first, span.second));
    }
    return out;
}

static inline ParsedSST parse_sst_bytes(std::vector<uint8_t> bytes)
{
    ParsedSST parsed;
    gp_fail_if(bytes.size() < sizeof(SSTFooter), "SST file is too small");
    parsed.file_bytes = std::move(bytes);
    std::memcpy(&parsed.footer,
                parsed.file_bytes.data() + parsed.file_bytes.size() - sizeof(SSTFooter),
                sizeof(SSTFooter));

    gp_fail_if(parsed.footer.magic != GP_SST_MAGIC, "SST magic mismatch");
    gp_fail_if(parsed.footer.version != GP_SST_VERSION, "SST version mismatch");
    gp_fail_if(parsed.footer.key_bytes != GP_KEY_BYTES, "Unexpected key size");
    gp_fail_if(parsed.footer.value_bytes != GP_VALUE_BYTES, "Unexpected value size");
    gp_fail_if(parsed.footer.restart_interval != GP_RESTART_INTERVAL, "Unexpected restart interval");
    gp_fail_if(parsed.footer.data_block_size != GP_DATA_BLOCK_BYTES, "Unexpected data block size");

    parsed.data_blocks.resize(parsed.footer.num_data_blocks);
    parsed.filter_blocks.resize(parsed.footer.num_data_blocks);
    if (!parsed.data_blocks.empty()) {
        std::memcpy(parsed.data_blocks.data(),
                    parsed.file_bytes.data() + parsed.footer.data_meta_offset,
                    parsed.data_blocks.size() * sizeof(DataBlockMeta));
        std::memcpy(parsed.filter_blocks.data(),
                    parsed.file_bytes.data() + parsed.footer.filter_meta_offset,
                    parsed.filter_blocks.size() * sizeof(FilterBlockMeta));
    }
    return parsed;
}

static inline void write_binary_file(const std::string& path, const std::vector<uint8_t>& bytes)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    gp_fail_if(!f, "Failed to open '" + path + "' for writing");
    size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    gp_fail_if(written != bytes.size(), "Short write to '" + path + "'");
}

static inline void write_binary_file_span(const std::string& path, const uint8_t* bytes, size_t size)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    gp_fail_if(!f, "Failed to open '" + path + "' for writing");
    size_t written = size == 0 ? 0 : std::fwrite(bytes, 1, size, f);
    std::fclose(f);
    gp_fail_if(written != size, "Short write to '" + path + "'");
}

static inline std::vector<uint8_t> read_binary_file(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    gp_fail_if(!f, "Failed to open '" + path + "' for reading");
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    gp_fail_if(len < 0, "Failed to measure '" + path + "'");
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t)len);
    size_t read = std::fread(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    gp_fail_if(read != bytes.size(), "Short read from '" + path + "'");
    return bytes;
}

static inline ParsedSST read_sst_file(const std::string& path)
{
    return parse_sst_bytes(read_binary_file(path));
}

static inline SSTBuildArtifacts artifacts_from_serialized_sst(std::vector<uint8_t> file_bytes)
{
    ParsedSST parsed = parse_sst_bytes(std::move(file_bytes));
    SSTBuildArtifacts artifacts;
    artifacts.file_bytes = std::move(parsed.file_bytes);
    artifacts.data_meta = std::move(parsed.data_blocks);
    artifacts.filter_meta = std::move(parsed.filter_blocks);
    return artifacts;
}

struct DeviceSSTFileLayout {
    uint64_t buffer_offset = 0;
    uint64_t total_size = 0;
    uint64_t filter_region_offset = 0;
    uint32_t filter_region_size = 0;
    uint64_t index_region_offset = 0;
    uint32_t index_region_size = 0;
    uint64_t filter_meta_offset = 0;
    uint32_t filter_meta_size = 0;
    uint64_t data_meta_offset = 0;
    uint32_t data_meta_size = 0;
    uint32_t num_data_blocks = 0;
    uint32_t total_kv = 0;
};

struct DeviceSSTBlockTask {
    uint32_t file_index = 0;
    uint32_t local_block_index = 0;
    uint32_t global_block_index = 0;
    uint64_t data_dst_offset = 0;
    uint64_t filter_dst_offset = 0;
    uint32_t block_size = 0;
    uint32_t filter_src_offset = 0;
    uint32_t filter_length = 0;
    uint32_t local_first_kv = 0;
    uint32_t num_kv = 0;
};

struct DeviceAssembleSSTResult {
    SSTBuildSet output;
    SerializedSSTHostSet serialized_output;
    float       kernel_ms = 0.0f;
    float       wall_ms = 0.0f;
    float       h2d_ms = 0.0f;
    float       d2h_ms = 0.0f;
    size_t      h2d_bytes = 0;
    size_t      d2h_bytes = 0;
};

__global__ static void assemble_sst_blocks_kernel(uint8_t*                    all_file_bytes,
                                                  const DeviceSSTFileLayout*  file_layouts,
                                                  const DeviceSSTBlockTask*   tasks,
                                                  int                         num_tasks,
                                                  const uint8_t*              packed_blocks,
                                                  const uint8_t*              filter_bytes,
                                                  const Key128*               largest_keys)
{
    int task_idx = (int)blockIdx.x;
    if (task_idx >= num_tasks) return;

    DeviceSSTBlockTask task = tasks[task_idx];
    DeviceSSTFileLayout layout = file_layouts[task.file_index];
    uint8_t* file = all_file_bytes + layout.buffer_offset;

    if (threadIdx.x == 0) {
        DataBlockMeta* data_meta =
            reinterpret_cast<DataBlockMeta*>(file + layout.data_meta_offset);
        data_meta[task.local_block_index].offset = task.data_dst_offset;
        data_meta[task.local_block_index].size = task.block_size;
        data_meta[task.local_block_index].first_kv = task.local_first_kv;
        data_meta[task.local_block_index].num_kv = task.num_kv;

        FilterBlockMeta* filter_meta =
            reinterpret_cast<FilterBlockMeta*>(file + layout.filter_meta_offset);
        filter_meta[task.local_block_index].offset = task.filter_dst_offset;
        filter_meta[task.local_block_index].size =
            (uint32_t)(sizeof(FilterBlockHeader) + task.filter_length);
        filter_meta[task.local_block_index].byte_vector_len =
            task.num_kv * GP_BLOOM_BITS_PER_KEY;
        filter_meta[task.local_block_index].num_keys = task.num_kv;

        IndexEntry* index_entries =
            reinterpret_cast<IndexEntry*>(file + layout.index_region_offset);
        index_entries[task.local_block_index].largest_key = largest_keys[task.global_block_index];
        index_entries[task.local_block_index].data_offset = task.data_dst_offset;
        index_entries[task.local_block_index].data_size = task.block_size;
        index_entries[task.local_block_index].num_kv = task.num_kv;

        FilterBlockHeader* hdr =
            reinterpret_cast<FilterBlockHeader*>(file + task.filter_dst_offset);
        hdr->num_keys = task.num_kv;
        hdr->byte_vector_len = task.num_kv * GP_BLOOM_BITS_PER_KEY;
        hdr->bitvec_len = task.filter_length;
        hdr->reserved = 0;
    }

    const uint8_t* block_src =
        packed_blocks + (size_t)task.global_block_index * GP_DATA_BLOCK_BYTES;
    for (uint32_t i = (uint32_t)threadIdx.x; i < task.block_size; i += (uint32_t)blockDim.x) {
        file[task.data_dst_offset + i] = block_src[i];
    }

    uint8_t* filter_dst = file + task.filter_dst_offset + sizeof(FilterBlockHeader);
    const uint8_t* filter_src = filter_bytes + task.filter_src_offset;
    for (uint32_t i = (uint32_t)threadIdx.x; i < task.filter_length; i += (uint32_t)blockDim.x) {
        filter_dst[i] = filter_src[i];
    }
}

__global__ static void write_sst_footers_kernel(uint8_t*                   all_file_bytes,
                                                const DeviceSSTFileLayout* file_layouts,
                                                int                        num_files)
{
    int file_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (file_idx >= num_files) return;

    DeviceSSTFileLayout layout = file_layouts[file_idx];
    uint8_t* file = all_file_bytes + layout.buffer_offset;
    union {
        SSTFooter footer;
        uint8_t   bytes[sizeof(SSTFooter)];
    } footer_image = {};
    footer_image.footer.magic = GP_SST_MAGIC;
    footer_image.footer.version = GP_SST_VERSION;
    footer_image.footer.key_bytes = GP_KEY_BYTES;
    footer_image.footer.value_bytes = GP_VALUE_BYTES;
    footer_image.footer.restart_interval = GP_RESTART_INTERVAL;
    footer_image.footer.data_block_size = GP_DATA_BLOCK_BYTES;
    footer_image.footer.bloom_bits_per_key = GP_BLOOM_BITS_PER_KEY;
    footer_image.footer.bloom_hashes = GP_BLOOM_HASHES;
    footer_image.footer.num_data_blocks = layout.num_data_blocks;
    footer_image.footer.total_kv = layout.total_kv;
    footer_image.footer.filter_meta_offset = layout.filter_meta_offset;
    footer_image.footer.filter_meta_size = layout.filter_meta_size;
    footer_image.footer.data_meta_offset = layout.data_meta_offset;
    footer_image.footer.data_meta_size = layout.data_meta_size;
    footer_image.footer.filter_region_offset = layout.filter_region_offset;
    footer_image.footer.filter_region_size = layout.filter_region_size;
    footer_image.footer.index_region_offset = layout.index_region_offset;
    footer_image.footer.index_region_size = layout.index_region_size;

    uint8_t* footer_dst = file + layout.total_size - sizeof(SSTFooter);
    for (size_t i = 0; i < sizeof(SSTFooter); ++i) footer_dst[i] = footer_image.bytes[i];
}

static inline DeviceAssembleSSTResult
assemble_sst_files_from_spans_on_device(const std::vector<DataBlockPlanEntry>&      plans,
                                        const std::vector<uint32_t>&                 block_sizes,
                                        const uint8_t*                               d_packed_blocks,
                                        const Key128*                                d_largest_keys,
                                        const uint8_t*                               d_filter_bytes,
                                        const std::vector<uint32_t>&                 filter_offsets,
                                        const std::vector<uint32_t>&                 filter_lengths,
                                        const std::vector<std::pair<size_t, size_t>>& spans,
                                        bool                                          materialize_output = true)
{
    DeviceAssembleSSTResult result;
    if (spans.empty()) return result;

    std::vector<DeviceSSTFileLayout> layouts(spans.size());
    std::vector<DeviceSSTBlockTask> tasks;
    tasks.reserve(plans.size());

    uint64_t total_output_bytes = 0;
    for (size_t file_idx = 0; file_idx < spans.size(); ++file_idx) {
        size_t block_begin = spans[file_idx].first;
        size_t block_end = spans[file_idx].second;
        gp_fail_if(block_begin >= block_end, "invalid SST block range");

        size_t block_count = block_end - block_begin;
        size_t kv_begin = plans[block_begin].first_kv;
        size_t kv_end = plans[block_end - 1].first_kv + plans[block_end - 1].num_kv;

        uint64_t data_region_size = 0;
        uint64_t filter_region_size = 0;
        for (size_t global = block_begin; global < block_end; ++global) {
            data_region_size += block_sizes[global];
            filter_region_size += sizeof(FilterBlockHeader) + filter_lengths[global];
        }

        DeviceSSTFileLayout& layout = layouts[file_idx];
        layout.buffer_offset = total_output_bytes;
        layout.filter_region_offset = data_region_size;
        layout.filter_region_size = (uint32_t)filter_region_size;
        layout.index_region_offset = layout.filter_region_offset + filter_region_size;
        layout.index_region_size = (uint32_t)(block_count * sizeof(IndexEntry));
        layout.filter_meta_offset = layout.index_region_offset + layout.index_region_size;
        layout.filter_meta_size = (uint32_t)(block_count * sizeof(FilterBlockMeta));
        layout.data_meta_offset = layout.filter_meta_offset + layout.filter_meta_size;
        layout.data_meta_size = (uint32_t)(block_count * sizeof(DataBlockMeta));
        layout.num_data_blocks = (uint32_t)block_count;
        layout.total_kv = (uint32_t)(kv_end - kv_begin);
        layout.total_size = layout.data_meta_offset + layout.data_meta_size + sizeof(SSTFooter);

        uint64_t running_data_offset = 0;
        uint64_t running_filter_offset = layout.filter_region_offset;
        for (size_t local = 0; local < block_count; ++local) {
            size_t global = block_begin + local;
            DeviceSSTBlockTask task{};
            task.file_index = (uint32_t)file_idx;
            task.local_block_index = (uint32_t)local;
            task.global_block_index = (uint32_t)global;
            task.data_dst_offset = running_data_offset;
            task.filter_dst_offset = running_filter_offset;
            task.block_size = block_sizes[global];
            task.filter_src_offset = filter_offsets[global];
            task.filter_length = filter_lengths[global];
            task.local_first_kv = (uint32_t)(plans[global].first_kv - kv_begin);
            task.num_kv = plans[global].num_kv;
            tasks.push_back(task);

            running_data_offset += block_sizes[global];
            running_filter_offset += sizeof(FilterBlockHeader) + filter_lengths[global];
        }

        total_output_bytes += layout.total_size;
    }

    uint8_t* d_all_files = nullptr;
    DeviceSSTFileLayout* d_layouts = nullptr;
    DeviceSSTBlockTask* d_tasks = nullptr;
    cudaMalloc(&d_all_files, (size_t)std::max<uint64_t>(total_output_bytes, 1u));
    cudaMalloc(&d_layouts, layouts.size() * sizeof(DeviceSSTFileLayout));
    cudaMalloc(&d_tasks, tasks.size() * sizeof(DeviceSSTBlockTask));

    auto h2d_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_layouts, layouts.data(),
               layouts.size() * sizeof(DeviceSSTFileLayout), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tasks, tasks.data(),
               tasks.size() * sizeof(DeviceSSTBlockTask), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = layouts.size() * sizeof(DeviceSSTFileLayout)
                     + tasks.size() * sizeof(DeviceSSTBlockTask);

    auto wall_start = std::chrono::steady_clock::now();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    assemble_sst_blocks_kernel<<<(int)tasks.size(), 256>>>(
        d_all_files, d_layouts, d_tasks, (int)tasks.size(), d_packed_blocks, d_filter_bytes, d_largest_keys);
    int footer_block = 128;
    int footer_grid = ((int)layouts.size() + footer_block - 1) / footer_block;
    write_sst_footers_kernel<<<footer_grid, footer_block>>>(d_all_files, d_layouts, (int)layouts.size());
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    uint8_t* h_all_files = nullptr;
    cudaMallocHost(&h_all_files, (size_t)std::max<uint64_t>(total_output_bytes, 1u));
    auto d2h_start = std::chrono::steady_clock::now();
    cudaMemcpy(h_all_files, d_all_files,
               (size_t)total_output_bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::steady_clock::now();
    result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
    result.d2h_bytes = (size_t)total_output_bytes;
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    if (materialize_output) {
        result.output.files.reserve(layouts.size());
        for (const auto& layout : layouts) {
            SSTBuildArtifacts artifacts;
            artifacts.file_bytes.resize((size_t)layout.total_size);
            std::memcpy(artifacts.file_bytes.data(),
                        h_all_files + layout.buffer_offset,
                        (size_t)layout.total_size);

            artifacts.index_entries.resize(layout.num_data_blocks);
            if (!artifacts.index_entries.empty()) {
                std::memcpy(artifacts.index_entries.data(),
                            artifacts.file_bytes.data() + layout.index_region_offset,
                            (size_t)layout.index_region_size);
            }

            artifacts.filter_meta.resize(layout.num_data_blocks);
            if (!artifacts.filter_meta.empty()) {
                std::memcpy(artifacts.filter_meta.data(),
                            artifacts.file_bytes.data() + layout.filter_meta_offset,
                            (size_t)layout.filter_meta_size);
            }

            artifacts.data_meta.resize(layout.num_data_blocks);
            if (!artifacts.data_meta.empty()) {
                std::memcpy(artifacts.data_meta.data(),
                            artifacts.file_bytes.data() + layout.data_meta_offset,
                            (size_t)layout.data_meta_size);
            }

            result.output.files.push_back(std::move(artifacts));
        }
    } else {
        result.serialized_output.all_file_bytes.data = h_all_files;
        result.serialized_output.all_file_bytes.size = (size_t)total_output_bytes;
        result.serialized_output.file_offsets.reserve(layouts.size());
        result.serialized_output.file_sizes.reserve(layouts.size());
        result.serialized_output.file_blocks.reserve(layouts.size());
        for (const auto& layout : layouts) {
            result.serialized_output.file_offsets.push_back(layout.buffer_offset);
            result.serialized_output.file_sizes.push_back(layout.total_size);
            result.serialized_output.file_blocks.push_back(layout.num_data_blocks);
        }
        h_all_files = nullptr;
    }

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    if (h_all_files) cudaFreeHost(h_all_files);
    cudaFree(d_tasks);
    cudaFree(d_layouts);
    cudaFree(d_all_files);
    return result;
}

struct DeviceAssembleSSTUntimedResult {
    SSTBuildSet output;
    SerializedSSTHostSet serialized_output;
};

static inline DeviceAssembleSSTUntimedResult
assemble_sst_files_untimed_on_device(const std::vector<DataBlockPlanEntry>&      plans,
                                     const std::vector<uint32_t>&                 block_sizes,
                                     const uint8_t*                               d_packed_blocks,
                                     const Key128*                                d_largest_keys,
                                     const uint8_t*                               d_filter_bytes,
                                     const std::vector<uint32_t>&                 filter_offsets,
                                     const std::vector<uint32_t>&                 filter_lengths,
                                     const std::vector<std::pair<size_t, size_t>>& spans,
                                     bool                                          materialize_output = true)
{
    DeviceAssembleSSTUntimedResult result;
    if (spans.empty()) return result;

    std::vector<DeviceSSTFileLayout> layouts(spans.size());
    std::vector<DeviceSSTBlockTask> tasks;
    tasks.reserve(plans.size());

    uint64_t total_output_bytes = 0;
    for (size_t file_idx = 0; file_idx < spans.size(); ++file_idx) {
        size_t block_begin = spans[file_idx].first;
        size_t block_end = spans[file_idx].second;
        gp_fail_if(block_begin >= block_end, "invalid SST block range");

        size_t block_count = block_end - block_begin;
        size_t kv_begin = plans[block_begin].first_kv;
        size_t kv_end = plans[block_end - 1].first_kv + plans[block_end - 1].num_kv;

        uint64_t data_region_size = 0;
        uint64_t filter_region_size = 0;
        for (size_t global = block_begin; global < block_end; ++global) {
            data_region_size += block_sizes[global];
            filter_region_size += sizeof(FilterBlockHeader) + filter_lengths[global];
        }

        DeviceSSTFileLayout& layout = layouts[file_idx];
        layout.buffer_offset = total_output_bytes;
        layout.filter_region_offset = data_region_size;
        layout.filter_region_size = (uint32_t)filter_region_size;
        layout.index_region_offset = layout.filter_region_offset + filter_region_size;
        layout.index_region_size = (uint32_t)(block_count * sizeof(IndexEntry));
        layout.filter_meta_offset = layout.index_region_offset + layout.index_region_size;
        layout.filter_meta_size = (uint32_t)(block_count * sizeof(FilterBlockMeta));
        layout.data_meta_offset = layout.filter_meta_offset + layout.filter_meta_size;
        layout.data_meta_size = (uint32_t)(block_count * sizeof(DataBlockMeta));
        layout.num_data_blocks = (uint32_t)block_count;
        layout.total_kv = (uint32_t)(kv_end - kv_begin);
        layout.total_size = layout.data_meta_offset + layout.data_meta_size + sizeof(SSTFooter);

        uint64_t running_data_offset = 0;
        uint64_t running_filter_offset = layout.filter_region_offset;
        for (size_t local = 0; local < block_count; ++local) {
            size_t global = block_begin + local;
            DeviceSSTBlockTask task{};
            task.file_index = (uint32_t)file_idx;
            task.local_block_index = (uint32_t)local;
            task.global_block_index = (uint32_t)global;
            task.data_dst_offset = running_data_offset;
            task.filter_dst_offset = running_filter_offset;
            task.block_size = block_sizes[global];
            task.filter_src_offset = filter_offsets[global];
            task.filter_length = filter_lengths[global];
            task.local_first_kv = (uint32_t)(plans[global].first_kv - kv_begin);
            task.num_kv = plans[global].num_kv;
            tasks.push_back(task);

            running_data_offset += block_sizes[global];
            running_filter_offset += sizeof(FilterBlockHeader) + filter_lengths[global];
        }
        total_output_bytes += layout.total_size;
    }

    uint8_t* d_all_files = nullptr;
    DeviceSSTFileLayout* d_layouts = nullptr;
    DeviceSSTBlockTask* d_tasks = nullptr;
    cudaMalloc(&d_all_files, (size_t)std::max<uint64_t>(total_output_bytes, 1u));
    cudaMalloc(&d_layouts, layouts.size() * sizeof(DeviceSSTFileLayout));
    cudaMalloc(&d_tasks, tasks.size() * sizeof(DeviceSSTBlockTask));

    cudaMemcpy(d_layouts, layouts.data(),
               layouts.size() * sizeof(DeviceSSTFileLayout), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tasks, tasks.data(),
               tasks.size() * sizeof(DeviceSSTBlockTask), cudaMemcpyHostToDevice);

    assemble_sst_blocks_kernel<<<(int)tasks.size(), 256>>>(
        d_all_files, d_layouts, d_tasks, (int)tasks.size(), d_packed_blocks, d_filter_bytes, d_largest_keys);
    int footer_block = 128;
    int footer_grid = ((int)layouts.size() + footer_block - 1) / footer_block;
    write_sst_footers_kernel<<<footer_grid, footer_block>>>(d_all_files, d_layouts, (int)layouts.size());
    cudaDeviceSynchronize();

    uint8_t* h_all_files = nullptr;
    cudaMallocHost(&h_all_files, (size_t)std::max<uint64_t>(total_output_bytes, 1u));
    cudaMemcpy(h_all_files, d_all_files, (size_t)total_output_bytes, cudaMemcpyDeviceToHost);

    if (materialize_output) {
        result.output.files.reserve(layouts.size());
        for (const auto& layout : layouts) {
            SSTBuildArtifacts artifacts;
            artifacts.file_bytes.resize((size_t)layout.total_size);
            std::memcpy(artifacts.file_bytes.data(),
                        h_all_files + layout.buffer_offset,
                        (size_t)layout.total_size);
            artifacts.index_entries.resize(layout.num_data_blocks);
            if (!artifacts.index_entries.empty())
                std::memcpy(artifacts.index_entries.data(),
                            artifacts.file_bytes.data() + layout.index_region_offset,
                            (size_t)layout.index_region_size);
            artifacts.filter_meta.resize(layout.num_data_blocks);
            if (!artifacts.filter_meta.empty())
                std::memcpy(artifacts.filter_meta.data(),
                            artifacts.file_bytes.data() + layout.filter_meta_offset,
                            (size_t)layout.filter_meta_size);
            artifacts.data_meta.resize(layout.num_data_blocks);
            if (!artifacts.data_meta.empty())
                std::memcpy(artifacts.data_meta.data(),
                            artifacts.file_bytes.data() + layout.data_meta_offset,
                            (size_t)layout.data_meta_size);
            result.output.files.push_back(std::move(artifacts));
        }
    } else {
        result.serialized_output.all_file_bytes.data = h_all_files;
        result.serialized_output.all_file_bytes.size = (size_t)total_output_bytes;
        result.serialized_output.file_offsets.reserve(layouts.size());
        result.serialized_output.file_sizes.reserve(layouts.size());
        result.serialized_output.file_blocks.reserve(layouts.size());
        for (const auto& layout : layouts) {
            result.serialized_output.file_offsets.push_back(layout.buffer_offset);
            result.serialized_output.file_sizes.push_back(layout.total_size);
            result.serialized_output.file_blocks.push_back(layout.num_data_blocks);
        }
        h_all_files = nullptr;
    }

    if (h_all_files) cudaFreeHost(h_all_files);
    cudaFree(d_tasks);
    cudaFree(d_layouts);
    cudaFree(d_all_files);
    return result;
}

static inline std::vector<DataBlockPlanEntry> plans_from_parsed(const ParsedSST& parsed)
{
    std::vector<DataBlockPlanEntry> plans(parsed.data_blocks.size());
    for (size_t i = 0; i < parsed.data_blocks.size(); ++i) {
        plans[i].first_kv = parsed.data_blocks[i].first_kv;
        plans[i].num_kv = parsed.data_blocks[i].num_kv;
        plans[i].serialized_size = parsed.data_blocks[i].size;
    }
    return plans;
}

static inline std::vector<uint32_t> block_offsets_from_parsed(const ParsedSST& parsed)
{
    std::vector<uint32_t> offsets(parsed.data_blocks.size());
    for (size_t i = 0; i < parsed.data_blocks.size(); ++i) offsets[i] = (uint32_t)parsed.data_blocks[i].offset;
    return offsets;
}

static inline std::vector<KVPair> cpu_unpack_sst(const ParsedSST& parsed)
{
    return cpu_unpack_all(parsed.file_bytes,
                          block_offsets_from_parsed(parsed),
                          plans_from_parsed(parsed),
                          parsed.footer.total_kv);
}
