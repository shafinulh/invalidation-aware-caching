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

static inline SSTBuildSet assemble_sst_files_targeted(const std::vector<KVPair>&            kv_array,
                                                      const PackResult&                      pack_result,
                                                      const std::vector<uint8_t>&           filter_bytes,
                                                      const std::vector<uint32_t>&          filter_offsets,
                                                      const std::vector<uint32_t>&          filter_lengths,
                                                      size_t                                 target_file_bytes = GP_TARGET_FILE_BYTES)
{
    SSTBuildSet out;
    std::vector<std::pair<size_t, size_t>> spans =
        partition_output_blocks(pack_result, filter_lengths, target_file_bytes);
    out.files.reserve(spans.size());
    for (const auto& span : spans) {
        out.files.push_back(assemble_sst_file_range(kv_array, pack_result, filter_bytes,
                                                    filter_offsets, filter_lengths,
                                                    span.first, span.second));
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
