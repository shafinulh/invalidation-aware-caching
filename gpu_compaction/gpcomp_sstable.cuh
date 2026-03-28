#pragma once

#include "gpcomp_bloom.cuh"
#include "gpcomp_pack.cuh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
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

static constexpr size_t GP_DIRECT_IO_ALIGNMENT = 4096;
static constexpr uint64_t GP_DIRECT_IO_TRAILER_MAGIC = 0x4750444954524149ULL; /* "GPDITRAI" */

struct __attribute__((packed)) DirectIoTrailer {
    uint64_t magic;
    uint64_t logical_size;
};

static inline bool gpcomp_direct_io_enabled()
{
    const char* value = std::getenv("GPCOMP_DIRECT_IO");
    if (!value || value[0] == '\0') return true;
    return std::strcmp(value, "0") != 0
        && std::strcmp(value, "false") != 0
        && std::strcmp(value, "FALSE") != 0;
}

static inline size_t gp_align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static inline void strip_direct_io_trailer_if_present(std::vector<uint8_t>& bytes,
                                                      const std::string&    path)
{
    if (bytes.size() < sizeof(DirectIoTrailer)) return;

    DirectIoTrailer trailer{};
    std::memcpy(&trailer,
                bytes.data() + bytes.size() - sizeof(DirectIoTrailer),
                sizeof(DirectIoTrailer));
    if (trailer.magic != GP_DIRECT_IO_TRAILER_MAGIC) return;

    gp_fail_if(trailer.logical_size > bytes.size() - sizeof(DirectIoTrailer),
               "Corrupt direct-I/O trailer in '" + path + "'");
    bytes.resize((size_t)trailer.logical_size);
}

static inline void write_binary_file_direct(const std::string& path, const uint8_t* bytes, size_t size)
{
    const size_t physical_size = gp_align_up(size + sizeof(DirectIoTrailer), GP_DIRECT_IO_ALIGNMENT);
    void* aligned_buf = nullptr;
    gp_fail_if(posix_memalign(&aligned_buf, GP_DIRECT_IO_ALIGNMENT, physical_size) != 0,
               "Failed to allocate aligned direct-I/O buffer for '" + path + "'");
    std::memset(aligned_buf, 0, physical_size);
    if (size > 0) {
        std::memcpy(aligned_buf, bytes, size);
    }

    DirectIoTrailer trailer{};
    trailer.magic = GP_DIRECT_IO_TRAILER_MAGIC;
    trailer.logical_size = (uint64_t)size;
    std::memcpy((uint8_t*)aligned_buf + physical_size - sizeof(DirectIoTrailer),
                &trailer,
                sizeof(DirectIoTrailer));

    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        std::free(aligned_buf);
        gp_fail_if(true, "Failed to open '" + path + "' for direct write: " + std::string(std::strerror(errno)));
    }

    size_t written_total = 0;
    while (written_total < physical_size) {
        ssize_t written = write(fd,
                                (const uint8_t*)aligned_buf + written_total,
                                physical_size - written_total);
        if (written < 0) {
            int saved_errno = errno;
            close(fd);
            std::free(aligned_buf);
            gp_fail_if(true, "Direct write failed for '" + path + "': " + std::string(std::strerror(saved_errno)));
        }
        written_total += (size_t)written;
    }

    if (fdatasync(fd) != 0) {
        int saved_errno = errno;
        close(fd);
        std::free(aligned_buf);
        gp_fail_if(true, "fdatasync failed for '" + path + "': " + std::string(std::strerror(saved_errno)));
    }
    close(fd);
    std::free(aligned_buf);
}

static inline std::vector<uint8_t> read_binary_file_direct(const std::string& path)
{
    struct stat st{};
    gp_fail_if(stat(path.c_str(), &st) != 0, "Failed to stat '" + path + "'");
    gp_fail_if(st.st_size < (off_t)sizeof(DirectIoTrailer),
               "Direct-I/O file too small to contain trailer: '" + path + "'");
    gp_fail_if((size_t)st.st_size % GP_DIRECT_IO_ALIGNMENT != 0,
               "Direct-I/O file size is not aligned; regenerate '" + path + "' with direct-I/O-enabled datagen");

    const size_t physical_size = (size_t)st.st_size;
    void* aligned_buf = nullptr;
    gp_fail_if(posix_memalign(&aligned_buf, GP_DIRECT_IO_ALIGNMENT, physical_size) != 0,
               "Failed to allocate aligned direct-I/O read buffer for '" + path + "'");

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::free(aligned_buf);
        gp_fail_if(true, "Failed to open '" + path + "' for direct read: " + std::string(std::strerror(errno)));
    }

    size_t read_total = 0;
    while (read_total < physical_size) {
        ssize_t read_now = read(fd, (uint8_t*)aligned_buf + read_total, physical_size - read_total);
        if (read_now < 0) {
            int saved_errno = errno;
            close(fd);
            std::free(aligned_buf);
            gp_fail_if(true, "Direct read failed for '" + path + "': " + std::string(std::strerror(saved_errno)));
        }
        gp_fail_if(read_now == 0, "Unexpected EOF during direct read of '" + path + "'");
        read_total += (size_t)read_now;
    }

    close(fd);
    std::vector<uint8_t> bytes(physical_size);
    std::memcpy(bytes.data(), aligned_buf, physical_size);
    std::free(aligned_buf);
    strip_direct_io_trailer_if_present(bytes, path);
    return bytes;
}

struct RegisteredDirectReadBuffer {
    uint8_t* data = nullptr;
    size_t   logical_size = 0;
    size_t   physical_size = 0;
    bool     registered = false;

    RegisteredDirectReadBuffer() = default;
    RegisteredDirectReadBuffer(const RegisteredDirectReadBuffer&) = delete;
    RegisteredDirectReadBuffer& operator=(const RegisteredDirectReadBuffer&) = delete;

    RegisteredDirectReadBuffer(RegisteredDirectReadBuffer&& other) noexcept
        : data(other.data),
          logical_size(other.logical_size),
          physical_size(other.physical_size),
          registered(other.registered)
    {
        other.data = nullptr;
        other.logical_size = 0;
        other.physical_size = 0;
        other.registered = false;
    }

    RegisteredDirectReadBuffer& operator=(RegisteredDirectReadBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            data = other.data;
            logical_size = other.logical_size;
            physical_size = other.physical_size;
            registered = other.registered;
            other.data = nullptr;
            other.logical_size = 0;
            other.physical_size = 0;
            other.registered = false;
        }
        return *this;
    }

    ~RegisteredDirectReadBuffer() { reset(); }

    void reset()
    {
        if (data) {
            if (registered) cudaHostUnregister(data);
            std::free(data);
        }
        data = nullptr;
        logical_size = 0;
        physical_size = 0;
        registered = false;
    }
};

static inline RegisteredDirectReadBuffer read_binary_file_direct_registered(const std::string& path)
{
    RegisteredDirectReadBuffer result;

    struct stat st{};
    gp_fail_if(stat(path.c_str(), &st) != 0,
               "Failed to stat '" + path + "': " + std::string(std::strerror(errno)));
    gp_fail_if(st.st_size < (off_t)sizeof(SSTFooter),
               "SST file is too small: '" + path + "'");

    result.physical_size = (size_t)st.st_size;
    result.logical_size = result.physical_size;

    gp_fail_if(posix_memalign((void**)&result.data, GP_DIRECT_IO_ALIGNMENT, result.physical_size) != 0,
               "Failed to allocate aligned read buffer for '" + path + "'");

    int flags = O_RDONLY;
    if (gpcomp_direct_io_enabled()) flags |= O_DIRECT;
    int fd = open(path.c_str(), flags);
    if (fd < 0 && (flags & O_DIRECT)) {
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd < 0) {
        result.reset();
        gp_fail_if(true, "Failed to open '" + path + "' for read: " + std::string(std::strerror(errno)));
    }

    size_t read_total = 0;
    while (read_total < result.physical_size) {
        ssize_t read_now = read(fd, result.data + read_total, result.physical_size - read_total);
        if (read_now < 0) {
            int saved_errno = errno;
            close(fd);
            result.reset();
            gp_fail_if(true, "Read failed for '" + path + "': " + std::string(std::strerror(saved_errno)));
        }
        gp_fail_if(read_now == 0, "Unexpected EOF during read of '" + path + "'");
        read_total += (size_t)read_now;
    }
    close(fd);

    if (result.physical_size >= sizeof(DirectIoTrailer)) {
        DirectIoTrailer trailer{};
        std::memcpy(&trailer,
                    result.data + result.physical_size - sizeof(DirectIoTrailer),
                    sizeof(DirectIoTrailer));
        if (trailer.magic == GP_DIRECT_IO_TRAILER_MAGIC) {
            gp_fail_if(trailer.logical_size > result.physical_size - sizeof(DirectIoTrailer),
                       "Corrupt direct-I/O trailer in '" + path + "'");
            result.logical_size = (size_t)trailer.logical_size;
        }
    }

    cudaError_t register_err =
        cudaHostRegister(result.data, result.physical_size, cudaHostRegisterDefault);
    gp_fail_if(register_err != cudaSuccess,
               "Failed to register pinned read buffer for '" + path + "': "
                   + std::string(cudaGetErrorString(register_err)));
    result.registered = true;
    return result;
}

static inline void ensure_registered_direct_read_buffer(uint8_t*& data,
                                                        size_t&   capacity,
                                                        bool&     registered,
                                                        size_t    required_size)
{
    if (capacity >= required_size && data) return;

    if (data) {
        if (registered) cudaHostUnregister(data);
        std::free(data);
        data = nullptr;
        capacity = 0;
        registered = false;
    }

    gp_fail_if(posix_memalign((void**)&data, GP_DIRECT_IO_ALIGNMENT, required_size) != 0,
               "Failed to allocate aligned reusable read buffer");
    cudaError_t register_err = cudaHostRegister(data, required_size, cudaHostRegisterDefault);
    gp_fail_if(register_err != cudaSuccess,
               "Failed to register reusable pinned read buffer: "
                   + std::string(cudaGetErrorString(register_err)));
    capacity = required_size;
    registered = true;
}

static inline void read_binary_file_direct_into_registered(const std::string& path,
                                                           uint8_t*           data,
                                                           size_t             physical_size)
{
    int flags = O_RDONLY;
    if (gpcomp_direct_io_enabled()) flags |= O_DIRECT;
    int fd = open(path.c_str(), flags);
    if (fd < 0 && (flags & O_DIRECT)) {
        fd = open(path.c_str(), O_RDONLY);
    }
    gp_fail_if(fd < 0, "Failed to open '" + path + "' for read: " + std::string(std::strerror(errno)));

    size_t read_total = 0;
    while (read_total < physical_size) {
        ssize_t read_now = read(fd, data + read_total, physical_size - read_total);
        if (read_now < 0) {
            int saved_errno = errno;
            close(fd);
            gp_fail_if(true, "Read failed for '" + path + "': " + std::string(std::strerror(saved_errno)));
        }
        gp_fail_if(read_now == 0, "Unexpected EOF during read of '" + path + "'");
        read_total += (size_t)read_now;
    }
    close(fd);
}

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

static inline std::vector<uint8_t> build_cpu_filter_bytes(const std::vector<KVRef>& ref_array,
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
        std::fill(byte_vector.begin(), byte_vector.end(), 0);
        for (uint32_t j = 0; j < plans[i].num_kv; ++j) {
            const Key128& key = ref_array[plans[i].first_kv + j].key;
            for (int k = 1; k <= GP_BLOOM_HASHES; ++k) {
                uint32_t h = bloom_hash_key(key, k);
                byte_vector[h % (uint32_t)byte_vector_len] = 1;
            }
        }
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
    if (gpcomp_direct_io_enabled()) {
        write_binary_file_direct(path, bytes.data(), bytes.size());
        return;
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    gp_fail_if(!f, "Failed to open '" + path + "' for writing");
    size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    gp_fail_if(written != bytes.size(), "Short write to '" + path + "'");
}

static inline void write_binary_file_span(const std::string& path, const uint8_t* bytes, size_t size)
{
    if (gpcomp_direct_io_enabled()) {
        write_binary_file_direct(path, bytes, size);
        return;
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    gp_fail_if(!f, "Failed to open '" + path + "' for writing");
    size_t written = size == 0 ? 0 : std::fwrite(bytes, 1, size, f);
    std::fclose(f);
    gp_fail_if(written != size, "Short write to '" + path + "'");
}

static inline std::vector<uint8_t> read_binary_file(const std::string& path)
{
    if (gpcomp_direct_io_enabled()) {
        return read_binary_file_direct(path);
    }
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
    strip_direct_io_trailer_if_present(bytes, path);
    return bytes;
}

static inline ParsedSST read_sst_file(const std::string& path)
{
    return parse_sst_bytes(read_binary_file(path));
}

struct ParsedSSTMetadataOnly {
    ParsedSST parsed;
    size_t    logical_size = 0;
    size_t    physical_size = 0;
};

static inline void pread_full_or_fail(int fd,
                                      void* data,
                                      size_t size,
                                      off_t offset,
                                      const std::string& path)
{
    uint8_t* dst = (uint8_t*)data;
    size_t total = 0;
    while (total < size) {
        ssize_t read_now = pread(fd, dst + total, size - total, offset + (off_t)total);
        if (read_now < 0) {
            gp_fail_if(true, "pread failed for '" + path + "': " + std::string(std::strerror(errno)));
        }
        gp_fail_if(read_now == 0, "Unexpected EOF during pread of '" + path + "'");
        total += (size_t)read_now;
    }
}

static inline ParsedSSTMetadataOnly read_sst_file_metadata_only(const std::string& path)
{
    ParsedSSTMetadataOnly meta;

    struct stat st{};
    gp_fail_if(stat(path.c_str(), &st) != 0,
               "Failed to stat '" + path + "': " + std::string(std::strerror(errno)));
    gp_fail_if(st.st_size < (off_t)sizeof(SSTFooter),
               "SST file is too small: '" + path + "'");

    meta.physical_size = (size_t)st.st_size;
    meta.logical_size = meta.physical_size;

    int fd = open(path.c_str(), O_RDONLY);
    gp_fail_if(fd < 0, "Failed to open '" + path + "': " + std::string(std::strerror(errno)));

    if (meta.physical_size >= sizeof(DirectIoTrailer)) {
        DirectIoTrailer trailer{};
        pread_full_or_fail(fd, &trailer, sizeof(trailer),
                           (off_t)(meta.physical_size - sizeof(DirectIoTrailer)), path);
        if (trailer.magic == GP_DIRECT_IO_TRAILER_MAGIC) {
            gp_fail_if(trailer.logical_size > meta.physical_size - sizeof(DirectIoTrailer),
                       "Corrupt direct-I/O trailer in '" + path + "'");
            meta.logical_size = (size_t)trailer.logical_size;
        }
    }

    gp_fail_if(meta.logical_size < sizeof(SSTFooter),
               "Logical SST size is too small: '" + path + "'");

    pread_full_or_fail(fd, &meta.parsed.footer, sizeof(SSTFooter),
                       (off_t)(meta.logical_size - sizeof(SSTFooter)), path);

    gp_fail_if(meta.parsed.footer.magic != GP_SST_MAGIC, "SST magic mismatch in '" + path + "'");
    gp_fail_if(meta.parsed.footer.version != GP_SST_VERSION, "SST version mismatch in '" + path + "'");
    gp_fail_if(meta.parsed.footer.key_bytes != GP_KEY_BYTES, "Unexpected key size in '" + path + "'");
    gp_fail_if(meta.parsed.footer.value_bytes != GP_VALUE_BYTES, "Unexpected value size in '" + path + "'");
    gp_fail_if(meta.parsed.footer.restart_interval != GP_RESTART_INTERVAL,
               "Unexpected restart interval in '" + path + "'");
    gp_fail_if(meta.parsed.footer.data_block_size != GP_DATA_BLOCK_BYTES,
               "Unexpected data block size in '" + path + "'");

    meta.parsed.data_blocks.resize(meta.parsed.footer.num_data_blocks);
    meta.parsed.filter_blocks.resize(meta.parsed.footer.num_data_blocks);

    if (!meta.parsed.data_blocks.empty()) {
        gp_fail_if(meta.parsed.footer.data_meta_offset
                       + meta.parsed.data_blocks.size() * sizeof(DataBlockMeta)
                       > meta.logical_size,
                   "Data block metadata out of bounds in '" + path + "'");
        gp_fail_if(meta.parsed.footer.filter_meta_offset
                       + meta.parsed.filter_blocks.size() * sizeof(FilterBlockMeta)
                       > meta.logical_size,
                   "Filter block metadata out of bounds in '" + path + "'");

        pread_full_or_fail(fd,
                           meta.parsed.data_blocks.data(),
                           meta.parsed.data_blocks.size() * sizeof(DataBlockMeta),
                           (off_t)meta.parsed.footer.data_meta_offset,
                           path);
        pread_full_or_fail(fd,
                           meta.parsed.filter_blocks.data(),
                           meta.parsed.filter_blocks.size() * sizeof(FilterBlockMeta),
                           (off_t)meta.parsed.footer.filter_meta_offset,
                           path);
    }

    close(fd);
    return meta;
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

struct StreamedSSTWriteResult {
    float       kernel_ms = 0.0f;
    float       h2d_ms = 0.0f;
    float       d2h_ms = 0.0f;
    size_t      h2d_bytes = 0;
    size_t      d2h_bytes = 0;
    double      write_ms = 0.0;
    double      output_window_ms = 0.0;
    double      output_active_ms = 0.0;
    double      output_idle_ms = 0.0;
    double      output_d2h_ms = 0.0;
    double      stream_overlap_pct = 0.0;
    size_t      output_bytes = 0;
    size_t      output_blocks = 0;
    size_t      output_files = 0;
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

struct StreamedSSTFilePlan {
    DeviceSSTFileLayout        layout{};
    std::vector<DeviceSSTBlockTask> tasks;
    size_t                     block_begin = 0;
    size_t                     block_end = 0;
};

static inline StreamedSSTFilePlan
build_streamed_sst_file_plan(const std::vector<DataBlockPlanEntry>& plans,
                             const std::vector<uint32_t>&           block_sizes,
                             const std::vector<uint32_t>&           filter_offsets,
                             const std::vector<uint32_t>&           filter_lengths,
                             const std::pair<size_t, size_t>&       span,
                             size_t                                 file_index,
                             uint64_t                               buffer_offset = 0)
{
    StreamedSSTFilePlan file_plan;
    file_plan.block_begin = span.first;
    file_plan.block_end = span.second;

    gp_fail_if(span.first >= span.second, "invalid SST block range");
    size_t block_count = span.second - span.first;
    size_t kv_begin = plans[span.first].first_kv;
    size_t kv_end = plans[span.second - 1].first_kv + plans[span.second - 1].num_kv;

    uint64_t data_region_size = 0;
    uint64_t filter_region_size = 0;
    for (size_t global = span.first; global < span.second; ++global) {
        data_region_size += block_sizes[global];
        filter_region_size += sizeof(FilterBlockHeader) + filter_lengths[global];
    }

    DeviceSSTFileLayout& layout = file_plan.layout;
    layout.buffer_offset = buffer_offset;
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

    file_plan.tasks.reserve(block_count);
    uint64_t running_data_offset = 0;
    uint64_t running_filter_offset = layout.filter_region_offset;
    for (size_t local = 0; local < block_count; ++local) {
        size_t global = span.first + local;
        DeviceSSTBlockTask task{};
        task.file_index = 0;
        task.local_block_index = (uint32_t)local;
        task.global_block_index = (uint32_t)global;
        task.data_dst_offset = running_data_offset;
        task.filter_dst_offset = running_filter_offset;
        task.block_size = block_sizes[global];
        task.filter_src_offset = filter_offsets[global];
        task.filter_length = filter_lengths[global];
        task.local_first_kv = (uint32_t)(plans[global].first_kv - kv_begin);
        task.num_kv = plans[global].num_kv;
        file_plan.tasks.push_back(task);

        running_data_offset += block_sizes[global];
        running_filter_offset += sizeof(FilterBlockHeader) + filter_lengths[global];
    }

    return file_plan;
}

static inline StreamedSSTWriteResult
assemble_and_write_sst_files_from_spans_streaming(const std::vector<DataBlockPlanEntry>&       plans,
                                                  const std::vector<uint32_t>&                  block_sizes,
                                                  const uint8_t*                                d_packed_blocks,
                                                  const Key128*                                 d_largest_keys,
                                                  const uint8_t*                                d_filter_bytes,
                                                  const std::vector<uint32_t>&                  filter_offsets,
                                                  const std::vector<uint32_t>&                  filter_lengths,
                                                  const std::vector<std::pair<size_t, size_t>>& spans,
                                                  const std::string&                            out_dir,
                                                  const std::string&                            prefix)
{
    StreamedSSTWriteResult result;
    if (spans.empty()) return result;

    std::vector<StreamedSSTFilePlan> file_plans;
    file_plans.reserve(spans.size());
    for (size_t file_idx = 0; file_idx < spans.size(); ++file_idx) {
        file_plans.push_back(build_streamed_sst_file_plan(
            plans, block_sizes, filter_offsets, filter_lengths, spans[file_idx], file_idx));
        result.output_bytes += (size_t)file_plans.back().layout.total_size;
        result.output_blocks += spans[file_idx].second - spans[file_idx].first;
    }
    result.output_files = file_plans.size();

    struct StreamSlot {
        uint8_t*            d_file = nullptr;
        size_t              d_capacity = 0;
        uint8_t*            h_file = nullptr;
        size_t              h_capacity = 0;
        DeviceSSTFileLayout* d_layout = nullptr;
        DeviceSSTBlockTask* d_tasks = nullptr;
        size_t              task_capacity = 0;
        cudaStream_t        stream = nullptr;
        cudaEvent_t         kernel_start = nullptr;
        cudaEvent_t         kernel_stop = nullptr;
        cudaEvent_t         d2h_start = nullptr;
        cudaEvent_t         d2h_stop = nullptr;
        bool                pending = false;
        bool                queued = false;
        bool                writing = false;
        bool                metrics_recorded = false;
        size_t              pending_file_index = 0;
        size_t              pending_size = 0;
        std::string         pending_path;
    };

    auto destroy_slot = [](StreamSlot& slot) {
        if (slot.d2h_stop) cudaEventDestroy(slot.d2h_stop);
        if (slot.d2h_start) cudaEventDestroy(slot.d2h_start);
        if (slot.kernel_stop) cudaEventDestroy(slot.kernel_stop);
        if (slot.kernel_start) cudaEventDestroy(slot.kernel_start);
        if (slot.stream) cudaStreamDestroy(slot.stream);
        if (slot.d_tasks) cudaFree(slot.d_tasks);
        if (slot.d_layout) cudaFree(slot.d_layout);
        if (slot.h_file) cudaFreeHost(slot.h_file);
        if (slot.d_file) cudaFree(slot.d_file);
        slot = StreamSlot{};
    };

    auto ensure_slot_capacity = [](StreamSlot& slot, size_t byte_capacity, size_t task_count) {
        if (!slot.stream) cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking);
        if (!slot.kernel_start) cudaEventCreate(&slot.kernel_start);
        if (!slot.kernel_stop) cudaEventCreate(&slot.kernel_stop);
        if (!slot.d2h_start) cudaEventCreate(&slot.d2h_start);
        if (!slot.d2h_stop) cudaEventCreate(&slot.d2h_stop);

        if (slot.d_capacity < byte_capacity) {
            if (slot.d_file) cudaFree(slot.d_file);
            cudaMalloc(&slot.d_file, (size_t)std::max(byte_capacity, (size_t)1));
            slot.d_capacity = byte_capacity;
        }
        if (slot.h_capacity < byte_capacity) {
            if (slot.h_file) cudaFreeHost(slot.h_file);
            cudaMallocHost(&slot.h_file, (size_t)std::max(byte_capacity, (size_t)1));
            slot.h_capacity = byte_capacity;
        }
        if (slot.task_capacity < task_count) {
            if (slot.d_tasks) cudaFree(slot.d_tasks);
            cudaMalloc(&slot.d_tasks, std::max(task_count, (size_t)1) * sizeof(DeviceSSTBlockTask));
            slot.task_capacity = task_count;
        }
        if (!slot.d_layout) cudaMalloc(&slot.d_layout, sizeof(DeviceSSTFileLayout));
    };

    static constexpr int kStreamSlotDepth = 3;
    StreamSlot slots[kStreamSlotDepth];
    std::mutex writer_mu;
    std::condition_variable writer_cv;
    std::condition_variable slot_cv;
    std::deque<int> write_queue;
    bool shutdown_writer = false;
    double write_ms_total = 0.0;

    std::thread writer_thread([&]() {
        for (;;) {
            int slot_idx = -1;
            {
                std::unique_lock<std::mutex> lock(writer_mu);
                writer_cv.wait(lock, [&]() { return shutdown_writer || !write_queue.empty(); });
                if (write_queue.empty()) {
                    if (shutdown_writer) break;
                    continue;
                }
                slot_idx = write_queue.front();
                write_queue.pop_front();
            }

            StreamSlot& slot = slots[slot_idx];
            auto write_start = std::chrono::steady_clock::now();
            write_binary_file_span(slot.pending_path, slot.h_file, slot.pending_size);
            auto write_end = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(writer_mu);
                write_ms_total += std::chrono::duration<double, std::milli>(write_end - write_start).count();
                slot.pending = false;
                slot.queued = false;
                slot.writing = false;
                slot.metrics_recorded = false;
                slot.pending_size = 0;
                slot.pending_path.clear();
            }
            slot_cv.notify_all();
        }
    });

    auto queue_slot_for_write = [&](int slot_idx, bool wait_for_completion) {
        StreamSlot& slot = slots[slot_idx];
        {
            std::lock_guard<std::mutex> lock(writer_mu);
            if (!slot.pending || slot.queued || slot.writing) return;
        }

        if (wait_for_completion) {
            cudaEventSynchronize(slot.d2h_stop);
        } else {
            cudaError_t status = cudaEventQuery(slot.d2h_stop);
            gp_fail_if(status != cudaSuccess && status != cudaErrorNotReady,
                       "cudaEventQuery failed while queueing streamed SST write");
            if (status == cudaErrorNotReady) return;
        }

        if (!slot.metrics_recorded) {
            float kernel_ms = 0.0f;
            float d2h_ms = 0.0f;
            cudaEventElapsedTime(&kernel_ms, slot.kernel_start, slot.kernel_stop);
            cudaEventElapsedTime(&d2h_ms, slot.d2h_start, slot.d2h_stop);
            result.kernel_ms += kernel_ms;
            result.d2h_ms += d2h_ms;
            result.output_d2h_ms += d2h_ms;
            result.d2h_bytes += slot.pending_size;
        }

        {
            std::lock_guard<std::mutex> lock(writer_mu);
            if (!slot.pending || slot.queued || slot.writing) return;
            slot.metrics_recorded = true;
            slot.queued = true;
            slot.writing = true;
            write_queue.push_back(slot_idx);
        }
        writer_cv.notify_one();
    };

    auto window_start = std::chrono::steady_clock::now();
    bool launched_any = false;

    for (size_t file_idx = 0; file_idx < file_plans.size(); ++file_idx) {
        const int slot_idx = (int)(file_idx % (size_t)kStreamSlotDepth);
        StreamSlot& slot = slots[slot_idx];
        queue_slot_for_write(slot_idx, true);
        {
            std::unique_lock<std::mutex> lock(writer_mu);
            slot_cv.wait(lock, [&]() { return !slot.pending && !slot.queued && !slot.writing; });
        }

        const StreamedSSTFilePlan& file_plan = file_plans[file_idx];
        ensure_slot_capacity(slot, (size_t)file_plan.layout.total_size, file_plan.tasks.size());

        auto h2d_start = std::chrono::steady_clock::now();
        cudaMemcpyAsync(slot.d_layout, &file_plan.layout, sizeof(DeviceSSTFileLayout),
                        cudaMemcpyHostToDevice, slot.stream);
        if (!file_plan.tasks.empty()) {
            cudaMemcpyAsync(slot.d_tasks, file_plan.tasks.data(),
                            file_plan.tasks.size() * sizeof(DeviceSSTBlockTask),
                            cudaMemcpyHostToDevice, slot.stream);
        }
        auto h2d_end = std::chrono::steady_clock::now();
        result.h2d_ms += (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
        result.h2d_bytes += sizeof(DeviceSSTFileLayout)
                          + file_plan.tasks.size() * sizeof(DeviceSSTBlockTask);

        if (!launched_any) {
            window_start = std::chrono::steady_clock::now();
            launched_any = true;
        }

        cudaEventRecord(slot.kernel_start, slot.stream);
        assemble_sst_blocks_kernel<<<(int)file_plan.tasks.size(), 256, 0, slot.stream>>>(
            slot.d_file, slot.d_layout, slot.d_tasks, (int)file_plan.tasks.size(),
            d_packed_blocks, d_filter_bytes, d_largest_keys);
        write_sst_footers_kernel<<<1, 1, 0, slot.stream>>>(slot.d_file, slot.d_layout, 1);
        cudaEventRecord(slot.kernel_stop, slot.stream);

        cudaEventRecord(slot.d2h_start, slot.stream);
        cudaMemcpyAsync(slot.h_file, slot.d_file, (size_t)file_plan.layout.total_size,
                        cudaMemcpyDeviceToHost, slot.stream);
        cudaEventRecord(slot.d2h_stop, slot.stream);

        char name[256];
        std::snprintf(name, sizeof(name), "%s_%04zu.sst", prefix.c_str(), file_idx);
        {
            std::lock_guard<std::mutex> lock(writer_mu);
            slot.pending = true;
            slot.queued = false;
            slot.writing = false;
            slot.metrics_recorded = false;
            slot.pending_file_index = file_idx;
            slot.pending_size = (size_t)file_plan.layout.total_size;
            slot.pending_path = out_dir + "/" + name;
        }

        for (int i = 0; i < kStreamSlotDepth; ++i) {
            if (i == slot_idx) continue;
            queue_slot_for_write(i, false);
        }
    }

    for (int i = 0; i < kStreamSlotDepth; ++i) queue_slot_for_write(i, true);
    {
        std::unique_lock<std::mutex> lock(writer_mu);
        shutdown_writer = true;
    }
    writer_cv.notify_one();
    writer_thread.join();
    result.write_ms = write_ms_total;
    auto window_end = std::chrono::steady_clock::now();

    result.output_window_ms = launched_any
        ? std::chrono::duration<double, std::milli>(window_end - window_start).count()
        : 0.0;
    result.output_active_ms = result.output_d2h_ms + result.write_ms;
    result.output_idle_ms = std::max(0.0, result.output_window_ms - result.output_active_ms);
    if (result.output_active_ms > 0.0 && result.output_window_ms > 0.0) {
        double overlap_ms = std::max(0.0, result.output_active_ms - result.output_window_ms);
        result.stream_overlap_pct = (overlap_ms / result.output_active_ms) * 100.0;
    }

    for (int i = 0; i < kStreamSlotDepth; ++i) destroy_slot(slots[i]);
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

static inline std::vector<KVRef> cpu_unpack_sst_refs(const ParsedSST& parsed, uint32_t source_sst)
{
    return cpu_unpack_all_refs(parsed.file_bytes,
                               block_offsets_from_parsed(parsed),
                               plans_from_parsed(parsed),
                               parsed.footer.total_kv,
                               source_sst);
}

static inline KVPair materialize_kv_from_ref(const ParsedSST& parsed, const KVRef& ref)
{
    KVPair kv{};
    kv.key = ref.key;
    gp_fail_if(ref.value_size != GP_VALUE_BYTES, "Unexpected KVRef value size during materialization");
    gp_fail_if((size_t)ref.value_offset + (size_t)ref.value_size > parsed.file_bytes.size(),
               "KVRef value offset out of bounds during materialization");
    std::memcpy(kv.value.bytes, parsed.file_bytes.data() + ref.value_offset, GP_VALUE_BYTES);
    return kv;
}
