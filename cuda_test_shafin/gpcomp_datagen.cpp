#include "gpcomp_dummy_data.cuh"

#include <cerrno>
#include <string>
#include <sys/stat.h>

static uint32_t tune_entries_per_sst(uint64_t seed)
{
    uint32_t low = 1;
    uint32_t high = 2 * GP_TARGET_FILE_BYTES / (GP_KEY_BYTES + GP_VALUE_BYTES);
    if (high < 100) high = 100;
    while (build_cpu_sst(high, 0, seed).file_bytes.size() < GP_TARGET_FILE_BYTES) high *= 2;

    uint32_t best_entries = low;
    uint64_t best_diff = UINT64_MAX;
    while (low <= high) {
        uint32_t mid = low + (high - low) / 2;
        uint64_t size = build_cpu_sst(mid, 0, seed).file_bytes.size();
        uint64_t diff = (size > GP_TARGET_FILE_BYTES) ? (size - GP_TARGET_FILE_BYTES)
                                                      : (GP_TARGET_FILE_BYTES - size);
        if (diff < best_diff) {
            best_diff = diff;
            best_entries = mid;
        }
        if (size < GP_TARGET_FILE_BYTES) low = mid + 1;
        else if (mid == 0) break;
        else high = mid - 1;
    }
    return best_entries;
}

static void ensure_dir(const std::string& dir)
{
    struct stat st{};
    if (stat(dir.c_str(), &st) == 0) return;
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create output directory");
    }
}

int main(int argc, char** argv)
{
    std::string out_dir = "dataset";
    uint64_t seed = 23;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("missing CLI argument");
            return argv[++i];
        };
        if (arg == "--out_dir") out_dir = next();
        else if (arg == "--seed") seed = std::stoull(next());
        else if (arg == "--help") {
            std::printf("Usage: %s [--out_dir DIR] [--seed N]\n", argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    ensure_dir(out_dir);
    uint32_t entries_per_sst = tune_entries_per_sst(seed);
    std::printf("Generating %d dummy SSTs, target file size %d bytes, entries_per_sst=%u\n",
                GP_NUM_INPUT_SSTS, GP_TARGET_FILE_BYTES, entries_per_sst);

    FILE* meta = std::fopen((out_dir + "/dataset.meta").c_str(), "w");
    gp_fail_if(!meta, "failed to open dataset.meta");
    std::fprintf(meta, "num_sst=%d\n", GP_NUM_INPUT_SSTS);
    std::fprintf(meta, "entries_per_sst=%u\n", entries_per_sst);
    std::fprintf(meta, "key_bytes=%d\n", GP_KEY_BYTES);
    std::fprintf(meta, "value_bytes=%d\n", GP_VALUE_BYTES);
    std::fprintf(meta, "restart_interval=%d\n", GP_RESTART_INTERVAL);
    std::fprintf(meta, "data_block_bytes=%d\n", GP_DATA_BLOCK_BYTES);
    std::fprintf(meta, "target_file_bytes=%d\n", GP_TARGET_FILE_BYTES);
    std::fprintf(meta, "bloom_bits_per_key=%d\n", GP_BLOOM_BITS_PER_KEY);
    std::fprintf(meta, "bloom_hashes=%d\n", GP_BLOOM_HASHES);
    std::fprintf(meta, "seed=%llu\n", (unsigned long long)seed);

    for (uint32_t sst = 0; sst < GP_NUM_INPUT_SSTS; ++sst) {
        SSTBuildArtifacts build = build_cpu_sst(entries_per_sst, sst, seed);
        char path[256];
        std::snprintf(path, sizeof(path), "%s/sst_%04u.sst", out_dir.c_str(), sst);
        write_binary_file(path, build.file_bytes);
        std::printf("  %s  size=%zu bytes  blocks=%zu  kv=%u\n",
                    path, build.file_bytes.size(), build.data_meta.size(), entries_per_sst);
        std::fprintf(meta, "sst_%u_bytes=%zu\n", sst, build.file_bytes.size());
        std::fprintf(meta, "sst_%u_blocks=%zu\n", sst, build.data_meta.size());
    }
    std::fclose(meta);

    return 0;
}
