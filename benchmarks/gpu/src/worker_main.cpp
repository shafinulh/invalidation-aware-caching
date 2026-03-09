/*
 * worker_main.cpp — RocksDB I/O and main logic for gpu_compaction_worker.
 *
 * Compiled by g++ (supports C++20 RocksDB headers).
 * GPU work is delegated to the extern "C" gpu_merge_worker() function
 * defined in worker_gpu.cu (compiled by nvcc).
 *
 * Usage:
 *   ./gpu_compaction_worker                          \
 *       --input_sst path1 [path2 ...]               \
 *       --output_dir /path/to/output/dir            \
 *       [--key_prefix  N]   (bytes before numeric key suffix, default 3)
 *       [--device      N]   (GPU device index, default 0)
 *       [--no_compression]  (disable SST compression, default on)
 *
 * On success prints one machine-readable metrics line:
 *   GPU_COMPACTION_WORKER_METRICS total_us=... sst_read_us=... \
 *       h2d_us=... merge_kernel_us=... d2h_us=... sst_write_us=... \
 *       input_files=N input_keys=N output_keys=N output_file=<path>
 */

#include <rocksdb/sst_file_reader.h>
#include <rocksdb/sst_file_writer.h>
#include <rocksdb/options.h>
#include <rocksdb/iterator.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/slice_transform.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* gpcomp_common.cuh defines the KVPair struct — include it as plain C++.
 * It has no CUDA-specific code at the struct definition level. */
#include "../../cuda_test/gpcomp_common.cuh"

/* Forward declaration of the GPU merge function (defined in worker_gpu.cu). */
extern "C" int gpu_merge_worker(KVPair** sst_arrays, const int* sst_sizes,
                                 int num_ssts, KVPair* output,
                                 long long total_count,
                                 double* h2d_us_out, double* kernel_us_out);

/* ---- Utilities ---------------------------------------------------------- */

static double now_us() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::micro>(
               clk::now().time_since_epoch()).count();
}

/*
 * make_sort_key — encode a user key into a uint64_t that preserves the
 * lexicographic ordering of the original key byte string.
 *
 * Layout (64 bits):
 *   bits 63 … KEY_SORT_SHIFT   : normalized user-key integer
 *   bits KEY_SORT_SHIFT-1 … 0  : (MAX_SSTS - sst_index) — newest SST → 0
 */
static const int KEY_SORT_SHIFT = 8;   /* supports up to 256 SSTs */
static const int MAX_SSTS       = 256;

static uint64_t make_sort_key(uint64_t user_key_int, int sst_index) {
    uint64_t age = (uint64_t)(MAX_SSTS - 1 - (sst_index & (MAX_SSTS - 1)));
    return (user_key_int << KEY_SORT_SHIFT) | age;
}

/* Extract the user-key integer from the sort key (strip sst-age bits). */
static uint64_t sort_key_to_user_int(uint64_t sort_key) {
    return sort_key >> KEY_SORT_SHIFT;
}

/* ---- Per-SST KV store --------------------------------------------------- */

struct KVEntry {
    std::string key;
    std::string value;
};

/* ---- main --------------------------------------------------------------- */

int main(int argc, char** argv) {

    std::vector<std::string> input_sst_paths;
    std::string output_dir;
    int key_prefix     = 3;   /* "key" prefix in "key%06d" */
    int gpu_device     = 0;
    bool no_compression = false;

    /* ---- argument parsing -------------------------------------------- */
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input_sst") {
            /* collect paths until next -- option */
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                input_sst_paths.push_back(argv[++i]);
            }
        } else if (arg == "--output_dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--key_prefix" && i + 1 < argc) {
            key_prefix = atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            gpu_device = atoi(argv[++i]);
        } else if (arg == "--no_compression") {
            no_compression = true;
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "usage: %s --input_sst p1 [p2 ...] --output_dir dir\n"
                "          [--key_prefix N] [--device N] [--no_compression]\n",
                argv[0]);
            return EXIT_SUCCESS;
        }
    }

    if (input_sst_paths.empty()) {
        fprintf(stderr, "error: --input_sst is required\n");
        return EXIT_FAILURE;
    }
    if (output_dir.empty()) {
        fprintf(stderr, "error: --output_dir is required\n");
        return EXIT_FAILURE;
    }

    const int num_ssts = static_cast<int>(input_sst_paths.size());

    /* ---- Step 1: SstFileReader — extract KV pairs -------------------- */

    const double t_read_start = now_us();

    ROCKSDB_NAMESPACE::Options options;
    options.create_if_missing = true;
    if (no_compression) {
        options.compression = rocksdb::kNoCompression;
    }

    /* per-SST KV storage (key and value strings kept on CPU for SstFileWriter) */
    std::vector<std::vector<KVEntry>> sst_entries(num_ssts);
    /* KVPair arrays for GPU (sort_key + encoded index) */
    std::vector<std::vector<KVPair>> sst_kv_pairs(num_ssts);

    size_t total_input_keys = 0;

    for (int s = 0; s < num_ssts; ++s) {
        ROCKSDB_NAMESPACE::SstFileReader reader(options);
        ROCKSDB_NAMESPACE::Status status = reader.Open(input_sst_paths[s]);
        if (!status.ok()) {
            fprintf(stderr, "SstFileReader::Open(%s): %s\n",
                    input_sst_paths[s].c_str(),
                    status.ToString().c_str());
            return EXIT_FAILURE;
        }

        ROCKSDB_NAMESPACE::ReadOptions ropts;
        ropts.total_order_seek = true;
        std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
            reader.NewIterator(ropts));

        if (!iter) {
            fprintf(stderr, "SstFileReader::NewIterator returned null for %s\n",
                    input_sst_paths[s].c_str());
            return EXIT_FAILURE;
        }

        uint32_t pos = 0;
        for (iter->SeekToFirst(); iter->Valid(); iter->Next(), ++pos) {
            /* Store full key/value strings for SstFileWriter later */
            KVEntry entry;
            entry.key   = iter->key().ToString();
            entry.value = iter->value().ToString();
            sst_entries[s].push_back(entry);

            /* Parse numeric sort key from key string (skip prefix bytes) */
            const char* key_data = entry.key.data();
            size_t      key_size = entry.key.size();
            uint64_t    user_int = 0;

            if (static_cast<int>(key_size) > key_prefix) {
                char* end_ptr = nullptr;
                user_int = strtoull(key_data + key_prefix, &end_ptr, 10);
            }

            /* Encode: sort_key encodes user-key order + SST freshness.
             * value   encodes (sst_index, position) for reverse lookup. */
            KVPair kv;
            kv.key   = make_sort_key(user_int, s);
            kv.value = ((uint64_t)(uint32_t)s << 32) | (uint64_t)(uint32_t)pos;
            sst_kv_pairs[s].push_back(kv);
            ++total_input_keys;
        }

        if (!iter->status().ok()) {
            fprintf(stderr, "iterator error on %s: %s\n",
                    input_sst_paths[s].c_str(),
                    iter->status().ToString().c_str());
            return EXIT_FAILURE;
        }
    }

    const double t_read_end = now_us();
    const double sst_read_us = t_read_end - t_read_start;

    /* ---- Step 2: GPU merge ------------------------------------------- */

    /* Build raw-pointer arrays expected by gpu_merge_worker */
    std::vector<KVPair*> h_sst_ptrs(num_ssts);
    std::vector<int>     h_sst_sizes(num_ssts);
    for (int s = 0; s < num_ssts; ++s) {
        h_sst_ptrs[s]  = sst_kv_pairs[s].data();
        h_sst_sizes[s] = static_cast<int>(sst_kv_pairs[s].size());
    }

    /* Allocate output buffer */
    std::vector<KVPair> h_output(total_input_keys);

    double h2d_us        = 0.0;
    double merge_kernel_us = 0.0;
    const double t_kernel_start = now_us();
    int rc = gpu_merge_worker(h_sst_ptrs.data(), h_sst_sizes.data(),
                              num_ssts, h_output.data(),
                              static_cast<long long>(total_input_keys),
                              &h2d_us, &merge_kernel_us);
    merge_kernel_us = now_us() - t_kernel_start;

    if (rc != 0) {
        fprintf(stderr, "gpu_merge_worker failed (rc=%d)\n", rc);
        return EXIT_FAILURE;
    }

    const double d2h_us = 0.0; /* accounted inside merge_kernel_us */

    /* ---- Step 3: CPU dedup — resolve same-user-key across SSTs -------- */

    std::vector<KVPair> deduped;
    deduped.reserve(total_input_keys);
    uint64_t prev_user_int = UINT64_MAX;

    for (const KVPair& kv : h_output) {
        uint64_t user_int = sort_key_to_user_int(kv.key);
        if (user_int != prev_user_int) {
            deduped.push_back(kv);
            prev_user_int = user_int;
        }
    }

    const size_t output_keys = deduped.size();

    /* ---- Step 4: SstFileWriter — produce output SST ------------------- */

    const double t_write_start = now_us();

    /* Create output directory if it doesn't exist */
    {
        ROCKSDB_NAMESPACE::Status s =
            options.env->CreateDirIfMissing(output_dir);
        if (!s.ok() && !s.IsNotFound()) {
            fprintf(stderr, "CreateDirIfMissing(%s): %s\n",
                    output_dir.c_str(), s.ToString().c_str());
            return EXIT_FAILURE;
        }
    }

    const std::string output_sst_path = output_dir + "/gpu_merged_000000.sst";

    ROCKSDB_NAMESPACE::EnvOptions env_options;
    ROCKSDB_NAMESPACE::SstFileWriter writer(env_options, options);

    ROCKSDB_NAMESPACE::Status ws = writer.Open(output_sst_path);
    if (!ws.ok()) {
        fprintf(stderr, "SstFileWriter::Open(%s): %s\n",
                output_sst_path.c_str(), ws.ToString().c_str());
        return EXIT_FAILURE;
    }

    for (const KVPair& kv : deduped) {
        uint32_t sst_idx = static_cast<uint32_t>(kv.value >> 32);
        uint32_t pos_idx = static_cast<uint32_t>(kv.value & 0xFFFFFFFF);

        if (sst_idx >= static_cast<uint32_t>(num_ssts) ||
            pos_idx >= sst_entries[sst_idx].size()) {
            fprintf(stderr,
                    "out-of-bounds index: sst_idx=%u pos=%u\n",
                    sst_idx, pos_idx);
            return EXIT_FAILURE;
        }

        const KVEntry& entry = sst_entries[sst_idx][pos_idx];
        ws = writer.Put(ROCKSDB_NAMESPACE::Slice(entry.key),
                        ROCKSDB_NAMESPACE::Slice(entry.value));
        if (!ws.ok()) {
            fprintf(stderr, "SstFileWriter::Put: %s\n",
                    ws.ToString().c_str());
            return EXIT_FAILURE;
        }
    }

    ROCKSDB_NAMESPACE::ExternalSstFileInfo sst_info;
    ws = writer.Finish(&sst_info);
    if (!ws.ok()) {
        fprintf(stderr, "SstFileWriter::Finish: %s\n",
                ws.ToString().c_str());
        return EXIT_FAILURE;
    }

    const double t_write_end   = now_us();
    const double sst_write_us  = t_write_end - t_write_start;
    const double total_us      = t_write_end - t_read_start;

    /* ---- Print metrics ----------------------------------------------- */

    printf(
        "GPU_COMPACTION_WORKER_METRICS "
        "total_us=%.1f "
        "sst_read_us=%.1f "
        "h2d_us=%.1f "
        "merge_kernel_us=%.1f "
        "d2h_us=%.1f "
        "sst_write_us=%.1f "
        "input_files=%d "
        "input_keys=%zu "
        "output_keys=%zu "
        "output_file=%s\n",
        total_us,
        sst_read_us,
        h2d_us,
        merge_kernel_us,
        d2h_us,
        sst_write_us,
        num_ssts,
        total_input_keys,
        output_keys,
        output_sst_path.c_str());

    (void)gpu_device; /* currently used by gpu_merge_worker, not cudaSetDevice() */
    return EXIT_SUCCESS;
}
