/*
 * gpcomp_bench.cu
 *
 * End-to-end kernel benchmark for the two GPComp algorithms using the
 * synthetic SST dataset produced by gpcomp_datagen.
 *
 * What is measured
 * ----------------
 *   Merge (Algorithm 1)
 *     Loads all input SST binary files and feeds them to launch_merge().
 *     Measures kernel-only time using CUDA events, and total time including
 *     host↔device transfers.  Validates the output against a CPU merge-sort.
 *
 *   Bloom filter (Algorithm 2)
 *     The bloom_filter_kernel runs as a single block and keeps the entire
 *     ByteVector in shared memory, so it operates *per SST data block* —
 *     matching RocksDB's BlockBasedTableBuilder which builds one filter per
 *     data block (BLOCK_SIZE = 32 KB in the GP-Comp paper settings).
 *
 *     With BLOCK_SIZE=32KB, KEY_SIZE=16B, VALUE_SIZE=64B, overhead~20B:
 *       keys_per_block ≈ 32768 / (16+64+20) ≈ 327
 *       byte_vector_len = 327 * 10        ≈ 3270  (fits in 48 KB shared mem)
 *       block_dim = ceil(3270/8) = 409    → 416 threads  (< 1024 max)
 *
 *     The benchmark splits the merged output into blocks of this size,
 *     re-uses a single pair of device buffers for all blocks, and accumulates
 *     kernel-only time across all blocks.  This is identical to what a GPU-
 *     accelerated BlockBasedTableBuilder would do during compaction output.
 *
 * Usage
 * -----
 *   ./gpcomp_bench [--dataset DIR] [--block_size BYTES] [--fpr_samples N]
 *
 *   --dataset     DIR    path to dataset directory    [default: ./dataset]
 *   --block_size  BYTES  SST data block size in bytes [default: 32768  = 32 KB]
 *   --key_size    BYTES  key size in bytes            [default: 16]
 *   --value_size  BYTES  value size in bytes          [default: 64]
 *   --overhead    BYTES  per-entry SST overhead bytes [default: 20]
 *   --fpr_samples N      non-member keys for FPR test [default: 10000]
 *   --help
 */

#include "gpcomp_merge.cuh"
#include "gpcomp_bloom.cuh"
#include "gpcomp_pack.cuh"
#include "gpcomp_cufile.cuh"
/* gpcomp_common.cuh is pulled in by the headers above */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

/* -------------------------------------------------------------------------
 * Inline xorshift64 – replaces std::mt19937_64 which may be absent on
 * older system libstdc++ versions paired with nvcc.
 * ---------------------------------------------------------------------- */
static inline uint64_t xorshift64(uint64_t& state)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

/* -------------------------------------------------------------------------
 * CUDA error-check helper
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* -------------------------------------------------------------------------
 * dataset.meta parser – returns value for a given key string
 * ---------------------------------------------------------------------- */
struct DatasetMeta {
    int      num_sst          = 0;
    int      keys_per_sst     = 0;
    uint64_t total_keys       = 0;
    uint64_t key_space        = 0;
    uint64_t seed             = 23;
    int      bloom_bits       = 10;
    int      bloom_K          = 7;
    uint64_t bloom_byte_vector_len = 0;       /* total – used only for display */
    std::vector<int> sst_sizes;               /* actual sizes after dedup */
};

static DatasetMeta load_meta(const std::string& meta_path)
{
    FILE* f = fopen(meta_path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s': %s\n",
                meta_path.c_str(), strerror(errno));
        exit(1);
    }

    DatasetMeta m;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char key[128]; char val[128];
        if (sscanf(line, "%127[^=]=%127s", key, val) != 2) continue;
        std::string k(key), v(val);
        if      (k == "num_sst")               m.num_sst              = std::stoi(v);
        else if (k == "keys_per_sst")          m.keys_per_sst         = std::stoi(v);
        else if (k == "total_keys")            m.total_keys           = std::stoull(v);
        else if (k == "key_space")             m.key_space            = std::stoull(v);
        else if (k == "seed")                  m.seed                 = std::stoull(v);
        else if (k == "bloom_bits")            m.bloom_bits           = std::stoi(v);
        else if (k == "bloom_K")               m.bloom_K              = std::stoi(v);
        else if (k == "bloom_byte_vector_len") m.bloom_byte_vector_len = std::stoull(v);
        else if (k.rfind("sst_", 0) == 0 && k.find("_size") != std::string::npos)
            m.sst_sizes.push_back(std::stoi(v));
    }
    fclose(f);

    if (m.num_sst == 0 || m.total_keys == 0) {
        fprintf(stderr, "error: dataset.meta is missing required fields\n");
        exit(1);
    }
    return m;
}

/* -------------------------------------------------------------------------
 * Load one SST binary file into a vector
 * ---------------------------------------------------------------------- */
static std::vector<KVPair> load_sst_bin(const std::string& path, int expected_size)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s': %s\n",
                path.c_str(), strerror(errno));
        exit(1);
    }
    std::vector<KVPair> v((size_t)expected_size);
    size_t n = fread(v.data(), sizeof(KVPair), (size_t)expected_size, f);
    fclose(f);
    if ((int)n != expected_size) {
        fprintf(stderr, "error: '%s': expected %d pairs, read %zu\n",
                path.c_str(), expected_size, n);
        exit(1);
    }
    return v;
}

/* -------------------------------------------------------------------------
 * CPU reference merge (concat + std::sort) – used for validation
 * ---------------------------------------------------------------------- */
static std::vector<KVPair>
cpu_merge_reference(const std::vector<std::vector<KVPair>>& ssts)
{
    std::vector<KVPair> out;
    for (const auto& s : ssts) {
        out.insert(out.end(), s.begin(), s.end());
    }
    std::sort(out.begin(), out.end(),
              [](const KVPair& a, const KVPair& b){ return a.key < b.key; });
    return out;
}

/* -------------------------------------------------------------------------
 * CUDA event timer helper
 * ---------------------------------------------------------------------- */
struct CudaTimer {
    cudaEvent_t start_, stop_;
    CudaTimer()  { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~CudaTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    float stop_ms() {                                       /* call after kernel */
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

/* -------------------------------------------------------------------------
 * parse_count – accept human-readable counts: 10M, 200M, 1B, 500K, etc.
 *
 * Strips optional trailing suffix (K/M/B, case-insensitive) and returns
 * the corresponding integer.  Plain integers ("10000000") still work.
 * ---------------------------------------------------------------------- */
static int64_t parse_count(const char* s)
{
    char* end = nullptr;
    double val = strtod(s, &end);
    if (end && (*end == 'k' || *end == 'K'))      val *= 1e3;
    else if (end && (*end == 'm' || *end == 'M')) val *= 1e6;
    else if (end && (*end == 'b' || *end == 'B')) val *= 1e9;
    return (int64_t)val;
}

/* -------------------------------------------------------------------------
 * Print a horizontal rule
 * ---------------------------------------------------------------------- */
static void hr() { printf("─────────────────────────────────────────────────────────\n"); }

/* -------------------------------------------------------------------------
 * RunStats – min / mean / sample-stddev over N repeated timed runs
 * ---------------------------------------------------------------------- */
struct RunStats {
    double min    = 0;
    double mean   = 0;
    double stddev = 0;
    static RunStats from(const std::vector<double>& v) {
        RunStats s;
        s.min  = *std::min_element(v.begin(), v.end());
        double sum = 0; for (double x : v) sum += x;
        s.mean = sum / (double)v.size();
        double var = 0;
        for (double x : v) var += (x - s.mean) * (x - s.mean);
        s.stddev = (v.size() > 1) ? std::sqrt(var / (double)(v.size() - 1)) : 0.0;
        return s;
    }
};

static void print_stat(const char* label, const RunStats& s, uint64_t keys)
{
    printf("  %-36s  min=%7.2f  mean=%7.2f ± %5.2f ms  (%7.1f M keys/s at min)\n",
           label, s.min, s.mean, s.stddev,
           (double)keys / s.min / 1e3);
}

/* -------------------------------------------------------------------------
 * Result structs – returned by each benchmark for the combined section
 * ---------------------------------------------------------------------- */
struct MergeResult {
    RunStats cpu;             /* std::sort                             */
    RunStats kernel;          /* GPU kernel-only                       */
    RunStats gpu_wall;        /* GPU H2D + kernel + D2H  (heap src)    */
    RunStats gpu_wall_pinned; /* GPU pinned async H2D + kernel + D2H   */
    RunStats h2d_pinned;      /* pinned H2D-only                       */
};

struct BloomResult {
    RunStats cpu;        /* CPU per-block bloom               */
    RunStats kernel;     /* GPU kernel-only (pre-transferred) */
    RunStats batched;    /* GPU 1×H2D + grid + 1×D2H          */
};

struct PackUnpackResult {
    RunStats cpu_pack;     /* CPU pack (serialise)               */
    RunStats cpu_unpack;   /* CPU unpack (parse)                 */
    RunStats gpu_pack_k;   /* GPU pack kernel-only               */
    RunStats gpu_pack_w;   /* GPU pack wall (H2D+k+D2H)         */
    RunStats gpu_unpack_k; /* GPU unpack kernel-only             */
    RunStats gpu_unpack_w; /* GPU unpack wall (H2D+k+D2H)       */
};

struct GDSMergeResult {
    RunStats gds_load;   /* disk→device via cuFile (compat/native) or pinned H2D */
    RunStats kernel;     /* GPU merge kernel-only (data already on device)        */
    RunStats wall;       /* gds_load + kernel + D2H                               */
    bool     validated;
};

/* =========================================================================
 * BENCHMARK 1 – Merge kernel
 * ======================================================================= */
static MergeResult bench_merge(const std::vector<std::vector<KVPair>>& ssts,
                               const DatasetMeta& meta,
                               double io_ms,
                               int runs)
{
    hr();
    printf("BENCHMARK 1 – Merge kernel (Algorithm 1)\n");
    hr();

    int num_sst   = (int)ssts.size();
    uint64_t total = meta.total_keys;

    /* Build pointer + size arrays for launch_merge */
    std::vector<const KVPair*> h_ptrs(num_sst);
    std::vector<int>           h_sizes(num_sst);
    for (int i = 0; i < num_sst; ++i) {
        h_ptrs[i]  = ssts[i].data();
        h_sizes[i] = (int)ssts[i].size();
    }

    std::vector<KVPair> gpu_output(total);

    /* ---- CPU sort: run `runs` times, collect stats ---- */
    printf("  Running CPU sort (%d runs) ...\n", runs); fflush(stdout);
    std::vector<KVPair> cpu_ref;
    std::vector<double> cpu_s(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        cpu_ref = cpu_merge_reference(ssts);
        cpu_s[r] = std::chrono::duration<double,std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    }
    RunStats cpu_stats = RunStats::from(cpu_s);

    /* ---- GPU: allocate device buffers once, time `runs` passes ---- */
    printf("  Running GPU merge (%d runs) ...\n", runs); fflush(stdout);

    /* Build prefix-sum offsets on host */
    std::vector<int> h_offsets_v(num_sst + 1);
    h_offsets_v[0] = 0;
    for (int i = 0; i < num_sst; ++i)
        h_offsets_v[i + 1] = h_offsets_v[i] + h_sizes[i];

    /* Allocate per-SST device buffers */
    std::vector<KVPair*> d_sst_v(num_sst);
    for (int i = 0; i < num_sst; ++i)
        CUDA_CHECK(cudaMalloc(&d_sst_v[i], (size_t)h_sizes[i] * sizeof(KVPair)));

    KVPair **d_sst_arrays;  int *d_sst_sizes_dev, *d_sst_offsets_dev;  KVPair *d_output;
    CUDA_CHECK(cudaMalloc(&d_sst_arrays,      (size_t)num_sst       * sizeof(KVPair*)));
    CUDA_CHECK(cudaMalloc(&d_sst_sizes_dev,   (size_t)num_sst       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sst_offsets_dev, (size_t)(num_sst + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output,          (size_t)total         * sizeof(KVPair)));
    CUDA_CHECK(cudaMemcpy(d_sst_arrays,      d_sst_v.data(),       (size_t)num_sst       * sizeof(KVPair*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sst_sizes_dev,   h_sizes.data(),       (size_t)num_sst       * sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sst_offsets_dev, h_offsets_v.data(),   (size_t)(num_sst + 1) * sizeof(int),     cudaMemcpyHostToDevice));

    const int MERGE_BLOCK = 256;
    int merge_grid = ((int)total + MERGE_BLOCK - 1) / MERGE_BLOCK;

    /* Warm-up: H2D + kernel to prime JIT/caches */
    for (int i = 0; i < num_sst; ++i)
        CUDA_CHECK(cudaMemcpy(d_sst_v[i], h_ptrs[i],
                              (size_t)h_sizes[i] * sizeof(KVPair), cudaMemcpyHostToDevice));
    merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_sst_arrays, d_sst_sizes_dev,
                                              d_sst_offsets_dev, num_sst, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- Multi-run timed passes ---- */
    std::vector<double> kernel_s(runs), wall_s(runs);
    for (int r = 0; r < runs; ++r) {
        auto wall_t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < num_sst; ++i)
            CUDA_CHECK(cudaMemcpy(d_sst_v[i], h_ptrs[i],
                                  (size_t)h_sizes[i] * sizeof(KVPair), cudaMemcpyHostToDevice));

        CudaTimer ktimer;
        ktimer.start();
        merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_sst_arrays, d_sst_sizes_dev,
                                                  d_sst_offsets_dev, num_sst, d_output);
        kernel_s[r] = (double)ktimer.stop_ms();

        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output,
                              (size_t)total * sizeof(KVPair), cudaMemcpyDeviceToHost));
        wall_s[r] = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now() - wall_t0).count();
    }
    RunStats kernel_stats = RunStats::from(kernel_s);
    RunStats wall_stats   = RunStats::from(wall_s);

    /* ---- Pinned + async H2D variant ----------------------------------------
     * Each SST is copied into a pinned host buffer (cudaMallocHost) and then
     * transferred to the device with cudaMemcpyAsync on a dedicated per-SST
     * stream.  All copies are dispatched before DeviceSynchronize so the GPU's
     * single H2D copy engine can chain them back-to-back with minimal overhead,
     * and the pinned src eliminates the driver's heap→staging bounce copy.
     * -------------------------------------------------------------------- */
    RunStats pinned_wall_stats, pinned_h2d_stats;
    {
        printf("  Running GPU merge with pinned+async H2D (%d runs) ...\n", runs);
        fflush(stdout);

        /* Pinned host mirrors, one per SST */
        std::vector<KVPair*> h_pin(num_sst, nullptr);
        for (int i = 0; i < num_sst; ++i) {
            CUDA_CHECK(cudaMallocHost(&h_pin[i], (size_t)h_sizes[i] * sizeof(KVPair)));
            memcpy(h_pin[i], h_ptrs[i], (size_t)h_sizes[i] * sizeof(KVPair));
        }

        /* New device buffers – reuse existing d_sst_sizes_dev / d_sst_offsets_dev */
        std::vector<KVPair*> d_pin(num_sst);
        for (int i = 0; i < num_sst; ++i)
            CUDA_CHECK(cudaMalloc(&d_pin[i], (size_t)h_sizes[i] * sizeof(KVPair)));
        KVPair **d_pin_arr;
        CUDA_CHECK(cudaMalloc(&d_pin_arr, (size_t)num_sst * sizeof(KVPair*)));
        CUDA_CHECK(cudaMemcpy(d_pin_arr, d_pin.data(),
                              (size_t)num_sst * sizeof(KVPair*), cudaMemcpyHostToDevice));
        KVPair *d_out2;
        CUDA_CHECK(cudaMalloc(&d_out2, (size_t)total * sizeof(KVPair)));

        /* One H2D stream per SST so copies are dispatched in parallel */
        std::vector<cudaStream_t> h2d_streams(num_sst);
        for (int i = 0; i < num_sst; ++i)
            CUDA_CHECK(cudaStreamCreate(&h2d_streams[i]));

        /* Warm-up */
        for (int i = 0; i < num_sst; ++i)
            CUDA_CHECK(cudaMemcpyAsync(d_pin[i], h_pin[i],
                                       (size_t)h_sizes[i] * sizeof(KVPair),
                                       cudaMemcpyHostToDevice, h2d_streams[i]));
        CUDA_CHECK(cudaDeviceSynchronize());
        merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_pin_arr, d_sst_sizes_dev,
                                                  d_sst_offsets_dev, num_sst, d_out2);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> pw_s(runs), ph2d_s(runs);
        for (int r = 0; r < runs; ++r) {
            auto t0 = std::chrono::steady_clock::now();

            /* Async H2D on parallel streams – single DeviceSync waits for all */
            auto h2d_t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < num_sst; ++i)
                CUDA_CHECK(cudaMemcpyAsync(d_pin[i], h_pin[i],
                                           (size_t)h_sizes[i] * sizeof(KVPair),
                                           cudaMemcpyHostToDevice, h2d_streams[i]));
            CUDA_CHECK(cudaDeviceSynchronize());
            ph2d_s[r] = std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now() - h2d_t0).count();

            CudaTimer ktimer; ktimer.start();
            merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_pin_arr, d_sst_sizes_dev,
                                                      d_sst_offsets_dev, num_sst, d_out2);
            (void)ktimer.stop_ms();

            CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_out2,
                                  (size_t)total * sizeof(KVPair),
                                  cudaMemcpyDeviceToHost));
            pw_s[r] = std::chrono::duration<double,std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
        }
        pinned_wall_stats = RunStats::from(pw_s);
        pinned_h2d_stats  = RunStats::from(ph2d_s);

        for (int i = 0; i < num_sst; ++i) {
            cudaFreeHost(h_pin[i]);
            cudaFree(d_pin[i]);
            cudaStreamDestroy(h2d_streams[i]);
        }
        cudaFree(d_pin_arr);
        cudaFree(d_out2);
    }

    /* Cleanup */
    for (int i = 0; i < num_sst; ++i) cudaFree(d_sst_v[i]);
    cudaFree(d_sst_arrays);  cudaFree(d_sst_sizes_dev);
    cudaFree(d_sst_offsets_dev);  cudaFree(d_output);

    /* ---- Validate ---- */
    bool ok = true;
    if (gpu_output.size() != cpu_ref.size()) {
        ok = false;
        fprintf(stderr, "  FAIL: output size mismatch (%zu vs %zu)\n",
                gpu_output.size(), cpu_ref.size());
    } else {
        for (size_t i = 0; i < gpu_output.size(); ++i) {
            if (gpu_output[i].key   != cpu_ref[i].key ||
                gpu_output[i].value != cpu_ref[i].value) {
                fprintf(stderr,
                    "  FAIL: mismatch at index %zu: "
                    "GPU {key=%llu, val=%llu} vs CPU {key=%llu, val=%llu}\n",
                    i,
                    (unsigned long long)gpu_output[i].key,
                    (unsigned long long)gpu_output[i].value,
                    (unsigned long long)cpu_ref[i].key,
                    (unsigned long long)cpu_ref[i].value);
                ok = false;
                break;
            }
        }
    }

    /* ---- Print results ---- */
    double data_gb      = (double)total * sizeof(KVPair) / (1 << 30);
    double cpu_total_min = io_ms + cpu_stats.min;
    printf("\n");
    printf("  Input:   %d SSTs × %d keys = %llu total  (%.1f MB)   runs=%d\n",
           num_sst, (int)ssts[0].size(),
           (unsigned long long)total,
           (double)total * sizeof(KVPair) / (1 << 20), runs);
    printf("  CPU I/O (disk read):            %.2f ms\n", io_ms);
    print_stat("CPU sort",           cpu_stats,    total);
    print_stat("GPU kernel-only",    kernel_stats, total);
    print_stat("GPU wall (H2D+k+D2H)",wall_stats,  total);
    print_stat("GPU wall (pinned async H2D+k+D2H)", pinned_wall_stats, total);
    printf("  (GPU kernel BW at min: %.2f GB/s)\n", data_gb / (kernel_stats.min / 1e3));
    printf("  I/O breakdown (min):  disk %.2f ms  |  H2D(pinned) %.2f ms  |"
           "  kernel %.2f ms  |  D2H ~%.2f ms\n",
           io_ms, pinned_h2d_stats.min, kernel_stats.min,
           pinned_wall_stats.min - pinned_h2d_stats.min - kernel_stats.min);
    printf("  Wall savings (pinned vs heap H2D):   %.2f ms  (%.2f×)\n",
           wall_stats.min - pinned_wall_stats.min,
           wall_stats.min / pinned_wall_stats.min);
    printf("  Speedup kernel vs CPU sort (min):  %.2f×\n",
           cpu_stats.min / kernel_stats.min);
    printf("  Speedup wall   vs CPU+I/O  (min):  %.2f×\n",
           cpu_total_min / wall_stats.min);
    printf("  Speedup pinned vs CPU+I/O  (min):  %.2f×\n",
           cpu_total_min / pinned_wall_stats.min);
    printf("  Validation:             %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    MergeResult r;
    r.cpu              = cpu_stats;
    r.kernel           = kernel_stats;
    r.gpu_wall         = wall_stats;
    r.gpu_wall_pinned  = pinned_wall_stats;
    r.h2d_pinned       = pinned_h2d_stats;
    return r;
}

/* =========================================================================
 * BENCHMARK 2 – Bloom filter kernel (per-block mode)
 * ======================================================================= */
static BloomResult bench_bloom(const std::vector<KVPair>& merged,
                               const DatasetMeta& meta,
                               int block_size_bytes,
                               int key_size_bytes,
                               int value_size_bytes,
                               int overhead_bytes,
                               int fpr_samples,
                               int runs)
{
    hr();
    printf("BENCHMARK 2 – Bloom filter kernel (Algorithm 2, per-block mode)\n");
    hr();

    int bytes_per_entry = key_size_bytes + value_size_bytes + overhead_bytes;
    int keys_per_block  = block_size_bytes / bytes_per_entry;
    if (keys_per_block <= 0) keys_per_block = 1;

    int K               = meta.bloom_K;
    int bloom_bits      = meta.bloom_bits;

    /* byte_vector_len per block = keys_per_block × bloom_bits (bits = bytes here) */
    int byte_vector_len = keys_per_block * bloom_bits;
    int bitvec_len      = (byte_vector_len + 7) / 8;

    /* thread count = max(keys_per_block, bitvec_len) rounded to warp */
    int block_dim = (keys_per_block > bitvec_len ? keys_per_block : bitvec_len);
    block_dim = ((block_dim + 31) / 32) * 32;

    uint64_t total_keys = (uint64_t)merged.size();
    int      num_blocks = (int)((total_keys + keys_per_block - 1) / keys_per_block);

    printf("  Block size:         %d bytes  (BLOCK_SIZE from GP-Comp paper)\n",
           block_size_bytes);
    printf("  Keys/block:         %d  (%d+%d+%d B: key+value+overhead)\n",
           keys_per_block, key_size_bytes, value_size_bytes, overhead_bytes);
    printf("  Bloom params:       K=%d  bits/key=%d  byte_vector_len=%d\n",
           K, bloom_bits, byte_vector_len);
    printf("  Launch config:      <<<1, %d>>>  shared=%d B\n",
           block_dim, byte_vector_len);
    printf("  Total blocks:       %d  (%llu keys total)\n",
           num_blocks, (unsigned long long)total_keys);

    /* Check shared memory */
    {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        if ((size_t)byte_vector_len > prop.sharedMemPerBlock ||
            block_dim > prop.maxThreadsPerBlock) {
            fprintf(stderr,
                "\n  ERROR: bloom config exceeds device limits:\n"
                "         byte_vector_len=%d B  (device max shared=%zu B)\n"
                "         block_dim=%d threads  (device max=%d)\n"
                "  Try --block_size with a smaller value.\n",
                byte_vector_len, prop.sharedMemPerBlock,
                block_dim, prop.maxThreadsPerBlock);
            return BloomResult{};
        }
        printf("  Device sharedMem:   %zu KB  (%s)\n",
               prop.sharedMemPerBlock / 1024,
               (size_t)byte_vector_len <= prop.sharedMemPerBlock ? "fits ✓" : "OVERFLOW ✗");
    }
    printf("\n");

    size_t shared_bytes = (size_t)byte_vector_len * sizeof(uint8_t);

    /* ---- Pre-transfer all keys once; warm-up kernel-only pass ---- */
    KVPair  *d_all_keys;
    uint8_t *d_bitvec;
    CUDA_CHECK(cudaMalloc(&d_all_keys, total_keys * sizeof(KVPair)));
    CUDA_CHECK(cudaMalloc(&d_bitvec,   (size_t)bitvec_len * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_all_keys, merged.data(),
                          total_keys * sizeof(KVPair), cudaMemcpyHostToDevice));
    {
        int block_keys = std::min(keys_per_block, (int)total_keys);
        bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
            d_all_keys, block_keys, K, byte_vector_len, d_bitvec);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---- Kernel-only: `runs` timed passes (CUDA events, data already on device) ---- */
    printf("  Running GPU bloom kernel-only (%d runs) ...\n", runs); fflush(stdout);
    std::vector<double> ko_s(runs);
    for (int r = 0; r < runs; ++r) {
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));
        for (int b = 0; b < num_blocks; ++b) {
            int offset     = b * keys_per_block;
            int block_keys = std::min(keys_per_block, (int)(total_keys - offset));
            bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
                d_all_keys + offset, block_keys, K, byte_vector_len, d_bitvec);
        }
        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        ko_s[r] = (double)ms;
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    RunStats kernel_stats = RunStats::from(ko_s);
    cudaFree(d_all_keys);
    cudaFree(d_bitvec);

    /* ---- CPU bloom: warm-up then `runs` timed passes ---- */
    printf("  Running CPU bloom (%d runs) ...\n", runs); fflush(stdout);
    std::vector<uint8_t> cpu_bv_tmp(byte_vector_len);
    std::vector<uint8_t> cpu_packed_tmp(bitvec_len);
    {
        cpu_build_byte_vector(merged.data(), std::min(keys_per_block, (int)total_keys),
                              K, byte_vector_len, cpu_bv_tmp.data());
        cpu_pack_bit_vector(cpu_bv_tmp.data(), byte_vector_len, cpu_packed_tmp.data());
    }
    std::vector<double> cpu_s(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < num_blocks; ++b) {
            int offset     = b * keys_per_block;
            int block_keys = std::min(keys_per_block, (int)(total_keys - offset));
            cpu_build_byte_vector(merged.data() + offset, block_keys,
                                  K, byte_vector_len, cpu_bv_tmp.data());
            cpu_pack_bit_vector(cpu_bv_tmp.data(), byte_vector_len, cpu_packed_tmp.data());
        }
        cpu_s[r] = std::chrono::duration<double,std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    }
    RunStats cpu_stats = RunStats::from(cpu_s);

    /* ---- Single serial pass to collect per-block bitvecs for validation ---- */
    printf("  Building per-block bitvecs for validation ...\n"); fflush(stdout);
    std::vector<std::vector<uint8_t>> all_bitvecs(num_blocks,
                                                   std::vector<uint8_t>(bitvec_len, 0));
    {
        KVPair *d_block;
        CUDA_CHECK(cudaMalloc(&d_block,  (size_t)keys_per_block * sizeof(KVPair)));
        CUDA_CHECK(cudaMalloc(&d_bitvec, (size_t)bitvec_len     * sizeof(uint8_t)));
        for (int b = 0; b < num_blocks; ++b) {
            int offset     = b * keys_per_block;
            int block_keys = std::min(keys_per_block, (int)(total_keys - offset));
            CUDA_CHECK(cudaMemcpy(d_block, merged.data() + offset,
                                  block_keys * sizeof(KVPair), cudaMemcpyHostToDevice));
            bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
                d_block, block_keys, K, byte_vector_len, d_bitvec);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(all_bitvecs[b].data(), d_bitvec,
                                  bitvec_len * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        }
        cudaFree(d_block);
        cudaFree(d_bitvec);
    }

    /* ---- Batched wall pass: `runs` timed passes ---- */
    printf("  Running GPU bloom batched wall (%d runs) ...\n", runs); fflush(stdout);
    KVPair  *d_all_keys_b;
    uint8_t *d_all_bitvecs_b;
    CUDA_CHECK(cudaMalloc(&d_all_keys_b,    total_keys                      * sizeof(KVPair)));
    CUDA_CHECK(cudaMalloc(&d_all_bitvecs_b, (size_t)num_blocks * bitvec_len * sizeof(uint8_t)));
    /* warm-up */
    CUDA_CHECK(cudaMemcpy(d_all_keys_b, merged.data(),
                          total_keys * sizeof(KVPair), cudaMemcpyHostToDevice));
    bloom_filter_kernel_batched<<<num_blocks, block_dim, shared_bytes>>>(
        d_all_keys_b, keys_per_block, (int)total_keys, K, byte_vector_len, d_all_bitvecs_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> batched_flat((size_t)num_blocks * bitvec_len);
    std::vector<double> batched_s(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(d_all_keys_b, merged.data(),
                              total_keys * sizeof(KVPair), cudaMemcpyHostToDevice));
        bloom_filter_kernel_batched<<<num_blocks, block_dim, shared_bytes>>>(
            d_all_keys_b, keys_per_block, (int)total_keys, K, byte_vector_len, d_all_bitvecs_b);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(batched_flat.data(), d_all_bitvecs_b,
                              (size_t)num_blocks * bitvec_len * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));
        batched_s[r] = std::chrono::duration<double,std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
    }
    RunStats batched_stats = RunStats::from(batched_s);
    cudaFree(d_all_keys_b);
    cudaFree(d_all_bitvecs_b);

    /* Cross-validate batched vs per-block output */
    bool batched_ok = true;
    for (int b = 0; b < num_blocks && batched_ok; ++b) {
        const uint8_t *batch_bv = batched_flat.data() + (size_t)b * bitvec_len;
        for (int i = 0; i < bitvec_len && batched_ok; ++i) {
            if (batch_bv[i] != all_bitvecs[b][i]) {
                fprintf(stderr,
                    "  BATCHED MISMATCH: block %d byte %d: batched=0x%02x per-block=0x%02x\n",
                    b, i, batch_bv[i], all_bitvecs[b][i]);
                batched_ok = false;
            }
        }
    }

    /* ---- Validate: no false negatives ---- */
    printf("  Validating (no false negatives) ...\n"); fflush(stdout);
    bool no_fn = true;
    for (int b = 0; b < num_blocks && no_fn; ++b) {
        int offset     = b * keys_per_block;
        int block_keys = std::min(keys_per_block, (int)(total_keys - offset));
        for (int i = 0; i < block_keys && no_fn; ++i) {
            uint64_t key = merged[(size_t)(offset + i)].key;
            /* Check all K hash positions */
            bool found = true;
            for (int k = 1; k <= K; ++k) {
                /* cpu_bloom_hash from gpcomp_common.cuh */
                uint32_t h   = cpu_bloom_hash(key, k);
                int      pos = (int)(h % (uint32_t)byte_vector_len);
                int      byte_idx = pos / 8;
                int      bit_idx  = pos % 8;
                if (!(all_bitvecs[b][byte_idx] & (uint8_t)(1 << bit_idx))) {
                    found = false;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "  FALSE NEGATIVE: key %llu missing in block %d\n",
                        (unsigned long long)key, b);
                no_fn = false;
            }
        }
    }

    /* ---- Validate: false positive rate ---- */
    printf("  Estimating false positive rate (%d non-member samples per block) ...\n",
           fpr_samples); fflush(stdout);

    /* Build a set of all keys in the merged output for membership tests */
    std::unordered_set<uint64_t> key_set;
    key_set.reserve(merged.size() * 2);
    for (const auto& kv : merged) key_set.insert(kv.key);

    /* Sample non-members from [key_space, key_space * 2) — guaranteed outside dataset */
    uint64_t rng_state = meta.seed ^ 0xabcdef0123456789ULL;
    uint64_t nm_range  = meta.key_space;   /* offset into [key_space, 2*key_space) */

    long long fp_count = 0;
    long long total_samples = 0;
    for (int b = 0; b < num_blocks; ++b) {
        for (int s = 0; s < fpr_samples; ++s) {
            uint64_t k = meta.key_space + (xorshift64(rng_state) % nm_range);
            /* Check all K hash positions in block b's filter */
            bool passes = true;
            for (int ki = 1; ki <= K; ++ki) {
                uint32_t h       = cpu_bloom_hash(k, ki);
                int      pos     = (int)(h % (uint32_t)byte_vector_len);
                int      byte_i  = pos / 8;
                int      bit_i   = pos % 8;
                if (!(all_bitvecs[b][byte_i] & (uint8_t)(1 << bit_i))) {
                    passes = false;
                    break;
                }
            }
            if (passes) ++fp_count;
            ++total_samples;
        }
    }
    double fpr = (double)fp_count / (double)total_samples;
    /* Theoretical FPR: (1 - e^{-K/m})^K where m = bits/key */
    double m = (double)bloom_bits;
    double k_d = (double)K;
    double p_theoretical = pow(1.0 - exp(-k_d / m), k_d);

    /* ---- Print results ---- */
    printf("\n");
    printf("  Bloom config: %d blocks / %d keys/block / K=%d / bvlen=%d   runs=%d\n",
           num_blocks, keys_per_block, K, byte_vector_len, runs);
    print_stat("CPU bloom (per-block)",      cpu_stats,    total_keys);
    print_stat("GPU kernel-only (no xfer)",  kernel_stats, total_keys);
    print_stat("GPU batched wall (1×H2D+grid+1×D2H)", batched_stats, total_keys);
    printf("  Speedup kernel    vs CPU (min): %.2f×\n",
           cpu_stats.min / kernel_stats.min);
    printf("  Speedup batched   vs CPU (min): %.2f×\n",
           cpu_stats.min / batched_stats.min);
    printf("  No false negatives:  %s\n", no_fn      ? "PASS ✓" : "FAIL ✗");
    printf("  Batched vs serial match:  %s\n", batched_ok ? "PASS ✓" : "FAIL ✗");
    printf("  FPR (measured):               %.4f%%\n", fpr * 100.0);
    printf("  FPR (theoretical):            %.4f%%  "
           "[K=%d, bits/key=%d, p=(1-e^{-K/m})^K]\n",
           p_theoretical * 100.0, K, bloom_bits);

    BloomResult r;
    r.cpu     = cpu_stats;
    r.kernel  = kernel_stats;
    r.batched = batched_stats;
    return r;
}

/* =========================================================================
 * BENCHMARK 4 – GDS I/O + Merge kernel  (disk → device → merged output)
 *
 * Loads all SST binary files directly into device memory using the cuFile
 * abstraction layer (gpcomp_cufile.cuh), then runs the merge kernel with
 * the data already on the device – eliminating the separate H2D transfer.
 *
 *   GDS_COMPAT (RTX 3070): cuFile uses a pinned bounce-buffer in the driver;
 *              avoids the OS page-cache copy-in that plain fread incurs.
 *   GDS_NATIVE (A30):      cuFileRead issues a PCIe DMA directly to VRAM,
 *              so the load is the only I/O operation and H2D → 0 ns.
 *   GDS_DISABLED:          falls back to pinned fread + cudaMemcpy (same as
 *              compat but without the cuFile driver overhead).
 *
 * This benchmark reports:
 *   GDS I/O   – disk-to-device load time (replaces POSIX I/O + H2D)
 *   kernel    – GPU merge kernel-only (CUDA events, data pre-loaded)
 *   wall      – GDS I/O + kernel + D2H (end-to-end per-run)
 *
 * Comparison against Benchmark 1:
 *   Traditional path: io_ms (POSIX host load) + mr.gpu_wall (H2D+k+D2H)
 *   GDS path:         result.wall             (load+k+D2H)
 * ======================================================================= */
static GDSMergeResult bench_gds_io_merge(
        const std::string&              dataset_dir,
        const DatasetMeta&              meta,
        const std::vector<KVPair>&      cpu_reference,  /* validation oracle */
        const MergeResult&              mr_posix,       /* Benchmark 1 stats */
        double                          posix_io_ms,    /* measured disk I/O  */
        int                             runs,
        GDSMode                         mode)
{
    hr();
    printf("BENCHMARK 4 – GDS I/O + Merge  [%s]\n", gds_mode_name(mode));
    hr();

    int      num_sst = meta.num_sst;
    uint64_t total   = meta.total_keys;

    /* Build size / offset tables */
    std::vector<int> h_sizes(num_sst);
    for (int i = 0; i < num_sst; ++i) h_sizes[i] = meta.sst_sizes[i];

    std::vector<int> h_offsets(num_sst + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < num_sst; ++i)
        h_offsets[i + 1] = h_offsets[i] + h_sizes[i];

    /* Persistent device buffers for all SST arrays */
    std::vector<KVPair*> d_sst_v(num_sst);
    for (int i = 0; i < num_sst; ++i)
        CUDA_CHECK(cudaMalloc(&d_sst_v[i],
                              (size_t)h_sizes[i] * sizeof(KVPair)));

    KVPair **d_sst_arrays;
    int     *d_sst_sizes_dev, *d_sst_offsets_dev;
    KVPair  *d_output;
    CUDA_CHECK(cudaMalloc(&d_sst_arrays,
                          (size_t)num_sst       * sizeof(KVPair*)));
    CUDA_CHECK(cudaMalloc(&d_sst_sizes_dev,
                          (size_t)num_sst       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sst_offsets_dev,
                          (size_t)(num_sst + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output,
                          (size_t)total         * sizeof(KVPair)));

    CUDA_CHECK(cudaMemcpy(d_sst_arrays,      d_sst_v.data(),
               (size_t)num_sst       * sizeof(KVPair*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sst_sizes_dev,   h_sizes.data(),
               (size_t)num_sst       * sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sst_offsets_dev, h_offsets.data(),
               (size_t)(num_sst + 1) * sizeof(int),     cudaMemcpyHostToDevice));

    const int MERGE_BLOCK = 256;
    int merge_grid = ((int)total + MERGE_BLOCK - 1) / MERGE_BLOCK;

    /* Build SST file path list */
    std::vector<std::string> sst_paths(num_sst);
    for (int i = 0; i < num_sst; ++i) {
        char fname[64];
        snprintf(fname, sizeof(fname), "sst_%04d.bin", i);
        sst_paths[i] = dataset_dir + "/" + fname;
    }

    /* Warm-up: one full GDS load + kernel pass, not timed */
    printf("  Warm-up (1 GDS load + kernel pass) ...\n"); fflush(stdout);
    for (int i = 0; i < num_sst; ++i)
        gpcomp_gds_load_to_device(sst_paths[i].c_str(),
                                  d_sst_v[i],
                                  (size_t)h_sizes[i] * sizeof(KVPair));
    merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_sst_arrays, d_sst_sizes_dev,
                                              d_sst_offsets_dev, num_sst, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    printf("  Running GDS I/O + merge (%d runs) ...\n", runs); fflush(stdout);
    std::vector<double> load_s(runs), kernel_s(runs), wall_s(runs);
    std::vector<KVPair> gpu_output(total);

    for (int r = 0; r < runs; ++r) {
        auto wall_t0 = std::chrono::steady_clock::now();

        /* GDS load: disk → device (replaces POSIX I/O + cudaMemcpy H2D) */
        auto load_t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < num_sst; ++i) {
            if (gpcomp_gds_load_to_device(sst_paths[i].c_str(),
                                          d_sst_v[i],
                                          (size_t)h_sizes[i] * sizeof(KVPair)) != 0) {
                fprintf(stderr, "  GDS load failed for %s\n",
                        sst_paths[i].c_str());
            }
        }
        load_s[r] = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - load_t0).count();

        /* Merge kernel – data is already on device */
        CudaTimer ktimer;
        ktimer.start();
        merge_kernel<<<merge_grid, MERGE_BLOCK>>>(d_sst_arrays, d_sst_sizes_dev,
                                                  d_sst_offsets_dev, num_sst,
                                                  d_output);
        kernel_s[r] = (double)ktimer.stop_ms();

        /* D2H for output */
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output,
                              (size_t)total * sizeof(KVPair),
                              cudaMemcpyDeviceToHost));

        wall_s[r] = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wall_t0).count();
    }

    /* Cleanup device buffers */
    for (int i = 0; i < num_sst; ++i) cudaFree(d_sst_v[i]);
    cudaFree(d_sst_arrays);
    cudaFree(d_sst_sizes_dev);
    cudaFree(d_sst_offsets_dev);
    cudaFree(d_output);

    /* Validate against CPU oracle */
    bool ok = (gpu_output.size() == cpu_reference.size());
    if (ok) {
        for (size_t i = 0; i < gpu_output.size(); ++i) {
            if (gpu_output[i].key   != cpu_reference[i].key ||
                gpu_output[i].value != cpu_reference[i].value) {
                fprintf(stderr,
                    "  GDS FAIL: mismatch at %zu: "
                    "GPU {key=%llu, val=%llu} vs CPU {key=%llu, val=%llu}\n",
                    i,
                    (unsigned long long)gpu_output[i].key,
                    (unsigned long long)gpu_output[i].value,
                    (unsigned long long)cpu_reference[i].key,
                    (unsigned long long)cpu_reference[i].value);
                ok = false;
                break;
            }
        }
    }

    RunStats load_stats   = RunStats::from(load_s);
    RunStats kernel_stats = RunStats::from(kernel_s);
    RunStats wall_stats   = RunStats::from(wall_s);

    double data_mb = (double)total * sizeof(KVPair) / (1 << 20);

    printf("\n");
    printf("  Input:  %d SSTs × %d keys = %llu total  (%.1f MB)   runs=%d\n",
           num_sst, num_sst ? (int)(total / num_sst) : 0,
           (unsigned long long)total, data_mb, runs);
    print_stat("GDS I/O  (disk→device)",         load_stats,   total);
    print_stat("GPU merge kernel-only",           kernel_stats, total);
    print_stat("GDS wall (I/O + kernel + D2H)",   wall_stats,   total);
    printf("  GDS I/O bandwidth at min:  %.1f MB/s\n",
           data_mb / (load_stats.min / 1e3));
    printf("  Validation:                %s\n", ok ? "PASS ✓" : "FAIL ✗");

    /* ---- Comparison vs traditional POSIX path ---- */
    double posix_wall_min  = posix_io_ms + mr_posix.gpu_wall.min;
    double posix_wall_mean = posix_io_ms + mr_posix.gpu_wall.mean;
    double gds_wall_min    = wall_stats.min;
    double gds_wall_mean   = wall_stats.mean;

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  I/O + Merge path comparison (min / mean)               │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │  POSIX I/O + H2D + kernel + D2H:  %7.2f / %7.2f ms  │\n",
           posix_wall_min, posix_wall_mean);
    printf("  │    (%s + %s)\n",
           "disk→host (POSIX)", "cudaMemcpy H2D");
    printf("  │  GDS %-11s + kernel + D2H:  %7.2f / %7.2f ms  │\n",
           s_gds_mode == GDS_NATIVE ? "(native)" : "(compat)",
           gds_wall_min, gds_wall_mean);
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │  Speedup (min):   %.2f×    Speedup (mean):  %.2f×       │\n",
           posix_wall_min  / gds_wall_min,
           posix_wall_mean / gds_wall_mean);
    printf("  │  I/O savings (min):  %.2f ms   "
           "I/O savings (mean):  %.2f ms  │\n",
           posix_io_ms - load_stats.min,
           posix_io_ms - load_stats.mean);
    printf("  └─────────────────────────────────────────────────────────┘\n\n");

    GDSMergeResult result;
    result.gds_load  = load_stats;
    result.kernel    = kernel_stats;
    result.wall      = wall_stats;
    result.validated = ok;
    return result;
}

/* =========================================================================
 * BENCHMARK 5 – Pack / Unpack kernels (serialise ↔ data blocks)
 *
 * Pack:   sorted KVPair array → prefix-compressed SST data blocks (Fig. 8)
 * Unpack: SST data blocks → flat KVPair array                    (Fig. 7)
 *
 * Measured independently and in a round-trip (pack → unpack) to verify
 * correctness.  Both CPU and GPU paths are timed.
 * ======================================================================= */
static PackUnpackResult bench_pack_unpack(
        const std::vector<KVPair>& merged,
        int    keys_per_block,
        int    restart_interval,
        int    runs)
{
    hr();
    printf("BENCHMARK 5 – Pack / Unpack kernels (Fig. 7 & 8)\n");
    hr();

    int total_keys = (int)merged.size();
    int num_blocks = (total_keys + keys_per_block - 1) / keys_per_block;

    printf("  Total keys:         %d\n", total_keys);
    printf("  Keys/block:         %d\n", keys_per_block);
    printf("  Restart interval:   %d\n", restart_interval);
    printf("  Data blocks:        %d\n", num_blocks);
    printf("\n");

    /* ── CPU Pack baseline ─────────────────────────────────────────── */
    printf("  Running CPU pack (%d runs) ...\n", runs); fflush(stdout);
    std::vector<double> cpu_pack_s(runs);
    std::vector<std::vector<uint8_t>> cpu_blocks(num_blocks);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < num_blocks; ++b) {
            int offset = b * keys_per_block;
            int bk     = std::min(keys_per_block, total_keys - offset);
            cpu_blocks[b] = cpu_pack_block(merged.data() + offset, bk,
                                            restart_interval);
        }
        cpu_pack_s[r] = std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    }
    RunStats cpu_pack_stats = RunStats::from(cpu_pack_s);

    /* ── CPU Unpack baseline ───────────────────────────────────────── */
    printf("  Running CPU unpack (%d runs) ...\n", runs); fflush(stdout);
    std::vector<KVPair> cpu_unpacked(total_keys);
    std::vector<double> cpu_unpack_s(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        int kv_offset = 0;
        for (int b = 0; b < num_blocks; ++b) {
            int n = cpu_unpack_block(cpu_blocks[b].data(),
                                     cpu_blocks[b].size(),
                                     cpu_unpacked.data() + kv_offset);
            kv_offset += n;
        }
        cpu_unpack_s[r] = std::chrono::duration<double,std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    }
    RunStats cpu_unpack_stats = RunStats::from(cpu_unpack_s);

    /* Validate CPU round-trip */
    bool cpu_rt_ok = true;
    for (int i = 0; i < total_keys && cpu_rt_ok; ++i) {
        if (cpu_unpacked[i].key != merged[i].key ||
            cpu_unpacked[i].value != merged[i].value) {
            fprintf(stderr, "  CPU round-trip MISMATCH at %d: "
                    "{%llu,%llu} vs {%llu,%llu}\n", i,
                    (unsigned long long)cpu_unpacked[i].key,
                    (unsigned long long)cpu_unpacked[i].value,
                    (unsigned long long)merged[i].key,
                    (unsigned long long)merged[i].value);
            cpu_rt_ok = false;
        }
    }

    /* ── GPU Pack ──────────────────────────────────────────────────── */
    printf("  Running GPU pack (%d runs) ...\n", runs); fflush(stdout);

    /* Warm-up */
    PackResult warmup_pr;
    launch_pack(merged.data(), total_keys, keys_per_block,
                restart_interval, warmup_pr);

    /* Device buffers for timed runs */
    int max_restarts    = (keys_per_block + restart_interval - 1) / restart_interval;
    int max_block_bytes = PACK_HEADER_BYTES
                        + keys_per_block * PACK_MAX_ENTRY_BYTES
                        + max_restarts * (int)sizeof(uint32_t);

    KVPair   *d_kv;
    uint8_t  *d_blocks;
    uint32_t *d_block_sizes;
    CUDA_CHECK(cudaMalloc(&d_kv,          (size_t)total_keys * sizeof(KVPair)));
    CUDA_CHECK(cudaMalloc(&d_blocks,      (size_t)num_blocks * max_block_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_sizes, (size_t)num_blocks * sizeof(uint32_t)));

    int smem_bytes = 3 * max_restarts * (int)sizeof(uint32_t);
    int block_dim  = ((max_restarts + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    std::vector<double> gpu_pack_k(runs), gpu_pack_w(runs);
    PackResult gpu_pr;
    gpu_pr.num_blocks = num_blocks;
    gpu_pr.block_sizes.resize(num_blocks);

    for (int r = 0; r < runs; ++r) {
        auto wall_t0 = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(d_kv, merged.data(),
                              (size_t)total_keys * sizeof(KVPair),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_blocks, 0, (size_t)num_blocks * max_block_bytes));

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));

        pack_kernel<<<num_blocks, block_dim, smem_bytes>>>(
            d_kv, total_keys, keys_per_block, restart_interval,
            d_blocks, d_block_sizes, max_block_bytes);

        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float kms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kms, ev0, ev1));
        gpu_pack_k[r] = (double)kms;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        CUDA_CHECK(cudaMemcpy(gpu_pr.block_sizes.data(), d_block_sizes,
                              (size_t)num_blocks * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        gpu_pack_w[r] = std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now() - wall_t0).count();
    }
    RunStats gpu_pack_k_stats = RunStats::from(gpu_pack_k);
    RunStats gpu_pack_w_stats = RunStats::from(gpu_pack_w);

    /* Download last pack result for unpack benchmark */
    std::vector<uint8_t> raw_blocks((size_t)num_blocks * max_block_bytes);
    CUDA_CHECK(cudaMemcpy(raw_blocks.data(), d_blocks,
                          (size_t)num_blocks * max_block_bytes,
                          cudaMemcpyDeviceToHost));
    cudaFree(d_kv);
    cudaFree(d_blocks);
    cudaFree(d_block_sizes);

    /* Compact GPU packed blocks */
    gpu_pr.block_offsets.resize(num_blocks);
    size_t total_bytes = 0;
    for (int b = 0; b < num_blocks; ++b)
        total_bytes += gpu_pr.block_sizes[b];
    gpu_pr.block_buf.resize(total_bytes);
    size_t off = 0;
    for (int b = 0; b < num_blocks; ++b) {
        gpu_pr.block_offsets[b] = (uint32_t)off;
        memcpy(gpu_pr.block_buf.data() + off,
               raw_blocks.data() + (size_t)b * max_block_bytes,
               gpu_pr.block_sizes[b]);
        off += gpu_pr.block_sizes[b];
    }

    /* ── GPU Unpack ────────────────────────────────────────────────── */
    printf("  Running GPU unpack (%d runs) ...\n", runs); fflush(stdout);

    /* Warm-up */
    {
        std::vector<KVPair> tmp(total_keys);
        launch_unpack(gpu_pr.block_buf.data(), gpu_pr.block_buf.size(),
                      gpu_pr.block_offsets.data(), num_blocks,
                      keys_per_block, total_keys, tmp.data());
    }

    uint8_t  *d_blk_buf;
    uint32_t *d_blk_offsets;
    KVPair   *d_kv_out;
    CUDA_CHECK(cudaMalloc(&d_blk_buf,     gpu_pr.block_buf.size()));
    CUDA_CHECK(cudaMalloc(&d_blk_offsets, (size_t)num_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_kv_out,      (size_t)total_keys * sizeof(KVPair)));

    int unpack_max_restarts = max_restarts;
    int unpack_block_dim    = ((unpack_max_restarts + 31) / 32) * 32;
    if (unpack_block_dim < 32) unpack_block_dim = 32;

    std::vector<KVPair> gpu_unpacked(total_keys);
    std::vector<double> gpu_unpack_k(runs), gpu_unpack_w(runs);

    for (int r = 0; r < runs; ++r) {
        auto wall_t0 = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(d_blk_buf, gpu_pr.block_buf.data(),
                              gpu_pr.block_buf.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_blk_offsets, gpu_pr.block_offsets.data(),
                              (size_t)num_blocks * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kv_out, 0, (size_t)total_keys * sizeof(KVPair)));

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));

        unpack_kernel<<<num_blocks, unpack_block_dim>>>(
            d_blk_buf, d_blk_offsets, num_blocks,
            keys_per_block, d_kv_out, total_keys);

        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float kms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kms, ev0, ev1));
        gpu_unpack_k[r] = (double)kms;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        CUDA_CHECK(cudaMemcpy(gpu_unpacked.data(), d_kv_out,
                              (size_t)total_keys * sizeof(KVPair),
                              cudaMemcpyDeviceToHost));

        gpu_unpack_w[r] = std::chrono::duration<double,std::milli>(
                              std::chrono::steady_clock::now() - wall_t0).count();
    }
    RunStats gpu_unpack_k_stats = RunStats::from(gpu_unpack_k);
    RunStats gpu_unpack_w_stats = RunStats::from(gpu_unpack_w);

    cudaFree(d_blk_buf);
    cudaFree(d_blk_offsets);
    cudaFree(d_kv_out);

    /* ── Validate GPU round-trip ───────────────────────────────────── */
    bool gpu_rt_ok = true;
    for (int i = 0; i < total_keys && gpu_rt_ok; ++i) {
        if (gpu_unpacked[i].key != merged[i].key ||
            gpu_unpacked[i].value != merged[i].value) {
            fprintf(stderr, "  GPU round-trip MISMATCH at %d: "
                    "{%llu,%llu} vs {%llu,%llu}\n", i,
                    (unsigned long long)gpu_unpacked[i].key,
                    (unsigned long long)gpu_unpacked[i].value,
                    (unsigned long long)merged[i].key,
                    (unsigned long long)merged[i].value);
            gpu_rt_ok = false;
        }
    }

    /* Compression ratio */
    size_t raw_bytes    = (size_t)total_keys * sizeof(KVPair);
    size_t packed_bytes = gpu_pr.block_buf.size();
    double ratio        = (double)packed_bytes / (double)raw_bytes;

    /* ── Print results ─────────────────────────────────────────────── */
    printf("\n");
    printf("  Block config:  %d blocks × %d entries  restart_interval=%d   runs=%d\n",
           num_blocks, keys_per_block, restart_interval, runs);
    printf("  Raw size:      %zu bytes  (%.1f MB)\n", raw_bytes,
           (double)raw_bytes / (1 << 20));
    printf("  Packed size:   %zu bytes  (%.1f MB)  ratio=%.3f\n",
           packed_bytes, (double)packed_bytes / (1 << 20), ratio);
    printf("\n");
    print_stat("CPU pack",                    cpu_pack_stats,   (uint64_t)total_keys);
    print_stat("CPU unpack",                  cpu_unpack_stats, (uint64_t)total_keys);
    print_stat("GPU pack kernel-only",        gpu_pack_k_stats, (uint64_t)total_keys);
    print_stat("GPU pack wall (H2D+k+D2H)",  gpu_pack_w_stats, (uint64_t)total_keys);
    print_stat("GPU unpack kernel-only",      gpu_unpack_k_stats, (uint64_t)total_keys);
    print_stat("GPU unpack wall (H2D+k+D2H)",gpu_unpack_w_stats, (uint64_t)total_keys);
    printf("  Pack speedup kernel vs CPU (min):   %.2f×\n",
           cpu_pack_stats.min / gpu_pack_k_stats.min);
    printf("  Unpack speedup kernel vs CPU (min): %.2f×\n",
           cpu_unpack_stats.min / gpu_unpack_k_stats.min);
    printf("  CPU round-trip:   %s\n", cpu_rt_ok  ? "PASS ✓" : "FAIL ✗");
    printf("  GPU round-trip:   %s\n", gpu_rt_ok  ? "PASS ✓" : "FAIL ✗");
    printf("\n");

    PackUnpackResult pur;
    pur.cpu_pack     = cpu_pack_stats;
    pur.cpu_unpack   = cpu_unpack_stats;
    pur.gpu_pack_k   = gpu_pack_k_stats;
    pur.gpu_pack_w   = gpu_pack_w_stats;
    pur.gpu_unpack_k = gpu_unpack_k_stats;
    pur.gpu_unpack_w = gpu_unpack_w_stats;
    return pur;
}

/* =========================================================================
 * BENCHMARK 6 – Full GPU compaction pipeline
 *
 * Unpack (parse input SSTs) → Merge → Bloom → Pack (serialise output SST)
 *
 * This models the complete L0→L1 compaction on GPU:
 *   1. Unpack input SST data blocks into flat KVPair arrays
 *   2. Merge all KVPair arrays into one sorted array
 *   3. Build bloom filters for each output data block
 *   4. Pack sorted array into output SST data blocks
 *
 * Measures are:
 *   CPU pipeline:  CPU sort + CPU bloom + CPU pack
 *   GPU pipeline:  GPU merge + GPU batched bloom + GPU pack  (wall times)
 *   GPU pipeline kernel-only: kernel times excluding H2D/D2H
 * ======================================================================= */
static void bench_full_pipeline(
        const std::vector<std::vector<KVPair>>& ssts,
        const DatasetMeta&    meta,
        const MergeResult&    mr,
        const BloomResult&    br,
        const PackUnpackResult& pur,
        double                io_ms,
        int                   keys_per_block,
        int                   restart_interval,
        int                   runs,
        int                   compaction_rounds)
{
    hr();
    printf("BENCHMARK 6 – Full GPU compaction pipeline (Unpack → Merge → Bloom → Pack)\n");
    hr();

    uint64_t total_keys = meta.total_keys;
    int num_blocks = (int)((total_keys + keys_per_block - 1) / keys_per_block);

    printf("  Pipeline stages:\n");
    printf("    1. Unpack  – parse %d input SST blocks → flat KVPair arrays\n",
           meta.num_sst);
    printf("    2. Merge   – %llu keys from %d SSTs\n",
           (unsigned long long)total_keys, meta.num_sst);
    printf("    3. Bloom   – generate bloom filters for %d output blocks\n",
           num_blocks);
    printf("    4. Pack    – serialise sorted output into %d data blocks\n",
           num_blocks);
    printf("\n");

    /* Per-stage timing summary (from individual benchmarks) */
    printf("  Per-stage timing (best-of-%d):\n", runs);
    printf("  %-44s  %9s  %9s\n", "Stage", "CPU(ms)", "GPU wall(ms)");
    printf("  %-44s  %9s  %9s\n",
           "────────────────────────────────────────────",
           "---------", "---------");
    printf("  %-44s  %9.2f  %9.2f\n",
           "1. Unpack  (parse input blocks)",
           pur.cpu_unpack.min, pur.gpu_unpack_w.min);
    printf("  %-44s  %9.2f  %9.2f\n",
           "2. Merge   (sort/merge all keys)",
           mr.cpu.min, mr.gpu_wall.min);
    printf("  %-44s  %9.2f  %9.2f\n",
           "3. Bloom   (build per-block filters)",
           br.cpu.min, br.batched.min);
    printf("  %-44s  %9.2f  %9.2f\n",
           "4. Pack    (serialise output blocks)",
           pur.cpu_pack.min, pur.gpu_pack_w.min);
    printf("\n");

    /* Totals */
    double cpu_total_min  = pur.cpu_unpack.min + mr.cpu.min
                          + br.cpu.min + pur.cpu_pack.min;
    double cpu_total_mean = pur.cpu_unpack.mean + mr.cpu.mean
                          + br.cpu.mean + pur.cpu_pack.mean;
    double gpu_total_min  = pur.gpu_unpack_w.min + mr.gpu_wall.min
                          + br.batched.min + pur.gpu_pack_w.min;
    double gpu_total_mean = pur.gpu_unpack_w.mean + mr.gpu_wall.mean
                          + br.batched.mean + pur.gpu_pack_w.mean;
    double gpu_kern_min   = pur.gpu_unpack_k.min + mr.kernel.min
                          + br.kernel.min + pur.gpu_pack_k.min;
    double gpu_kern_mean  = pur.gpu_unpack_k.mean + mr.kernel.mean
                          + br.kernel.mean + pur.gpu_pack_k.mean;

    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │  Full pipeline comparison (min / mean)                      │\n");
    printf("  ├─────────────────────────────────────────────────────────────┤\n");
    printf("  │  CPU total (unpack+sort+bloom+pack):  %7.2f / %7.2f ms  │\n",
           cpu_total_min, cpu_total_mean);
    printf("  │  GPU wall  (unpack+merge+bloom+pack): %7.2f / %7.2f ms  │\n",
           gpu_total_min, gpu_total_mean);
    printf("  │  GPU kernel-only:                     %7.2f / %7.2f ms  │\n",
           gpu_kern_min, gpu_kern_mean);
    printf("  ├─────────────────────────────────────────────────────────────┤\n");
    printf("  │  Pipeline speedup wall  (min): %.2f×    (mean): %.2f×      │\n",
           cpu_total_min / gpu_total_min,
           cpu_total_mean / gpu_total_mean);
    printf("  │  Pipeline speedup kern  (min): %.2f×    (mean): %.2f×      │\n",
           cpu_total_min / gpu_kern_min,
           cpu_total_mean / gpu_kern_mean);
    printf("  │  Throughput CPU  (min): %.2f M keys/s                      │\n",
           (double)total_keys / cpu_total_min / 1e3);
    printf("  │  Throughput GPU  (min): %.2f M keys/s                      │\n",
           (double)total_keys / gpu_total_min / 1e3);
    printf("  └─────────────────────────────────────────────────────────────┘\n");

    /* ── I/O Analysis ─────────────────────────────────────────────────────
     * Break down exactly where time is spent across the full round.
     * H2D is the merge input transfer (pinned measurement from Benchmark 1).
     * D2H is estimated as (pinned_wall - H2D - kernel).
     * -------------------------------------------------------------------- */
    {
        double h2d_ms   = mr.h2d_pinned.min;          /* merge input H2D     */
        double kern_ms  = gpu_kern_min;                /* all kernels         */
        double d2h_ms   = mr.gpu_wall_pinned.min - h2d_ms - mr.kernel.min;
        if (d2h_ms < 0) d2h_ms = 0;
        double xfer_ms  = h2d_ms + d2h_ms;
        double round_ms = io_ms + gpu_total_min;      /* entire GPU round    */

        printf("\n  I/O analysis (where time goes per GPU round):\n");
        printf("  %-38s  %8s  %7s\n", "Component", "ms (min)", "% of round");
        printf("  %-38s  %8s  %7s\n",
               "──────────────────────────────────────", "--------", "----------");
        printf("  %-38s  %8.2f  %6.1f%%\n", "Disk read (fread, NFS)",
               io_ms,   100.0 * io_ms   / round_ms);
        printf("  %-38s  %8.2f  %6.1f%%\n", "H2D transfer (merge input, pinned)",
               h2d_ms,  100.0 * h2d_ms  / round_ms);
        printf("  %-38s  %8.2f  %6.1f%%\n", "GPU kernels (unpack+merge+bloom+pack)",
               kern_ms, 100.0 * kern_ms / round_ms);
        printf("  %-38s  %8.2f  %6.1f%%\n", "D2H transfer (merge output, est.)",
               d2h_ms,  100.0 * d2h_ms  / round_ms);
        printf("  %-38s  %8.2f  %6.1f%%\n", "Other GPU wall overhead",
               gpu_total_min - kern_ms - xfer_ms,
               100.0 * (gpu_total_min - kern_ms - xfer_ms) / round_ms);
        printf("  %-38s  %8.2f  %6.1f%%\n", "TOTAL GPU round (I/O + wall)",
               round_ms, 100.0);
        printf("\n");
        printf("  Bottleneck:  ");
        double max_comp = io_ms;
        const char* bottleneck = "disk I/O (NFS fread)";
        if (h2d_ms  > max_comp) { max_comp = h2d_ms;  bottleneck = "PCIe H2D transfer"; }
        if (kern_ms > max_comp) { max_comp = kern_ms; bottleneck = "GPU kernels";        }
        printf("%s  (%.2f ms = %.1f%% of round)\n",
               bottleneck, max_comp, 100.0 * max_comp / round_ms);

        /* ── Projected pipelined throughput ─────────────────────────────
         * Double-buffered pipelining overlaps disk I/O of round N+1 with
         * GPU processing of round N.  The per-round bottleneck becomes
         * max(disk, GPU_wall) instead of disk + GPU_wall.
         * ---------------------------------------------------------------- */
        double pipe_round_ms = (io_ms > gpu_total_min) ? io_ms : gpu_total_min;
        printf("\n  Projected pipelined round (overlap disk I/O with GPU):\n");
        printf("    per-round bottleneck: max(disk %.2f, GPU %.2f) = %.2f ms\n",
               io_ms, gpu_total_min, pipe_round_ms);
        printf("    per-round speedup vs serial:  %.2f×\n", round_ms / pipe_round_ms);
        if (compaction_rounds > 0) {
            double cpu_total_round = io_ms + cpu_total_min;
            double pipe_total_ms   = (double)compaction_rounds * pipe_round_ms;
            double serial_total_ms = (double)compaction_rounds * round_ms;
            double cpu_sim_ms      = (double)compaction_rounds * cpu_total_round;
            printf("    aggregate %d rounds:  %.1f ms  vs serial %.1f ms  (%.2f× faster)\n",
                   compaction_rounds, pipe_total_ms, serial_total_ms,
                   serial_total_ms / pipe_total_ms);
            printf("    speedup vs CPU total:      %.2f×\n",
                   cpu_sim_ms / pipe_total_ms);
        }
        printf("\n");
    }

    /* ── Pinned H2D variant for full pipeline ────────────────────────────
     * Replace mr.gpu_wall with mr.gpu_wall_pinned (pinned async H2D)
     * to see how pinned memory improves the complete wall time.
     * -------------------------------------------------------------------- */
    {
        double gpu_pinned_min  = pur.gpu_unpack_w.min + mr.gpu_wall_pinned.min
                               + br.batched.min + pur.gpu_pack_w.min;
        double gpu_pinned_mean = pur.gpu_unpack_w.mean + mr.gpu_wall_pinned.mean
                               + br.batched.mean + pur.gpu_pack_w.mean;
        printf("  With pinned async H2D for merge (B1 optimisation):\n");
        printf("  %-44s  %9s  %9s\n", "Stage", "heap(ms)", "pinned(ms)");
        printf("  %-44s  %9s  %9s\n",
               "────────────────────────────────────────────",
               "---------", "---------");
        printf("  %-44s  %9.2f  %9.2f\n", "1. Unpack",
               pur.gpu_unpack_w.min, pur.gpu_unpack_w.min);
        printf("  %-44s  %9.2f  %9.2f\n", "2. Merge (H2D+k+D2H)",
               mr.gpu_wall.min, mr.gpu_wall_pinned.min);
        printf("  %-44s  %9.2f  %9.2f\n", "3. Bloom",
               br.batched.min, br.batched.min);
        printf("  %-44s  %9.2f  %9.2f\n", "4. Pack",
               pur.gpu_pack_w.min, pur.gpu_pack_w.min);
        printf("  %-44s  %9.2f  %9.2f\n", "TOTAL GPU wall",
               gpu_total_min, gpu_pinned_min);
        printf("  %-44s  %9s  %9.2f×\n", "Speedup (wall pinned vs wall heap)",
               "-", gpu_total_min / gpu_pinned_min);
        (void)gpu_pinned_mean;
    }

    /* Including I/O */
    printf("\n  With disk I/O (%.2f ms):\n", io_ms);
    printf("    CPU pipeline total:     %.2f ms  (I/O + CPU compute)\n",
           io_ms + cpu_total_min);
    printf("    GPU pipeline total:     %.2f ms  (I/O + GPU wall)\n",
           io_ms + gpu_total_min);
    printf("    GPU pinned total:       %.2f ms  (I/O + GPU pinned wall)\n",
           io_ms + pur.gpu_unpack_w.min + mr.gpu_wall_pinned.min
                 + br.batched.min + pur.gpu_pack_w.min);
    printf("    End-to-end speedup (heap):   %.2f×\n",
           (io_ms + cpu_total_min) / (io_ms + gpu_total_min));
    printf("    End-to-end speedup (pinned): %.2f×\n",
           (io_ms + cpu_total_min) /
           (io_ms + pur.gpu_unpack_w.min + mr.gpu_wall_pinned.min
                  + br.batched.min + pur.gpu_pack_w.min));

    /* Compaction round simulation (if requested) */
    if (compaction_rounds > 0) {
        printf("\n  Compaction simulation (%d rounds × %llu keys = %.1f M total keys):\n",
               compaction_rounds, (unsigned long long)total_keys,
               (double)compaction_rounds * total_keys / 1e6);

        double gpu_pinned_min  = pur.gpu_unpack_w.min + mr.gpu_wall_pinned.min
                               + br.batched.min + pur.gpu_pack_w.min;
        double cpu_sim         = (double)compaction_rounds * (io_ms + cpu_total_min);
        double gpu_sim         = (double)compaction_rounds * (io_ms + gpu_total_min);
        double gpu_pinned_sim  = (double)compaction_rounds * (io_ms + gpu_pinned_min);

        printf("    CPU total:       %.1f ms  (%.2f s)  throughput: %.2f M keys/s\n",
               cpu_sim, cpu_sim / 1e3,
               (double)compaction_rounds * total_keys / cpu_sim / 1e3);
        printf("    GPU total:       %.1f ms  (%.2f s)  throughput: %.2f M keys/s\n",
               gpu_sim, gpu_sim / 1e3,
               (double)compaction_rounds * total_keys / gpu_sim / 1e3);
        printf("    GPU pinned H2D:  %.1f ms  (%.2f s)  throughput: %.2f M keys/s\n",
               gpu_pinned_sim, gpu_pinned_sim / 1e3,
               (double)compaction_rounds * total_keys / gpu_pinned_sim / 1e3);
        printf("    Speedup (heap):   %.2f×\n", cpu_sim / gpu_sim);
        printf("    Speedup (pinned): %.2f×\n", cpu_sim / gpu_pinned_sim);
        printf("    Time saved (heap):   %.1f ms  (%.2f s)\n",
               cpu_sim - gpu_sim, (cpu_sim - gpu_sim) / 1e3);
        printf("    Time saved (pinned): %.1f ms  (%.2f s)\n",
               cpu_sim - gpu_pinned_sim, (cpu_sim - gpu_pinned_sim) / 1e3);
    }
    printf("\n");
}

/* =========================================================================
 * BENCHMARK 3 – Total compaction job (Alg1 + Alg2 end-to-end)
 * ======================================================================= */
static void bench_total_compaction(const MergeResult& mr,
                                   const BloomResult&  br,
                                   double              io_ms,
                                   uint64_t            total_keys,
                                   int                 runs,
                                   int                 compaction_rounds)
{
    if (compaction_rounds <= 0) return;

    hr();
    printf("BENCHMARK 3 – fillrandom compaction simulation (Merge + Bloom)\n");
    hr();

    uint64_t total_simulated_keys = (uint64_t)compaction_rounds * total_keys;

    printf("  Compaction model:\n");
    printf("    keys/compaction round = %llu  (%d SSTs flushed from MemTable)\n",
           (unsigned long long)total_keys, 4);
    printf("    compaction rounds     = %d\n", compaction_rounds);
    printf("    total simulated keys  = %llu  (%.1f M)\n",
           (unsigned long long)total_simulated_keys,
           (double)total_simulated_keys / 1e6);
    printf("\n");

    /* Per-round timing from Benchmarks 1 & 2 */
    double cpu_round_mean  = mr.cpu.mean  + br.cpu.mean;
    double cpu_round_min   = mr.cpu.min   + br.cpu.min;
    double gpu_round_mean  = mr.gpu_wall.mean + br.batched.mean;
    double gpu_round_min   = mr.gpu_wall.min  + br.batched.min;
    double io_round        = io_ms;

    double cpu_total_ms    = (double)compaction_rounds * (io_round + cpu_round_mean);
    double gpu_total_ms    = (double)compaction_rounds * (io_round + gpu_round_mean);
    double cpu_total_min   = (double)compaction_rounds * (io_round + cpu_round_min);
    double gpu_total_min   = (double)compaction_rounds * (io_round + gpu_round_min);

    double time_saved_mean = cpu_total_ms - gpu_total_ms;
    double time_saved_min  = cpu_total_min - gpu_total_min;

    auto tput_total = [&](double total_ms) {
        return (double)total_simulated_keys / total_ms / 1e3;
    };

    printf("  Per-round timing (best-of-%d):\n", runs);
    printf("  %-40s  %9s  %9s\n", "Path", "mean(ms)", "min(ms)");
    printf("  %-40s  %9s  %9s\n",
           "────────────────────────────────────────", "---------", "---------");
    printf("  %-40s  %9.2f  %9.2f\n",
           "I/O (disk read, per round)",
           io_round, io_round);
    printf("  %-40s  %9.2f  %9.2f\n",
           "CPU compute/round  (sort+bloom)",
           cpu_round_mean, cpu_round_min);
    printf("  %-40s  %9.2f  %9.2f\n",
           "GPU wall/round     (H2D+merge+bloom+D2H)",
           gpu_round_mean, gpu_round_min);
    printf("  %-40s  %9.2f  %9.2f\n",
           "CPU total/round    (I/O+compute)",
           io_round + cpu_round_mean, io_round + cpu_round_min);
    printf("  %-40s  %9.2f  %9.2f\n",
           "GPU total/round    (I/O+wall)",
           io_round + gpu_round_mean, io_round + gpu_round_min);
    printf("\n");

    printf("  Aggregate over %d compaction rounds:\n", compaction_rounds);
    printf("  %-40s  %9s  %9s  %12s\n",
           "Path", "mean(ms)", "min(ms)", "M keys/s");
    printf("  %-40s  %9s  %9s  %12s\n",
           "────────────────────────────────────────",
           "---------", "---------", "------------");
    printf("  %-40s  %9.1f  %9.1f  %12.2f\n",
           "CPU total  (I/O + sort + bloom)",
           cpu_total_ms, cpu_total_min,
           tput_total(cpu_total_ms));
    printf("  %-40s  %9.1f  %9.1f  %12.2f\n",
           "GPU total  (I/O + merge + batched bloom)",
           gpu_total_ms, gpu_total_min,
           tput_total(gpu_total_ms));
    printf("\n");
    printf("  Speedup (mean): %.2f×\n",
           cpu_total_ms / gpu_total_ms);
    printf("  Speedup (min):  %.2f×\n",
           cpu_total_min / gpu_total_min);
    printf("  Time saved (mean): %.1f ms  ( %.2f s )\n",
           time_saved_mean, time_saved_mean / 1000.0);
    printf("  Time saved (min):  %.1f ms  ( %.2f s )\n",
           time_saved_min,  time_saved_min  / 1000.0);
}

/* =========================================================================
 * main
 * ======================================================================= */
static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "  --dataset         DIR    path to dataset directory    [default: ./dataset]\n"
        "  --block_size      BYTES  SST data block size (bytes)  [default: 32768]\n"
        "  --key_size        BYTES  key size (bytes)             [default: 16]\n"
        "  --value_size      BYTES  value size (bytes)           [default: 64]\n"
        "  --overhead        BYTES  per-entry SST overhead bytes [default: 20]\n"
        "  --restart_interval N     prefix-compress restart interval  [default: 16]\n"
        "  --fpr_samples     N      non-member samples for FPR   [default: 10000]\n"
        "  --runs            N      timed repetitions per section [default: 5]\n"
        "  --compaction_rounds N    simulate N compaction rounds  [default: 0 = skip]\n"
        "  --fillrandom_keys N      auto-compute rounds for N total keys (overrides --compaction_rounds)\n"
        "                           Accepts K/M/B suffixes: 10M = 10,000,000  200M  1B  500K\n"
        "  --gds                    Enable Benchmark 4: GDS I/O + Merge via cuFile\n"
        "                           (requires -DGPCOMP_USE_CUFILE build; falls back to\n"
        "                            pinned cudaMemcpy when cuFile is not compiled in)\n"
        "  --compat-mode            Force cuFile Compatibility Mode before driver open\n"
        "                           (use on RTX 3070 / any GPU without native GDS;\n"
        "                            implies --gds)\n"
        "  --help\n",
        prog);
}

int main(int argc, char* argv[])
{
    std::string dataset_dir  = "dataset";
    int block_size_bytes     = 32768;
    int key_size_bytes       = 16;
    int value_size_bytes     = 64;
    int overhead_bytes       = 20;
    int fpr_samples          = 10000;
    int runs                 = 5;
    int restart_interval     = PACK_RESTART_INTERVAL;  /* 16 */
    int compaction_rounds    = 0;   /* 0 = skip Benchmark 3           */
    int64_t fillrandom_keys  = 0;   /* if >0, overrides compaction_rounds */
    bool use_gds             = false;
    bool force_compat        = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--dataset")     dataset_dir      = next();
        else if (a == "--block_size")  block_size_bytes  = std::stoi(next());
        else if (a == "--key_size")    key_size_bytes    = std::stoi(next());
        else if (a == "--value_size")  value_size_bytes  = std::stoi(next());
        else if (a == "--overhead")    overhead_bytes    = std::stoi(next());
        else if (a == "--restart_interval") restart_interval = std::stoi(next());
        else if (a == "--fpr_samples") fpr_samples       = std::stoi(next());
        else if (a == "--runs")            runs              = std::stoi(next());
        else if (a == "--compaction_rounds") compaction_rounds = std::stoi(next());
        else if (a == "--fillrandom_keys")   fillrandom_keys   = parse_count(next());
        else if (a == "--gds")              use_gds           = true;
        else if (a == "--compat-mode")    { force_compat = true; use_gds = true; }
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown option '%s'\n", a.c_str()); return 1; }
    }

    /* ---- GDS / cuFile initialisation (before any I/O) ---- */
    GDSMode gds_mode = GDS_DISABLED;
    if (use_gds) {
        if (force_compat)
            gpcomp_gds_force_compat_mode();   /* writes /tmp JSON + sets env-var */
        gds_mode = gpcomp_gds_init();
    }

    /* ---- GPU info ---- */
    {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        printf("═══════════════════════════════════════════════════════════\n");
        printf("  GPComp kernel benchmark\n");
        printf("  GPU: %s  |  %.0f GB global  |  %zu KB shared/block  |  %d SMs\n",
               prop.name,
               (double)prop.totalGlobalMem / (1 << 30),
               prop.sharedMemPerBlock / 1024,
               prop.multiProcessorCount);
        printf("  GDS mode: %s\n", gds_mode_name(gds_mode));
        printf("  Dataset: %s\n", dataset_dir.c_str());
        printf("═══════════════════════════════════════════════════════════\n\n");
    }

    /* ---- Load metadata ---- */
    DatasetMeta meta = load_meta(dataset_dir + "/dataset.meta");

    printf("Dataset summary:\n");
    printf("  num_sst=%d  keys_per_sst=%d  total_keys=%llu\n",
           meta.num_sst, meta.keys_per_sst, (unsigned long long)meta.total_keys);
    printf("  key_space=%llu  bloom_bits=%d  K=%d\n\n",
           (unsigned long long)meta.key_space, meta.bloom_bits, meta.bloom_K);

    /* ---- Load SSTs (measure disk I/O time) ---- */
    printf("Loading SST files ...\n");
    std::vector<std::vector<KVPair>> ssts(meta.num_sst);
    auto io_t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < meta.num_sst; ++s) {
        char fname[64];
        snprintf(fname, sizeof(fname), "sst_%04d.bin", s);
        ssts[s] = load_sst_bin(dataset_dir + "/" + fname, meta.sst_sizes[s]);
    }
    double io_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - io_t0).count();
    printf("  Loaded %d SST files in %.1f ms\n\n", meta.num_sst, io_ms);

    /* ================================================================
     * Benchmark 1: Merge
     * ================================================================ */
    MergeResult mr = bench_merge(ssts, meta, io_ms, runs);

    /* ================================================================
     * Compute merged output (CPU) – input for bloom benchmark
     * ================================================================ */
    printf("Computing CPU merge for bloom benchmark input ...\n");
    std::vector<KVPair> merged = cpu_merge_reference(ssts);
    printf("  %zu keys in merged output\n\n", merged.size());

    /* ================================================================
     * Benchmark 2: Bloom filter
     * ================================================================ */
    BloomResult br = bench_bloom(merged, meta,
                                 block_size_bytes, key_size_bytes, value_size_bytes,
                                 overhead_bytes, fpr_samples, runs);

    /* ================================================================
     * Benchmark 5: Pack / Unpack (serialise ↔ data blocks)
     * ================================================================ */
    int bytes_per_entry = key_size_bytes + value_size_bytes + overhead_bytes;
    int pack_keys_per_block = block_size_bytes / bytes_per_entry;
    if (pack_keys_per_block <= 0) pack_keys_per_block = 1;

    PackUnpackResult pur = bench_pack_unpack(merged, pack_keys_per_block,
                                             restart_interval, runs);

    /* ================================================================
     * Benchmark 6: Full GPU compaction pipeline
     * ================================================================ */
    if (fillrandom_keys > 0) {
        compaction_rounds = (int)(((int64_t)fillrandom_keys + (int64_t)meta.total_keys - 1)
                                  / (int64_t)meta.total_keys);
        printf("  (--fillrandom_keys %lld → %d compaction rounds of %llu keys each)\n",
               (long long)fillrandom_keys,
               compaction_rounds,
               (unsigned long long)meta.total_keys);
    }
    bench_full_pipeline(ssts, meta, mr, br, pur, io_ms,
                        pack_keys_per_block, restart_interval,
                        runs, compaction_rounds);

    /* ================================================================
     * Benchmark 3: Total compaction job (legacy: Merge + Bloom only)
     * ================================================================ */
    bench_total_compaction(mr, br, io_ms, meta.total_keys, runs, compaction_rounds);

    /* ================================================================
     * Benchmark 4 (optional): GDS I/O + Merge
     *
     * Enabled by --gds (any GPU, pinned-memory fallback) or
     *            --compat-mode (RTX 3070 / no native GDS hardware).
     * On A30 with native GDS hardware: build with GDS=1 and omit
     * --compat-mode; cuFileDriverOpen() will activate NATIVE mode.
     * ================================================================ */
    if (use_gds) {
        bench_gds_io_merge(dataset_dir, meta, merged, mr, io_ms, runs, gds_mode);
    }

    gpcomp_gds_cleanup();

    hr();
    printf("Done.\n");
    return 0;
}
