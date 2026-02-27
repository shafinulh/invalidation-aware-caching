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
/* gpcomp_common.cuh is pulled in by both headers above */

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
 * Print a horizontal rule
 * ---------------------------------------------------------------------- */
static void hr() { printf("─────────────────────────────────────────────────────────\n"); }

/* =========================================================================
 * BENCHMARK 1 – Merge kernel
 * ======================================================================= */
static void bench_merge(const std::vector<std::vector<KVPair>>& ssts,
                        const DatasetMeta& meta)
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

    /* ---- CPU reference (for validation) ---- */
    printf("  Computing CPU reference ...\n"); fflush(stdout);
    auto cpu_t0 = std::chrono::steady_clock::now();
    std::vector<KVPair> cpu_ref = cpu_merge_reference(ssts);
    double cpu_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - cpu_t0).count();
    printf("  CPU merge-sort:   %.1f ms  (%.1f M keys/s)\n",
           cpu_ms, (double)total / cpu_ms / 1e3);

    /* ---- GPU: total time (H2D + kernel + D2H) ---- */
    printf("  Running GPU merge (including H2D + D2H) ...\n"); fflush(stdout);

    auto gpu_wall_t0 = std::chrono::steady_clock::now();
    int rc = launch_merge(const_cast<KVPair* const *>(h_ptrs.data()),
                          h_sizes.data(), num_sst, gpu_output.data());
    double gpu_wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - gpu_wall_t0).count();

    if (rc != 0) {
        fprintf(stderr, "  ERROR: launch_merge returned %d\n", rc);
        return;
    }

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
    double data_gb = (double)total * sizeof(KVPair) / (1 << 30);
    printf("\n");
    printf("  Input:            %d SSTs × %d keys = %llu total keys  (%.1f MB)\n",
           num_sst, (int)ssts[0].size(),
           (unsigned long long)total,
           (double)total * sizeof(KVPair) / (1 << 20));
    printf("  GPU total (wall): %.2f ms → %.1f M keys/s  (%.2f GB/s)\n",
           gpu_wall_ms,
           (double)total / gpu_wall_ms / 1e3,
           data_gb / (gpu_wall_ms / 1e3));
    printf("  CPU reference:    %.2f ms → %.1f M keys/s\n",
           cpu_ms, (double)total / cpu_ms / 1e3);
    printf("  Speedup (wall):   %.2f×\n", cpu_ms / gpu_wall_ms);
    printf("  Validation:       %s\n\n", ok ? "PASS ✓" : "FAIL ✗");
}

/* =========================================================================
 * BENCHMARK 2 – Bloom filter kernel (per-block mode)
 * ======================================================================= */
static void bench_bloom(const std::vector<KVPair>& merged,
                         const DatasetMeta& meta,
                         int block_size_bytes,
                         int key_size_bytes,
                         int value_size_bytes,
                         int overhead_bytes,
                         int fpr_samples)
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
            return;
        }
        printf("  Device sharedMem:   %zu KB  (%s)\n",
               prop.sharedMemPerBlock / 1024,
               (size_t)byte_vector_len <= prop.sharedMemPerBlock ? "fits ✓" : "OVERFLOW ✗");
    }
    printf("\n");

    /* ---- Allocate device buffers (reused across all blocks) ---- */
    KVPair  *d_block;
    uint8_t *d_bitvec;
    CUDA_CHECK(cudaMalloc(&d_block,  (size_t)keys_per_block * sizeof(KVPair)));
    CUDA_CHECK(cudaMalloc(&d_bitvec, (size_t)bitvec_len     * sizeof(uint8_t)));

    size_t shared_bytes = (size_t)byte_vector_len * sizeof(uint8_t);

    /* We store bit vectors to validate later */
    std::vector<std::vector<uint8_t>> all_bitvecs(num_blocks,
                                                   std::vector<uint8_t>(bitvec_len, 0));

    /* ---- Benchmark loop: kernel timing only (no H2D/D2H) ---- */
    CudaTimer timer;

    /* Warm-up pass (one block) */
    {
        int block_keys = std::min(keys_per_block, (int)total_keys);
        CUDA_CHECK(cudaMemcpy(d_block, merged.data(),
                              block_keys * sizeof(KVPair), cudaMemcpyHostToDevice));
        bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
            d_block, block_keys, K, byte_vector_len, d_bitvec);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* Timed pass – kernel time only (device-side CUDA events)  */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    for (int b = 0; b < num_blocks; ++b) {
        int offset      = b * keys_per_block;
        int block_keys  = std::min(keys_per_block, (int)(total_keys - offset));

        bloom_filter_kernel<<<1, block_dim, shared_bytes>>>(
            d_block, block_keys, K, byte_vector_len, d_bitvec);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float kernel_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    /* Full timed pass with H2D + kernel + D2H (wall time) */
    auto wall_t0 = std::chrono::steady_clock::now();
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
    double wall_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - wall_t0).count();

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_block);
    cudaFree(d_bitvec);

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
    printf("  GPU kernel-only:    %.2f ms → %.1f M keys/s\n",
           kernel_ms, (double)total_keys / kernel_ms / 1e3);
    printf("  GPU total (wall):   %.2f ms → %.1f M keys/s  (includes H2D+D2H)\n",
           wall_ms,   (double)total_keys / wall_ms   / 1e3);
    printf("  No false negatives: %s\n", no_fn ? "PASS ✓" : "FAIL ✗");
    printf("  FPR (measured):     %.4f%%\n",    fpr * 100.0);
    printf("  FPR (theoretical):  %.4f%%  "
           "  [K=%d, bits/key=%d, p=(1-e^{-K/m})^K]\n",
           p_theoretical * 100.0, K, bloom_bits);
}

/* =========================================================================
 * main
 * ======================================================================= */
static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "  --dataset     DIR    path to dataset directory    [default: ./dataset]\n"
        "  --block_size  BYTES  SST data block size (bytes)  [default: 32768]\n"
        "  --key_size    BYTES  key size (bytes)             [default: 16]\n"
        "  --value_size  BYTES  value size (bytes)           [default: 64]\n"
        "  --overhead    BYTES  per-entry SST overhead bytes [default: 20]\n"
        "  --fpr_samples N      non-member samples for FPR   [default: 10000]\n"
        "  --help\n",
        prog);
}

int main(int argc, char* argv[])
{
    std::string dataset_dir  = "dataset";
    int block_size_bytes     = 32768;   /* BLOCK_SIZE from benchmark_common.sh */
    int key_size_bytes       = 16;      /* KEY_SIZE from benchmark_common.sh */
    int value_size_bytes     = 64;      /* middle of VALUE_SIZES range */
    int overhead_bytes       = 20;      /* typical RocksDB block-based table overhead */
    int fpr_samples          = 10000;

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
        else if (a == "--fpr_samples") fpr_samples       = std::stoi(next());
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown option '%s'\n", a.c_str()); return 1; }
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

    /* ---- Load SSTs ---- */
    printf("Loading SST files ...\n");
    std::vector<std::vector<KVPair>> ssts(meta.num_sst);
    for (int s = 0; s < meta.num_sst; ++s) {
        char fname[64];
        snprintf(fname, sizeof(fname), "sst_%04d.bin", s);
        ssts[s] = load_sst_bin(dataset_dir + "/" + fname, meta.sst_sizes[s]);
    }
    printf("  Loaded %d SST files\n\n", meta.num_sst);

    /* ================================================================
     * Benchmark 1: Merge
     * ================================================================ */
    bench_merge(ssts, meta);

    /* ================================================================
     * Compute merged output (CPU) – input for bloom benchmark
     * ================================================================ */
    printf("Computing CPU merge for bloom benchmark input ...\n");
    std::vector<KVPair> merged = cpu_merge_reference(ssts);
    printf("  %zu keys in merged output\n\n", merged.size());

    /* ================================================================
     * Benchmark 2: Bloom filter
     * ================================================================ */
    bench_bloom(merged, meta,
                block_size_bytes, key_size_bytes, value_size_bytes,
                overhead_bytes, fpr_samples);

    hr();
    printf("Done.\n");
    return 0;
}
