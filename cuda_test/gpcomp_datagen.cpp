/*
 * gpcomp_datagen.cpp
 *
 * Synthetic SST dataset generator that mimics db_bench fillrandom behavior.
 *
 * Compaction scenario modelled
 * -----------------------------
 * RocksDB with the GP-Comp paper settings flushes each ~8 MB memtable to an
 * L0 SST file.  With LEVEL0_FILE_NUM_COMPACTION_TRIGGER=4 the compaction
 * scheduler fires when 4 L0 files accumulate and merges them into L1.
 * Algorithm 1 of GPComp accelerates exactly this merge step.
 *
 * db_bench fillrandom key generation
 * ------------------------------------
 *   key = thread->rand.Next() % FLAGS_num          (random uniform integer)
 *   formatted as a zero-padded decimal string of length KEY_SIZE=16
 *
 * We model this as a random uniform draw of uint64_t integers from
 * [0, key_space).  Each write gets a unique sequence number in real RocksDB,
 * so two writes with the same user key produce two *different* internal keys.
 * We replicate this by generating globally unique keys spread across SSTs.
 *
 * Layout of keys across SSTs
 * ---------------------------
 * Keys are assigned to SSTs uniformly at random (not partitioned by range) so
 * each SST file covers approximately the full key range – exactly what L0 files
 * look like before compaction.  Within each SST the keys are sorted ascending.
 *
 * Output
 * ------
 *   <out_dir>/dataset.meta      – text parameter summary
 *   <out_dir>/sst_NNNN.bin      – raw KVPair array (sorted, binary)
 *
 * KVPair layout (matches gpcomp_common.cuh)
 * ------------------------------------------
 *   struct KVPair { uint64_t key; uint64_t value; };   // 16 bytes
 *
 * The value field stores a fast deterministic hash of (key, sst_id) so that
 * merge-correctness checks can verify no entry was dropped or corrupted.
 *
 * Usage
 * -----
 *   ./gpcomp_datagen [options]
 *
 *   --num_sst       N     number of input SST files    [default: 4]
 *   --keys_per_sst  N     writes per SST before sort   [default: 81920]
 *   --key_space     N     draw keys from [0, N)        [default: auto = 10×total]
 *   --seed          N     PRNG seed                    [default: 23]
 *   --bloom_bits    N     bits-per-key for bloom meta  [default: 10]
 *   --out_dir       DIR   output directory             [default: ./dataset]
 *   --large               preset: 8 SSTs × 524288 keys (stress test)
 *   --help
 *
 * Default parameters match GP-Comp paper settings from benchmark_common.sh:
 *   write_buffer_size=8MB, target_file_size_base=8MB, value_size=64,
 *   level0_file_num_compaction_trigger=4, bloom_bits=10, seed=23
 *
 *   keys_per_sst = floor(8 MB / (key_size + value_size + overhead))
 *               ≈ floor(8 388 608 / (16 + 64 + 20)) = 83 886  → rounded to 81 920
 */

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>

/* -------------------------------------------------------------------------
 * KVPair – must match gpcomp_common.cuh exactly
 * ---------------------------------------------------------------------- */
struct KVPair {
    uint64_t key;
    uint64_t value;
};

/* -------------------------------------------------------------------------
 * Deterministic value hash: mirrors what a real SST value would contain.
 * Uses Knuth multiplicative hash mixed with the SST id.
 * ---------------------------------------------------------------------- */
static inline uint64_t make_value(uint64_t key, int sst_id)
{
    uint64_t h = key * 0x9e3779b97f4a7c15ULL ^ ((uint64_t)sst_id * 0xdeadbeefcafe1234ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

/* -------------------------------------------------------------------------
 * mkdir -p (single level, or already exists)
 * ---------------------------------------------------------------------- */
static void mkdir_p(const std::string& dir)
{
    struct stat st{};
    if (stat(dir.c_str(), &st) == 0) return;       // already exists
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: cannot create directory '%s': %s\n",
                dir.c_str(), strerror(errno));
        exit(1);
    }
}

/* -------------------------------------------------------------------------
 * Write a binary array of KVPairs to a file
 * ---------------------------------------------------------------------- */
static void write_sst_bin(const std::string& path,
                          const std::vector<KVPair>& pairs)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s' for writing: %s\n",
                path.c_str(), strerror(errno));
        exit(1);
    }
    size_t written = fwrite(pairs.data(), sizeof(KVPair), pairs.size(), f);
    if (written != pairs.size()) {
        fprintf(stderr, "error: short write to '%s' (%zu / %zu pairs)\n",
                path.c_str(), written, pairs.size());
        exit(1);
    }
    fclose(f);
}

/* -------------------------------------------------------------------------
 * Print usage
 * ---------------------------------------------------------------------- */
static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --num_sst      N    number of input SST files    [default: 4]\n"
        "  --keys_per_sst N    keys written per SST         [default: 81920]\n"
        "  --key_space    N    draw keys from [0, N)        [default: 10×total]\n"
        "  --seed         N    PRNG seed                    [default: 23]\n"
        "  --bloom_bits   N    bits-per-key stored in meta  [default: 10]\n"
        "  --out_dir      DIR  output directory             [default: ./dataset]\n"
        "  --large             preset: 8 SSTs x 524288 keys (stress test)\n"
        "  --help              print this message\n"
        "\n"
        "Defaults match GP-Comp paper settings:\n"
        "  write_buffer_size=8MB, target_file_size_base=8MB, value_size=64,\n"
        "  level0_file_num_compaction_trigger=4, bloom_bits=10, seed=23\n",
        prog);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    /* ---- defaults matching GP-Comp / benchmark_common.sh ---- */
    int         num_sst      = 4;
    int         keys_per_sst = 81920;   // 8MB SST @ key=16B value=64B overhead=20B
    uint64_t    key_space    = 0;       // 0 = auto (10× total keys)
    uint64_t    seed         = 23;      // SEED in benchmark_common.sh
    int         bloom_bits   = 10;      // BLOOM_BITS in benchmark_common.sh
    std::string out_dir      = "dataset";

    /* ---- parse CLI ---- */
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };

        if (a == "--num_sst")           num_sst      = std::stoi(next());
        else if (a == "--keys_per_sst") keys_per_sst = std::stoi(next());
        else if (a == "--key_space")    key_space    = std::stoull(next());
        else if (a == "--seed")         seed         = std::stoull(next());
        else if (a == "--bloom_bits")   bloom_bits   = std::stoi(next());
        else if (a == "--out_dir")      out_dir      = next();
        else if (a == "--large") {
            num_sst      = 8;
            keys_per_sst = 524288;   // 512K × 8 = 4M keys
        }
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown option '%s'\n", a.c_str()); return 1; }
    }

    if (num_sst <= 0 || keys_per_sst <= 0) {
        fprintf(stderr, "error: num_sst and keys_per_sst must be positive\n");
        return 1;
    }

    const uint64_t total_keys = (uint64_t)num_sst * keys_per_sst;

    /* key_space must be at least total_keys (need unique keys) */
    if (key_space == 0)
        key_space = total_keys * 10;   // 10% fill rate, like a production db
    if (key_space < total_keys) {
        fprintf(stderr, "error: key_space (%llu) < total_keys (%llu). "
                "Cannot generate enough unique keys.\n",
                (unsigned long long)key_space,
                (unsigned long long)total_keys);
        return 1;
    }

    /* ---- bloom filter derived parameters (stored in meta, used by bench) ---- */
    /* optimal K for a given bits-per-key: K = bits_per_key * ln(2) ≈ 0.693 */
    const int bloom_K = std::max(1, (int)((double)bloom_bits * 0.6931 + 0.5));

    /* ---- header ---- */
    printf("GPComp dataset generator – mimicking db_bench fillrandom\n");
    printf("  GP-Comp paper settings: write_buffer=8MB, target_file=8MB,\n");
    printf("                          value_size~64B, L0_trigger=%d, bloom_bits=%d\n",
           num_sst, bloom_bits);
    printf("\n");
    printf("  num_sst      = %d\n",              num_sst);
    printf("  keys_per_sst = %d  (writes before flush)\n", keys_per_sst);
    printf("  total_keys   = %llu\n",            (unsigned long long)total_keys);
    printf("  key_space    = %llu  (%.1f%% fill)\n",
           (unsigned long long)key_space,
           100.0 * (double)total_keys / (double)key_space);
    printf("  seed         = %llu\n",            (unsigned long long)seed);
    printf("  bloom_bits   = %d  →  K = %d\n",  bloom_bits, bloom_K);
    printf("  out_dir      = %s\n\n",            out_dir.c_str());
    fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();

    /* ================================================================
     * Step 1 – Sample total_keys UNIQUE keys from [0, key_space)
     *
     * For large key_space vs total_keys (≥10× here) Fisher-Yates on a
     * dense vector wastes memory.  We use reservoir sampling via a hash
     * set: draw k random values, retry on collision.  The expected number
     * of retries is < total_keys/(key_space-total_keys) × total_keys < 0.1×
     * total_keys for our default 10× ratio.
     * ================================================================ */
    printf("Step 1/3  Sampling %llu unique keys from [0, %llu) ...\n",
           (unsigned long long)total_keys, (unsigned long long)key_space);
    fflush(stdout);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> key_dist(0, key_space - 1);

    /* Pre-allocate a flat vector; we'll assign to SSTs in the next step. */
    std::vector<uint64_t> all_keys;
    all_keys.reserve(total_keys);

    {
        /* Use an unordered_set for O(1) duplicate detection.
         * Load factor 0.5 ⇒ ~total_keys * 16 bytes of overhead. For 4M keys
         * that's ~64 MB, acceptable.  For --large (4M keys) same estimate. */
        std::unordered_set<uint64_t> seen;
        seen.reserve((size_t)(total_keys * 2));   // load factor ~0.5

        while (all_keys.size() < total_keys) {
            uint64_t k = key_dist(rng);
            if (seen.insert(k).second)
                all_keys.push_back(k);
        }
    }

    /* ================================================================
     * Step 2 – Assign keys to SSTs and sort each SST
     *
     * Keys are randomly shuffled then split into equal slices
     * (slice i = [i*keys_per_sst, (i+1)*keys_per_sst)).
     *
     * This means every SST covers approximately the full key range,
     * exactly like L0 SST files just before a compaction.  Each key
     * still appears in exactly one SST (unique internal key invariant).
     * ================================================================ */
    printf("Step 2/3  Shuffling and partitioning into %d SSTs ...\n", num_sst);
    fflush(stdout);

    std::shuffle(all_keys.begin(), all_keys.end(), rng);

    /* Build sorted KVPair arrays per SST */
    std::vector<std::vector<KVPair>> sst_arrays(num_sst);
    for (int s = 0; s < num_sst; ++s) {
        int start = s * keys_per_sst;
        int end   = start + keys_per_sst;
        sst_arrays[s].resize(keys_per_sst);
        for (int i = start; i < end; ++i)
            sst_arrays[s][i - start] = { all_keys[i], make_value(all_keys[i], s) };
        std::sort(sst_arrays[s].begin(), sst_arrays[s].end(),
                  [](const KVPair& a, const KVPair& b){ return a.key < b.key; });
    }

    /* ================================================================
     * Step 3 – Write dataset to disk
     * ================================================================ */
    printf("Step 3/3  Writing dataset to '%s/' ...\n", out_dir.c_str());
    fflush(stdout);

    mkdir_p(out_dir);

    /* Write per-SST binary files */
    for (int s = 0; s < num_sst; ++s) {
        char fname[64];
        snprintf(fname, sizeof(fname), "sst_%04d.bin", s);
        write_sst_bin(out_dir + "/" + fname, sst_arrays[s]);
    }

    /* Write metadata text file */
    {
        std::string meta_path = out_dir + "/dataset.meta";
        FILE* mf = fopen(meta_path.c_str(), "w");
        if (!mf) {
            fprintf(stderr, "error: cannot write '%s': %s\n",
                    meta_path.c_str(), strerror(errno));
            return 1;
        }
        fprintf(mf, "# GPComp synthetic dataset – generated by gpcomp_datagen\n");
        fprintf(mf, "# Mimics db_bench fillrandom with GP-Comp paper settings\n");
        fprintf(mf, "#\n");
        fprintf(mf, "# GP-Comp paper settings used:\n");
        fprintf(mf, "#   write_buffer_size  = 8388608  (8 MB)\n");
        fprintf(mf, "#   target_file_size   = 8388608  (8 MB)\n");
        fprintf(mf, "#   level0_trigger     = %d\n", num_sst);
        fprintf(mf, "#   bloom_bits         = %d\n", bloom_bits);
        fprintf(mf, "#   seed               = %llu\n", (unsigned long long)seed);
        fprintf(mf, "#\n");
        fprintf(mf, "num_sst=%d\n",                    num_sst);
        fprintf(mf, "keys_per_sst=%d\n",               keys_per_sst);
        fprintf(mf, "total_keys=%llu\n",               (unsigned long long)total_keys);
        fprintf(mf, "key_space=%llu\n",                (unsigned long long)key_space);
        fprintf(mf, "seed=%llu\n",                     (unsigned long long)seed);
        fprintf(mf, "bloom_bits=%d\n",                 bloom_bits);
        fprintf(mf, "bloom_K=%d\n",                    bloom_K);
        /* byte_vector_len for the bloom filter kernel:
         * number of bits = total_keys * bloom_bits.
         * byte_vector_len = that many bytes (ByteVector, 1 byte per bit slot). */
        fprintf(mf, "bloom_byte_vector_len=%llu\n",
                    (unsigned long long)total_keys * bloom_bits);
        for (int s = 0; s < num_sst; ++s)
            fprintf(mf, "sst_%04d_size=%d\n", s, (int)sst_arrays[s].size());
        fclose(mf);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    /* ---- summary ---- */
    uint64_t disk_bytes = total_keys * sizeof(KVPair);
    printf("\nDone in %.2f s\n", elapsed);
    printf("  Dataset written to  %s/\n", out_dir.c_str());
    printf("  Files:              %d sst_NNNN.bin  +  dataset.meta\n", num_sst);
    printf("  SST size on disk:   %.1f MB each  (raw KVPair binary)\n",
           (double)sst_arrays[0].size() * sizeof(KVPair) / (1 << 20));
    printf("  Total data:         %.1f MB  (%llu KVPairs × 16 B)\n",
           (double)disk_bytes / (1 << 20), (unsigned long long)total_keys);
    printf("\nKey statistics (simulating L0 compaction overlap):\n");
    /* Show that SST key ranges overlap like real L0 files */
    uint64_t min0 = sst_arrays[0].front().key, max0 = sst_arrays[0].back().key;
    uint64_t min1 = sst_arrays[1].front().key, max1 = sst_arrays[1].back().key;
    printf("  SST 0 key range:    [%llu, %llu]\n",
           (unsigned long long)min0, (unsigned long long)max0);
    printf("  SST 1 key range:    [%llu, %llu]\n",
           (unsigned long long)min1, (unsigned long long)max1);
    if (num_sst > 2) {
        uint64_t min2 = sst_arrays[2].front().key, max2 = sst_arrays[2].back().key;
        printf("  SST 2 key range:    [%llu, %llu]\n",
               (unsigned long long)min2, (unsigned long long)max2);
    }
    printf("  (Ranges overlap like real L0 files – binary search is fully exercised)\n");
    printf("\nRun gpcomp_bench to benchmark the kernels on this dataset.\n");

    return 0;
}
