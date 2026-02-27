/*
 * gpcomp_tests.cu
 *
 * Unit-test suite for the GPComp CUDA kernels.
 *
 *   Kernel 1 – merge_kernel     (Algorithm 1, gpcomp_merge.cuh)
 *   Kernel 2 – bloom_filter_kernel (Algorithm 2, gpcomp_bloom.cuh)
 *
 * Both headers are included directly so this file compiles as a single,
 * self-contained translation unit with no external link dependencies.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -o gpcomp_unit_tests gpcomp_tests.cu
 *
 * Run:
 *   ./gpcomp_unit_tests
 *
 * A non-zero exit code means at least one assertion failed.
 */

#include "gpcomp_merge.cuh"
#include "gpcomp_bloom.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* =========================================================================
 * Lightweight test framework
 * ====================================================================== */

static int s_passed  = 0;
static int s_failed  = 0;
static int s_skipped = 0;

/* Evaluate `cond`.  Print PASS/FAIL with a description and update counters.
 * Returns true if the condition held. */
static bool check(bool condition, const char *description)
{
    if (condition) {
        printf("    [PASS] %s\n", description);
        ++s_passed;
    } else {
        printf("    [FAIL] %s\n", description);
        ++s_failed;
    }
    return condition;
}

/* Thin wrapper so CHECK() can be written in-line inside test functions. */
#define CHECK(cond, msg)  check(!!(cond), (msg))

/* Run one named test function and print a header/footer around it. */
static void run_test(const char *name, bool (*fn)())
{
    printf("\n--- %s\n", name);
    bool ok = fn();
    printf("    => %s\n", ok ? "PASSED" : "FAILED");
}

/* =========================================================================
 * Helpers used by multiple tests
 * ====================================================================== */

/* Returns true if the keys in output[0..n-1] are in non-decreasing order. */
static bool is_sorted(const KVPair *arr, int n)
{
    for (int i = 1; i < n; ++i)
        if (arr[i].key < arr[i-1].key) return false;
    return true;
}

/* Returns the value associated with key k in arr[0..n-1], or UINT64_MAX. */
static uint64_t find_value(const KVPair *arr, int n, uint64_t k)
{
    for (int i = 0; i < n; ++i)
        if (arr[i].key == k) return arr[i].value;
    return UINT64_MAX;
}

/* Count set bits in a BitVector of bitvec_len bytes. */
static int count_set_bits(const uint8_t *bv, int bitvec_len)
{
    int count = 0;
    for (int i = 0; i < bitvec_len; ++i)
        for (int j = 0; j < 8; ++j)
            if ((bv[i] >> j) & 1) ++count;
    return count;
}

/* =========================================================================
 * === MERGE TESTS ==========================================================
 * ====================================================================== */

/* --------------------------------------------------------------------- *
 * T-M1: Single array passthrough                                         *
 *                                                                         *
 * Merging a single sorted array should reproduce the input exactly.      *
 * --------------------------------------------------------------------- */
static bool test_merge_passthrough()
{
    KVPair sst0[] = {{1,100},{3,300},{5,500},{7,700},{9,900}};
    const int SZ  = 5;

    KVPair *arrays[1] = {sst0};
    int     sizes[1]  = {SZ};
    KVPair  out[SZ];

    launch_merge(arrays, sizes, 1, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, SZ),             "output is sorted");
    ok &= CHECK(out[0].key == 1  && out[0].value == 100, "pair[0] correct");
    ok &= CHECK(out[2].key == 5  && out[2].value == 500, "pair[2] correct");
    ok &= CHECK(out[4].key == 9  && out[4].value == 900, "pair[4] correct");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M2: Two perfectly interleaved arrays                                 *
 *                                                                         *
 * SST_0 = {1,3,5,7,9}  SST_1 = {2,4,6,8,10} → {1,2,…,10}               *
 * --------------------------------------------------------------------- */
static bool test_merge_two_interleaved()
{
    KVPair sst0[] = {{1,10},{3,30},{5,50},{7,70},{9,90}};
    KVPair sst1[] = {{2,20},{4,40},{6,60},{8,80},{10,100}};
    const int SZ  = 5;

    KVPair *arrays[2] = {sst0, sst1};
    int     sizes[2]  = {SZ, SZ};
    KVPair  out[10];

    launch_merge(arrays, sizes, 2, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, 10), "output is sorted");
    for (int k = 1; k <= 10; ++k)
        ok &= CHECK(out[k-1].key == (uint64_t)k,
                    "sequential keys present");
    /* Value tracking: odd keys come from sst0 (value = key*10). */
    ok &= CHECK(find_value(out, 10, 1) == 10,  "value for key 1  preserved");
    ok &= CHECK(find_value(out, 10, 6) == 60,  "value for key 6  preserved");
    ok &= CHECK(find_value(out, 10, 10) == 100,"value for key 10 preserved");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M3: Three round-robin arrays                                          *
 *                                                                         *
 * SST_0={1,4,7}  SST_1={2,5,8}  SST_2={3,6,9} → {1,…,9}                *
 * --------------------------------------------------------------------- */
static bool test_merge_three_round_robin()
{
    KVPair sst0[] = {{1,10},{4,40},{7,70}};
    KVPair sst1[] = {{2,20},{5,50},{8,80}};
    KVPair sst2[] = {{3,30},{6,60},{9,90}};

    KVPair *arrays[3] = {sst0, sst1, sst2};
    int     sizes[3]  = {3, 3, 3};
    KVPair  out[9];

    launch_merge(arrays, sizes, 3, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, 9), "output is sorted");
    for (int k = 1; k <= 9; ++k)
        ok &= CHECK(out[k-1].key == (uint64_t)k, "key in position");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M4: Unequal array sizes                                               *
 *                                                                         *
 * SST_0={1,10,100} (3)  SST_1={2,3,50,75,99} (5)  SST_2={11,15} (2)    *
 * --------------------------------------------------------------------- */
static bool test_merge_unequal_sizes()
{
    KVPair sst0[] = {{1,1},{10,10},{100,100}};
    KVPair sst1[] = {{2,2},{3,3},{50,50},{75,75},{99,99}};
    KVPair sst2[] = {{11,11},{15,15}};
    const int TOTAL = 3 + 5 + 2;

    KVPair *arrays[3] = {sst0, sst1, sst2};
    int     sizes[3]  = {3, 5, 2};
    KVPair  out[TOTAL];

    launch_merge(arrays, sizes, 3, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, TOTAL), "output is sorted");
    ok &= CHECK(out[0].key == 1,   "smallest key at front");
    ok &= CHECK(out[TOTAL-1].key == 100, "largest key at back");

    /* Value preservation: value == key for all pairs in this test. */
    bool vals_ok = true;
    for (int i = 0; i < TOTAL; ++i)
        if (out[i].key != out[i].value) { vals_ok = false; break; }
    ok &= CHECK(vals_ok, "values track their keys");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M5: Five single-element arrays                                        *
 *                                                                         *
 * Arrays: {5},{3},{1},{4},{2} (each unsorted relative to the others)      *
 * Expected output: {1,2,3,4,5}                                            *
 * --------------------------------------------------------------------- */
static bool test_merge_single_element_arrays()
{
    KVPair sst0[] = {{5,50}};
    KVPair sst1[] = {{3,30}};
    KVPair sst2[] = {{1,10}};
    KVPair sst3[] = {{4,40}};
    KVPair sst4[] = {{2,20}};

    KVPair *arrays[5] = {sst0, sst1, sst2, sst3, sst4};
    int     sizes[5]  = {1, 1, 1, 1, 1};
    KVPair  out[5];

    launch_merge(arrays, sizes, 5, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, 5), "output is sorted");
    for (int k = 1; k <= 5; ++k)
        ok &= CHECK(out[k-1].key == (uint64_t)k &&
                    out[k-1].value == (uint64_t)(k*10),
                    "key-value pair correct");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M6: All elements in one array, empty others                          *
 *                                                                         *
 * SST_0 = {10,20,30,40,50}  SST_1 = {}  SST_2 = {}                      *
 * --------------------------------------------------------------------- */
static bool test_merge_empty_companions()
{
    KVPair sst0[] = {{10,1},{20,2},{30,3},{40,4},{50,5}};
    KVPair sst1[1]; /* unused */
    KVPair sst2[1]; /* unused */

    KVPair *arrays[3] = {sst0, sst1, sst2};
    int     sizes[3]  = {5, 0, 0};
    KVPair  out[5];

    launch_merge(arrays, sizes, 3, out);

    bool ok = true;
    ok &= CHECK(is_sorted(out, 5), "output is sorted");
    ok &= CHECK(out[0].key == 10 && out[4].key == 50, "endpoints correct");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-M7: Large arrays – correctness and value association                  *
 *                                                                         *
 * 3 arrays of 300 elements (arithmetic sequences offset by 3).            *
 * SST_0 keys: 1,4,7,…,898   value = key                                  *
 * SST_1 keys: 2,5,8,…,899   value = key*2                               *
 * SST_2 keys: 3,6,9,…,900   value = key*3                               *
 * Merged: 1,2,…,900 (sorted), with per-source values preserved.          *
 * --------------------------------------------------------------------- */
static bool test_merge_large()
{
    const int SZ    = 300;
    const int TOTAL = 3 * SZ;

    std::vector<KVPair> sst0(SZ), sst1(SZ), sst2(SZ);
    for (int i = 0; i < SZ; ++i) {
        sst0[i] = {(uint64_t)(1 + 3*i), (uint64_t)(1 + 3*i)};
        sst1[i] = {(uint64_t)(2 + 3*i), (uint64_t)2*(2 + 3*i)};
        sst2[i] = {(uint64_t)(3 + 3*i), (uint64_t)3*(3 + 3*i)};
    }

    KVPair *arrays[3] = {sst0.data(), sst1.data(), sst2.data()};
    int     sizes[3]  = {SZ, SZ, SZ};
    std::vector<KVPair> out(TOTAL);

    launch_merge(arrays, sizes, 3, out.data());

    bool ok = true;
    ok &= CHECK(is_sorted(out.data(), TOTAL), "900-pair output is sorted");

    /* Spot-check: key k should be at index k-1, value depends on source. */
    bool vals_ok = true;
    for (int k = 1; k <= TOTAL; ++k) {
        if (out[k-1].key != (uint64_t)k) { vals_ok = false; break; }
        uint64_t expected_val;
        if      (k % 3 == 1) expected_val = (uint64_t)k;       /* SST_0 */
        else if (k % 3 == 2) expected_val = (uint64_t)k * 2;   /* SST_1 */
        else                 expected_val = (uint64_t)k * 3;   /* SST_2 */
        if (out[k-1].value != expected_val) { vals_ok = false; break; }
    }
    ok &= CHECK(vals_ok, "all 900 values correctly track their source array");
    return ok;
}

/* =========================================================================
 * === BLOOM FILTER TESTS ===================================================
 * ====================================================================== */

/* --------------------------------------------------------------------- *
 * T-B1: All inserted keys must be found (soundness / no false negatives)  *
 *                                                                         *
 * K=3, 10 keys, byte_vector_len=64.                                       *
 * --------------------------------------------------------------------- */
static bool test_bloom_all_inserted_present()
{
    const int N   = 10, K = 3, BVL = 64;
    KVPair arr[N];
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1) * 100, 0};

    uint8_t bv[(BVL + 7) / 8] = {};
    launch_bloom_filter(arr, N, K, BVL, bv);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        char msg[64];
        snprintf(msg, sizeof(msg), "key %llu found",
                 (unsigned long long)arr[i].key);
        ok &= CHECK(cpu_bloom_check_bit(bv, BVL, K, arr[i].key), msg);
    }
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B2: Empty input → BitVector must be all-zero bytes                   *
 * --------------------------------------------------------------------- */
static bool test_bloom_empty_input()
{
    const int K = 3, BVL = 64;
    const int bitvec_len = (BVL + 7) / 8;

    uint8_t bv[bitvec_len];
    memset(bv, 0xFF, bitvec_len);              /* pre-fill to detect zeros */

    launch_bloom_filter(nullptr, 0, K, BVL, bv);

    bool all_zero = true;
    for (int i = 0; i < bitvec_len; ++i)
        if (bv[i] != 0) { all_zero = false; break; }

    bool ok = true;
    ok &= CHECK(all_zero, "empty input → all-zero BitVector");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B3: K=1 – exactly one bit set per key; query that bit                 *
 *                                                                         *
 * With K=1 and a single key we can compute the expected bit position on   *
 * the CPU and verify the GPU set exactly that bit.                        *
 * --------------------------------------------------------------------- */
static bool test_bloom_k1_single_key()
{
    const int K = 1, BVL = 128;
    const int bitvec_len = (BVL + 7) / 8;

    KVPair arr[1] = {{42, 0}};
    uint8_t bv[bitvec_len];
    memset(bv, 0, bitvec_len);

    launch_bloom_filter(arr, 1, K, BVL, bv);

    /* CPU oracle: compute expected bit position. */
    uint32_t h   = cpu_bloom_hash(42, 1);
    int      pos = (int)(h % (uint32_t)BVL);

    bool ok = true;
    ok &= CHECK(cpu_bloom_check_bit(bv, BVL, K, 42), "inserted key present");

    /* Verify the exact bit is set. */
    bool bit_set = (bv[pos / 8] >> (pos % 8)) & 1;
    ok &= CHECK(bit_set, "correct bit position is set");

    /* Verify total set-bit count == 1 (K=1 and a single key). */
    ok &= CHECK(count_set_bits(bv, bitvec_len) == 1,
                "exactly 1 bit set for K=1, 1 key");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B4: K=7 – all inserted keys still produce positive lookups           *
 * --------------------------------------------------------------------- */
static bool test_bloom_k7_soundness()
{
    const int N = 20, K = 7, BVL = 512;
    KVPair arr[N];
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1) * 31337, 0};

    uint8_t bv[(BVL + 7) / 8] = {};
    launch_bloom_filter(arr, N, K, BVL, bv);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        char msg[64];
        snprintf(msg, sizeof(msg), "key %llu present (K=7)",
                 (unsigned long long)arr[i].key);
        ok &= CHECK(cpu_bloom_check_bit(bv, BVL, K, arr[i].key), msg);
    }
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B5: Bit-packing correctness – GPU output must exactly match CPU      *
 *        oracle (cpu_build_byte_vector + cpu_pack_bit_vector)            *
 *                                                                         *
 * Tests both the hash/ByteVector phase and the compaction phase in one   *
 * shot by doing a byte-level comparison.                                  *
 * --------------------------------------------------------------------- */
static bool test_bloom_bit_packing_oracle()
{
    const int N = 25, K = 3, BVL = 128;
    const int bitvec_len = (BVL + 7) / 8;

    KVPair arr[N];
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1) * 997, 0};

    /* GPU result. */
    uint8_t gpu_bv[bitvec_len];
    memset(gpu_bv, 0, bitvec_len);
    launch_bloom_filter(arr, N, K, BVL, gpu_bv);

    /* CPU oracle. */
    std::vector<uint8_t> byte_vec(BVL);
    uint8_t cpu_bv[bitvec_len];
    cpu_build_byte_vector(arr, N, K, BVL, byte_vec.data());
    cpu_pack_bit_vector(byte_vec.data(), BVL, cpu_bv);

    bool match = (memcmp(gpu_bv, cpu_bv, bitvec_len) == 0);
    bool ok    = CHECK(match, "GPU BitVector == CPU oracle BitVector");

    if (!match) {
        /* Print mismatching bytes for diagnostics. */
        for (int i = 0; i < bitvec_len; ++i)
            if (gpu_bv[i] != cpu_bv[i])
                printf("      byte[%d]: gpu=0x%02x  cpu=0x%02x\n",
                       i, gpu_bv[i], cpu_bv[i]);
    }
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B6: Non-power-of-two byte_vector_len – boundary handling             *
 *                                                                         *
 * byte_vector_len=100 → 13 full bytes (104 bits) with top 4 bits padded. *
 * The last BitVector byte should only reflect ByteVector[96..99];        *
 * bits [4..7] of byte[12] must be zero.                                  *
 * --------------------------------------------------------------------- */
static bool test_bloom_non_pow2_width()
{
    const int N = 15, K = 3, BVL = 100;
    const int bitvec_len = (BVL + 7) / 8;   /* = 13 */

    KVPair arr[N];
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1) * 1009, 0};

    uint8_t gpu_bv[bitvec_len];
    memset(gpu_bv, 0, bitvec_len);
    launch_bloom_filter(arr, N, K, BVL, gpu_bv);

    /* CPU oracle. */
    std::vector<uint8_t> byte_vec(BVL);
    uint8_t cpu_bv[bitvec_len];
    cpu_build_byte_vector(arr, N, K, BVL, byte_vec.data());
    cpu_pack_bit_vector(byte_vec.data(), BVL, cpu_bv);

    bool ok = true;
    ok &= CHECK(memcmp(gpu_bv, cpu_bv, bitvec_len) == 0,
                "non-pow2 width: GPU == CPU oracle");
    /* Padding bits [4..7] of the last byte must be 0. */
    ok &= CHECK((gpu_bv[bitvec_len - 1] & 0xF0) == 0,
                "padding bits in last byte are zero");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B7: Large byte_vector_len = 1024 bits (128-byte BitVector)           *
 * --------------------------------------------------------------------- */
static bool test_bloom_large_width()
{
    const int N = 50, K = 5, BVL = 1024;
    const int bitvec_len = (BVL + 7) / 8;

    std::vector<KVPair> arr(N);
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1) * 7919, 0};

    std::vector<uint8_t> gpu_bv(bitvec_len, 0);
    launch_bloom_filter(arr.data(), N, K, BVL, gpu_bv.data());

    /* Oracle. */
    std::vector<uint8_t> byte_vec(BVL);
    std::vector<uint8_t> cpu_bv(bitvec_len);
    cpu_build_byte_vector(arr.data(), N, K, BVL, byte_vec.data());
    cpu_pack_bit_vector(byte_vec.data(), BVL, cpu_bv.data());

    bool ok = true;
    ok &= CHECK(memcmp(gpu_bv.data(), cpu_bv.data(), bitvec_len) == 0,
                "1024-bit filter: GPU == CPU oracle");
    /* Soundness: all inserted keys must be found. */
    for (int i = 0; i < N; ++i)
        ok &= CHECK(cpu_bloom_check_bit(gpu_bv.data(), BVL, K, arr[i].key),
                    "large-width: inserted key present");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B8: Filter saturation – insert many keys until all bits set          *
 *                                                                         *
 * With BVL=32 bits and 1000 keys * 3 hashes, every slot is virtually     *
 * guaranteed to be set → BitVector must be 0xFF for all 4 bytes.         *
 * --------------------------------------------------------------------- */
static bool test_bloom_full_saturation()
{
    const int N = 1000, K = 3, BVL = 32;
    const int bitvec_len = (BVL + 7) / 8;

    std::vector<KVPair> arr(N);
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1), 0};

    std::vector<uint8_t> bv(bitvec_len, 0);
    launch_bloom_filter(arr.data(), N, K, BVL, bv.data());

    bool all_set = true;
    for (int i = 0; i < bitvec_len; ++i)
        if (bv[i] != 0xFF) { all_set = false; break; }

    bool ok = CHECK(all_set, "all 32 bits set after 1000-key saturation");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B9: False positive rate is within theoretical bound                   *
 *                                                                         *
 * n=100 inserted keys, m=2000 bits wide, K=7.                             *
 * Theory: FPR ≈ (1 - e^{-K·n/m})^K ≈ 0.03% → expect ≪ 5% on 1000 probes *
 * --------------------------------------------------------------------- */
static bool test_bloom_false_positive_rate()
{
    const int N    = 100;
    const int K    = 7;
    const int BVL  = 2000;   /* bits */
    const int PROBES = 1000;
    const int bitvec_len = (BVL + 7) / 8;

    /* Insert keys 1..N. */
    std::vector<KVPair> arr(N);
    for (int i = 0; i < N; ++i) arr[i] = {(uint64_t)(i + 1), 0};

    std::vector<uint8_t> bv(bitvec_len, 0);
    launch_bloom_filter(arr.data(), N, K, BVL, bv.data());

    /* Soundness: all N inserted keys must be found. */
    bool ok = true;
    for (int i = 0; i < N; ++i)
        ok &= CHECK(cpu_bloom_check_bit(bv.data(), BVL, K, arr[i].key),
                    "FPR test: inserted key present");

    /* Query PROBES absent keys (keys starting far beyond the inserted set). */
    int fp_count = 0;
    for (int i = 0; i < PROBES; ++i) {
        uint64_t absent_key = (uint64_t)(100001 + i);
        if (cpu_bloom_check_bit(bv.data(), BVL, K, absent_key)) ++fp_count;
    }

    double fpr = (double)fp_count / PROBES;
    printf("    FPR: %d/%d = %.4f%%  (threshold 5%%)\n",
           fp_count, PROBES, fpr * 100.0);
    ok &= CHECK(fpr < 0.05, "false positive rate < 5%");
    return ok;
}

/* --------------------------------------------------------------------- *
 * T-B10: Single key, K=3 – verify exactly K≤3 bits set                  *
 *                                                                         *
 * With one key and K=3, the GPU sets at most 3 bit-positions (fewer if   *
 * any two of the K hashes collide to the same slot).                     *
 * --------------------------------------------------------------------- */
static bool test_bloom_single_key_k3()
{
    const int K = 3, BVL = 256;
    const int bitvec_len = (BVL + 7) / 8;

    KVPair arr[1] = {{12345678ULL, 0}};
    uint8_t bv[bitvec_len];
    memset(bv, 0, bitvec_len);

    launch_bloom_filter(arr, 1, K, BVL, bv);

    int bits = count_set_bits(bv, bitvec_len);

    bool ok = true;
    ok &= CHECK(bits >= 1 && bits <= K,
                "set-bit count in [1, K] for single key");
    ok &= CHECK(cpu_bloom_check_bit(bv, BVL, K, arr[0].key),
                "single key is found after insertion");
    return ok;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main()
{
    printf("==========================================================\n");
    printf("  GPComp CUDA Kernel Unit Tests\n");
    printf("==========================================================\n");

    /* Merge kernel tests */
    run_test("T-M1 | Merge: single-array passthrough",       test_merge_passthrough);
    run_test("T-M2 | Merge: two interleaved arrays",         test_merge_two_interleaved);
    run_test("T-M3 | Merge: three round-robin arrays",       test_merge_three_round_robin);
    run_test("T-M4 | Merge: unequal array sizes",            test_merge_unequal_sizes);
    run_test("T-M5 | Merge: five single-element arrays",     test_merge_single_element_arrays);
    run_test("T-M6 | Merge: empty companion arrays",         test_merge_empty_companions);
    run_test("T-M7 | Merge: large arrays (900 pairs total)", test_merge_large);

    /* Bloom filter tests */
    run_test("T-B1 | Bloom: all inserted keys present (K=3)",   test_bloom_all_inserted_present);
    run_test("T-B2 | Bloom: empty input → all-zero BitVector",  test_bloom_empty_input);
    run_test("T-B3 | Bloom: K=1, single key, exact bit check",  test_bloom_k1_single_key);
    run_test("T-B4 | Bloom: K=7 soundness (20 keys)",           test_bloom_k7_soundness);
    run_test("T-B5 | Bloom: bit-packing vs CPU oracle",         test_bloom_bit_packing_oracle);
    run_test("T-B6 | Bloom: non-power-of-two width (100 bits)", test_bloom_non_pow2_width);
    run_test("T-B7 | Bloom: large width (1024 bits)",           test_bloom_large_width);
    run_test("T-B8 | Bloom: full saturation",                   test_bloom_full_saturation);
    run_test("T-B9 | Bloom: false positive rate < 5%",          test_bloom_false_positive_rate);
    run_test("T-B10| Bloom: single key K=3, bit-count check",   test_bloom_single_key_k3);

    printf("\n==========================================================\n");
    printf("  Results: %d passed   %d failed   %d skipped\n",
           s_passed, s_failed, s_skipped);
    printf("==========================================================\n");

    return (s_failed == 0) ? 0 : 1;
}
