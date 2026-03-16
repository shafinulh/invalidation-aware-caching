/*
 * gpcomp_bloom.cu  –  driver / self-test for Kernel 2.
 *
 * All kernel logic lives in gpcomp_bloom.cuh.
 * Build: nvcc -O2 -std=c++17 -o bloom_test gpcomp_bloom.cu
 */

#include "gpcomp_bloom.cuh"
#include <cstdio>
#include <cstring>

int main()
{
    /* Insert keys 10, 20, 30, 40, 50 into a Bloom filter. */
    const int ARRAY_SIZE    = 5;
    const int K             = 3;     /* hash functions */
    const int BYTE_VEC_LEN  = 64;    /* 64 bit-slots → 8 byte BitVector */

    KVPair h_array[ARRAY_SIZE] = {
        {10, 0}, {20, 0}, {30, 0}, {40, 0}, {50, 0}
    };

    int     bit_vector_len = (BYTE_VEC_LEN + 7) / 8;
    uint8_t h_bit_vector[8];
    memset(h_bit_vector, 0, sizeof(h_bit_vector));

    launch_bloom_filter(h_array, ARRAY_SIZE, K, BYTE_VEC_LEN, h_bit_vector);

    printf("BitVector (%d bytes): ", bit_vector_len);
    for (int i = 0; i < bit_vector_len; ++i)
        printf("%02x ", h_bit_vector[i]);
    printf("\n");

    /* Positive queries – must all report present. */
    uint64_t inserted_keys[]  = {10, 20, 30, 40, 50};
    /* Negative queries – should (usually) report absent. */
    uint64_t absent_keys[]    = {1, 3, 7, 11, 99};
    bool all_positive_ok = true;

    printf("\nPositive lookups (all should be PRESENT):\n");
    for (uint64_t k : inserted_keys) {
        int hit = cpu_bloom_check_bit(h_bit_vector, BYTE_VEC_LEN, K, k);
        printf("  key=%llu -> %s\n", (unsigned long long)k,
               hit ? "PRESENT" : "ABSENT (ERROR)");
        if (!hit) all_positive_ok = false;
    }

    printf("\nNegative lookups (false positives possible but unlikely):\n");
    for (uint64_t k : absent_keys) {
        int hit = cpu_bloom_check_bit(h_bit_vector, BYTE_VEC_LEN, K, k);
        printf("  key=%llu -> %s\n", (unsigned long long)k,
               hit ? "false positive" : "absent (correct)");
    }

    printf("\nBloom filter self-test %s.\n",
           all_positive_ok ? "PASSED" : "FAILED");
    return all_positive_ok ? 0 : 1;
}
