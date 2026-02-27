/*
 * gpcomp_merge.cu  –  driver / self-test for Kernel 1.
 *
 * All kernel logic lives in gpcomp_merge.cuh.
 * Build: nvcc -O2 -std=c++17 -o merge_test gpcomp_merge.cu
 */

#include "gpcomp_merge.cuh"
#include <cstdio>

int main()
{
    /*
     * Three sorted SST arrays to merge:
     *   SST_0 : keys { 1, 4, 7 }
     *   SST_1 : keys { 2, 5, 8 }
     *   SST_2 : keys { 3, 6, 9 }
     * Expected merged order: 1 2 3 4 5 6 7 8 9
     */
    const int N = 3, SZ = 3;
    KVPair sst0[SZ] = {{1,10},{4,40},{7,70}};
    KVPair sst1[SZ] = {{2,20},{5,50},{8,80}};
    KVPair sst2[SZ] = {{3,30},{6,60},{9,90}};

    KVPair *h_arrays[N] = {sst0, sst1, sst2};
    int    sizes[N]     = {SZ, SZ, SZ};
    KVPair output[N * SZ];

    launch_merge(h_arrays, sizes, N, output);

    printf("Merged output (%d pairs):\n", N * SZ);
    bool ok = true;
    for (int i = 0; i < N * SZ; ++i) {
        printf("  [%d] key=%llu  value=%llu\n",
               i, (unsigned long long)output[i].key,
                  (unsigned long long)output[i].value);
        if (i > 0 && output[i].key < output[i-1].key) {
            ok = false;
            fprintf(stderr, "  ERROR: out-of-order at index %d\n", i);
        }
    }
    printf("Merge %s.\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
