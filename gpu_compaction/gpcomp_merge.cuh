#pragma once

#include "gpcomp_common.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

__device__ static int device_key_compare(const Key128& a, const Key128& b)
{
    for (int i = 0; i < GP_KEY_BYTES; ++i) {
        if (a.bytes[i] < b.bytes[i]) return -1;
        if (a.bytes[i] > b.bytes[i]) return 1;
    }
    return 0;
}

__device__ static int binary_search_lower_bound(const KVPair* arr,
                                                int           size,
                                                const Key128& query_key)
{
    int lo = 0;
    int hi = size;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (device_key_compare(arr[mid].key, query_key) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ static int binary_search_lower_bound_ref(const KVRef* arr,
                                                    int          size,
                                                    const Key128& query_key)
{
    int lo = 0;
    int hi = size;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (device_key_compare(arr[mid].key, query_key) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* Algorithm 1 from the paper: one thread computes the final rank of one KV. */
__global__ void merge_kernel(KVPair* const* __restrict__ sst_arrays,
                             const int*     __restrict__ sst_sizes,
                             const int*     __restrict__ sst_offsets,
                             int                          num_arrays,
                             KVPair*       __restrict__  output)
{
    int global_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = sst_offsets[num_arrays];
    if (global_idx >= total) return;

    int array_id = 0;
    while (array_id + 1 < num_arrays && global_idx >= sst_offsets[array_id + 1]) ++array_id;

    int local_idx = global_idx - sst_offsets[array_id];
    KVPair pair = sst_arrays[array_id][local_idx];

    int final_idx = 0;
    for (int i = 0; i < num_arrays; ++i) {
        int rank = (i == array_id)
                 ? local_idx
                 : binary_search_lower_bound(sst_arrays[i], sst_sizes[i], pair.key);
        final_idx += rank;
    }

    output[final_idx] = pair;
}

__global__ void merge_ref_kernel(KVRef* const* __restrict__ sst_arrays,
                                 const int*    __restrict__ sst_sizes,
                                 const int*    __restrict__ sst_offsets,
                                 int                         num_arrays,
                                 KVRef*      __restrict__   output)
{
    int global_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = sst_offsets[num_arrays];
    if (global_idx >= total) return;

    int array_id = 0;
    while (array_id + 1 < num_arrays && global_idx >= sst_offsets[array_id + 1]) ++array_id;

    int local_idx = global_idx - sst_offsets[array_id];
    KVRef ref = sst_arrays[array_id][local_idx];

    int final_idx = 0;
    for (int i = 0; i < num_arrays; ++i) {
        int rank = (i == array_id)
                 ? local_idx
                 : binary_search_lower_bound_ref(sst_arrays[i], sst_sizes[i], ref.key);
        final_idx += rank;
    }

    output[final_idx] = ref;
}

static inline int launch_merge(KVPair* const* h_sst_arrays,
                               const int*      h_sst_sizes,
                               int             num_arrays,
                               KVPair*         h_output)
{
    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i) h_offsets[i + 1] = h_offsets[i] + h_sst_sizes[i];
    int total = h_offsets[num_arrays];

    std::vector<KVPair*> d_host_ptrs((size_t)num_arrays, nullptr);
    for (int i = 0; i < num_arrays; ++i) {
        cudaMalloc(&d_host_ptrs[i], (size_t)std::max(h_sst_sizes[i], 1) * sizeof(KVPair));
        if (h_sst_sizes[i] > 0) {
            cudaMemcpy(d_host_ptrs[i], h_sst_arrays[i],
                       (size_t)h_sst_sizes[i] * sizeof(KVPair), cudaMemcpyHostToDevice);
        }
    }

    KVPair** d_sst_arrays = nullptr;
    int*     d_sst_sizes = nullptr;
    int*     d_sst_offsets = nullptr;
    KVPair*  d_output = nullptr;

    cudaMalloc(&d_sst_arrays, (size_t)num_arrays * sizeof(KVPair*));
    cudaMalloc(&d_sst_sizes, (size_t)num_arrays * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)num_arrays + 1) * sizeof(int));
    cudaMalloc(&d_output, (size_t)std::max(total, 1) * sizeof(KVPair));

    cudaMemcpy(d_sst_arrays, d_host_ptrs.data(),
               (size_t)num_arrays * sizeof(KVPair*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sst_sizes, h_sst_sizes,
               (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sst_offsets, h_offsets.data(),
               ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (total + block - 1) / block;
    merge_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[merge_kernel] launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    if (total > 0) {
        cudaMemcpy(h_output, d_output, (size_t)total * sizeof(KVPair), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_output);
    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    for (KVPair* ptr : d_host_ptrs) cudaFree(ptr);

    return (int)err;
}

struct MergeTimedResult {
    std::vector<KVPair> merged;
    float               kernel_ms = 0.0f;
    float               wall_ms = 0.0f;
    float               h2d_ms = 0.0f;
    float               d2h_ms = 0.0f;
    size_t              h2d_bytes = 0;
    size_t              d2h_bytes = 0;
};

struct DeviceMergeTimedResult {
    std::vector<KVPair> merged;
    KVPair*             d_output = nullptr;
    int                 total = 0;
    float               kernel_ms = 0.0f;
    float               wall_ms = 0.0f;
    float               h2d_ms = 0.0f;
    float               d2h_ms = 0.0f;
    size_t              h2d_bytes = 0;
    size_t              d2h_bytes = 0;
};

struct DeviceMergeRefTimedResult {
    std::vector<KVRef> merged;
    KVRef*             d_output = nullptr;
    int                total = 0;
    float              kernel_ms = 0.0f;
    float              wall_ms = 0.0f;
    float              h2d_ms = 0.0f;
    float              d2h_ms = 0.0f;
    size_t             h2d_bytes = 0;
    size_t             d2h_bytes = 0;
};

static inline MergeTimedResult launch_merge_timed(const std::vector<std::vector<KVPair>>& arrays)
{
    MergeTimedResult result;
    int num_arrays = (int)arrays.size();
    std::vector<KVPair*> ptrs((size_t)num_arrays);
    std::vector<int> sizes((size_t)num_arrays);
    int total = 0;
    for (int i = 0; i < num_arrays; ++i) {
        ptrs[i] = const_cast<KVPair*>(arrays[i].data());
        sizes[i] = (int)arrays[i].size();
        total += sizes[i];
    }
    result.merged.resize((size_t)total);

    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i) h_offsets[i + 1] = h_offsets[i] + sizes[i];

    std::vector<KVPair*> d_host_ptrs((size_t)num_arrays, nullptr);
    KVPair** d_sst_arrays = nullptr;
    int* d_sst_sizes = nullptr;
    int* d_sst_offsets = nullptr;
    KVPair* d_output = nullptr;

    for (int i = 0; i < num_arrays; ++i) {
        cudaMalloc(&d_host_ptrs[i], (size_t)std::max(sizes[i], 1) * sizeof(KVPair));
        if (sizes[i] > 0) {
            cudaMemcpy(d_host_ptrs[i], arrays[i].data(),
                       (size_t)sizes[i] * sizeof(KVPair), cudaMemcpyHostToDevice);
        }
    }
    cudaMalloc(&d_sst_arrays, (size_t)num_arrays * sizeof(KVPair*));
    cudaMalloc(&d_sst_sizes, (size_t)num_arrays * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)num_arrays + 1) * sizeof(int));
    cudaMalloc(&d_output, (size_t)std::max(total, 1) * sizeof(KVPair));

    auto wall_start = std::chrono::steady_clock::now();
    auto h2d_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_sst_arrays, d_host_ptrs.data(), (size_t)num_arrays * sizeof(KVPair*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sst_sizes, sizes.data(), (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sst_offsets, h_offsets.data(), ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = (size_t)num_arrays * (sizeof(KVPair*) + sizeof(int))
                     + ((size_t)num_arrays + 1) * sizeof(int);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    int block = 256;
    int grid = (total + block - 1) / block;
    cudaEventRecord(ev0, 0);
    merge_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, d_output);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    if (total > 0) {
        auto d2h_start = std::chrono::steady_clock::now();
        cudaMemcpy(result.merged.data(), d_output, (size_t)total * sizeof(KVPair), cudaMemcpyDeviceToHost);
        auto d2h_end = std::chrono::steady_clock::now();
        result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        result.d2h_bytes = (size_t)total * sizeof(KVPair);
    }
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_output);
    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    for (KVPair* ptr : d_host_ptrs) cudaFree(ptr);
    return result;
}

static inline DeviceMergeTimedResult
launch_merge_timed_from_device(const std::vector<KVPair*>& d_arrays,
                               const std::vector<int>&     sizes,
                               bool                        copy_to_host = true)
{
    DeviceMergeTimedResult result;
    int num_arrays = (int)d_arrays.size();
    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i) {
        h_offsets[i + 1] = h_offsets[i] + sizes[i];
    }
    result.total = h_offsets[num_arrays];
    if (copy_to_host) result.merged.resize((size_t)result.total);

    KVPair** d_sst_arrays = nullptr;
    int*     d_sst_sizes = nullptr;
    int*     d_sst_offsets = nullptr;

    cudaMalloc(&d_sst_arrays, (size_t)std::max(num_arrays, 1) * sizeof(KVPair*));
    cudaMalloc(&d_sst_sizes, (size_t)std::max(num_arrays, 1) * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)std::max(num_arrays, 1) + 1) * sizeof(int));
    cudaMalloc(&result.d_output, (size_t)std::max(result.total, 1) * sizeof(KVPair));

    auto wall_start = std::chrono::steady_clock::now();
    auto h2d_start = std::chrono::steady_clock::now();
    if (num_arrays > 0) {
        cudaMemcpy(d_sst_arrays, d_arrays.data(), (size_t)num_arrays * sizeof(KVPair*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sst_sizes, sizes.data(), (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_sst_offsets, h_offsets.data(), ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = (size_t)num_arrays * (sizeof(KVPair*) + sizeof(int))
                     + ((size_t)num_arrays + 1) * sizeof(int);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    int block = 256;
    int grid = (result.total + block - 1) / block;
    cudaEventRecord(ev0, 0);
    if (result.total > 0) {
        merge_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, result.d_output);
    }
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    if (copy_to_host && result.total > 0) {
        auto d2h_start = std::chrono::steady_clock::now();
        cudaMemcpy(result.merged.data(), result.d_output,
                   (size_t)result.total * sizeof(KVPair), cudaMemcpyDeviceToHost);
        auto d2h_end = std::chrono::steady_clock::now();
        result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        result.d2h_bytes = (size_t)result.total * sizeof(KVPair);
    }
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    return result;
}

static inline DeviceMergeRefTimedResult
launch_merge_refs_timed_from_device(const std::vector<KVRef*>& d_arrays,
                                    const std::vector<int>&    sizes,
                                    bool                       copy_to_host = true)
{
    DeviceMergeRefTimedResult result;
    int num_arrays = (int)d_arrays.size();
    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i) h_offsets[i + 1] = h_offsets[i] + sizes[i];
    result.total = h_offsets[num_arrays];
    if (copy_to_host) result.merged.resize((size_t)result.total);

    KVRef** d_sst_arrays = nullptr;
    int*    d_sst_sizes = nullptr;
    int*    d_sst_offsets = nullptr;

    cudaMalloc(&d_sst_arrays, (size_t)std::max(num_arrays, 1) * sizeof(KVRef*));
    cudaMalloc(&d_sst_sizes, (size_t)std::max(num_arrays, 1) * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)std::max(num_arrays, 1) + 1) * sizeof(int));
    cudaMalloc(&result.d_output, (size_t)std::max(result.total, 1) * sizeof(KVRef));

    auto wall_start = std::chrono::steady_clock::now();
    auto h2d_start = std::chrono::steady_clock::now();
    if (num_arrays > 0) {
        cudaMemcpy(d_sst_arrays, d_arrays.data(), (size_t)num_arrays * sizeof(KVRef*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sst_sizes, sizes.data(), (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_sst_offsets, h_offsets.data(), ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::steady_clock::now();
    result.h2d_ms = (float)std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.h2d_bytes = (size_t)num_arrays * (sizeof(KVRef*) + sizeof(int))
                     + ((size_t)num_arrays + 1) * sizeof(int);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    int block = 256;
    int grid = (result.total + block - 1) / block;
    cudaEventRecord(ev0, 0);
    if (result.total > 0) {
        merge_ref_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, result.d_output);
    }
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&result.kernel_ms, ev0, ev1);

    if (copy_to_host && result.total > 0) {
        auto d2h_start = std::chrono::steady_clock::now();
        cudaMemcpy(result.merged.data(), result.d_output,
                   (size_t)result.total * sizeof(KVRef), cudaMemcpyDeviceToHost);
        auto d2h_end = std::chrono::steady_clock::now();
        result.d2h_ms = (float)std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        result.d2h_bytes = (size_t)result.total * sizeof(KVRef);
    }
    auto wall_end = std::chrono::steady_clock::now();
    result.wall_ms = (float)std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(ev1);
    cudaEventDestroy(ev0);
    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    return result;
}

struct DeviceMergeUntimedResult {
    KVPair* d_output = nullptr;
    int     total = 0;
};

struct DeviceMergeRefUntimedResult {
    KVRef* d_output = nullptr;
    int    total = 0;
};

static inline DeviceMergeUntimedResult
launch_merge_untimed_from_device(const std::vector<KVPair*>& d_arrays,
                                 const std::vector<int>&     sizes)
{
    DeviceMergeUntimedResult result;
    int num_arrays = (int)d_arrays.size();
    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i)
        h_offsets[i + 1] = h_offsets[i] + sizes[i];
    result.total = h_offsets[num_arrays];

    KVPair** d_sst_arrays = nullptr;
    int*     d_sst_sizes = nullptr;
    int*     d_sst_offsets = nullptr;
    cudaMalloc(&d_sst_arrays, (size_t)std::max(num_arrays, 1) * sizeof(KVPair*));
    cudaMalloc(&d_sst_sizes, (size_t)std::max(num_arrays, 1) * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)std::max(num_arrays, 1) + 1) * sizeof(int));
    cudaMalloc(&result.d_output, (size_t)std::max(result.total, 1) * sizeof(KVPair));

    if (num_arrays > 0) {
        cudaMemcpy(d_sst_arrays, d_arrays.data(), (size_t)num_arrays * sizeof(KVPair*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sst_sizes, sizes.data(), (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_sst_offsets, h_offsets.data(), ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (result.total + block - 1) / block;
    if (result.total > 0)
        merge_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, result.d_output);
    cudaDeviceSynchronize();

    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    return result;
}

static inline DeviceMergeRefUntimedResult
launch_merge_refs_untimed_from_device(const std::vector<KVRef*>& d_arrays,
                                      const std::vector<int>&    sizes)
{
    DeviceMergeRefUntimedResult result;
    int num_arrays = (int)d_arrays.size();
    std::vector<int> h_offsets((size_t)num_arrays + 1, 0);
    for (int i = 0; i < num_arrays; ++i)
        h_offsets[i + 1] = h_offsets[i] + sizes[i];
    result.total = h_offsets[num_arrays];

    KVRef** d_sst_arrays = nullptr;
    int*    d_sst_sizes = nullptr;
    int*    d_sst_offsets = nullptr;
    cudaMalloc(&d_sst_arrays, (size_t)std::max(num_arrays, 1) * sizeof(KVRef*));
    cudaMalloc(&d_sst_sizes, (size_t)std::max(num_arrays, 1) * sizeof(int));
    cudaMalloc(&d_sst_offsets, ((size_t)std::max(num_arrays, 1) + 1) * sizeof(int));
    cudaMalloc(&result.d_output, (size_t)std::max(result.total, 1) * sizeof(KVRef));

    if (num_arrays > 0) {
        cudaMemcpy(d_sst_arrays, d_arrays.data(), (size_t)num_arrays * sizeof(KVRef*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sst_sizes, sizes.data(), (size_t)num_arrays * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_sst_offsets, h_offsets.data(), ((size_t)num_arrays + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (result.total + block - 1) / block;
    if (result.total > 0)
        merge_ref_kernel<<<grid, block>>>(d_sst_arrays, d_sst_sizes, d_sst_offsets, num_arrays, result.d_output);
    cudaDeviceSynchronize();

    cudaFree(d_sst_offsets);
    cudaFree(d_sst_sizes);
    cudaFree(d_sst_arrays);
    return result;
}
