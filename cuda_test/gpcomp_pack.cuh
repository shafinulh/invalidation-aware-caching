/*
 * gpcomp_pack.cuh
 *
 * GPU kernels for the Pack and Unpack units of GPComp (Figures 7 & 8).
 *
 *   Unpack (Fig. 7): Parse serialised SST data blocks → flat KVPair arrays.
 *   Pack   (Fig. 8): Sorted KVPair array → serialised SST data blocks with
 *                    prefix compression.
 *
 * Serialised Data Block Format (simplified RocksDB BlockBuilder)
 * ──────────────────────────────────────────────────────────────
 *   ┌──────────────────────────────────────────────┐
 *   │  PackBlockHeader  (16 bytes)                  │
 *   ├──────────────────────────────────────────────┤
 *   │  Entry data       (variable bytes)            │
 *   │    entry₀: shared(1B) unshared(1B)            │
 *   │            delta_key[unshared] value[8B]      │
 *   │    entry₁: ...                                │
 *   ├──────────────────────────────────────────────┤
 *   │  Restart offsets   (num_restarts × 4 bytes)   │
 *   └──────────────────────────────────────────────┘
 *
 * Keys are stored as 8-byte big-endian uint64_t so lexicographic byte
 * comparison matches integer comparison.  At restart points (every
 * restart_interval entries) the full key is stored (shared=0).  Between
 * restarts, a shared-prefix byte count allows delta encoding.
 *
 * Include in exactly one translation unit (same convention as the other
 * gpcomp_*.cuh headers — static/inline helpers plus __global__ kernels).
 */
#pragma once

#include "gpcomp_common.cuh"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

/* ========================================================================= */
/* Constants                                                                  */
/* ========================================================================= */

static constexpr int PACK_KEY_BYTES        = 8;   /* sizeof(uint64_t) */
static constexpr int PACK_VALUE_BYTES      = 8;   /* sizeof(uint64_t) */
static constexpr int PACK_RESTART_INTERVAL = 16;  /* RocksDB default  */

/* Max serialised entry size: 1(shared) + 1(unshared) + 8(key) + 8(val) */
static constexpr int PACK_MAX_ENTRY_BYTES  = 2 + PACK_KEY_BYTES + PACK_VALUE_BYTES;

/* ========================================================================= */
/* On-disk block header                                                       */
/* ========================================================================= */

struct __attribute__((packed)) PackBlockHeader {
    uint32_t num_entries;         /* KV pairs in this block              */
    uint32_t restart_interval;    /* entries between restart points      */
    uint32_t num_restarts;        /* # restart offsets at end of block   */
    uint32_t data_size;           /* entry-data section size in bytes    */
};

static constexpr int PACK_HEADER_BYTES = (int)sizeof(PackBlockHeader);

/* ========================================================================= */
/* Host/device helpers: uint64_t ↔ big-endian bytes                           */
/* ========================================================================= */

__host__ __device__ static inline void u64_to_be(uint64_t v, uint8_t out[8])
{
    out[0] = (uint8_t)(v >> 56);  out[1] = (uint8_t)(v >> 48);
    out[2] = (uint8_t)(v >> 40);  out[3] = (uint8_t)(v >> 32);
    out[4] = (uint8_t)(v >> 24);  out[5] = (uint8_t)(v >> 16);
    out[6] = (uint8_t)(v >>  8);  out[7] = (uint8_t)(v);
}

__host__ __device__ static inline uint64_t be_to_u64(const uint8_t in[8])
{
    return ((uint64_t)in[0] << 56) | ((uint64_t)in[1] << 48) |
           ((uint64_t)in[2] << 40) | ((uint64_t)in[3] << 32) |
           ((uint64_t)in[4] << 24) | ((uint64_t)in[5] << 16) |
           ((uint64_t)in[6] <<  8) | ((uint64_t)in[7]);
}

/* Shared key prefix length between two big-endian byte arrays. */
__host__ __device__ static inline int key_shared_prefix(
        const uint8_t a[PACK_KEY_BYTES],
        const uint8_t b[PACK_KEY_BYTES])
{
    int s = 0;
    for (int i = 0; i < PACK_KEY_BYTES; ++i) {
        if (a[i] != b[i]) break;
        ++s;
    }
    return s;
}

/* ========================================================================= *
 * CPU Pack – serialise a KVPair slice into a single data block.              *
 *                                                                             *
 * Returns the serialised block in a vector<uint8_t>.                          *
 * ========================================================================= */

static inline std::vector<uint8_t>
cpu_pack_block(const KVPair *kv, int n, int restart_interval = PACK_RESTART_INTERVAL)
{
    if (n <= 0) return {};

    int num_restarts = (n + restart_interval - 1) / restart_interval;

    /* --- Phase 1: serialise entries into a temporary buffer -------------- */
    std::vector<uint8_t> entry_data;
    entry_data.reserve((size_t)n * PACK_MAX_ENTRY_BYTES);

    std::vector<uint32_t> restart_offsets;
    restart_offsets.reserve(num_restarts);

    uint8_t prev_key[PACK_KEY_BYTES] = {};

    for (int i = 0; i < n; ++i) {
        uint8_t cur_key[PACK_KEY_BYTES];
        u64_to_be(kv[i].key, cur_key);

        int shared = 0;
        if (i % restart_interval != 0) {
            shared = key_shared_prefix(prev_key, cur_key);
        } else {
            restart_offsets.push_back((uint32_t)entry_data.size());
        }
        int unshared = PACK_KEY_BYTES - shared;

        entry_data.push_back((uint8_t)shared);
        entry_data.push_back((uint8_t)unshared);
        entry_data.insert(entry_data.end(), cur_key + shared, cur_key + PACK_KEY_BYTES);

        uint8_t val_bytes[PACK_VALUE_BYTES];
        u64_to_be(kv[i].value, val_bytes);
        entry_data.insert(entry_data.end(), val_bytes, val_bytes + PACK_VALUE_BYTES);

        memcpy(prev_key, cur_key, PACK_KEY_BYTES);
    }

    /* --- Phase 2: assemble block = header + entry_data + restarts ------- */
    uint32_t data_size = (uint32_t)entry_data.size();
    PackBlockHeader hdr;
    hdr.num_entries      = (uint32_t)n;
    hdr.restart_interval = (uint32_t)restart_interval;
    hdr.num_restarts     = (uint32_t)restart_offsets.size();
    hdr.data_size        = data_size;

    size_t total = PACK_HEADER_BYTES + data_size
                 + restart_offsets.size() * sizeof(uint32_t);
    std::vector<uint8_t> block(total);

    memcpy(block.data(), &hdr, PACK_HEADER_BYTES);
    memcpy(block.data() + PACK_HEADER_BYTES, entry_data.data(), data_size);
    memcpy(block.data() + PACK_HEADER_BYTES + data_size,
           restart_offsets.data(),
           restart_offsets.size() * sizeof(uint32_t));

    return block;
}

/* ========================================================================= *
 * CPU Unpack – parse a serialised data block back to KVPair array.           *
 *                                                                             *
 * Returns the number of KV pairs written to `out` (which must hold at least  *
 * hdr.num_entries elements).                                                  *
 * ========================================================================= */

static inline int
cpu_unpack_block(const uint8_t *block, size_t block_len, KVPair *out)
{
    if (block_len < (size_t)PACK_HEADER_BYTES) return 0;

    PackBlockHeader hdr;
    memcpy(&hdr, block, PACK_HEADER_BYTES);

    const uint8_t *data = block + PACK_HEADER_BYTES;
    const uint8_t *data_end = data + hdr.data_size;

    uint8_t prev_key[PACK_KEY_BYTES] = {};
    int count = 0;

    const uint8_t *p = data;
    for (uint32_t i = 0; i < hdr.num_entries && p + 2 <= data_end; ++i) {
        int shared   = p[0];
        int unshared = p[1];
        p += 2;

        if (p + unshared + PACK_VALUE_BYTES > data_end) break;

        uint8_t cur_key[PACK_KEY_BYTES];
        memcpy(cur_key, prev_key, shared);
        memcpy(cur_key + shared, p, unshared);
        p += unshared;

        out[count].key = be_to_u64(cur_key);

        uint8_t val_bytes[PACK_VALUE_BYTES];
        memcpy(val_bytes, p, PACK_VALUE_BYTES);
        p += PACK_VALUE_BYTES;

        out[count].value = be_to_u64(val_bytes);

        memcpy(prev_key, cur_key, PACK_KEY_BYTES);
        ++count;
    }
    return count;
}

/* ========================================================================= *
 * GPU Pack Kernel                                                             *
 *                                                                             *
 * Grid:  <<<num_data_blocks, threads_per_block>>>                             *
 * Each CUDA block serialises one data block of keys_per_block entries.        *
 *                                                                             *
 * Strategy: one thread per restart interval.  Thread t handles entries         *
 * [t*restart_interval .. min((t+1)*restart_interval, block_entries)).         *
 *                                                                             *
 * Two passes using shared memory:                                             *
 *   Pass 1 – each thread computes the byte count for its interval.            *
 *   Prefix-sum in shared memory → byte offsets.                               *
 *   Pass 2 – each thread writes its entries at the computed offset.           *
 *   Thread 0 writes the header and restart offsets.                            *
 * ========================================================================= */

__global__ void pack_kernel(
        const KVPair * __restrict__ kv_array,   /* sorted input         */
        int                         total_keys,
        int                         keys_per_block,
        int                         restart_interval,
        uint8_t      * __restrict__ block_out,  /* output buffer        */
        uint32_t     * __restrict__ block_sizes,/* per-block byte count */
        int                         max_block_bytes) /* allocated per block */
{
    int block_id    = (int)blockIdx.x;
    int tid         = (int)threadIdx.x;

    int block_start  = block_id * keys_per_block;
    int block_entries = total_keys - block_start;
    if (block_entries > keys_per_block) block_entries = keys_per_block;
    if (block_entries <= 0) return;

    int num_restarts = (block_entries + restart_interval - 1) / restart_interval;

    /* Shared memory layout:
     *   uint32_t interval_bytes[num_restarts]  – byte count per interval
     *   uint32_t interval_offsets[num_restarts] – exclusive prefix sum
     *   uint32_t restart_offsets[num_restarts]  – restart byte offsets
     *   (all accessed via a single uint32_t* base)
     */
    extern __shared__ uint32_t smem[];
    uint32_t *interval_bytes   = smem;
    uint32_t *interval_offsets = smem + num_restarts;
    uint32_t *restart_off_smem = smem + 2 * num_restarts;

    const KVPair *block_kv = kv_array + block_start;

    /* ── Pass 1: compute byte count for each restart interval ────────── */
    if (tid < num_restarts) {
        int e_start = tid * restart_interval;
        int e_end   = e_start + restart_interval;
        if (e_end > block_entries) e_end = block_entries;

        uint32_t bytes = 0;
        uint8_t prev_key[PACK_KEY_BYTES];

        for (int e = e_start; e < e_end; ++e) {
            uint8_t cur_key[PACK_KEY_BYTES];
            u64_to_be(block_kv[e].key, cur_key);

            int shared = 0;
            if (e != e_start) {
                /* compute shared prefix with previous key */
                for (int b = 0; b < PACK_KEY_BYTES; ++b) {
                    if (prev_key[b] != cur_key[b]) break;
                    ++shared;
                }
            }
            int unshared = PACK_KEY_BYTES - shared;
            bytes += (uint32_t)(2 + unshared + PACK_VALUE_BYTES);

            /* remember for pass 2 */
            for (int b = 0; b < PACK_KEY_BYTES; ++b) prev_key[b] = cur_key[b];
        }
        interval_bytes[tid] = bytes;
    }
    __syncthreads();

    /* ── Exclusive prefix sum (serial by thread 0 – num_restarts ≤ 64) ─ */
    if (tid == 0) {
        uint32_t sum = 0;
        for (int r = 0; r < num_restarts; ++r) {
            restart_off_smem[r] = sum;   /* restart byte offset */
            interval_offsets[r] = sum;
            sum += interval_bytes[r];
        }
        /* interval_offsets now holds the exclusive prefix sum */
    }
    __syncthreads();

    /* ── Pass 2: write entries ───────────────────────────────────────── */
    if (tid < num_restarts) {
        /* Output pointer: after header for entry data section */
        uint8_t *out_base = block_out + (size_t)block_id * max_block_bytes
                          + PACK_HEADER_BYTES;
        uint8_t *p = out_base + interval_offsets[tid];

        int e_start = tid * restart_interval;
        int e_end   = e_start + restart_interval;
        if (e_end > block_entries) e_end = block_entries;

        uint8_t prev_key[PACK_KEY_BYTES];

        for (int e = e_start; e < e_end; ++e) {
            uint8_t cur_key[PACK_KEY_BYTES], val_bytes[PACK_VALUE_BYTES];
            u64_to_be(block_kv[e].key, cur_key);
            u64_to_be(block_kv[e].value, val_bytes);

            int shared = 0;
            if (e != e_start) {
                for (int b = 0; b < PACK_KEY_BYTES; ++b) {
                    if (prev_key[b] != cur_key[b]) break;
                    ++shared;
                }
            }
            int unshared = PACK_KEY_BYTES - shared;

            *p++ = (uint8_t)shared;
            *p++ = (uint8_t)unshared;
            for (int b = 0; b < unshared; ++b)
                *p++ = cur_key[shared + b];
            for (int b = 0; b < PACK_VALUE_BYTES; ++b)
                *p++ = val_bytes[b];

            for (int b = 0; b < PACK_KEY_BYTES; ++b) prev_key[b] = cur_key[b];
        }
    }
    __syncthreads();

    /* ── Thread 0: write header and restart offsets ─────────────────── */
    if (tid == 0) {
        uint8_t *blk = block_out + (size_t)block_id * max_block_bytes;

        /* Compute total data size */
        uint32_t data_size = 0;
        for (int r = 0; r < num_restarts; ++r)
            data_size += interval_bytes[r];

        PackBlockHeader hdr;
        hdr.num_entries      = (uint32_t)block_entries;
        hdr.restart_interval = (uint32_t)restart_interval;
        hdr.num_restarts     = (uint32_t)num_restarts;
        hdr.data_size        = data_size;

        memcpy(blk, &hdr, PACK_HEADER_BYTES);

        /* Write restart offsets after entry data (byte-by-byte to avoid
         * misaligned uint32_t writes – data_size is not always 4-aligned) */
        uint8_t *restart_dst = blk + PACK_HEADER_BYTES + data_size;
        for (int r = 0; r < num_restarts; ++r)
            memcpy(restart_dst + r * sizeof(uint32_t),
                   &restart_off_smem[r], sizeof(uint32_t));

        /* Record total block size */
        block_sizes[block_id] = PACK_HEADER_BYTES + data_size
                              + (uint32_t)num_restarts * sizeof(uint32_t);
    }
}

/* ========================================================================= *
 * GPU Unpack Kernel                                                           *
 *                                                                             *
 * Grid:  <<<num_data_blocks, threads_per_block>>>                             *
 * Each CUDA block parses one serialised data block.                           *
 *                                                                             *
 * Strategy: one thread per restart interval.  Thread t reads the restart      *
 * offset, then sequentially parses restart_interval entries from that point,  *
 * reconstructing keys and writing KVPairs to the output array.                *
 * ========================================================================= */

__global__ void unpack_kernel(
        const uint8_t * __restrict__  block_buf,   /* serialised blocks     */
        const uint32_t * __restrict__ block_offsets,/* byte offset per block */
        int                           num_blocks,
        int                           keys_per_block,
        KVPair        * __restrict__  kv_out,       /* flat output array     */
        int                           total_keys)
{
    int block_id = (int)blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = (int)threadIdx.x;

    /* Locate our block in the buffer */
    const uint8_t *blk = block_buf + block_offsets[block_id];

    /* Read header */
    PackBlockHeader hdr;
    memcpy(&hdr, blk, PACK_HEADER_BYTES);

    int block_entries    = (int)hdr.num_entries;
    int restart_interval = (int)hdr.restart_interval;
    int num_restarts     = (int)hdr.num_restarts;

    if (tid >= num_restarts) return;

    /* Read restart offsets (after entry data – use memcpy to avoid
     * misaligned uint32_t reads; data_size is not always 4-aligned) */
    const uint8_t *restart_src = blk + PACK_HEADER_BYTES + hdr.data_size;
    uint32_t my_restart_off;
    memcpy(&my_restart_off, restart_src + tid * sizeof(uint32_t),
           sizeof(uint32_t));

    /* Parse entries from this restart point */
    const uint8_t *data = blk + PACK_HEADER_BYTES;
    const uint8_t *p    = data + my_restart_off;
    const uint8_t *data_end = data + hdr.data_size;

    int e_start = tid * restart_interval;
    int e_end   = e_start + restart_interval;
    if (e_end > block_entries) e_end = block_entries;

    int kv_base = block_id * keys_per_block + e_start;

    uint8_t prev_key[PACK_KEY_BYTES] = {};

    for (int e = e_start; e < e_end && p + 2 <= data_end; ++e) {
        int shared   = p[0];
        int unshared = p[1];
        p += 2;

        if (p + unshared + PACK_VALUE_BYTES > data_end) break;

        uint8_t cur_key[PACK_KEY_BYTES];
        /* Copy shared prefix from previous key */
        for (int b = 0; b < shared; ++b)
            cur_key[b] = prev_key[b];
        /* Copy delta bytes */
        for (int b = 0; b < unshared; ++b)
            cur_key[shared + b] = p[b];
        p += unshared;

        uint8_t val_bytes[PACK_VALUE_BYTES];
        for (int b = 0; b < PACK_VALUE_BYTES; ++b)
            val_bytes[b] = p[b];
        p += PACK_VALUE_BYTES;

        int out_idx = kv_base + (e - e_start);
        if (out_idx < total_keys) {
            kv_out[out_idx].key   = be_to_u64(cur_key);
            kv_out[out_idx].value = be_to_u64(val_bytes);
        }

        for (int b = 0; b < PACK_KEY_BYTES; ++b)
            prev_key[b] = cur_key[b];
    }
}

/* ========================================================================= *
 * Host launcher: Pack (KVPair array → serialised blocks)                     *
 *                                                                             *
 * Splits kv_array into data blocks of keys_per_block entries, serialises      *
 * each on the GPU, and returns the concatenated block buffer + offset array.  *
 *                                                                             *
 * Returns 0 on success.                                                       *
 * ========================================================================= */

struct PackResult {
    std::vector<uint8_t>  block_buf;     /* concatenated serialised blocks */
    std::vector<uint32_t> block_offsets; /* byte offset of each block      */
    std::vector<uint32_t> block_sizes;   /* byte count of each block       */
    int                   num_blocks;
};

static int launch_pack(const KVPair *h_kv, int total_keys,
                       int keys_per_block, int restart_interval,
                       PackResult &result)
{
    int num_blocks = (total_keys + keys_per_block - 1) / keys_per_block;
    result.num_blocks = num_blocks;

    /* Max block size: header + max_entries*max_entry + max_restarts*4 */
    int max_restarts    = (keys_per_block + restart_interval - 1) / restart_interval;
    int max_block_bytes = PACK_HEADER_BYTES
                        + keys_per_block * PACK_MAX_ENTRY_BYTES
                        + max_restarts * (int)sizeof(uint32_t);

    /* Device allocations */
    KVPair   *d_kv;
    uint8_t  *d_blocks;
    uint32_t *d_block_sizes;

    cudaMalloc(&d_kv,          (size_t)total_keys * sizeof(KVPair));
    cudaMalloc(&d_blocks,      (size_t)num_blocks * max_block_bytes);
    cudaMalloc(&d_block_sizes, (size_t)num_blocks * sizeof(uint32_t));

    cudaMemcpy(d_kv, h_kv, (size_t)total_keys * sizeof(KVPair),
               cudaMemcpyHostToDevice);
    cudaMemset(d_blocks, 0, (size_t)num_blocks * max_block_bytes);

    /* Shared memory: 3 × num_restarts × sizeof(uint32_t) */
    int smem_bytes = 3 * max_restarts * (int)sizeof(uint32_t);

    /* Block dim: one thread per restart interval (max_restarts),
     * rounded up to warp size */
    int block_dim = ((max_restarts + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    pack_kernel<<<num_blocks, block_dim, smem_bytes>>>(
        d_kv, total_keys, keys_per_block, restart_interval,
        d_blocks, d_block_sizes, max_block_bytes);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pack_kernel] launch error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    /* Copy back block sizes */
    result.block_sizes.resize(num_blocks);
    cudaMemcpy(result.block_sizes.data(), d_block_sizes,
               (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Copy back block data (only the actual bytes, not max padding) */
    std::vector<uint8_t> raw_blocks((size_t)num_blocks * max_block_bytes);
    cudaMemcpy(raw_blocks.data(), d_blocks,
               (size_t)num_blocks * max_block_bytes, cudaMemcpyDeviceToHost);

    /* Compact: concatenate blocks tightly, record offsets */
    result.block_offsets.resize(num_blocks);
    size_t total_bytes = 0;
    for (int b = 0; b < num_blocks; ++b)
        total_bytes += result.block_sizes[b];

    result.block_buf.resize(total_bytes);
    size_t off = 0;
    for (int b = 0; b < num_blocks; ++b) {
        result.block_offsets[b] = (uint32_t)off;
        memcpy(result.block_buf.data() + off,
               raw_blocks.data() + (size_t)b * max_block_bytes,
               result.block_sizes[b]);
        off += result.block_sizes[b];
    }

    cudaFree(d_kv);
    cudaFree(d_blocks);
    cudaFree(d_block_sizes);

    return (int)err;
}

/* ========================================================================= *
 * Host launcher: Unpack (serialised blocks → KVPair array)                   *
 *                                                                             *
 * Takes a concatenated block buffer + offsets (as produced by launch_pack)    *
 * and returns the unpacked KVPair array.                                      *
 *                                                                             *
 * Returns 0 on success.                                                       *
 * ========================================================================= */

static int launch_unpack(const uint8_t  *h_block_buf,
                         size_t          block_buf_len,
                         const uint32_t *h_block_offsets,
                         int             num_blocks,
                         int             keys_per_block,
                         int             total_keys,
                         KVPair         *h_kv_out)
{
    uint8_t  *d_block_buf;
    uint32_t *d_block_offsets;
    KVPair   *d_kv_out;

    cudaMalloc(&d_block_buf,     block_buf_len);
    cudaMalloc(&d_block_offsets, (size_t)num_blocks * sizeof(uint32_t));
    cudaMalloc(&d_kv_out,        (size_t)total_keys * sizeof(KVPair));

    cudaMemcpy(d_block_buf, h_block_buf, block_buf_len,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets,
               (size_t)num_blocks * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemset(d_kv_out, 0, (size_t)total_keys * sizeof(KVPair));

    /* Block dim: one thread per restart interval */
    int max_restarts = (keys_per_block + PACK_RESTART_INTERVAL - 1)
                     / PACK_RESTART_INTERVAL;
    int block_dim = ((max_restarts + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    unpack_kernel<<<num_blocks, block_dim>>>(
        d_block_buf, d_block_offsets, num_blocks,
        keys_per_block, d_kv_out, total_keys);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[unpack_kernel] launch error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_kv_out, d_kv_out, (size_t)total_keys * sizeof(KVPair),
               cudaMemcpyDeviceToHost);

    cudaFree(d_block_buf);
    cudaFree(d_block_offsets);
    cudaFree(d_kv_out);

    return (int)err;
}

/* ========================================================================= *
 * Combined host launcher: Pack with timing (for benchmarks)                   *
 *                                                                             *
 * Same as launch_pack but returns kernel-only and wall timing.  Allocates     *
 * device memory, uploads data, runs the kernel, downloads result.             *
 * ========================================================================= */

struct PackTimedResult {
    PackResult result;
    float      kernel_ms;  /* GPU kernel-only (CUDA events) */
    float      wall_ms;    /* H2D + kernel + D2H            */
};

static PackTimedResult launch_pack_timed(
        const KVPair *h_kv, int total_keys,
        int keys_per_block, int restart_interval)
{
    PackTimedResult tr;
    int num_blocks = (total_keys + keys_per_block - 1) / keys_per_block;
    tr.result.num_blocks = num_blocks;

    int max_restarts    = (keys_per_block + restart_interval - 1) / restart_interval;
    int max_block_bytes = PACK_HEADER_BYTES
                        + keys_per_block * PACK_MAX_ENTRY_BYTES
                        + max_restarts * (int)sizeof(uint32_t);

    KVPair   *d_kv;
    uint8_t  *d_blocks;
    uint32_t *d_block_sizes;
    cudaMalloc(&d_kv,          (size_t)total_keys * sizeof(KVPair));
    cudaMalloc(&d_blocks,      (size_t)num_blocks * max_block_bytes);
    cudaMalloc(&d_block_sizes, (size_t)num_blocks * sizeof(uint32_t));
    cudaMemset(d_blocks, 0, (size_t)num_blocks * max_block_bytes);

    int smem_bytes = 3 * max_restarts * (int)sizeof(uint32_t);
    int block_dim  = ((max_restarts + 31) / 32) * 32;
    if (block_dim < 32) block_dim = 32;

    /* ---------- Wall start ---------- */
    auto wall_t0 = std::chrono::steady_clock::now();

    cudaMemcpy(d_kv, h_kv, (size_t)total_keys * sizeof(KVPair),
               cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);

    pack_kernel<<<num_blocks, block_dim, smem_bytes>>>(
        d_kv, total_keys, keys_per_block, restart_interval,
        d_blocks, d_block_sizes, max_block_bytes);

    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&tr.kernel_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    /* Download block sizes */
    tr.result.block_sizes.resize(num_blocks);
    cudaMemcpy(tr.result.block_sizes.data(), d_block_sizes,
               (size_t)num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Download raw blocks & compact */
    std::vector<uint8_t> raw_blocks((size_t)num_blocks * max_block_bytes);
    cudaMemcpy(raw_blocks.data(), d_blocks,
               (size_t)num_blocks * max_block_bytes, cudaMemcpyDeviceToHost);

    auto wall_t1 = std::chrono::steady_clock::now();
    tr.wall_ms = (float)std::chrono::duration<double, std::milli>(
                     wall_t1 - wall_t0).count();

    /* Compact */
    tr.result.block_offsets.resize(num_blocks);
    size_t total_bytes = 0;
    for (int b = 0; b < num_blocks; ++b)
        total_bytes += tr.result.block_sizes[b];
    tr.result.block_buf.resize(total_bytes);
    size_t off = 0;
    for (int b = 0; b < num_blocks; ++b) {
        tr.result.block_offsets[b] = (uint32_t)off;
        memcpy(tr.result.block_buf.data() + off,
               raw_blocks.data() + (size_t)b * max_block_bytes,
               tr.result.block_sizes[b]);
        off += tr.result.block_sizes[b];
    }

    cudaFree(d_kv);
    cudaFree(d_blocks);
    cudaFree(d_block_sizes);

    return tr;
}
