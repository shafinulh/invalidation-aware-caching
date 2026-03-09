/*
 * gpcomp_cufile.cuh
 *
 * cuFile / GPUDirect Storage (GDS) abstraction layer for GPComp.
 *
 * Conditional compilation
 * -----------------------
 *   -DGPCOMP_USE_CUFILE   link against libcufile (NVIDIA GDS package).
 *                         Requires cufile.h and libcufile.so.
 *   (default)             POSIX fread + cudaHostAlloc + cudaMemcpy fallback;
 *                         no cuFile dependency, works on any GPU.
 *
 * GPU compatibility matrix
 * ------------------------
 *   GDS_NATIVE  – A30, A100, H100, and other data-centre GPUs with
 *                 NVMe/NFS-over-RDMA support: cuFileRead issues a PCIe DMA
 *                 directly from storage to VRAM, bypassing the CPU entirely.
 *
 *   GDS_COMPAT  – RTX 3070 and every other GPU without native GDS hardware:
 *                 cuFile routes I/O through a kernel-managed pinned bounce
 *                 buffer.  Data still travels host→device, but the OS
 *                 page-cache copy-in is avoided and the transfer uses
 *                 DMA-mapped pinned memory instead of a pageable buffer.
 *
 *   GDS_DISABLED – cuFile not compiled in (no -DGPCOMP_USE_CUFILE).
 *                  Plain POSIX fread into cudaHostAlloc pinned memory,
 *                  then cudaMemcpy H2D.  This is already faster than
 *                  fread into a heap buffer.
 *
 * Typical workflow
 * ----------------
 *   // On RTX 3070 (or to test compat on any GPU):
 *   gpcomp_gds_force_compat_mode();          // before init
 *   GDSMode mode = gpcomp_gds_init();        // opens cuFile driver
 *   gpcomp_gds_load_to_device(path, d_ptr, bytes);
 *   gpcomp_gds_cleanup();                    // before process exit
 *
 * Include this header in exactly ONE translation unit (same convention as
 * gpcomp_merge.cuh / gpcomp_bloom.cuh – the static variables live here).
 */

#pragma once

#ifdef GPCOMP_USE_CUFILE
#  include <cufile.h>
#endif

#include <cuda_runtime.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

/* Re-use the including TU's CUDA_CHECK if present; otherwise define our own. */
#ifndef GPCOMP_CUFILE_CUDA_CHECK
#  define GPCOMP_CUFILE_CUDA_CHECK(call)                                      \
     do {                                                                      \
         cudaError_t _e = (call);                                              \
         if (_e != cudaSuccess) {                                              \
             fprintf(stderr, "[GDS] CUDA error %s:%d – %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(_e));              \
             exit(1);                                                          \
         }                                                                     \
     } while (0)
#endif

/* =========================================================================
 * GDS mode enum
 * ====================================================================== */

enum GDSMode {
    GDS_DISABLED = 0,   /* cuFile not compiled in; plain POSIX + pinned H2D  */
    GDS_COMPAT   = 1,   /* cuFile Compat Mode: driver-managed bounce buffer  */
    GDS_NATIVE   = 2,   /* cuFile Native Mode: PCIe DMA direct to VRAM       */
};

static GDSMode s_gds_mode = GDS_DISABLED;

static inline const char* gds_mode_name(GDSMode m)
{
    switch (m) {
        case GDS_NATIVE:  return "NATIVE GDS  (PCIe DMA → VRAM)";
        case GDS_COMPAT:  return "COMPAT MODE (pinned bounce-buffer via cuFile driver)";
        default:          return "DISABLED    (POSIX fread + pinned cudaMemcpy)";
    }
}

/* =========================================================================
 * gpcomp_gds_force_compat_mode()
 *
 * Writes a minimal cufile.json to /tmp and sets CUFILE_ENV_PATH_JSON so
 * cuFileDriverOpen() picks it up and activates Compatibility Mode.
 *
 * Must be called BEFORE gpcomp_gds_init().
 *
 * On hardware without native GDS support (RTX 3070) cuFile activates compat
 * mode automatically.  Calling this function makes the intent explicit and
 * allows Compat Mode testing on GDS-capable machines (A30, A100).
 * ====================================================================== */
static inline void gpcomp_gds_force_compat_mode()
{
#ifdef GPCOMP_USE_CUFILE
    const char *json_path = "/tmp/gpcomp_cufile_compat.json";
    FILE *f = fopen(json_path, "w");
    if (f) {
        fprintf(f,
            "{\n"
            "  \"properties\": {\n"
            "    \"allow_compat_mode\": true\n"
            "  }\n"
            "}\n");
        fclose(f);
        setenv("CUFILE_ENV_PATH_JSON", json_path, /*overwrite=*/1);
        fprintf(stdout,
            "[GDS] Compatibility Mode forced via %s\n", json_path);
    } else {
        fprintf(stderr,
            "[GDS] Warning: cannot write %s (%s); "
            "relying on automatic compat detection.\n",
            json_path, strerror(errno));
    }
#else
    fprintf(stdout,
        "[GDS] gpcomp_gds_force_compat_mode(): cuFile not compiled in"
        " (-DGPCOMP_USE_CUFILE absent) – no-op.\n");
#endif
}

/* =========================================================================
 * gpcomp_gds_init()
 *
 * Opens the cuFile driver with cuFileDriverOpen(), then queries
 * CUfileDrvProps_t to determine whether native GDS or compat mode is
 * active.  Sets the module-level s_gds_mode and returns it.
 *
 * When cuFile is not compiled in (no -DGPCOMP_USE_CUFILE), returns
 * GDS_DISABLED immediately.
 * ====================================================================== */
static inline GDSMode gpcomp_gds_init()
{
#ifdef GPCOMP_USE_CUFILE
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr,
            "[GDS] cuFileDriverOpen() failed (err=%d) – "
            "falling back to POSIX+pinned H2D\n", (int)st.err);
        s_gds_mode = GDS_DISABLED;
        return s_gds_mode;
    }

    /* Probe driver properties to distinguish native vs compat. */
    CUfileDrvProps_t props{};
    st = cuFileDriverGetProperties(&props);
    if (st.err == CU_FILE_SUCCESS &&
        (props.nvfs.dcontrolflags & CU_FILE_ALLOW_COMPAT_MODE)) {
        /* CU_FILE_ALLOW_COMPAT_MODE is SET ⟹ driver allows/uses compat mode.
         * This is the case for non-GDS hardware (RTX 3070) or when forced via JSON. */
        s_gds_mode = GDS_COMPAT;
    } else {
        /* Flag absent ⟹ driver opened in pure-native mode (A30/A100/H100). */
        s_gds_mode = GDS_NATIVE;
    }

    fprintf(stdout, "[GDS] Driver opened – %s\n", gds_mode_name(s_gds_mode));
#else
    s_gds_mode = GDS_DISABLED;
    fprintf(stdout, "[GDS] cuFile not compiled in – %s\n",
            gds_mode_name(s_gds_mode));
#endif
    return s_gds_mode;
}

/* =========================================================================
 * gpcomp_gds_cleanup()
 *
 * Closes the cuFile driver.  Call once before process exit.
 * ====================================================================== */
static inline void gpcomp_gds_cleanup()
{
#ifdef GPCOMP_USE_CUFILE
    if (s_gds_mode != GDS_DISABLED)
        cuFileDriverClose();
#endif
    s_gds_mode = GDS_DISABLED;
}

/* =========================================================================
 * gpcomp_gds_load_to_device()
 *
 * Loads `byte_count` bytes from `path` (starting at `file_offset`) directly
 * into the device buffer at `d_ptr`.
 *
 * cuFile path (GDS_NATIVE or GDS_COMPAT)
 * ----------------------------------------
 *   1. Open the file with O_DIRECT (or O_RDONLY if O_DIRECT is rejected,
 *      e.g. on NFS).
 *   2. Register the file descriptor with cuFileHandleRegister().
 *   3. Register the GPU buffer with cuFileBufRegister().
 *   4. Issue a single cuFileRead() call.
 *   5. Deregister both and close the file descriptor.
 *
 *   In COMPAT MODE the kernel driver copies data through a pinned host
 *   bounce buffer, so it works on any GPU including RTX 3070.
 *   In NATIVE MODE (A30/A100/H100) the copy goes DMA → VRAM directly.
 *
 * POSIX fallback (GDS_DISABLED or cuFile I/O error)
 * ---------------------------------------------------
 *   cudaHostAlloc → fread → cudaMemcpy(H2D) → cudaFreeHost.
 *   Using pinned memory avoids the extra pageable-to-pinned copy that a
 *   plain malloc + cudaMemcpy would trigger through the CUDA runtime.
 *
 * Returns 0 on success, -1 on error.
 * ====================================================================== */
static inline int gpcomp_gds_load_to_device(const char *path,
                                             void       *d_ptr,
                                             size_t      byte_count,
                                             off_t       file_offset = 0)
{
#ifdef GPCOMP_USE_CUFILE
    if (s_gds_mode != GDS_DISABLED) {
        bool ok = false;

        /* O_DIRECT is preferred; some file systems (NFS, tmpfs) reject it. */
        int fd = open(path, O_RDONLY | O_DIRECT);
        if (fd < 0)
            fd = open(path, O_RDONLY);

        if (fd < 0) {
            fprintf(stderr, "[GDS] open(%s): %s\n", path, strerror(errno));
        } else {
            CUfileDescr_t cf_descr{};
            cf_descr.handle.fd = fd;
            cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

            CUfileHandle_t cf_handle{};
            CUfileError_t  st = cuFileHandleRegister(&cf_handle, &cf_descr);
            if (st.err != CU_FILE_SUCCESS) {
                fprintf(stderr,
                    "[GDS] cuFileHandleRegister(%s) err=%d\n",
                    path, (int)st.err);
            } else {
                st = cuFileBufRegister(d_ptr, byte_count, 0);
                if (st.err != CU_FILE_SUCCESS) {
                    fprintf(stderr,
                        "[GDS] cuFileBufRegister err=%d\n", (int)st.err);
                } else {
                    ssize_t nread = cuFileRead(cf_handle, d_ptr,
                                               byte_count, file_offset, 0);
                    cuFileBufDeregister(d_ptr);
                    if (nread >= 0 && (size_t)nread == byte_count) {
                        ok = true;
                    } else {
                        fprintf(stderr,
                            "[GDS] cuFileRead(%s): got %zd, expected %zu\n",
                            path, nread, byte_count);
                    }
                }
                cuFileHandleDeregister(cf_handle);
            }
            close(fd);
        }

        if (ok) return 0;

        fprintf(stderr,
            "[GDS] cuFile I/O failed for '%s' – "
            "falling back to pinned cudaMemcpy\n", path);
    }
#endif /* GPCOMP_USE_CUFILE */

    /* ---- POSIX + pinned-memory fallback ---- */
    void *h_pinned = nullptr;
    GPCOMP_CUFILE_CUDA_CHECK(
        cudaHostAlloc(&h_pinned, byte_count, cudaHostAllocDefault));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[GDS] fopen(%s): %s\n", path, strerror(errno));
        cudaFreeHost(h_pinned);
        return -1;
    }
    if (file_offset > 0 && fseek(fp, (long)file_offset, SEEK_SET) != 0) {
        fprintf(stderr, "[GDS] fseek(%s): %s\n", path, strerror(errno));
        fclose(fp);
        cudaFreeHost(h_pinned);
        return -1;
    }
    size_t n = fread(h_pinned, 1, byte_count, fp);
    fclose(fp);

    if (n != byte_count) {
        fprintf(stderr,
            "[GDS] fread(%s): got %zu, expected %zu\n", path, n, byte_count);
        cudaFreeHost(h_pinned);
        return -1;
    }

    GPCOMP_CUFILE_CUDA_CHECK(
        cudaMemcpy(d_ptr, h_pinned, byte_count, cudaMemcpyHostToDevice));
    cudaFreeHost(h_pinned);
    return 0;
}
