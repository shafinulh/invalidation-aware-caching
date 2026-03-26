#!/bin/bash
# setup_local.sh — Download and extract cuFile (GDS) libraries into gds/local/
# No root required. Run once from the cuda_test/ or gds/ directory.
#
# Usage:
#   bash gds/setup_local.sh             # from cuda_test/
#   bash setup_local.sh                 # from gds/
#
# For CUDA versions other than 11.8, edit CUDA_DASH and CUFILE_VER below.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_DASH="11-8"           # e.g. 12-4 for CUDA 12.4
CUFILE_VER="1.4.0.31-1"   # deb package version
ARCH="amd64"
BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"

RT_DEB="libcufile-${CUDA_DASH}_${CUFILE_VER}_${ARCH}.deb"
DEV_DEB="libcufile-dev-${CUDA_DASH}_${CUFILE_VER}_${ARCH}.deb"

echo "==> Downloading cuFile packages (CUDA ${CUDA_DASH})..."
curl -fsSL -o "$SCRIPT_DIR/$RT_DEB"  "$BASE/$RT_DEB"
curl -fsSL -o "$SCRIPT_DIR/$DEV_DEB" "$BASE/$DEV_DEB"

echo "==> Extracting..."
rm -rf "$SCRIPT_DIR/extract_rt" "$SCRIPT_DIR/extract_dev"
dpkg-deb -x "$SCRIPT_DIR/$RT_DEB"  "$SCRIPT_DIR/extract_rt"
dpkg-deb -x "$SCRIPT_DIR/$DEV_DEB" "$SCRIPT_DIR/extract_dev"

RT="$SCRIPT_DIR/extract_rt/usr/local/cuda-${CUDA_DASH//-/.}/targets/x86_64-linux"
DEV="$SCRIPT_DIR/extract_dev/usr/local/cuda-${CUDA_DASH//-/.}/targets/x86_64-linux"
LOCAL="$SCRIPT_DIR/local"

mkdir -p "$LOCAL/include" "$LOCAL/lib"
cp "$DEV/include/cufile.h"           "$LOCAL/include/"
cp "$RT/lib/libcufile.so.1."*        "$LOCAL/lib/"  2>/dev/null || true
cp "$RT/lib/libcufile_rdma.so.1."*   "$LOCAL/lib/"  2>/dev/null || true

# Build SONAME symlinks
cd "$LOCAL/lib"
SO_VER=$(ls libcufile.so.*.* 2>/dev/null | head -1)
if [ -n "$SO_VER" ]; then
    # Detect embedded SONAME (may be .so.0 instead of .so.1)
    SONAME=$(objdump -p "$SO_VER" 2>/dev/null | awk '/SONAME/{print $2}')
    if [ -z "$SONAME" ]; then SONAME="libcufile.so.0"; fi
    ln -sf "$SO_VER" "$SONAME"
    ln -sf "$SO_VER" libcufile.so.1 2>/dev/null || true
    ln -sf libcufile.so.1 libcufile.so 2>/dev/null || true
fi
RDMA_VER=$(ls libcufile_rdma.so.*.* 2>/dev/null | head -1)
if [ -n "$RDMA_VER" ]; then
    ln -sf "$RDMA_VER" libcufile_rdma.so.1 2>/dev/null || true
    ln -sf libcufile_rdma.so.1 libcufile_rdma.so 2>/dev/null || true
fi

echo ""
echo "==> gds/local/ is ready:"
ls -lh "$LOCAL/include/" "$LOCAL/lib/"
echo ""
echo "Build with:  make GDS=1 gpcomp_bench"
