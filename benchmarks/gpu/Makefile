# Makefile for GPU benchmark tools (cuFile + GPU compaction worker)
#
# Prerequisites:
#   - CUDA toolkit (nvcc) with cuFile headers & libs
#   - GPU with CUDA support (GDS kernel driver optional; bounce-buffer fallback)
#   - librocksdb.a built in ../../rocksdb-gpu/  (required for gpu_compaction_worker)
#
# Usage:
#   make                        # build gpu_io_bench
#   make gpu_file_replay_bench  # cuFile I/O replay helper
#   make gpu_compaction_worker  # REAL GPU merge worker (links librocksdb)
#   make tools                  # build all three binaries
#   make clean                  # remove build artifacts

CUDA_HOME   ?= /usr/local/cuda
NVCC        := $(shell which nvcc 2>/dev/null || echo $(CUDA_HOME)/bin/nvcc)

# cuFile paths — fall back to a local GDS installation if /usr/local/cuda
# does not have cufile.h (common on GeForce systems).
GDS_LOCAL   ?= $(abspath ../../../cuda_test/gds/local)
CUFILE_INC  := $(if $(wildcard $(CUDA_HOME)/include/cufile.h),\
                    $(CUDA_HOME)/include,$(GDS_LOCAL)/include)
CUFILE_LIB  := $(if $(wildcard $(CUDA_HOME)/lib64/libcufile.so),\
                    $(CUDA_HOME)/lib64,$(GDS_LOCAL)/lib)

# Host compiler – CUDA 11.8 needs gcc ≤ 11; gcc-12 works with the
# --allow-unsupported-compiler flag on Ubuntu 24.04.
HOST_CXX    ?= $(shell command -v g++-12 2>/dev/null || echo g++)

# Compiler flags
NVCC_FLAGS  := -std=c++17 -O2 -Xcompiler -Wall \
               --allow-unsupported-compiler -ccbin $(HOST_CXX) \
               --expt-relaxed-constexpr \
               -Xcompiler -U_FORTIFY_SOURCE -Xcompiler -D_FORTIFY_SOURCE=0
INCLUDES    := -I$(CUFILE_INC)
LDFLAGS     := -L$(CUFILE_LIB) -lcufile -lcuda -lcudart

# ---- RocksDB worker settings -------------------------------------------
# Path to the pre-built RocksDB static library and headers.
ROCKSDB_ROOT  ?= $(abspath ../../rocksdb-gpu)
ROCKSDB_LIB   := $(ROCKSDB_ROOT)/librocksdb.a
ROCKSDB_INC   := -I$(ROCKSDB_ROOT)/include
# cuda_test/ headers (gpcomp_common.cuh, gpcomp_merge.cuh, etc.)
GPCOMP_INC    := -I$(abspath ../../cuda_test)

# RocksDB platform libraries (edit if your system differs)
# Use -l:libname.so.VER for libs without unversioned .so symlinks
ROCKSDB_PLATFORM_LIBS := -lpthread -lrt -ldl -lz \
	-l:libsnappy.so.1 -l:liblz4.so.1 -lzstd -lbz2

WORKER_INCLUDES := $(ROCKSDB_INC) $(GPCOMP_INC)
WORKER_LDFLAGS  := $(ROCKSDB_LIB) -lcuda -lcudart \
                   $(ROCKSDB_PLATFORM_LIBS)

SRC_DIR     := src
BUILD_DIR   := build
IO_TARGET   := gpu_io_bench
REPLAY_TARGET := gpu_file_replay_bench
WORKER_TARGET := gpu_compaction_worker

IO_OBJECT        := $(BUILD_DIR)/gpu_io_bench.o
REPLAY_OBJECT    := $(BUILD_DIR)/gpu_file_replay_bench.o
# Worker uses split compilation: CUDA kernel (nvcc) + RocksDB main (g++)
WORKER_GPU_OBJ   := $(BUILD_DIR)/worker_gpu.o
WORKER_MAIN_OBJ  := $(BUILD_DIR)/worker_main.o

.PHONY: all clean tools

all: $(IO_TARGET)

tools: $(IO_TARGET) $(REPLAY_TARGET) $(WORKER_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Worker GPU stub: compiled by nvcc (CUDA kernels only, no RocksDB headers)
$(WORKER_GPU_OBJ): $(SRC_DIR)/worker_gpu.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(GPCOMP_INC) -c $< -o $@

# Worker main: compiled by g++ (C++20 RocksDB headers, no CUDA device code)
$(WORKER_MAIN_OBJ): $(SRC_DIR)/worker_main.cpp | $(BUILD_DIR)
	$(HOST_CXX) -std=c++20 -O2 -Wall $(WORKER_INCLUDES) \
	    -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 \
	    -c $< -o $@

$(IO_TARGET): $(IO_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $(IO_OBJECT) $(LDFLAGS) -o $@

$(REPLAY_TARGET): $(REPLAY_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $(REPLAY_OBJECT) $(LDFLAGS) -o $@

# Link worker: nvcc as linker to pull in CUDA runtime, g++ objects, librocksdb
$(WORKER_TARGET): $(WORKER_GPU_OBJ) $(WORKER_MAIN_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(WORKER_GPU_OBJ) $(WORKER_MAIN_OBJ) \
	    $(WORKER_LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(IO_TARGET) $(REPLAY_TARGET) $(WORKER_TARGET)
