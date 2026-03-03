# Makefile for GPU IO benchmark (cuFile)
#
# Prerequisites:
#   - CUDA toolkit (nvcc) with cuFile headers & libs
#   - GPU with CUDA support (GDS kernel driver optional; bounce-buffer fallback)
#
# Usage:
#   make                     # build gpu_io_bench
#   make gpu_file_replay_bench
#   make tools              # build both GPU benchmark binaries
#   make clean              # remove build artifacts

CUDA_HOME   ?= /usr/local/cuda
NVCC        := $(CUDA_HOME)/bin/nvcc

# cuFile paths
CUFILE_INC  := $(CUDA_HOME)/include
CUFILE_LIB  := $(CUDA_HOME)/lib64

# Host compiler – CUDA 11.8 needs gcc ≤ 11; gcc-12 works with the
# --allow-unsupported-compiler flag on Ubuntu 24.04.
HOST_CXX    ?= $(shell command -v g++-12 2>/dev/null || echo g++)

# Compiler flags
NVCC_FLAGS  := -std=c++17 -O2 -Xcompiler -Wall \
               --allow-unsupported-compiler -ccbin $(HOST_CXX) \
               -Xcompiler -U_FORTIFY_SOURCE -Xcompiler -D_FORTIFY_SOURCE=0
INCLUDES    := -I$(CUFILE_INC)
LDFLAGS     := -L$(CUFILE_LIB) -lcufile -lcuda -lcudart

SRC_DIR     := src
BUILD_DIR   := build
IO_TARGET   := gpu_io_bench
REPLAY_TARGET := gpu_file_replay_bench

IO_OBJECT   := $(BUILD_DIR)/gpu_io_bench.o
REPLAY_OBJECT := $(BUILD_DIR)/gpu_file_replay_bench.o

.PHONY: all clean tools

all: $(IO_TARGET)

tools: $(IO_TARGET) $(REPLAY_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(IO_TARGET): $(IO_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $(IO_OBJECT) $(LDFLAGS) -o $@

$(REPLAY_TARGET): $(REPLAY_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $(REPLAY_OBJECT) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(IO_TARGET) $(REPLAY_TARGET)
