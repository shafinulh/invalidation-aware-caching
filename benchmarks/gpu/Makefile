# Makefile for GPU IO benchmark (cuFile)
#
# Prerequisites:
#   - CUDA toolkit (nvcc) with cuFile headers & libs
#   - GPU with CUDA support (GDS kernel driver optional; bounce-buffer fallback)
#
# Usage:
#   make            # build gpu_io_bench
#   make clean      # remove build artifacts

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
TARGET      := gpu_io_bench

SOURCES     := $(SRC_DIR)/gpu_io_bench.cu
OBJECTS     := $(BUILD_DIR)/gpu_io_bench.o

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
