# Makefile – GPComp CUDA kernels
#
# Targets:
#   all               – build all binaries
#   merge_test        – smoke-test driver for Kernel 1
#   bloom_test        – smoke-test driver for Kernel 2
#   gpcomp_unit_tests – comprehensive unit-test suite (17 tests)
#   gpcomp_datagen    – synthetic SST dataset generator (mimics db_bench fillrandom)
#   gpcomp_bench      – end-to-end kernel benchmark on the generated dataset
#   test              – build gpcomp_unit_tests and run it
#   bench             – generate default dataset then run both kernel benchmarks
#   bench-large       – generate large dataset (8 SSTs × 512K keys) then benchmark
#   bench-gds         – generate dataset then benchmark with cuFile compat mode
#   bench-gds-native  – benchmark with cuFile native GDS (A30 / data-centre GPU)
#   clean             – remove all build artefacts (binaries + dataset/)
#
# GDS / cuFile build
# ------------------
# To enable GPUDirect Storage support, compile with GDS=1:
#
#   make GDS=1 gpcomp_bench
#
# This adds -DGPCOMP_USE_CUFILE and links against libcufile.
# Requires the NVIDIA GDS (GPUDirect Storage) package:
#
#   Ubuntu/Debian:  apt-get install nvidia-gds
#   RHEL/CentOS:    yum install nvidia-gds
#   Manual install: https://docs.nvidia.com/gpudirect-storage/
#
# On RTX 3070 (no native GDS hardware):
#   Use --compat-mode flag at runtime; cuFile routes I/O through a
#   pinned bounce-buffer (faster than pageable fread + cudaMemcpy).
#
# On A30 / A100 / H100 (native GDS hardware):
#   Run without --compat-mode; cuFileRead issues PCIe DMA to VRAM.

NVCC    := nvcc
CXX     := g++
CFLAGS  := -O2 -std=c++17

# GDS=1: compile with cuFile support.
# Uses locally extracted package under gds/local/ so no system install needed.
GDS ?= 0
GDS_LOCAL := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))gds/local
ifeq ($(GDS),1)
  CUFILE_INC   := -I$(GDS_LOCAL)/include
  CUFILE_RPATH := -Xlinker -rpath,$(GDS_LOCAL)/lib
  CUFILE_FLAGS := -DGPCOMP_USE_CUFILE $(CUFILE_INC) -L$(GDS_LOCAL)/lib $(CUFILE_RPATH) -lcufile
  GDS_TAG      := (GDS enabled, local=$(GDS_LOCAL))
else
  CUFILE_FLAGS :=
  GDS_TAG      :=
endif

HEADERS := gpcomp_common.cuh gpcomp_merge.cuh gpcomp_bloom.cuh gpcomp_cufile.cuh

.PHONY: all test bench bench-large bench-gds bench-gds-native clean

all: merge_test bloom_test gpcomp_unit_tests gpcomp_datagen gpcomp_bench

# --------------------------------------------------------------------- #
# Kernel 1 smoke-test driver                                              #
# --------------------------------------------------------------------- #
merge_test: gpcomp_merge.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_merge.cu

# --------------------------------------------------------------------- #
# Kernel 2 smoke-test driver                                              #
# --------------------------------------------------------------------- #
bloom_test: gpcomp_bloom.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_bloom.cu

# --------------------------------------------------------------------- #
# Unit-test suite (17 tests, self-contained single TU)                   #
# --------------------------------------------------------------------- #
gpcomp_unit_tests: gpcomp_tests.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_tests.cu

# --------------------------------------------------------------------- #
# Synthetic SST dataset generator (pure C++, no CUDA)                    #
#                                                                         #
# Mimics db_bench fillrandom with GP-Comp paper settings:                 #
#   write_buffer_size=8MB, target_file_size_base=8MB, value_size=64B,     #
#   level0_file_num_compaction_trigger=4, bloom_bits=10, seed=23          #
# --------------------------------------------------------------------- #
gpcomp_datagen: gpcomp_datagen.cpp
	$(CXX) $(CFLAGS) -o $@ gpcomp_datagen.cpp

# --------------------------------------------------------------------- #
# End-to-end kernel benchmark (reads dataset written by gpcomp_datagen)  #
# Build with GDS=1 to enable cuFile / GPUDirect Storage support.         #
# --------------------------------------------------------------------- #
gpcomp_bench: gpcomp_bench.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(CUFILE_FLAGS) -o $@ gpcomp_bench.cu

# --------------------------------------------------------------------- #
# Convenience: build & run the full unit-test suite                       #
# --------------------------------------------------------------------- #
test: gpcomp_unit_tests
	./gpcomp_unit_tests

# --------------------------------------------------------------------- #
# Convenience: generate default dataset (4 SSTs × 81920 keys) and bench  #
# --------------------------------------------------------------------- #
bench: gpcomp_datagen gpcomp_bench
	./gpcomp_datagen
	./gpcomp_bench

# --------------------------------------------------------------------- #
# Stress-test: 8 SSTs × 524288 keys (≈ L1→L2 compaction size)           #
# --------------------------------------------------------------------- #
bench-large: gpcomp_datagen gpcomp_bench
	./gpcomp_datagen --large
	./gpcomp_bench

# --------------------------------------------------------------------- #
# GDS bench (cuFile Compatibility Mode – RTX 3070 or any GPU)            #
#   Builds with GDS=1, generates dataset, runs Benchmark 4 with          #
#   --compat-mode flag (forces pinned bounce-buffer path via cuFile).     #
#   On hardware without native GDS the driver enters compat mode          #
#   automatically; --compat-mode makes the intent explicit and            #
#   guarantees compat mode even on GDS-capable machines for testing.      #
# --------------------------------------------------------------------- #
bench-gds: gpcomp_datagen
	$(MAKE) GDS=1 gpcomp_bench
	./gpcomp_datagen
	LD_LIBRARY_PATH=$(GDS_LOCAL)/lib:$$LD_LIBRARY_PATH ./gpcomp_bench --compat-mode

# --------------------------------------------------------------------- #
# GDS bench (Native GDS – A30 / A100 / H100)                             #
#   Run WITHOUT --compat-mode; cuFileDriverOpen() detects GDS hardware    #
#   and activates native DMA (PCIe Peer-to-Peer to VRAM).                #
# --------------------------------------------------------------------- #
bench-gds-native: gpcomp_datagen
	$(MAKE) GDS=1 gpcomp_bench
	./gpcomp_datagen
	LD_LIBRARY_PATH=$(GDS_LOCAL)/lib:$$LD_LIBRARY_PATH ./gpcomp_bench --gds

clean:
	rm -f merge_test bloom_test gpcomp_unit_tests gpcomp_datagen gpcomp_bench
	rm -rf dataset/
