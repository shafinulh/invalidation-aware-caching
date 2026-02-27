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
#   clean             – remove all build artefacts (binaries + dataset/)

NVCC    := nvcc
CXX     := g++
CFLAGS  := -O2 -std=c++17

HEADERS := gpcomp_common.cuh gpcomp_merge.cuh gpcomp_bloom.cuh

.PHONY: all test bench bench-large clean

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
# --------------------------------------------------------------------- #
gpcomp_bench: gpcomp_bench.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_bench.cu

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

clean:
	rm -f merge_test bloom_test gpcomp_unit_tests gpcomp_datagen gpcomp_bench
	rm -rf dataset/
