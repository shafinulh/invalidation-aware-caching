# Makefile – GPComp CUDA kernels
#
# Targets:
#   all               – build all three binaries
#   merge_test        – smoke-test driver for Kernel 1
#   bloom_test        – smoke-test driver for Kernel 2
#   gpcomp_unit_tests – comprehensive unit-test suite (17 tests)
#   test              – build gpcomp_unit_tests and run it
#   clean             – remove all build artefacts

NVCC    := nvcc
CFLAGS  := -O2 -std=c++17

HEADERS := gpcomp_common.cuh gpcomp_merge.cuh gpcomp_bloom.cuh

.PHONY: all test clean

all: merge_test bloom_test gpcomp_unit_tests

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
# Convenience: build & run the full suite                                 #
# --------------------------------------------------------------------- #
test: gpcomp_unit_tests
	./gpcomp_unit_tests

clean:
	rm -f merge_test bloom_test gpcomp_unit_tests
