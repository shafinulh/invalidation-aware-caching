NVCC   := nvcc
HOSTCXX := g++-12
CFLAGS := -O2 -std=c++17 -ccbin $(HOSTCXX) -allow-unsupported-compiler -Xcompiler -U_FORTIFY_SOURCE -Xcompiler -D_FORTIFY_SOURCE=0

HEADERS := gpcomp_common.cuh gpcomp_merge.cuh gpcomp_bloom.cuh gpcomp_pack.cuh gpcomp_sstable.cuh gpcomp_pipeline.cuh gpcomp_dummy_data.cuh

.PHONY: all test bench clean

all: gpcomp_datagen gpcomp_unit_tests gpcomp_bench

gpcomp_datagen: gpcomp_datagen.cpp $(HEADERS)
	$(NVCC) -x cu $(CFLAGS) -o $@ gpcomp_datagen.cpp

gpcomp_unit_tests: gpcomp_tests.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_tests.cu

gpcomp_bench: gpcomp_bench.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ gpcomp_bench.cu

test: gpcomp_unit_tests
	./gpcomp_unit_tests

bench: gpcomp_datagen gpcomp_bench
	./gpcomp_datagen --out_dir dataset
	./gpcomp_bench --dataset dataset

clean:
	rm -f gpcomp_datagen gpcomp_unit_tests gpcomp_bench
