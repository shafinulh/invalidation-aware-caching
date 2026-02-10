# db_bench scripts

## macOS build command:
```
cd ~/rocksdb-gpu

rm -f make_config.mk

CPATH="$(brew --prefix)/include" \
LIBRARY_PATH="$(brew --prefix)/lib" \
make -B -j 8 DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 db_bench
```

## Run
First set the env vars in `bench_env.sh`

```
./scripts/run_fillrandom.sh
./scripts/run_readwritemix.sh
```

## workload configs (change in the scripts)
- `THREADS=8`
- `VALUE_SIZES="32 64 128 256"`
- `NUM_KEYS=200000000` (FillRandom)
- `LOAD_NUM=50000000`, `MIX_NUM=100000000` (ReadWriteMix)
- `WRITE_RATIOS="10 20 50"` (ReadWriteMix)