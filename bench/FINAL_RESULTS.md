# Final throughput benchmark (2026-09-27)

These results use generated C++ executables built from the regular runtime. The dense model is `sentence-transformers/static-retrieval-mrl-en-v1` at revision `f60985c706f192d45d218078e49e5a8b6f15283a` (1,024 dimensions). The sparse model is the **query route** of `monkeypostulate/inference-free-splade-distilbert-base-uncased-nq` at revision `1499ddff8c6545b3f3c7d103249bfbce64da3271` (30,522 dimensions).

The host was an Intel Core i9-13900HX running Linux, with eight performance cores, 16 efficiency cores, and 32 logical CPUs. Both executables used `-O3 -march=native`. Input was the same 10,000 GooAQ questions (mean 43.38 characters; SHA-256 `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`). Output used the CLI's binary format and was directed to `/dev/null`. Timings include process startup, reading, tokenization, embedding, and output writes; model download and compilation are outside the timed region. Each configuration had one warmup and three timed runs; QPS is computed from the median wall time.

| Mode | Queries/run | Batch size | Dense QPS | Sparse query QPS |
| --- | ---: | ---: | ---: | ---: |
| One process, one thread | 1,000,000 | streaming | 521,259 | 1,227,407 |
| One process, four workers | 1,000,000 | 256 | 1,527,511 | 1,629,797 |
| One process, eight workers | 1,000,000 | 256 | 1,649,096 | 1,943,088 |
| One process, 16 workers | 1,000,000 | 4,096 | 1,940,708 | 2,020,958 |
| One process, 32 workers | 1,000,000 | 4,096 | 1,540,203 | 2,169,149 |
| Eight independent processes | 1,600,000 | streaming | 3,386,637 | 8,862,514 |
| 32 independent processes | 6,400,000 | streaming | 5,010,771 | 16,144,986 |

The single-thread runs were pinned to CPU 0. Four- and eight-worker runs were allowed on CPUs `0,2,4,6,8,10,12,14`, which are distinct performance cores; workers were not individually pinned. The 16- and 32-worker runs were allowed on CPUs `0-31`. The process runs pinned each worker to a separate logical CPU, taking eight performance cores first, then efficiency cores, then performance-core sibling threads. Each process handled 200,000 queries, so total queries per run differ from the single-process rows. This is sustained CLI throughput, not an in-memory encoder microbenchmark.

The fastest measured single-process dense setting was 16 workers; sparse was fastest with 32 workers. Separate processes scaled better, especially for the sparse model. On Linux, the executable's read-only model pages are file-backed and shared physically across processes; each process still owns its small mutable state. These figures are specific to this CPU, corpus, compiler settings, and output format.

Relative to the previous runtime on the same benchmark, reusing tokenization buffers and scanning ASCII in SIMD-sized chunks raised single-thread dense throughput from 466,939 to 521,259 QPS (11.6%) and sparse throughput from 1,064,234 to 1,227,407 QPS (15.3%). In paired runs, the SIMD scan alone typically changed throughput by only about 1–2%; most of the gain came from avoiding per-query allocations. These are end-to-end comparisons, including output handling.

## Reproduce

Build the compiler and generated executables:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --output build/embed-final-static --cxx ./bench/native-cxx.sh
./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --revision 1499ddff8c6545b3f3c7d103249bfbce64da3271 \
  --route query --output build/embed-final-sparse --cxx ./bench/native-cxx.sh
```

For either executable, run the single-process benchmark with the CPU set and worker options from the table. Examples:

```sh
taskset -c 0 python3 bench/benchmark.py build/embed-final-static \
  --questions 10000 --repeat 100 --runs 3 --format binary
taskset -c 0-31 python3 bench/benchmark.py build/embed-final-sparse \
  --questions 10000 --repeat 100 --runs 3 --format binary \
  --threads 16 --batch-size 4096
```

For aggregate process throughput, replace the executable path to test the other model:

```sh
python3 bench/parallel.py build/embed-final-static \
  --corpus build/bench/gooaq-10000.txt --repeat 20 --workers 8,32 \
  --cpus 8,12,0,2,4,6,10,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,1,3,5,7,9,11,13,15 \
  --runs 3
```
