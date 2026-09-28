# Initial throughput benchmark

Run on 2026-09-27 on Linux with one pinned performance core (CPU 8) of an Intel Core i9-13900HX. GCC 13.3 compiled the generated executable with `-O3 -DNDEBUG`. The model was `sentence-transformers/static-retrieval-mrl-en-v1` at commit `f60985c706f192d45d218078e49e5a8b6f15283a` (1,024 dimensions).

The input was the first 10,000 questions from the `pair/train` split of [sentence-transformers/gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq), fetched through its dataset viewer API. The saved UTF-8 question file had SHA-256 `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`. Mean length: 43.38 characters; median WordPiece count: 10; 95th percentile: 14. The exact slice used by Flower Computer was not specified in its [post](https://www.flowercomputer.com/posts/fast-static-embedding/).

| Program and output | Queries per timed run | Median QPS | Timed runs |
| --- | ---: | ---: | --- |
| Icicle, portable `-O3`, binary float32 | 100,000 | 102,576 | 0.9749, 0.9710, 0.9802, 0.9734, 0.9782 s |
| Icicle, `-O3 -march=native`, binary float32 | 100,000 | 113,921 | 0.8811, 0.8778, 0.8818, 0.8776, 0.8775 s |
| Icicle, portable `-O3`, JSON Lines | 10,000 | 5,637 | 1.7753, 1.7682, 1.7740 s |
| Sentence Transformers 5.4.1, batch size 256, one Torch thread | 10,000 | 39,417 | 0.2603, 0.2537, 0.2531 s |

The Icicle runs included reading UTF-8 lines from a file and writing output to `/dev/null`. They used one warmup run and report median wall-clock throughput. The Python reference measured `model.encode(...)` in memory, after a warmup, with `torch.set_num_threads(1)` and the process pinned to CPU 8. Those are different output paths, so the local speedup is approximate: 2.6× for the portable binary-output build and 2.9× for the native build.

GCC's optimization report showed 16-byte vectorization for the dense accumulate and divide loops in the portable build, and 32-byte vectors with `-march=native`. The native build improved throughput by about 11%. A separate token-ID debug path reached about 128k QPS on the same input. This suggests tokenizer and per-line overhead are a major limit; the debug path still formats token IDs as JSON, so it is not a pure tokenizer microbenchmark. Hardware performance counters were unavailable (`perf_event_paranoid=4`).

The portable and native executables produced bit-identical float32 output for the first 1,000 questions in this corpus.

Flower Computer [reports](https://www.flowercomputer.com/posts/fast-static-embedding/) 4.5 million batched QPS and 524,900 single-query QPS on an M4 Mac Mini. Icicle's 114k streaming QPS is roughly 40× below that batched figure. Hardware, exact corpus slice, batching method, and output work differ, so this is a directional comparison rather than a head-to-head result. We have not reproduced their benchmark harness.

## Reproduce

```sh
python3 bench/benchmark.py build/embed-retrieval --questions 10000 --repeat 10 --runs 5 --format binary --cpu 8
python3 bench/benchmark.py build/embed-retrieval --questions 10000 --repeat 1 --runs 3 --format jsonl --cpu 8
taskset -c 8 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 bench/reference.py build/bench/gooaq-10000.txt --batch-size 256 --runs 3
```

To reproduce the CPU-specific build on an x86 machine, pass `--cxx` a wrapper that invokes `c++ -march=native "$@"`. It produces a binary tied to that CPU family; use the ordinary build for a portable executable.

## WordPiece lookup change

On the same CPU, model, corpus, and binary-output benchmark, replacing the token hash table with a direct offset (double-array) trie gave these results. Each run processed 100,000 queries; the values are medians of five timed runs after warmup.

| WordPiece lookup | Median QPS | Timed runs |
| --- | ---: | --- |
| Hash table, native build | 113,983 | 0.8787, 0.8752, 0.8783, 0.8773, 0.8765 s |
| Sorted-edge trie, native build | 109,069 | 0.9172, 0.9160, 0.9149, 0.9187, 0.9169 s |
| Direct offset trie, native build | 117,894 | 0.8516, 0.8443, 0.8470, 0.8482, 0.8492 s |

The direct offset trie was about 3.4% faster than the hash table in this end-to-end test. Each byte transition uses `base[node] + byte` and checks the stored parent index; no token candidate strings or hash probes are needed. The sorted-edge trie was slower because it searched each node's outgoing edges. The direct offset build produced identical token IDs on 10,000 GooAQ questions and identical binary dense output on the first 1,000. The sparse model produced identical token IDs and binary output on all 10,000 questions. The offline test also checks longest WordPiece match and `##` continuation behavior.

The `--tokens` debug path, which includes JSON token-ID formatting but skips pooling, rose from 128,073 to 133,740 median QPS (about 4.4%). The modest end-to-end gain suggests token lookup was one cost among normalization, input/output, and dense pooling; this benchmark does not isolate those costs individually.

## ASCII normalization path

The tokenizer now handles ASCII segments directly and keeps the ICU path for Unicode. On the same native dense build and 100,000-query binary-output benchmark, throughput rose from 118,820 QPS (0.8416 s median) to 379,709 QPS (0.2634 s median), or 3.20×. The native sparse query build reached 597,449 QPS (0.1674 s median).

```sh
python3 bench/benchmark.py build/embed-offset-native --questions 10000 --repeat 10 --runs 5 --format binary --cpu 8
```
