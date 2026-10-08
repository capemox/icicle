# Batch throughput improvements (2026-10-08)

The original 1,024-dimensional float32 model now reaches **3.77 million queries/s
through the CLI** and **4.80 million queries/s in memory** on this machine.
Optional full-width int8 weights reach **4.56 million CLI queries/s**. These
figures are specific to the workload and hardware below. The default build
still uses the original full-width float32 weights.

[BATCH_RESULTS.json](BATCH_RESULTS.json) contains all final timings, model
metadata, corpus checksums, and quality measurements.

## What changed

- Workers claim contiguous chunks of queries from an atomic counter. Fast cores
  can take more work, and queries of different lengths cause less idle time.
  Chunk sizes adapt to the batch size, up to 32 queries. The previous fixed
  assignments were removed after the comparison.
- Each worker's scratch header occupies separate cache lines. All workers
  share the same read-only tokenizer tables and model weights.
- `--pipeline` uses three reusable batch buffers. Workers encode batch B while
  the main thread writes B-1 and reads B+1. Output order is preserved. Keeping
  input on the main thread also lets input validation failures unwind without
  waiting for a background reader blocked on stdin.
- Dense buffers are allocated without initializing them on the input thread.
  Each worker clears its own output rows before accumulating weights. This
  removes a redundant serial zero-fill and moves initial page access to the
  workers. Buffers retain capacity across batches.
- Batches can contain up to 65,536 queries. Each batch remains limited to
  64 MiB of input and 256 MiB of dense results; the effective query count is
  reduced for wider embeddings. Three pipeline slots can use up to 768 MiB
  for dense results alone. Larger batches can increase latency and slow the
  CLI, as the measurements below show.
- `--benchmark` loads and validates inputs before timing. It measures the
  existing native worker engine directly, without Python in the timed loop.
- Dense builds support optional `--dimension N` and `--quantize int8`.
  Precision and dimension changes are explicit build options.

## Measurement conditions

- Intel Core i9-13900HX: eight performance cores, 16 efficiency cores, 32
  logical CPUs; Linux, GCC, generated executables built with `-O3 -march=native`.
- All batch runs were allowed on CPUs 0–31. Single-thread comparisons used
  CPU 0. Worker threads were not individually pinned.
- Dense: `sentence-transformers/static-retrieval-mrl-en-v1`, revision
  `f60985c706f192d45d218078e49e5a8b6f15283a`, 30,522 vocabulary rows, 1,024
  original dimensions. Its graph does not apply L2 normalization.
- Sparse: the query route of
  `monkeypostulate/inference-free-splade-distilbert-base-uncased-nq`, revision
  `1499ddff8c6545b3f3c7d103249bfbce64da3271`.
- Models were compiled from local snapshots of those revisions. Recorded
  executable metadata therefore says `revision: local`; the source pins above
  identify the checkpoints.
- Input: the same first 10,000 GooAQ questions used in earlier reports,
  repeated 100 times, for one million queries per run. Mean length: 43.38
  characters. SHA-256:
  `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.
- One warmup and three timed runs per configuration; QPS uses median elapsed
  time. Runs were sequential, without overlapping compilation or test jobs.

**CLI timing** includes process startup, reading and validating UTF-8 lines,
tokenization, pooling, and binary output writes to `/dev/null`. The Python
harness prepares a file before timing and only launches/waits for the process.
JSON serialization, disk storage of embeddings, and pipe consumers will have
different costs.

**In-memory timing** includes tokenization, pooling, dispatch, result stores,
and any output-buffer growth. It excludes input loading/validation, worker
startup, serialization, and the final checksum. An optimizer barrier after
every batch retains its output stores. The corpus is held as strings and
repetitions reference those strings; neither mode caches token IDs or embeddings.
Repeated short queries and warm model pages favor throughput over cold-start
or long-document performance.

## Full-width dense results

All rows use 32 workers. Units are **millions of queries per second**.

| Stored weights | Batch size | In memory | CLI | CLI with pipeline |
| --- | ---: | ---: | ---: | ---: |
| Float32, 1,024 dimensions | 4,096 | 3.91 | 2.63 | 3.29 |
| Float32, 1,024 dimensions | 16,384 | 4.35 | 3.03 | **3.77** |
| Float32, 1,024 dimensions | 65,536 | **4.80** | 3.23 | 3.34 |
| Int8, 1,024 dimensions | 4,096 | 4.41 | 2.92 | 3.70 |
| Int8, 1,024 dimensions | 16,384 | 5.39 | 3.66 | **4.56** |
| Int8, 1,024 dimensions | 65,536 | **5.85** | 3.73 | 3.84 |

For the 4.56M int8 CLI result, the three runs took 0.219318, 0.217823, and
0.225091 seconds for one million queries. The float32 CLI result took
0.262148, 0.265290, and 0.266671 seconds.

The preceding executable measured **1.91M CLI QPS** with its previous best
16-worker, 4,096-query setting in this session. The new best float32 setting
is about **1.97×** faster. This compares the complete configurations, including
worker and batch counts. Single-thread CLI throughput was 515K previously and
505K now; the changes chiefly improve parallel throughput.

## Smaller dense and sparse results

All rows use 24 workers. Units are **millions of queries per second**.

| Model | Batch size | In memory | CLI with pipeline |
| --- | ---: | ---: | ---: |
| Float32, 512 dimensions | 16,384 | 7.30 | **5.97** |
| Float32, 512 dimensions | 65,536 | **8.05** | 4.92 |
| Int8, 512 dimensions | 16,384 | 8.18 | **6.35** |
| Int8, 512 dimensions | 65,536 | **8.62** | 5.42 |
| Sparse query encoder | 4,096 | 18.89 | 8.37 |
| Sparse query encoder | 16,384 | 17.10 | **9.37** |
| Sparse query encoder | 65,536 | **20.26** | 7.89 |

Sparse results emit a few token/weight pairs per query; dense results emit
512 or 1,024 float32 values. Their output and arithmetic costs differ.

## Compact weights and retrieval quality

Dimension truncation physically retains the first N coordinates of every row.
For Matryoshka models this permits a smaller retrieval representation; reduced
dimensions still need quality evaluation for the intended corpus.

Int8 storage uses symmetric quantization independently for each retained row:
`scale = max(abs(row)) / 127`, then `q = round(row / scale)`, limited to
[-127, 127]. All-zero rows use scale 1; subnormal rows get an underflow fallback.
Nonfinite retained weights are rejected. The runtime reconstructs each value
as `float(q) * scale` while adding rows, then averages and applies the model's
optional normalization. Accumulation and output remain float32. This reduces
weight storage and traffic; the cost includes int8-to-float conversion.

`bench/quality.py` evaluates the pinned
`zeta-alpha-ai/NanoMSMARCO` revision
`7b8ff22f2771dc65ac5b439f222eb19a1f56abda`: **50 queries and 5,043 documents**.
It embeds both sides, ranks by cosine similarity, and compares against qrels.
This is one NanoBEIR subset, not the full suite.

| Model | NDCG@10 | MRR@10 | Recall@10 | Mean cosine to float32 prefix |
| --- | ---: | ---: | ---: | ---: |
| Float32, 1,024 dimensions | 0.4041 | 0.3245 | 0.6600 | Reference |
| Int8, 1,024 dimensions | 0.4041 | 0.3245 | 0.6600 | 0.999978 |
| Float32, 512 dimensions | 0.3893 | 0.3173 | 0.6200 | 1.000000 |
| Int8, 512 dimensions | 0.3893 | 0.3173 | 0.6200 | 0.999980 |

Full-width int8 changed some rankings (99.8% mean top-10 overlap) despite
matching these aggregate metrics. It is approximate, and this small test
does not establish lossless retrieval quality. Truncating to 512 reduced
NDCG@10 by about 3.7% relative. The unnormalized 512-dimensional float32
outputs were byte-identical to the original vectors' prefixes on this test.
Executable sizes were approximately 121 MiB / 32 MiB for full float32 / int8,
and 61 MiB / 17 MiB for 512-dimensional float32 / int8.

The default full-width float32 stream and pipeline outputs also matched the
preceding executable byte-for-byte on 1,000 GooAQ queries, for both dense and
sparse models. Offline integration tests cover WordPiece and BPE, output
order, partial batches, invalid input while workers are active, input byte
limits, truncation, and independent int8 arithmetic expectations.

## Experiments without a consistent gain

Dynamic scheduling improved a preliminary 24-worker, 16,384-query in-memory
run from 3.30M to 3.96M QPS against fixed assignments. Avoiding duplicate dense
buffer initialization improved paired CLI runs, especially at large batches;
one 32-worker, 16,384-query pipeline comparison went from 3.05M to 3.73M QPS.
These intermediate measurements explain implementation choices; the tables
above use the final runtime.

The discarded LTO experiment measured 4.45M versus 4.35M in the final
32-worker, 16,384-query float32 in-memory comparison. Earlier comparisons
did not show a consistent gain. Its build flag, metadata, and test scaffolding
have been removed. The recorded JSON retains this historical measurement;
current builds do not expose an LTO option.

An experimental callback pooled rows as committed tokenizer pieces were
produced; another version accumulated into worker-local storage before copying
results. Neither gave a consistent improvement, and both prototypes were
removed. The current tokenizer still produces token IDs before pooling.

## Comparison with Flower Computer

The [Flower Computer post](https://www.flowercomputer.com/posts/fast-static-embedding/)
reports about 4.5M batched QPS on an M4 Mac mini. Its public
[Python API build configuration](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/api-py/Cargo.toml)
enables Rayon, 512 dimensions, and 8-bit weights. Its
[batch loader](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/benches/py/model_loader.py)
defaults to 65,536 queries; its
[benchmark](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/benches/py/bench.py)
loads data before timing `encode`. These source files were published after
the blog, so they do not prove which configuration produced its original figure.

Our compact build, parallel workers, and in-memory harness let us investigate
similar choices. Our CLI also measures input/output costs, while the native
harness excludes Python binding and returned-array creation. Hardware, corpus
selection, repetition, and timing boundaries differ. These results put Icicle
in the same broad throughput range; they do not establish a head-to-head win.

## Reproduce

Build each dense variant with the same pinned checkpoint and native compiler
wrapper. Repeat the command with `--quantize int8`, `--dimension 512`, both
options to produce the desired variant:

```sh
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --output build/embed-batch-final --cxx ./bench/native-cxx.sh
./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --revision 1499ddff8c6545b3f3c7d103249bfbce64da3271 \
  --route query --output build/embed-batch-final-sparse --cxx ./bench/native-cxx.sh
```

If the corpus file is absent, the existing benchmark script downloads the same
GooAQ slice. Its resulting checksum must match the one above:

```sh
python3 bench/benchmark.py build/embed-batch-final \
  --questions 10000 --repeat 1 --runs 1 --format binary
taskset -c 0-31 python3 bench/batch_sweep.py build/embed-batch-final \
  --threads 32 --batch-sizes 4096,16384,65536 > results.json
```

Use 24 workers and `--pipeline on` for the 512-dimensional and sparse sweeps.
On macOS omit `taskset`; the compiler wrapper uses portable compiler options
there.

For quality measurements, install NumPy and PyArrow and pass the full-width
float32 executable first, followed by the separately built compact variants:

```sh
python3 bench/quality.py build/embed-batch-final build/embed-batch-final-int8 \
  build/embed-batch-final-512 build/embed-batch-final-512-int8 > quality.json
```

Start with `--threads 32 --batch-size 16384 --pipeline` for CLI throughput on
this CPU, then sweep for your workload. Keep the default float32 build when
exact original weight values matter; evaluate int8 or fewer dimensions against
your retrieval data before adopting them. Linux correctness and installed
compiler tests passed here. macOS execution still requires the existing CI
runner; no macOS machine was available for this measurement.
