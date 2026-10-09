# Icicle: architecture, performance, and validation report

**Prepared:** 8 October 2026

For the 9 October comparison with Hugging Face defaults and full-array output, see [Throughput compared with Hugging Face](../bench/HF_COMPARISON.md).

**Scope:** the C++ implementation in this repository after experimental code cleanup

**Audience:** readers who want both an approachable explanation and enough detail to inspect the implementation

## 1. Summary

Icicle compiles a supported, already trained Hugging Face text encoder into a
native command line executable. That executable includes its model weights
and tokenizer tables. It accepts one UTF-8 query per input line and emits one
dense or sparse result, preserving input order.

The supported computation is deliberately small. A dense static encoder
averages fixed token vectors. An inference-free SPLADE query encoder retrieves
one learned scalar for each distinct token. Neither supported path runs a
transformer during query encoding.

The main performance improvements are compiled lookup tables, an ASCII fast
path, reusable memory, parallel batches, dynamic work distribution, pipelined
input/output, and binary result serialization. Optional int8 weight storage
and fewer output dimensions reduce memory traffic and computation further.

The strongest recorded CLI results on an Intel i9-13900HX were:

| Configuration | Queries per second |
| --- | ---: |
| Dense, original 1,024 dimensions, float32 | 3.77 million |
| Dense, 1,024 dimensions, int8 weights | 4.56 million |
| Dense, 512 dimensions, float32 | 5.97 million |
| Dense, 512 dimensions, int8 weights | 6.35 million |
| Inference-free SPLADE query encoder | 9.37 million |

These are aggregate throughput measurements using tuned worker/batch settings,
short repeated queries, and binary output to `/dev/null`. They are not latency
guarantees or measurements of saving embeddings to disk. The full-width
float32 model also reached 4.80 million queries/s in an in-memory benchmark
that excludes input handling and output serialization.

The default build retains full-width float32 weights. Compact formats are
explicit options. A small retrieval evaluation found unchanged aggregate
scores with int8, while using 512 dimensions reduced NDCG@10 by approximately
3.7% relative. Broader quality validation remains necessary.

The performance runs preceded the final removal of comparison-only switches.
Cleanup removed the LTO option and fixed worker scheduling without changing
the retained dynamic scheduling, tokenization, pooling, or quantization
algorithms. The numbers in this report are the recorded runs, not new runs of
the cleaned executable. Linux integration and installed-compiler tests passed
after cleanup. Actual macOS compilation and execution remain unverified here.

## 2. A short vocabulary

| Term | Meaning here |
| --- | --- |
| Token | A vocabulary unit, which can be a whole word, a word fragment, or punctuation |
| Token ID | The integer identifying a token in the model's vocabulary |
| Embedding | A vector of numbers used to represent text for retrieval |
| Dimension | The number of coordinates in a vector |
| Dense vector | An output containing every coordinate |
| Sparse vector | An output containing only active coordinates and their indexes |
| Pooling | Combining the token representations into one text representation |
| SIMD | CPU instructions that apply an operation to several values at once |
| QPS | Queries processed per second |
| Batch | Several independent queries handled together |
| Quantization | Approximating weights using a smaller numeric representation |

## 3. Dense static encoding

### 3.1 The data

The model is a matrix with one row per token ID. Each row is a fixed vector.
The benchmark model has 30,522 vocabulary rows and 1,024 coordinates per row.
Its float32 weight matrix occupies about 119.23 MiB before tokenizer tables
and executable code are added.

A token's row does not change based on surrounding words. Contextual
transformer encoders perform much more work to create a representation that
depends on the sentence. A static encoder has already learned the rows during
training; Icicle only reads and combines them at runtime.

### 3.2 The operation

For token IDs `t[0] ... t[K-1]`, a matrix `W`, and output dimension `D`:

```text
result[j] = (W[t[0], j] + ... + W[t[K-1], j]) / K
for j = 0 ... D-1
```

For an empty token sequence, the result is the zero vector. If the saved graph
contains a `Normalize` module, the runtime divides the pooled vector by its
Euclidean length, with a small lower bound to avoid division by zero. The
specific StaticMRL checkpoint benchmarked here does not contain that module.

For a two-dimensional example:

```text
hello -> [2, 0]
world -> [0, 4]

hello world -> ([2, 0] + [0, 4]) / 2 -> [1, 2]
```

Repeated tokens contribute repeatedly. After tokenization, averaging does not
preserve token order. A saved prompt is prepended before tokenization and can
therefore affect the result. `--prompt` overrides that prompt.

### 3.3 Cost

Ignoring tokenization, dense encoding performs approximately `K * D` additions
and reads `K * D` stored weights. Mean pooling performs another `D` divisions.
Optional normalization adds a length calculation and division pass.

For eight tokens and 1,024 float32 dimensions, fetching rows reads about
32 KiB of weights, before accounting for CPU cache reuse. Short queries have
small token counts; longer documents require more work. A 512-dimensional
build halves the row width, coordinate work, and dense output size.

## 4. Inference-free SPLADE query encoding

### 4.1 The data and operation

The supported query module is `SparseStaticEmbedding`. It contains one
learned scalar for each token ID, rather than a full dense vector per token.

For the set of token IDs present after tokenization:

```text
result[token_id] = learned_weight[token_id]
all other coordinates = 0
```

The implementation sorts the token ID list, removes duplicates, discards IDs
whose learned weight is zero, and emits the remaining `(ID, weight)` pairs.

For example:

```text
hello -> token ID 101, weight 1.5
world -> token ID 203, weight 2.5

hello hello world -> [(101, 1.5), (203, 2.5)]
```

There is no averaging and no sum for repeated terms. The conceptual vector has
30,522 dimensions in the benchmark, but the program never constructs that
dense vocabulary-sized output. It emits only active pairs, sorted by ID.

### 4.2 What inference-free means

This query path performs tokenization, deduplication, and scalar lookups. It
has learned weights, but it performs no neural network forward pass.

An asymmetric SPLADE model can still use a transformer to encode documents.
Icicle downloads and compiles the static query route only. A retrieval system
must supply compatible document vectors and an index separately. Icicle does
not build that index or perform search.

### 4.3 Why it is faster than the dense path

For the same token count, sparse encoding reads a few scalars and emits a few
pairs. Dense encoding reads and combines an entire vector for every token and
writes every output coordinate. Tokenization and I/O consequently make up a
larger share of the sparse path's total cost. The current measurements do not
provide a fresh per-stage timing breakdown; no exact percentages are inferred
from the throughput comparisons.

## 5. Build-time compilation

### 5.1 Reading the checkpoint

The C++ build tool accepts a Hub model ID or a local snapshot. For Hub builds,
it resolves the requested revision to a commit, invokes `curl` for the needed
files, and caches them by model/revision. The default cache is
`$XDG_CACHE_HOME/icicle` or `~/.cache/icicle`.

The importer checks the module graph, selects the supported dense or static
query adapter, validates tokenizer settings, and checks the safetensors tensor
rank, dtype, dimensions, offsets, and payload size. The current importer
requires a single float32 tensor; int8 is an output storage transformation,
not a supported checkpoint input dtype.

Arbitrary transformer modules, unsupported tokenizer pipelines, and unexpected
tensors are rejected. The tool reads model data without executing model
repository Python code.

### 5.2 Building lookup tables

The vocabulary becomes a direct-offset trie. A trie represents shared token
prefixes as a sequence of transitions. Each node stores a base offset, parent
index, and optional token ID. Advancing by one input byte uses:

```text
slot = nodes[current].base + input_byte
accept slot when it is in bounds and nodes[slot].parent == current
```

This avoids a runtime vocabulary hash table. WordPiece can walk the trie while
tracking its longest matching prefix without allocating candidate substrings.
BPE builds additional merge tables described in section 6.

### 5.3 Linking a native executable

Weights and lookup tables are packed into aligned binary data. A small
generated header records constants such as dimension, model kind, table
offsets, weight format, normalization, model identity, and revision. An
assembly `.incbin` directive embeds the packed data in a read-only section:
ELF `.rodata` on Linux or Mach-O `__TEXT,__const` on macOS.

This avoids generating millions of C++ floating-point literals. The C++
compiler optimizes a small runtime and model descriptor; the linker includes
the large weight blob as data. Normal builds use C++20, `-O3`, and threads.
The benchmark compiler wrapper additionally uses `-march=native` on Linux.

The OS maps the read-only data when the executable runs. Model files are not
parsed at startup, although accessing pages still incurs normal memory and
page-fault costs. The executable remains dependent on its platform's ICU and
C++ runtime libraries and is specific to its target OS and CPU architecture.

## 6. Tokenization

Tokenization is significant because the downstream math is so simple. It must
still preserve the saved tokenizer's behavior, including Unicode processing,
added tokens, unknown tokens, and truncation.

### 6.1 WordPiece

The supported pipeline uses `BertNormalizer`, `BertPreTokenizer`, and WordPiece
with the conventional `##` continuation prefix. The runtime normalizes text,
splits words and punctuation, and greedily chooses the longest vocabulary
prefix. Subsequent pieces use the continuation prefix. Failed word splits
roll back partial IDs and emit the configured unknown token.

ASCII inputs avoid conversion to ICU Unicode strings. Byte operations handle
ASCII cleaning, lowercasing, and splitting. A SIMD scan detects ASCII using
16-byte chunks with SSE2 on x86 or NEON on ARM, followed by a scalar tail.
This scan also helps validate ASCII input cheaply.

Non-ASCII inputs use ICU for Unicode categories, lowercasing, and optional
accent removal. Lowercasing follows per-code-point behavior to match the
supported tokenizer; whole-string ICU lowercasing can apply contextual rules
that alter token IDs. Replacing these operations with approximate byte rules
would risk changing embeddings.

The benchmark corpus was especially favorable to this fast path: 9,996 of its
10,000 queries were ASCII-only. This percentage should not be assumed for
multilingual production traffic.

### 6.2 ByteLevel BPE

The supported BPE subset has a null normalizer, a ByteLevel pre-tokenizer, and
standard deterministic merge rules. Bytes first map to ByteLevel vocabulary
symbols. The algorithm repeatedly merges the adjacent pair with the best
saved rank, until no eligible pair remains.

Retained optimizations are:

- ASCII pre-tokenization implements the supported ByteLevel splitting behavior
  directly; Unicode inputs retain ICU regex handling.
- A 256-entry byte-to-token mapping is prepared once per tokenizer.
- A per-left-token offset index narrows searches for general merge rules.
- A shared 256 by 256 initial pair table stores merge rank and result ID. Its
  two uint32 values per entry occupy exactly 512 KiB.
- Short pieces cache pair ranks and update neighboring pairs after a merge.
- Pieces with at least 128 symbols use stable linked indexes and a minimum
  heap. Versions invalidate stale candidates, and original position resolves
  equal-rank ties. This avoids repeated full scans and vector erases on long
  pieces. Workers reuse their heap and link storage.

This is a defined subset of BPE configurations. It does not cover every modern
tokenizer that happens to use BPE. In particular, arbitrary normalizer or
pre-tokenizer sequences, dropout, and byte fallback are unsupported.

### 6.3 Special tokens and truncation

The runtime does not automatically insert model post-processing tokens such
as CLS and SEP. Supported literal added tokens are recognized according to
their saved matching options. Unsupported matching rules fail during import.
Configured right truncation is respected; sparse model maximum-length metadata
can further bound the query token count.

## 7. Arithmetic and SIMD

Dense rows are contiguous arrays. Addition and division over their coordinates
are straightforward loops that permit compiler auto-vectorization. SIMD can
operate on several coordinates at once, reducing instruction overhead.

The dense dimension, model kind, and weight format are constants in generated
code. Dense, sparse, float32, and int8 choices do not require selecting a
generic machine-learning operator for each query. Unused runtime branches can
be eliminated by the compiler. No fast-math flag is enabled.

There are two distinct uses of SIMD: explicit instructions for ASCII detection
and compiler optimization of coordinate loops. The entire tokenizer/pooling
pipeline is not one SIMD operation. Trie transitions and BPE decisions contain
branches and dependencies, and Unicode handling calls ICU. The compiler does
not turn unknown future queries into precomputed embeddings.

An attempted callback fusion pooled weights as tokenizer pieces were committed.
It did not improve throughput consistently, so it was removed. The retained
runtime first produces token IDs, then performs pooling or sparse weight lookup.

## 8. Batching, threads, memory, and I/O

### 8.1 Shared data and separate scratch

All worker threads use one shared read-only weight blob and tokenizer object.
Each has its own token ID list, normalization storage, and BPE scratch buffers.
Workers reuse their storage across queries and batches. CPU caches can contain
copies of frequently accessed data, but the implementation does not allocate
a full model per worker.

Scratch headers are aligned to separate 64-byte cache lines. This reduces
false sharing: one core's updates to its scratch header need not invalidate a
neighboring core's independently modified header on the same cache line.

### 8.2 Work distribution

The worker pool persists across batches. The main thread wakes it for a batch;
workers claim contiguous query chunks from an atomic counter. Chunk size
adapts to the batch size and worker count, up to 32 queries.

Queries have unequal lengths, and this CPU has unequal performance and
efficiency cores. Dynamic claims allow a worker that finishes early to take
more queries. Chunking also avoids an atomic operation for every short query.
Each result has a fixed position in the batch, so completion order can vary
while output remains in original input order. A one-worker batch runs directly
without creating a thread pool.

### 8.3 Dense buffers

Dense output buffers retain capacity. Allocation leaves their floats
uninitialized; each worker explicitly clears its own row before adding
weights. Output is read only after workers finish. This avoids clearing a
large buffer twice and avoids serially touching all its pages on the input
thread before the workers use it.

### 8.4 Three-buffer pipeline

With `--pipeline`, work can overlap as follows:

```text
Worker pool:    encode batch B
Main thread:   write batch B-1, then read batch B+1
```

Three reusable slots hold the previous, current, and next batches. Reading and
writing both remain on the main thread; the design overlaps them with encoding
rather than assigning a separate thread to each I/O operation. This avoids an
input reader that cannot be joined while blocked on stdin after another stage
fails. Worker threads are joined before their buffers are destroyed during
normal completion or exceptions.

Pipelining can improve sustained throughput but increase output latency and
memory use. Very large batches were slower through the CLI in these runs,
even when their in-memory encoding throughput improved.

### 8.5 Accepted limits and serialization

| Limit or default | Value |
| --- | ---: |
| Default operation | One line at a time, one thread |
| Default batch size when batching is enabled | 256 queries |
| Maximum requested batch size | 65,536 queries |
| Maximum accepted input line | 16 MiB |
| Input text per batch | 64 MiB |
| Dense results per batch | 256 MiB |
| Native benchmark corpus | 256 MiB of text and 10 million lines |
| Native benchmark expanded references | 10 million |

Dense result limits reduce effective batch size when needed. Three pipeline
slots can hold up to 768 MiB of dense outputs, in addition to weights, input,
and worker scratch. These are component limits rather than a total RSS bound.

JSON Lines is convenient for inspection but requires decimal conversion and
many formatting operations. Binary dense output writes contiguous float32
coordinates. Binary sparse batches pack records in a reusable buffer capped
at 1 MiB, reducing ostream calls. Binary data is little-endian and has no
global file header; consumers need the model information to interpret it.

## 9. Optional compact weights

### 9.1 Dimension reduction

`--dimension N` retains the first N coordinates of every dense weight row.
This physically narrows rows; the runtime performs less addition, reads less
weight data, and emits smaller vectors. Matryoshka models are trained to allow
useful prefixes, but the quality of a chosen dimension still needs evaluation.
The build rejects a requested dimension larger than the checkpoint dimension.
Normalization, when configured, operates on the shortened vector.

### 9.2 Int8 quantization

`--quantize int8` applies symmetric quantization independently to each retained
row:

```text
scale = max(abs(row)) / 127
q[j] = round(row[j] / scale), clamped to [-127, 127]
reconstructed_value[j] = float(q[j]) * scale
```

Zero rows use scale 1. Subnormal rows have an underflow fallback. Nonfinite
retained weights are rejected. The signed bytes and one float32 scale per row
are stored in the executable. Runtime accumulation and output remain float32.

For example, `[0.2, -1.0]` uses a scale near `1/127` and reconstructs the first
coordinate near `0.19685`. This is an approximation. The benefit comes from
smaller rows and lower weight traffic, with an extra conversion/multiplication
cost during accumulation. Int8 does not shrink the float32 output records.

Approximate executable sizes were 121 MiB for full float32, 32 MiB for full
int8, 61 MiB for 512-dimensional float32, and 17 MiB for 512-dimensional int8.
Both compact options apply only to dense models and are opt-in.

## 10. Benchmark methodology and results

### 10.1 Workload and hardware

- Intel Core i9-13900HX: eight performance cores, 16 efficiency cores,
  32 logical CPUs; Linux; generated programs compiled with `-O3 -march=native`.
- Dense model: `sentence-transformers/static-retrieval-mrl-en-v1`, revision
  `f60985c706f192d45d218078e49e5a8b6f15283a`.
- Sparse query model:
  `monkeypostulate/inference-free-splade-distilbert-base-uncased-nq`, revision
  `1499ddff8c6545b3f3c7d103249bfbce64da3271`.
- First 10,000 GooAQ questions, repeated 100 times: one million queries per
  run, averaging 43.38 characters per unique query.
- Corpus SHA-256:
  `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.
- One warmup and three measured runs; reported QPS uses median elapsed time.
- Batch runs could use CPUs 0–31; workers were not individually pinned.
  Single-thread comparisons used CPU 0.
- Final performance runs were sequential without overlapping compilation or
  tests. Local snapshot builds reported `revision: local`; the source pins
  above identify the downloaded checkpoints.

### 10.2 Two timing boundaries

**CLI timing** includes process startup, line reading, UTF-8 validation,
tokenization, embedding, and binary writes to `/dev/null`. The Python harness
prepares a file before timing, launches the executable, and waits. Python
does not feed or process queries in the timed hot loop.

**In-memory timing** loads and validates the corpus before the clock starts,
creates repeated string references, and starts the worker pool outside timing.
The timed loop includes tokenization, pooling, dispatch, output stores, and
any buffer growth. It excludes I/O, validation, serialization, and a diagnostic
checksum. An optimizer barrier after every batch retains its stores.

Neither mode caches token IDs or query embeddings. Warm pages and repeated
access to common token rows can nevertheless favor cache behavior. In-memory
QPS must not be presented as CLI throughput. These short runs measure warm
throughput rather than sustained thermal behavior over many minutes.

### 10.3 Full-width dense results

All rows use 32 workers. Values are millions of queries per second.

| Weights and dimensions | Batch | In memory | CLI | CLI with pipeline |
| --- | ---: | ---: | ---: | ---: |
| Float32, 1,024 | 4,096 | 3.91 | 2.63 | 3.29 |
| Float32, 1,024 | 16,384 | 4.35 | 3.03 | 3.77 |
| Float32, 1,024 | 65,536 | 4.80 | 3.23 | 3.34 |
| Int8, 1,024 | 4,096 | 4.41 | 2.92 | 3.70 |
| Int8, 1,024 | 16,384 | 5.39 | 3.66 | 4.56 |
| Int8, 1,024 | 65,536 | 5.85 | 3.73 | 3.84 |

The full-width int8 CLI runs at batch 16,384 took 0.219318, 0.217823, and
0.225091 seconds per million queries. Float32 runs took 0.262148, 0.265290,
and 0.266671 seconds.

The preceding executable's previous best setting, 16 workers and batch 4,096,
measured 1.91M CLI QPS in the same session. The new best float32 configuration
was approximately 1.97 times faster. This comparison includes tuning worker
and batch counts; it is not an isolated speedup attributable to one change.
Single-thread CLI throughput measured 515K previously and 505K with the new
runtime. The latest major gains are in parallel execution and I/O overlap.

### 10.4 Smaller dense and sparse results

All rows use 24 workers. Values are millions of queries per second.

| Model | Batch | In memory | CLI with pipeline |
| --- | ---: | ---: | ---: |
| Float32, 512 dimensions | 16,384 | 7.30 | 5.97 |
| Float32, 512 dimensions | 65,536 | 8.05 | 4.92 |
| Int8, 512 dimensions | 16,384 | 8.18 | 6.35 |
| Int8, 512 dimensions | 65,536 | 8.62 | 5.42 |
| Sparse query encoder | 4,096 | 18.89 | 8.37 |
| Sparse query encoder | 16,384 | 17.10 | 9.37 |
| Sparse query encoder | 65,536 | 20.26 | 7.89 |

Sparse and dense have different output sizes and arithmetic. A higher sparse
QPS does not mean the same representation was computed more quickly.

### 10.5 Evidence for individual choices

Intermediate comparisons supported dynamic scheduling: a 24-worker,
16,384-query in-memory run improved from 3.30M with fixed assignments to 3.96M
with dynamic chunks. Removing duplicate dense buffer initialization improved
one paired 32-worker, batch-16,384 CLI pipeline comparison from 3.05M to 3.73M.
The final float32 batch-16,384 table also shows pipelining raising CLI throughput
from 3.03M to 3.77M. These measurements are not independent factors to multiply.

### 10.6 Limits of the throughput claim

The main limitations are different CPUs, query lengths, Unicode content,
tokenizers, weight working sets, compilers, batch settings, and output
destinations. This corpus is nearly all ASCII and repeats 10,000 short queries.
A production stream of novel or long documents can behave differently.

Each full-width dense result is 4,096 bytes. At 3.77M QPS, materializing those
results represents about 15.44 GB/s of output data. `/dev/null` accepts writes
without storing those bytes or transmitting them to another machine. A real
disk, pipe consumer, or network link can therefore become the throughput limit.

Throughput is also different from response latency. Waiting to fill a large
batch adds delay. Dividing one second by aggregate QPS does not measure a
single request's end-to-end latency.

## 11. Retrieval quality and tokenizer validation

### 11.1 Compact dense retrieval evaluation

The evaluation used `zeta-alpha-ai/NanoMSMARCO` revision
`7b8ff22f2771dc65ac5b439f222eb19a1f56abda`: 50 queries and 5,043 documents.
Both sides were embedded, normalized for cosine comparison, and ranked against
the dataset's relevance labels. This is one NanoBEIR subset, not the full suite.

| Model | NDCG@10 | MRR@10 | Recall@10 | Mean cosine to float32 prefix |
| --- | ---: | ---: | ---: | ---: |
| Float32, 1,024 | 0.4041 | 0.3245 | 0.6600 | Reference |
| Int8, 1,024 | 0.4041 | 0.3245 | 0.6600 | 0.999978 |
| Float32, 512 | 0.3893 | 0.3173 | 0.6200 | 1.000000 |
| Int8, 512 | 0.3893 | 0.3173 | 0.6200 | 0.999980 |

NDCG@10 rewards placing relevant documents near the top ten positions. MRR@10
tracks the reciprocal rank of the first relevant result within ten. Recall@10
measures the fraction of relevant documents found in those ten. Higher is
better for all three.

Full-width int8 had 99.8% mean top-ten overlap with float32: some rankings
changed even though aggregate metrics matched. Matching this small test does
not establish universally unchanged retrieval quality. Using 512 dimensions
reduced NDCG by about 0.0148 absolute, or 3.7% relative. The 512 float32 outputs
were byte-identical to full-width output prefixes on this unnormalized model.

### 11.2 Correctness checks

Offline fixtures cover dense and sparse computation, WordPiece and BPE,
Unicode normalization, unknown tokens, truncation, signed int8 arithmetic,
zero rows, binary framing, ordering, partial batches, pipeline failures, and
input limits. Long BPE cases are compared with a simple independent full-scan
oracle, including equal-rank ties and stale merge candidates.

Recorded checks also found byte-identical original float32 outputs against
the preceding executable on 1,000 GooAQ inputs for both dense and sparse,
including pipeline output. BPE optimization checks matched Hugging Face
Tokenizers on 1,508 additional inputs and preserved outputs against the
previous BPE runtime. These are specific parity checks rather than proof for
every possible tokenizer configuration.

### 11.3 BPE throughput is a separate result

The BPE benchmark used the real tokenizer from `FacebookAI/roberta-base`
revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, but its dense and sparse
weights were synthetic. It therefore measures throughput and tokenizer
behavior, not a trained BPE encoder's retrieval quality.

The retained byte-pair table, long-piece heap, and sparse output buffer improved
one-thread dense throughput from 227,843 to 291,299 QPS and sparse from 321,391
to 454,992. At 16 workers and batch 4,096, dense rose from 1.30M to 1.49M and
sparse from 1.19M to 2.41M. Those measurements preceded the later batch pipeline
work; they must not be substituted for the final WordPiece model figures.

A directly compatible trained BPE checkpoint remains an outstanding validation
item. Some candidate checkpoints require unsupported float16 weights or
added-token rules. Detailed BPE evidence is included in
[BPE benchmark notes](../bench/BPE_RESULTS.md).

## 12. Comparison with Flower Computer

The [Flower Computer article](https://www.flowercomputer.com/posts/fast-static-embedding/)
reports approximately 4.5M batched QPS and 524.9K single-query QPS on an M4
Mac mini. It describes embedding StaticMRL's data and simplifying its tokenizer
in a specialized Rust runtime. Icicle uses a similar deployment approach in
C++: compiled tokenizer tables, embedded weights, and small arithmetic kernels.

At pinned public source commit `1b0f9ddd10fcf7d2648cc8214fff800be2860583`, the
[Python API build configuration](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/api-py/Cargo.toml)
enables Rayon parallelism, 512 dimensions, and 8-bit weights. Its
[loader](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/benches/py/model_loader.py)
uses a default batch size of 65,536. Its
[benchmark](https://github.com/flowercomputers/bogkit/blob/1b0f9ddd10fcf7d2648cc8214fff800be2860583/ese/benches/py/bench.py)
loads GooAQ before timing calls to `encode` and discards early timing samples.
That timing includes its API call and returned result, but excludes corpus
loading. The public sources do not establish the historical blog's exact build.

Icicle's 512/int8 build explores similar settings. Its native harness excludes
Python binding and returned-array overhead; its CLI includes line parsing and
serialization. Hardware, workload selection, repetitions, and timing boundaries
also differ. The numerical results are in the same broad throughput range,
but this is not a controlled head-to-head comparison.

The article's vectorization claim motivates a small compiler-visible runtime.
For Icicle, the evidence supports coordinate-loop SIMD and explicit ASCII
scanning. It does not establish that every tokenizer decision is vectorized.
Its reported full NanoBEIR accuracy is also a different evaluation from
Icicle's single NanoMSMARCO subset.

## 13. Discarded experiments and cleanup

| Experiment | Finding and final state |
| --- | --- |
| Tokenizer callbacks that pool committed pieces immediately | No consistent throughput gain; prototype removed |
| Worker-local dense accumulator followed by a copy | No consistent gain; prototype removed |
| LTO build mode | Mixed results; build option, generated metadata, and test scaffolding removed |
| Fixed worker assignments | Dynamic chunks performed better; comparison fallback removed |
| Profiling instrumentation | Removed; no profiling runtime paths remain |

The LTO comparison recorded 4.45M versus 4.35M in-memory QPS in one final
configuration, but earlier comparisons did not consistently improve. That
small difference did not justify keeping the option in the cleaned tool.

Cleanup also removed 27 generated benchmark executables and temporary repeated
inputs, freeing 2.12 GiB. The compiler, benchmark inputs, validation tools, and
historical measurements remain. Generated model binaries can be rebuilt from
the pinned models. Recorded JSON still contains historical `lto` metadata;
current executables do not expose that option or field.

## 14. Compatibility and operational limits

| Area | Supported now |
| --- | --- |
| Dense graph | Sentence Transformers StaticEmbedding, optionally followed by Normalize |
| Sparse graph | Router query route containing one SparseStaticEmbedding module |
| WordPiece | BertNormalizer + BertPreTokenizer + conventional continuation prefix |
| BPE | Null normalizer + ByteLevel pre-tokenizer + standard deterministic merges |
| Checkpoint weights | Single F32 safetensors tensor |
| Compact output weights | Optional dense int8 and dimension prefixes |
| Outputs | JSON Lines, binary dense, binary sparse, token inspection |
| Platforms | Linux ELF and macOS Mach-O backends, C++20 and ICU |

Unsupported areas include Unigram, arbitrary tokenizer component sequences,
float16/bfloat16 checkpoint import, BPE dropout/byte fallback, several added-token
rules, transformer query routes, and the transformer document route. Some
limitations are independent: supporting BPE does not automatically make a
float16 BPE checkpoint importable.

Linux tests and installed-compiler tests passed after cleanup. CI includes
Linux and macOS jobs, but no macOS build or throughput measurement was performed
for this report. Copying the report to a Mac is a file transfer, not macOS
validation of the C++ tool.

## 15. Build and reproduce

Run commands from the repository root. Model compilation and data download
are outside the performance timings.

### 15.1 Linux

```sh
sudo apt-get install build-essential cmake libicu-dev curl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### 15.2 macOS

```sh
brew install cmake icu4c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix icu4c)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

These commands are provided for reproduction; they have not been executed on
the destination Mac as part of preparing or transferring this report.

### 15.3 Pinned model builds

```sh
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --output build/embed-float32 --cxx ./bench/native-cxx.sh

./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --quantize int8 --output build/embed-int8 --cxx ./bench/native-cxx.sh

./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --dimension 512 --output build/embed-512 --cxx ./bench/native-cxx.sh

./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision f60985c706f192d45d218078e49e5a8b6f15283a \
  --dimension 512 --quantize int8 \
  --output build/embed-512-int8 --cxx ./bench/native-cxx.sh

./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --revision 1499ddff8c6545b3f3c7d103249bfbce64da3271 --route query \
  --output build/embed-sparse --cxx ./bench/native-cxx.sh
```

The wrapper enables native CPU instructions on Linux and uses portable
compiler options on macOS. Do not assume a Linux binary runs on macOS, or
that a native build is suitable for an older CPU.

### 15.4 Corpus and sweeps

If the existing corpus is absent, this downloads the same GooAQ slice:

```sh
python3 bench/benchmark.py build/embed-float32 \
  --questions 10000 --repeat 1 --runs 1 --format binary
```

Verify its checksum against section 10. On the measured Linux host:

```sh
taskset -c 0-31 python3 bench/batch_sweep.py build/embed-float32 \
  --threads 32 --batch-sizes 4096,16384,65536 > float32-results.json

taskset -c 0-31 python3 bench/batch_sweep.py build/embed-int8 \
  --threads 32 --batch-sizes 4096,16384,65536 > int8-results.json

taskset -c 0-31 python3 bench/batch_sweep.py build/embed-512-int8 \
  --threads 24 --batch-sizes 16384,65536 --pipeline on > compact-results.json

taskset -c 0-31 python3 bench/batch_sweep.py build/embed-sparse \
  --threads 24 --batch-sizes 4096,16384,65536 --pipeline on > sparse-results.json
```

On macOS omit `taskset` and choose worker counts appropriate to that CPU. The
32-worker configuration is a measurement for the i9 host, not a universal
default. The sweep measures both native in-memory and CLI paths by default.

For a direct native timing run:

```sh
./build/embed-float32 --benchmark --threads 32 --batch-size 65536 \
  --benchmark-repeat 100 --benchmark-runs 3 < build/bench/gooaq-10000.txt
```

For a production CLI invocation, supply the actual desired input and output:

```sh
./build/embed-float32 --format binary --threads 32 --batch-size 16384 --pipeline \
  < queries.txt > embeddings.bin
```

### 15.5 Quality evaluation

Install NumPy and PyArrow in your chosen Python environment, then run:

```sh
python3 bench/quality.py build/embed-float32 build/embed-int8 \
  build/embed-512 build/embed-512-int8 > quality-results.json
```

The first executable is the reference. The script fetches the pinned dataset,
records file checksums, measures retrieval scores, and compares vectors and
top-ten rankings. Dataset revision and checksums are included in the attached
results rather than inferred from a moving dataset name.

## 16. Evidence and source map

The report bundle includes [recorded benchmark results](../bench/BATCH_RESULTS.json)
with raw elapsed times, model metadata, checksums, baseline comparisons, and
quality results. It also includes `source-snapshot.json`
with SHA-256 hashes of the implementation files at report preparation. The
report was originally prepared from working tree changes based on `c359724`;
that original base commit alone does not reproduce the described improvements.
The source snapshot records that preparation state, while subsequent repository
commits preserve the implementation and this report.

| Repository file | What to inspect |
| --- | --- |
| `src/cli/main.cpp` | Hub/local import, graph validation, safetensors, trie and BPE compilation, compact weights, linker invocation |
| `src/runtime/main.cpp` | Dense arithmetic, sparse lookup, worker pool, pipeline, output, native timing |
| `src/runtime/tokenizer.cpp` | WordPiece, Unicode handling, ByteLevel splitting and BPE merging |
| `src/runtime/tokenizer.hpp` | Data structures and explicit ASCII SIMD scan |
| `tests/test_local.py` | Offline fixtures and end-to-end expectations |
| `bench/batch_sweep.py` | Native and CLI batch measurements |
| `bench/quality.py` | Pinned NanoMSMARCO retrieval and vector comparisons |
| `bench/BATCH_RESULTS.json` | Recorded benchmark and quality data |
| `bench/BPE_RESULTS.md` | Separate BPE fixture measurements and parity checks |
| `CMakeLists.txt` | Compiler build and runtime resource installation |

The implementation and local measurements are the primary evidence for Icicle
claims. External Flower comparison claims link to the article or pinned source
next to the relevant statements. No external performance number is presented
as a measurement of Icicle.

## 17. Conclusions

Icicle's speed comes from matching the runtime to the model's small computation
and organizing CPU work efficiently. Embedded tables and fixed model constants
reduce setup and framework costs; tokenization improvements reduce work per
query; reuse, dynamic batches, and pipelining improve parallel throughput.

The full-width float32 mode retains the original weights and showed approximately
twice the preceding tuned CLI throughput. Full-width int8 reached the requested
4.5M range in these conditions, with unchanged aggregate scores on one small
retrieval subset. Smaller dimensions increase throughput further but have a
measured quality trade-off.

For use on a different machine, begin with the full-width float32 model and
measure representative queries and the real output destination. Evaluate
compact settings against the intended retrieval corpus. Remaining validation
priorities are macOS execution, longer and multilingual workloads, broader
retrieval evaluation, and a compatible trained BPE checkpoint.
