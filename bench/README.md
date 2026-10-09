# Benchmarks

## Current results

- [Batch throughput and quality](BATCH_OPTIMIZATIONS.md): latest measurements for dense float32, optional int8 and reduced dimensions, and inference-free SPLADE queries.
- [Raw results](BATCH_RESULTS.json): timings, model metadata, corpus checksums, and retrieval quality measurements.
- [ByteLevel BPE](BPE_RESULTS.md): tokenizer benchmarks using a real RoBERTa tokenizer with synthetic embedding weights.

The latest CLI measurements on an i9-13900HX were 3.77 million queries/s for 1,024-dimensional float32, 4.56 million for int8, and 9.37 million for sparse queries. These used short repeated questions, multiple workers, large batches, and binary output sent to `/dev/null`. They measure aggregate throughput, not single-request latency. In-memory results exclude input and output and are reported separately.

## Tools

| Script | Purpose |
| --- | --- |
| `benchmark.py` | Fetch/cache GooAQ questions and measure CLI throughput |
| `batch_sweep.py` | Compare batch sizes, worker counts, pipelining, and in-memory timing |
| `quality.py` | Compare dense model retrieval quality on NanoMSMARCO |
| `make_bpe_fixture.py` | Create synthetic weights for ByteLevel BPE tokenizer benchmarks |
| `reference.py` | Optional Python Sentence Transformers comparison |
| `parallel.py` | Compare independent worker processes |
| `native-cxx.sh` | Compile generated executables for the local CPU |

Run Python scripts with `--help` for options. The [batch report](BATCH_OPTIMIZATIONS.md) includes exact reproduction commands and explains optional Python dependencies. `native-cxx.sh` produces executables for the local CPU; use the default compiler for portable builds.

## Historical measurements

These reports document earlier runtime versions and experiments. Their results are not the current throughput figures:

- [Initial optimizations](RESULTS.md)
- [Early batching and process scaling](BATCHING.md)
- [September dense and sparse benchmark](FINAL_RESULTS.md)
