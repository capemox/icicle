# Throughput compared with Hugging Face (2026-10-09)

## Results

This compares the standard CPU `SentenceTransformer.encode()` API with Icicle using the same static model, float32 weights, original 1,024 dimensions, and query corpus. Neither model is quantized or dimension-reduced. All multicore configurations have access to the same 32 logical CPUs.

| Path | Batch size | Torch threads / native workers | Median seconds | Queries/s |
| --- | ---: | ---: | ---: | ---: |
| HF defaults | 32 | 24 | 3.126928 | 31,980 |
| HF best tested | 65,536 | 8 | 1.064007 | 93,984 |
| Icicle CLI → owned NumPy array | 16,384 | 32 | 0.513636 | 194,691 |
| Icicle binary CLI → /dev/null | 16,384 | 32 | 0.058010 | 1,723,842 |
| Icicle encoding only | 65,536 | 32 | 0.021011 | 4,759,511 |

**When both return a complete NumPy array, Icicle is 6.1× faster than Hugging Face defaults and 2.1× faster than the best tested Hugging Face configuration.** These ratios use the first three rows. The encoding-only and `/dev/null` rows have different output handling and should be interpreted separately.

The fastest tested HF setting used batch size 65,536 and eight Torch threads. The default used batch size 32 and 24 Torch threads on this host. “Best tested” means the configurations in the sweep below; it is not a claim about every possible configuration, GPU execution, or other backends.

## What was measured

- **Hardware:** Intel Core i9-13900HX, eight performance cores, 16 efficiency cores, 32 logical CPUs, Linux. All multicore runs were allowed on CPUs 0–31; workers were not individually pinned.
- **Model:** `sentence-transformers/static-retrieval-mrl-en-v1`, revision `f60985c706f192d45d218078e49e5a8b6f15283a`. Both programs loaded the same pinned snapshot. The model uses mean pooling without a Normalize module.
- **Native build:** GCC 13.3.0, `-O3 -march=native`, full-width float32. Runtime source is unchanged from commit `12078a0`.
- **Input:** first 10,000 GooAQ questions, repeated ten times: **100,000 queries per timed run**, mean 43.38 characters. Corpus SHA-256: `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.
- **Method:** one complete warmup and three measured runs per configuration; throughput uses median elapsed time. Configurations ran sequentially. No embedding or token-ID caching. Downloading, compilation, reading the corpus into the benchmark harness, and HF model loading were excluded.
- **Libraries:** Sentence Transformers 5.4.1, Torch 2.11.0, Tokenizers 0.22.2, NumPy 1.26.4, Safetensors 0.7.0, Hugging Face Hub 1.11.0. No `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS`, or `TOKENIZERS_PARALLELISM` overrides were present.

### Standard Hugging Face API

The default baseline uses the public API directly:

```python
model = SentenceTransformer(snapshot_path, device="cpu", local_files_only=True)
embeddings = model.encode(queries, show_progress_bar=False)
```

No batch size or Torch thread count is supplied for the default row. The timed call includes text sorting, tokenization, pooling, result assembly, and restoration of input order. It returns the entire `(100000, 1024)` float32 NumPy matrix. The previous matrix is freed before the next timing. Disabling the progress display avoids terminal output in the timed region.

The tuned configurations change only the public `batch_size` argument and `torch.set_num_threads()`. The model implementation, weights, precision, and output semantics are unchanged. Torch threads are not a total thread count: HF's tokenizer also uses its normal parallel execution. Each configuration runs in a fresh process.

Official references: [model card](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) and [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

### Icicle returning the same kind of result

The executable reads the prepared text through stdin and writes binary vectors through stdout. Python collects those bytes, reshapes them, and copies them into an owned, writable float32 NumPy array:

```python
raw = subprocess.check_output(command, input=query_bytes)
embeddings = np.frombuffer(raw, dtype="<f4").reshape(100000, 1024).copy()
```

All of this is timed, including executable startup, UTF-8 validation, tokenization, pooling, binary serialization, pipe transport, and final array allocation/copy. The result occupies **409,600,000 bytes (390.625 MiB)**, just like the HF output. Previous results are released before timing the next run.

This measures the existing CLI when called from Python. A native library API could avoid pipe transport and copying, but Icicle does not currently expose one. The full-array row therefore reflects a usable interface today.

### Native CLI and encoding-only modes

The `/dev/null` row includes executable startup, input reading, tokenization, pooling, and binary writes. It does not retain the resulting vectors in the caller.

The encoding-only row uses the existing native `--benchmark` mode. It excludes input parsing, UTF-8 validation, worker startup, serialization, transport, and the final checksum. Output stores remain observable to the optimizer, but batch buffers are reused instead of retaining one whole-corpus matrix. This explains why its millions-of-queries result is much higher than throughput returning a complete array through the CLI.

The earlier [batch report](BATCH_OPTIMIZATIONS.md) measured 3.77M float32 CLI queries/s with one million queries per run. This comparison uses 100,000. Fixed startup and pipeline costs occupy a larger fraction of the shorter CLI run; the reports record separate workloads.

## Vector validation

HF and Icicle outputs were **byte-identical on 1,006 queries**: the first 1,000 corpus questions plus empty input, repeated terms, accents, Chinese, Greek, and an emoji case. Maximum absolute error was 0.0. The check ran outside timing. The full-array path also produced the same shape, dtype, and final-vector checksum as the HF baseline.

## Hugging Face batch/thread sweep

All entries below return the complete array and use all 32 available logical CPUs. Values are queries/s. Thread counts refer to Torch; tokenizer threading is left at its default.

| Batch size | Default (24 threads) | 1 thread | 8 threads | 32 threads |
| --- | ---: | ---: | ---: | ---: |
| 256 | 57,768 | 62,369 | 71,801 | 11,098 |
| 4,096 | 79,694 | 79,033 | 86,270 | 69,518 |
| 16,384 | 89,056 | 82,180 | 88,517 | 86,893 |
| 65,536 | 93,196 | 82,910 | 93,984 | 91,660 |

## One-core control

These separate runs restrict both programs to CPU 0. HF retains its normal encode behavior and full-array output; native in-memory timing retains its batch-buffer behavior.

| Path | Batch size | Queries/s |
| --- | ---: | ---: |
| HF defaults | 32 | 33,795 |
| HF larger batch | 4,096 | 37,166 |
| Icicle encoding only | 32 | 555,651 |
| Icicle encoding only | 4,096 | 492,341 |

These controls show a native encoding advantage with one CPU core. They still have different result assembly costs, so their ratio is not a comparison returning the same whole-corpus array.

## Reproduce

On Linux, create an optional benchmark environment with the CPU Torch wheel:

```sh
python3 -m venv build/hf-bench-venv
build/hf-bench-venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
build/hf-bench-venv/bin/pip install sentence-transformers==5.4.1 numpy==1.26.4 \
  tokenizers==0.22.2 safetensors==0.7.0 huggingface-hub==1.11.0
```

Download the pinned snapshot once:

```sh
build/hf-bench-venv/bin/python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/static-retrieval-mrl-en-v1",
    revision="f60985c706f192d45d218078e49e5a8b6f15283a",
    local_dir="build/bench/hf-static-model",
    allow_patterns=["*.json", "*.safetensors"],
)
PYCODE
```

Build the compiler and the full-width float32 executable. If the GooAQ corpus is not already cached, the existing CLI harness can fetch it:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --local build/bench/hf-static-model --output build/embed-hf-comparison \
  --cxx ./bench/native-cxx.sh
python3 bench/benchmark.py build/embed-hf-comparison \
  --questions 10000 --repeat 1 --runs 1 --format binary
```

Run the comparison; it checks vector parity before timing, sweeps HF batch/thread settings, performs Linux one-core controls, and measures all three native output modes:

```sh
build/hf-bench-venv/bin/python bench/compare_hf.py build/embed-hf-comparison \
  --local-model build/bench/hf-static-model --repeat 10 --runs 3 \
  > build/bench/hf-comparison.json
```

To run only the stock HF baseline:

```sh
build/hf-bench-venv/bin/python bench/reference.py build/bench/gooaq-10000.txt \
  --local-model build/bench/hf-static-model --repeat 10 --runs 3
```

The recorded [HF_RESULTS.json](HF_RESULTS.json) includes all run times, version metadata, source file hashes, CPU affinity, parity results, and output shapes. Results depend on the CPU, query lengths, batching, output handling, and installed libraries; they do not establish the same speedup on macOS or for BPE tokenizers or sparse models.
