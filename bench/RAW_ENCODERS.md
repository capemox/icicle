# Raw dense and inference-free encoder throughput (2026-10-09)

## Results

Throughput in **queries per second**, using original float32 weights and full model dimensions:

| Model | Sentence Transformers raw | Icicle raw | Raw speedup | Icicle binary CLI |
| --- | ---: | ---: | ---: | ---: |
| Static dense, 1,024 dimensions | 144,398 | 4,812,912 | **33.3×** | 3,902,889 |
| Inference-free SPLADE query | 19,308 | 20,590,823 | **1,066.4×** | 9,644,773 |

The raw rows include tokenization and encoder computation in memory, retaining one batch of output. They exclude model loading, corpus parsing, stdout, transport to Python, and whole-corpus array assembly. CLI results additionally include reading/validating text, formatting binary vectors, and writing to `/dev/null`.

For dense encoding, both raw paths produce full float32 vector batches. For sparse encoding, Sentence Transformers' forward pass produces a dense vocabulary-sized intermediate; Icicle retains nonzero token IDs referencing its shared weight table. Its raw path reads weights while filtering zeros. Packing `(ID, weight)` pairs is included in the **CLI** measurement. The sparse raw speedup includes this difference in internal representation.

## Standard API defaults

These additional measurements use the normal public calls on the original 10,000 questions, with batch size 32 and default 24 Torch threads:

| Model | Public call | Queries/s |
| --- | --- | ---: |
| Static dense, 1,024 dimensions | `SentenceTransformer.encode()` | 28,150 |
| Inference-free SPLADE query | `SparseEncoder.encode_query()` | 10,206 |

Those API calls include sorting and collecting the full result: a NumPy array for dense encoding and a sparse COO tensor for sparse queries. The raw pipeline skips that wrapper and measures its standard preprocessing and forward methods directly. The earlier [full-array comparison](HF_COMPARISON.md) measures delivery of a complete dense NumPy array through the executable as well.

## Method and configuration

- **Host:** i9-13900HX, Linux, eight performance cores plus 16 efficiency cores, 32 logical CPUs. Every multicore configuration could use CPUs 0–31. No GPU or multi-process HF execution.
- **Input:** first 10,000 GooAQ questions repeated 100 times: **one million queries per final raw or CLI run**. Mean length 43.38 characters. SHA-256: `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.
- **Timing:** one complete warmup, three measured runs, median. Configurations ran sequentially. Inputs and batch lists were prepared before timing. Neither backend caches token IDs or embeddings.
- **Dense source:** `sentence-transformers/static-retrieval-mrl-en-v1`, revision `f60985c706f192d45d218078e49e5a8b6f15283a`.
- **Sparse source:** query route of `monkeypostulate/inference-free-splade-distilbert-base-uncased-nq`, revision `1499ddff8c6545b3f3c7d103249bfbce64da3271`. The document transformer was loaded outside timing by HF; only the static query route executed.
- **Native compiler:** GCC 13.3.0, `-O3 -march=native`; no int8, reduced dimensions, LTO, or relaxed math.
- **Libraries:** Sentence Transformers 5.4.1, Torch 2.11.0, Transformers 5.5.4, Tokenizers 0.22.2, NumPy 1.26.4. Dense uses Rust Tokenizers; the sparse HF tokenizer reports `is_fast=True`.

HF raw calls use `torch.inference_mode()` and the saved query prompt:

```python
# Dense
features = model.preprocess(batch, prompt=prompt)
vectors = model.forward(features)["sentence_embedding"]

# Inference-free sparse query, through the model's Router
features = model.preprocess(batch, prompt=prompt, task="query")
vectors = model.forward(features, task="query")["sentence_embedding"]
```

All intermediate/output tensor allocation remains timed. The library model code is unchanged. Results are batch-local, avoiding a million-row output array.

HF candidates used 1, 8, and default 24 Torch threads. Dense batch sizes were 256, 4,096, 16,384, and 65,536; sparse sizes were 32, 256, 1,024, and 4,096. Dense discovery runs used 100,000 queries and sparse discovery runs used 10,000. The fastest tested configuration for each was then measured on one million queries. Sparse batches were bounded to keep HF's dense intermediate memory manageable.

| Path | Dense configuration | Sparse configuration |
| --- | --- | --- |
| HF best tested raw | 8 Torch threads, batch 4,096 | 24 Torch threads, batch 32 |
| Icicle raw | 32 workers, batch 65,536 | 24 workers, batch 65,536 |
| Icicle CLI | 32 workers, batch 16,384, pipeline | 24 workers, batch 16,384, pipeline |

These are best tested single-process settings, not a global search across HF backends or process pools.

## Why sparse differs so much

The installed HF `SparseStaticEmbedding.forward()` allocates an int64 token-presence matrix of shape `(batch_size, 30522)`, scatters token presence into it, and multiplies it by the weight vector to produce a full dense intermediate. The standard API subsequently converts it to sparse output. The raw benchmark stops before that final conversion.

Icicle sorts and deduplicates a short token list, reads the corresponding scalar weights, and retains active IDs. It avoids traversing and filling 30,522 coordinates for every query. This is a substantial algorithm and representation difference, beyond SIMD or C++ versus Python.

Both implementations produced **byte-identical vectors on 1,006 validation queries per model**, including empty input, repeated terms, accents, Chinese, Greek, and emoji. Sparse native pairs were expanded into the conceptual vector outside timing for validation. Maximum absolute error was zero.

## Reproduce

Build the models from the pinned local snapshots with `--cxx ./bench/native-cxx.sh`. The [HF comparison report](HF_COMPARISON.md#reproduce) describes dependencies, dense snapshot download, and corpus preparation. To fetch the sparse snapshot for the normal HF loader:

```sh
python3 - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="monkeypostulate/inference-free-splade-distilbert-base-uncased-nq",
    revision="1499ddff8c6545b3f3c7d103249bfbce64da3271",
    local_dir="build/bench/hf-sparse-model",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
PYCODE
./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --local build/bench/hf-sparse-model --route query \
  --output build/embed-sparse-comparison --cxx ./bench/native-cxx.sh
```

Run each family separately so timed workloads do not overlap:

```sh
python3 bench/raw_reference.py dense \
  --local-model build/bench/hf-static-model --executable build/embed-hf-comparison \
  --batch-sizes 256,4096,16384,65536 --sweep-repeat 10 --final-repeat 100 --runs 3 \
  > build/bench/raw-dense.json
python3 bench/raw_reference.py sparse \
  --local-model build/bench/hf-sparse-model --executable build/embed-sparse-comparison \
  --batch-sizes 32,256,1024,4096 --sweep-repeat 1 --final-repeat 100 --runs 3 \
  > build/bench/raw-sparse.json
```

[RAW_ENCODER_RESULTS.json](RAW_ENCODER_RESULTS.json) contains the entire sweep, all run times, library versions, CPU affinity, parity checks, and harness hashes. These numbers apply to the measured CPU, two models, short repeated questions, and timing boundaries.

Sources: [dense model](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1), [sparse model and query API](https://huggingface.co/monkeypostulate/inference-free-splade-distilbert-base-uncased-nq), and [HF sparse static module](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sparse_encoder/modules/sparse_static_embedding.py).
