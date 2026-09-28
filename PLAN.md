# Icicle implementation plan

## Implementation status

The C++ compiler and generated runtime now build dense `StaticEmbedding` models with an optional `Normalize` module and inference-free SPLADE `SparseStaticEmbedding` query routes. Supported tokenizers are BERT WordPiece and ByteLevel BPE with a null normalizer and standard merge rules; weights must be float32 safetensors. It embeds the weight table and tokenizer lookup data into a native executable, supports JSON Lines and binary output, builds from a pinned Hugging Face revision or local snapshot, and has offline end-to-end tests. Linux builds and installed-tool tests pass. A macOS build/test workflow is included; it awaits execution on a macOS runner.

The next compatibility work is broader BPE pipelines, Unigram tokenizers, more module adapters, and float16/bfloat16 weight conversion. Performance work should use the binary output path when measuring encoder throughput, since dense JSON formatting dominates total time.

## Goal

Build `icicle`, a **C++ command line compiler** for supported Hugging Face static text encoders. Given a Hub model ID or a local snapshot, it emits a native command line executable containing the model's weights and tokenizer data. The emitted executable reads UTF-8 text from stdin and writes one embedding per input line. It needs no Hugging Face, Python, Rust, model file, or network access at runtime.

The first two model families are Sentence Transformers `StaticEmbedding` (dense) and the **query route only** of a Sentence Transformers inference-free SPLADE model using `SparseStaticEmbedding`. A conventional SPLADE transformer query route is outside this design. The document route of an asymmetric model is deliberately neither downloaded nor compiled.

The [Flower Computer post](https://www.flowercomputer.com/posts/fast-static-embedding/) motivates putting the lookup table and tokenizer into the program at build time and using a small specialized runtime. Its speed claims are a benchmark result for its implementation, not an acceptance target for Icicle.

## User interface

```sh
icicle inspect sentence-transformers/static-retrieval-mrl-en-v1
icicle build sentence-transformers/static-retrieval-mrl-en-v1 \
  --revision <commit-sha> --output ./embed-retrieval
icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --revision <commit-sha> --route query --output ./embed-query

printf 'first query\nsecond query\n' | ./embed-retrieval
printf 'first query\n' | ./embed-query
./embed-query --info
```

`build` also accepts `--local <snapshot-directory>`, `--cxx <compiler>`, and `--jobs N`. Resolve a Hub revision to an immutable commit before fetching files; record the resolved commit and source hashes in `--info`. Fetch only files required for the selected route. `inspect` reports the detected family, tokenizer components, tensor names/shapes, supported status, and reasons for any rejection.

The default wire format is JSON Lines: dense `{"embedding":[... ]}`; sparse `{"indices":[...],"values":[...],"dimension":30522}` with sorted token IDs and nonzero values. One input line yields one output line. Document empty-line behavior, invalid UTF-8 handling, maximum input size, and errors. Keep float32 values by default; do not silently quantize, normalize, truncate dimensions, or prune sparse terms. Honor a saved default prompt if the model defines one, report it in `inspect`, and expose an explicit prompt override.

## Model semantics to preserve

| Family | Recognize | Exact operation |
| --- | --- | --- |
| Dense | `modules.json` contains a supported `StaticEmbedding` module and no unsupported downstream modules | Apply the selected prompt, then run that model's `tokenizer.json` with special tokens disabled. Mean-pool the selected float32 token vectors. An empty token list yields a zero vector. |
| Sparse query | `modules.json`/`router_config.json` selects a query route consisting of `SparseStaticEmbedding` | Run the query tokenizer with special tokens disabled and the model's truncation rules. For each distinct token ID present, emit its learned scalar weight once. Discard zero weights in sparse output. Repeated query terms do not sum. |

These operations follow the upstream [`StaticEmbedding` implementation](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/modules/static_embedding.py) and [`SparseStaticEmbedding` implementation](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sparse_encoder/modules/sparse_static_embedding.py). Inference-free SPLADE models commonly have a separate transformer document route, as described in the [Sentence Transformers sparse model documentation](https://sbert.net/docs/sparse_encoder/pretrained_models.html). Do not treat arbitrary sparse encoders as inference-free.

## Architecture

1. **Hub/snapshot reader (build tool only).** Use C++ with libcurl for Hub access and a JSON parser for metadata. Resolve and pin the revision, cache downloaded files, enforce size limits, validate paths and hashes, and avoid executing repository code (`trust_remote_code` is never needed). Accept local snapshots without network access. Download only `modules.json`, route/config files, selected `tokenizer.json` plus tokenizer config when needed, and selected `.safetensors` weights.
2. **Model importer.** Inspect module graph and tensor metadata, select dense or sparse query adapter, validate tokenizer vocabulary against weight dimensions, and reject unexpected modules/tensors with a specific diagnostic. Parse safetensors with bounds, dtype, shape, and offset checks; support float32 first, then float16/bfloat16 conversion to float32 if needed. No PyTorch or Python dependency in the builder. The [safetensors format documentation](https://huggingface.co/docs/safetensors/index) is the format reference.
3. **Tokenizer compiler.** Translate supported `tokenizer.json` normalization, pretokenization, model, added-token, and truncation settings into compact C++ runtime tables. BERT normalizer + WordPiece covers the two named acceptance models; ByteLevel BPE covers GPT-2/RoBERTa-style tokenizers. Expand BPE pipeline support and add Unigram to cover more static checkpoints. Use ICU or generated Unicode tables for correct Unicode behavior. Never approximate an unsupported tokenizer component; fail during `inspect`/`build`. Compare token IDs against Hugging Face Tokenizers fixtures before using that tokenizer for embeddings. The [Tokenizers pipeline documentation](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html) defines the component order.
4. **Artifact/linker.** Pack validated tokenizer tables and model weights into a versioned, aligned binary image. Link that image into a C++ runtime object as read-only data: ELF and Mach-O backends first, with a generated C++ byte-array fallback where required. Avoid compiling 125 MB of float literals. Generate a tiny model descriptor containing offsets, dimensions, model kind, source revision, and checksums. Invoke the selected C++ compiler/linker to produce the requested executable. The runtime performs no model parsing or allocation proportional to the whole model.
5. **Runtime.** Share UTF-8 input, tokenizer, output, and error-handling code between families; specialize the hot path at build time. Dense: gather rows, accumulate in float32, divide by token count. Sparse: deduplicate token IDs with a reusable generation-stamped array or sort a short ID list, then emit nonzero weights in ID order. Buffer stdin/stdout, reuse per-line workspaces, and keep memory bounded. Make the generated program self-contained with respect to model assets; document unavoidable platform system-library dependencies.

Suggested repository layout:

```text
CMakeLists.txt
include/icicle/             # shared data structures and artifact format
src/cli/                   # inspect/build commands, Hub fetch, linker driver
src/import/                # module graph, safetensors, tokenizer import
src/tokenizer/             # C++ tokenizer runtime and compiled tables
src/runtime/               # generated executable main and dense/sparse kernels
tests/                     # tokenizer and embedding parity fixtures
bench/                     # repeatable throughput/latency harness
docs/                      # supported model and platform matrix
```

## Delivery sequence and gates

1. **Skeleton and format:** Set up CMake/C++20, `inspect`, local snapshot handling, typed errors, and a small versioned artifact descriptor. Gate: malformed metadata and safetensors are rejected without crashes.
2. **Dense reference path:** Import `sentence-transformers/static-retrieval-mrl-en-v1`; implement exact WordPiece tokenization and float32 mean pooling; build a runnable embedded-data executable. Gate: token IDs match Hugging Face Tokenizers and vectors match Sentence Transformers on a fixed corpus within documented float32 tolerance.
3. **Sparse query path:** Import `monkeypostulate/inference-free-splade-distilbert-base-uncased-nq`; select only the query route, embed its one-dimensional weights, and emit sorted sparse JSONL. Gate: token IDs, active dimensions, and weights match `SparseEncoder.encode_query` on the same corpus, including repeated terms and truncation.
4. **Hub workflow:** Add pinned Hub downloads, cache, local/offline mode, provenance via `--info`, and clear diagnostics for unsupported graphs/tokenizers. Gate: an executable built from a Hub ID runs with the network disabled and without model files.
5. **Broader static support:** Expand ByteLevel BPE to other BPE pipelines, add Unigram and other tokenizer components based on real checkpoints, then add adapters only for static formats whose forward behavior can be verified. Keep a tested model matrix; a model outside it must fail clearly. Gate: at least one additional dense tokenizer family passes end-to-end parity.
6. **Optimize and measure:** Measure tokenization, row gathering, pooling, serialization, and startup separately. Optimize the measured bottlenecks with compact lookup structures, SIMD where useful, and optional batch/parallel processing. Preserve the default float32 parity path; make quantization or relaxed math opt-in and measure quality drift. Gate: report single-query latency, streaming throughput, binary size, peak RSS, and compile time against the corresponding Python reference and an unoptimized C++ baseline on the same hardware/input corpus.

## Acceptance checks

- Both sample models build from an immutable Hub revision and a local snapshot, and both generated executables run offline with no model sidecar files.
- Golden cases cover empty text, repeated terms, unknown words, punctuation, Unicode/case/accent behavior, long input, malformed UTF-8, and multiple input lines.
- Dense output dimension and values match the selected reference model; sparse output uses the same vocabulary IDs and nonzero weights as its **query** route. Set numerical tolerance from observed float32 accumulation differences and keep it in the test report.
- Unsupported module graphs, tensor dtypes, and tokenizer features fail with actionable messages instead of producing plausible but incorrect embeddings.
- Benchmark results include reproducible commands and do not claim the post's reported speedup without measuring it.

## Main risks and decisions

- **Tokenizer fidelity is the primary correctness risk.** `tokenizer.json` is a pipeline, not merely a vocabulary. Treat each component as an explicitly supported feature and test against the upstream tokenizer. In particular, both reference modules disable automatic special tokens.
- **Static does not imply small.** The example dense checkpoint's safetensors file is about 125 MB; the compiled binary will have a similar weight-data floor unless an explicit compression mode is added. Link binary data directly to keep build time reasonable.
- **Portability needs platform-specific linking.** Implement Linux ELF and macOS Mach-O explicitly; do not imply one generated binary can run on another OS/architecture.
- **Accuracy precedes speed.** Read-only embedded weights and a thin runtime are the central optimization. SIMD, quantization, and alternate accumulation orders follow reference parity and benchmark evidence.
