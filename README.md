# Icicle

Icicle compiles a supported Hugging Face static text encoder into a native C++ command line program. The generated executable contains its tokenizer tables and float32 weights: give it one UTF-8 text per line on standard input, and it writes one dense or sparse result for each line. It needs no model files, Python, or network access when running. It does link to the platform's ICU libraries.

## Quick start

Install a C++20 compiler, CMake 3.20+, ICU development libraries (`uc` and `i18n`), and `curl`. On Debian or Ubuntu:

```sh
sudo apt-get install build-essential cmake libicu-dev curl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On macOS with Homebrew:

```sh
brew install cmake icu4c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix icu4c)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Build a dense encoder and stream queries through it:

```sh
./build/icicle inspect sentence-transformers/static-retrieval-mrl-en-v1
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 --output ./embed
printf 'hello world\nsecond query\n' | ./embed
```

The output is JSON Lines, with one `{"embedding":[...]}` object per input line. A build from a public Hub model resolves the requested revision to a commit and caches source files under `$XDG_CACHE_HOME/icicle` or `~/.cache/icicle`.

For an inference-free SPLADE query encoder:

```sh
./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --route query --output ./sparse-query
printf 'hello world\n' | ./sparse-query
```

Its output contains sorted token IDs and their nonzero weights: `{"indices":[...],"values":[...],"dimension":N}`. Repeated query terms contribute their weight once.

## Supported models

| Component | Supported form |
| --- | --- |
| Dense model | Sentence Transformers `StaticEmbedding`, optionally followed by `Normalize` |
| Sparse model | `SparseStaticEmbedding` on the **query route** of a Sentence Transformers `Router`; the document route is not compiled |
| Tokenizer | BERT normalizer + `BertPreTokenizer` + WordPiece, or a null normalizer + ByteLevel BPE with standard merge rules |
| Weights | One float32 safetensors tensor, with rows matching tokenizer IDs |
| Platforms | Linux ELF and macOS Mach-O builds; C++20 and ICU required |

Icicle rejects unsupported graph modules, tokenizer components, BPE options, or weight formats during inspection or build. In particular, this release does not implement Unigram, Metaspace, arbitrary BPE pre-tokenizer sequences, BPE dropout, byte fallback, or float16 weights. Sparse transformer document encoders are outside the static query route.

`icicle build MODEL --output FILE` accepts `--revision REV` to pin a Hub revision, `--local DIRECTORY` to build from a local model snapshot without Hub access, and `--cxx COMPILER` to select the compiler for the generated executable. `icicle inspect MODEL` accepts `--revision` and `--local` too.

## Generated executable

Each input line is one query. The executable validates UTF-8, applies the model's saved default prompt when present, tokenizes the text, and emits one result in input order. `--prompt TEXT` overrides that prompt. A line can be at most 16 MiB; a batch can hold at most 64 MiB of input text.

Useful options:

```sh
./embed --info                       # Model, revision, kind, dimension, prompt
printf 'hello\n' | ./embed --tokens  # Token IDs as JSON Lines
./embed --format binary < queries.txt > embeddings.bin
./embed --format binary --threads 4 --batch-size 256 \
  < queries.txt > embeddings.bin
```

Binary output is little-endian and has no file header. A dense record is `dimension` consecutive float32 values; use `--info` to learn its fixed size. A sparse record begins with a uint32 pair count, followed by that many `(uint32 token_id, float32 weight)` pairs. `--tokens` is available with JSON output only.

By default, the program processes and emits each line as it arrives. `--threads N` enables a worker pool when `N > 1`; `--batch-size B` also enables batching and defaults to 256 when omitted. Workers share read-only weights and tokenizer tables while keeping separate scratch buffers. Batch mode preserves input order, flushes a partial batch at end of input, and may wait for a batch to fill before writing output. For an interactive stream, use the default mode or a small batch size.

## How it works

At build time, Icicle checks the model graph, tokenizer configuration, and safetensors shape. It compiles the vocabulary into a direct-offset trie; BPE models also get a merge table and an offset index. It then embeds those read-only tables and the weights into an optimized C++ executable. At runtime, WordPiece uses an ASCII fast path with SSE2 or NEON scanning where available; ByteLevel BPE handles ASCII directly and uses ICU for Unicode text. Dense models average the selected weight rows and apply optional L2 normalization. Sparse models emit one learned scalar per distinct token ID, omitting zero weights.

## Tests and benchmarks

`ctest --test-dir build --output-on-failure` runs offline end-to-end fixtures for dense and sparse models, WordPiece and BPE tokenizers, binary output, and batching. The [build workflow](.github/workflows/build.yml) runs the same tests on Linux and macOS and checks the installed compiler.

Benchmark reports include their model revisions, corpus checksum, machine, output format, and reproduction commands:

- [WordPiece dense and sparse throughput](bench/FINAL_RESULTS.md)
- [ByteLevel BPE throughput and optimization](bench/BPE_RESULTS.md) — its RoBERTa tokenizer is real; its embedding weights are synthetic.
- [Batching investigation](bench/BATCHING.md) and [earlier optimization experiments](bench/RESULTS.md)

[PLAN.md](PLAN.md) describes the architecture and remaining compatibility work. The vendored JSON header retains its own [MIT license](include/nlohmann/LICENSE.MIT).
