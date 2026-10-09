# Icicle

Compile a supported Hugging Face static embedding model into a native C++ executable. Give it query text, one line at a time; get an embedding for each line.

The executable includes the model weights and tokenizer. It runs offline without Python or separate model files, and links to the system ICU libraries. Linux and macOS are supported.

## Build

You need a C++20 compiler, CMake 3.20+, ICU, and `curl`. Python 3 is needed only for tests.

**Ubuntu / Debian**

```sh
sudo apt-get install build-essential cmake libicu-dev curl python3
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**macOS** — install Apple's command line tools (`xcode-select --install`), then:

```sh
brew install cmake icu4c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix icu4c)"
cmake --build build --parallel
```

## Use

Build a dense encoder:

```sh
./build/icicle build sentence-transformers/static-retrieval-mrl-en-v1 --output ./embed
printf 'hello world\nhow do airplanes fly?\n' | ./embed
```

Output is JSON Lines: one `{"embedding":[...]}` per query.

Build an inference-free SPLADE query encoder:

```sh
./build/icicle build monkeypostulate/inference-free-splade-distilbert-base-uncased-nq \
  --route query --output ./sparse-query
printf 'hello world\n' | ./sparse-query
```

Sparse output contains `indices`, `values`, and `dimension`. Repeated tokens contribute once.

For larger workloads, use batching and binary output:

```sh
./embed --threads 8 --batch-size 256 --pipeline --format binary \
  < queries.txt > vectors.bin
```

Results stay in input order. Run `./embed --info` for model details or `./embed --help` for options. Binary output has no header; its format is described in the [command reference](docs/USAGE.md).

## How it works

Dense encoding looks up a fixed vector for each token and averages the vectors, then normalizes if the model requests it. Sparse encoding looks up one weight per distinct token and emits the nonzero entries. Neither path runs a transformer at query time.

Speed comes from compiled lookup tables, fast ASCII tokenization, compiler-vectorized arithmetic, reusable buffers, shared read-only weights, and parallel batches. Optional `--quantize int8` and `--dimension N` make dense models smaller, with potential quality changes.

Supported models use Sentence Transformers `StaticEmbedding` or a Router query route containing `SparseStaticEmbedding`, with float32 safetensors and supported WordPiece or ByteLevel BPE configurations. Unsupported configurations are rejected. Icicle emits vectors; a search index is separate.

## Tests and details

```sh
ctest --test-dir build --output-on-failure
```

- [Command reference](docs/USAGE.md)
- [Benchmarks and reproduction](bench/README.md)
- [Detailed explanation and validation](docs/ICICLE_REPORT.md)
