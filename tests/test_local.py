"""Offline end-to-end tests for model import, linking, and generated executables."""
import json
import math
import os
import random
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def safetensors(path, name, shape, values):
    payload = struct.pack("<" + "f" * len(values), *values)
    header = json.dumps({name: {"dtype": "F32", "shape": shape, "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def call(*args, input_text=None):
    return subprocess.run(args, input=input_text, text=True, capture_output=True, check=True).stdout


def token_json():
    return {
        "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "hello": 1, "world": 2,
                                                  "σ": 3, "work": 4, "##ing": 5, "hell": 6,
                                                  "##o": 7},
                  "unk_token": "[UNK]", "max_input_chars_per_word": 100},
        "normalizer": {"type": "BertNormalizer", "clean_text": True,
                       "handle_chinese_chars": True, "strip_accents": None, "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "truncation": None, "added_tokens": [],
    }


def bpe_token_json():
    # The 256 visible characters used by Hugging Face's ByteLevel pre-tokenizer.
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = 0
    alphabet = []
    for byte in range(256):
        codepoint = byte if byte in direct else 256 + extra
        extra += byte not in direct
        alphabet.append(chr(codepoint))
    vocab = {char: index for index, char in enumerate(sorted(alphabet))}
    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
              ("Ġ", "h"), ("Ġh", "e"), ("Ġhe", "l"), ("Ġhel", "l"), ("Ġhell", "o")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    return {
        "model": {"type": "BPE", "vocab": vocab,
                  "merges": [left + " " + right for left, right in merges],
                  "unk_token": None, "dropout": None},
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "truncation": None,
        "added_tokens": [{"id": len(vocab), "content": "<mask>", "single_word": False,
                          "lstrip": True, "rstrip": False, "normalized": False,
                          "special": True}],
    }


def reference_bpe(text, tokenizer):
    """Simple full-scan reference for a ByteLevel piece, without pre-tokenization."""
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet = {}
    extra = 0
    for byte in range(256):
        alphabet[byte] = chr(byte if byte in direct else 256 + extra)
        extra += byte not in direct
    model = tokenizer["model"]
    vocab = model["vocab"]
    symbols = [alphabet[b] if alphabet[b] in vocab else model.get("unk_token")
               for b in text.encode()]
    symbols = [s for s in symbols if s is not None]
    ranks = {tuple(pair.split(" ")): rank for rank, pair in enumerate(model["merges"])}
    while len(symbols) > 1:
        best = min(range(len(symbols) - 1),
                   key=lambda i: ranks.get((symbols[i], symbols[i + 1]), math.inf))
        if (symbols[best], symbols[best + 1]) not in ranks:
            break
        symbols[best:best + 2] = [symbols[best] + symbols[best + 1]]
    return [vocab[s] for s in symbols]


def main(compiler):
    with tempfile.TemporaryDirectory(prefix="icicle-test-") as td:
        root = Path(td)
        dense = root / "dense"
        dense.mkdir()
        save_json(dense / "modules.json", [
            {"type": "sentence_transformers.models.StaticEmbedding", "path": "0_StaticEmbedding"},
            {"type": "sentence_transformers.models.Normalize", "path": "1_Normalize"},
        ])
        save_json(dense / "0_StaticEmbedding" / "tokenizer.json", token_json())
        safetensors(dense / "0_StaticEmbedding" / "model.safetensors", "embedding.weight",
                    [8, 2], [0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        dense_bin = root / "dense-bin"
        call(compiler, "build", "fixture", "--local", str(dense), "--output", str(dense_bin))
        tokens = json.loads(call(str(dense_bin), "--tokens", input_text="Hello world\n"))
        assert tokens == {"tokens": [1, 2]}, tokens
        assert json.loads(call(str(dense_bin), "--tokens", input_text="Σ\n")) == {"tokens": [3]}
        assert json.loads(call(str(dense_bin), "--tokens", input_text="Héllo\n")) == {"tokens": [1]}
        assert json.loads(call(str(dense_bin), "--tokens", input_text="HE\x01LLO\tWORLD\x7f\n")) == {"tokens": [1, 2]}
        assert json.loads(call(str(dense_bin), "--tokens", input_text="working hello\n")) == {"tokens": [4, 5, 1]}
        mixed = "Héllo\nworking hello\nΣ\nHELLO\n"
        mixed_want = [{"tokens": [1]}, {"tokens": [4, 5, 1]}, {"tokens": [3]}, {"tokens": [1]}]
        for flags in ((), ("--threads", "2", "--batch-size", "3")):
            assert [json.loads(row) for row in call(str(dense_bin), *flags, "--tokens",
                                                      input_text=mixed).splitlines()] == mixed_want
        assert call(str(dense_bin), "--tokens", input_text="a" * 20 + "é\n")
        for prefix_length in (0, 15, 16, 23, 24):
            invalid = subprocess.run([str(dense_bin), "--tokens"],
                                     input=b"a" * prefix_length + b"\xff\n", capture_output=True)
            assert invalid.returncode != 0 and b"not valid UTF-8" in invalid.stderr
        rows = [json.loads(x) for x in call(str(dense_bin), input_text="Hello world\n\n").splitlines()]
        want = [1 / math.sqrt(5), 2 / math.sqrt(5)]
        assert all(abs(a - b) < 1e-6 for a, b in zip(rows[0]["embedding"], want)), rows[0]
        assert rows[1] == {"embedding": [0, 0]}, rows[1]
        dense_raw = subprocess.check_output([str(dense_bin), "--format", "binary"], input=b"hello world\n")
        raw_values = struct.unpack("<2f", dense_raw)
        assert all(abs(a - b) < 1e-6 for a, b in zip(raw_values, want)), raw_values
        dense_input = "Hello world\nΣ\nworking hello\n\n"
        assert call(str(dense_bin), "--threads", "2", "--batch-size", "2", input_text=dense_input) == \
            call(str(dense_bin), input_text=dense_input)
        assert call(str(dense_bin), "--threads", "2", "--batch-size", "3", "--tokens", input_text=dense_input) == \
            call(str(dense_bin), "--tokens", input_text=dense_input)
        raw_input = dense_input.encode()
        assert subprocess.check_output([str(dense_bin), "--threads", "2", "--batch-size", "3",
                                        "--format", "binary"], input=raw_input) == \
            subprocess.check_output([str(dense_bin), "--format", "binary"], input=raw_input)

        # Pipeline slots must preserve order and drain partial batches. Mixed
        # query lengths also exercise dynamic chunk scheduling and scratch reuse.
        inputs = (dense_input + "hello " * 70 + "\n") * 31
        for fmt in ("jsonl", "binary"):
            expected = subprocess.check_output([str(dense_bin), "--format", fmt], input=inputs.encode())
            for batch_size in (1, 33, 65536):
                actual = subprocess.check_output([str(dense_bin), "--format", fmt, "--threads", "4",
                            "--batch-size", str(batch_size), "--pipeline"],
                            input=inputs.encode(), timeout=20)
                assert actual == expected
        bad_input = subprocess.run([str(dense_bin), "--threads", "4", "--batch-size", "1", "--pipeline"],
                                   input=b"hello\n\xff\n", capture_output=True, timeout=10)
        assert bad_input.returncode and b"not valid UTF-8" in bad_input.stderr
        # The input byte cap can split a batch before its requested line count.
        # Carry the pending line into the next pipeline slot without losing it.
        large_input = (b"a" * (14 * 1024 * 1024) + b"\n") * 5
        assert subprocess.check_output([str(dense_bin), "--threads", "4", "--batch-size", "65536",
                                        "--pipeline", "--format", "binary"], input=large_input,
                                       timeout=30) == b"\0" * (5 * 2 * 4)
        del large_input
        bench = json.loads(call(str(dense_bin), "--benchmark", "--benchmark-runs", "2",
                                "--benchmark-repeat", "3", "--threads", "4", "--batch-size", "7",
                                input_text=dense_input))
        assert bench["queries_per_run"] == 12 and len(bench["runs_seconds"]) == 2
        assert bench["median_qps"] > 0 and math.isfinite(bench["checksum"])
        empty = subprocess.run([str(dense_bin), "--benchmark"], input=b"", capture_output=True)
        assert empty.returncode and b"at least one input line" in empty.stderr

        # Validate dimension truncation and signed per-row quantization against
        # independent arithmetic, including an all-zero row and negative values.
        compact = root / "compact-bin"
        save_json(dense / "modules.json", [{"type": "sentence_transformers.models.StaticEmbedding",
                                            "path": "0_StaticEmbedding"}])
        compact_weights = [0, 0, 0.2, -1.0, -0.35, 2.0] + [0.0] * 10
        safetensors(dense / "0_StaticEmbedding" / "model.safetensors", "embedding.weight", [8, 2], compact_weights)
        call(compiler, "build", "fixture", "--local", str(dense), "--output", str(compact), "--dimension", "1")
        result = json.loads(call(str(compact), input_text="hello world\n"))["embedding"]
        assert len(result) == 1 and abs(result[0] - (-0.075)) < 1e-6
        info = json.loads(call(str(compact), "--info"))
        assert info["dimension"] == 1 and info["source_dimension"] == 2
        call(compiler, "build", "fixture", "--local", str(dense), "--output", str(compact), "--quantize", "int8")
        info = json.loads(call(str(compact), "--info"))
        assert info["weight_dtype"] == "int8" and info["dimension"] == 2
        def dequantize(row):
            scale = max(map(abs, row)) / 127 or 1
            return [round(v / scale) * scale for v in row]
        a, b = dequantize([0.2, -1.0]), dequantize([-0.35, 2.0])
        want = [(x + y) / 2 for x, y in zip(a, b)]
        for flags in ((), ("--threads", "4", "--batch-size", "1", "--pipeline")):
            raw = subprocess.check_output([str(compact), "--format", "binary", *flags], input=b"hello world\n\n")
            actual = struct.unpack("<4f", raw)
            assert all(abs(x - y) < 1e-6 for x, y in zip(actual[:2], want)) and actual[2:] == (0, 0)
        invalid = subprocess.run([compiler, "build", "fixture", "--local", str(dense), "--output", str(compact),
                                  "--dimension", "3"], capture_output=True)
        assert invalid.returncode and b"exceeds source dimension" in invalid.stderr
        safetensors(dense / "0_StaticEmbedding" / "model.safetensors", "embedding.weight",
                    [8, 2], [float("nan")] + [0.0] * 15)
        invalid = subprocess.run([compiler, "build", "fixture", "--local", str(dense), "--output", str(compact),
                                  "--quantize", "int8"], capture_output=True)
        assert invalid.returncode and b"finite weights" in invalid.stderr

        sparse = root / "sparse"
        sparse.mkdir()
        save_json(sparse / "modules.json", [{"type": "sentence_transformers.models.Router", "path": ""}])
        save_json(sparse / "router_config.json", {
            "structure": {"query": ["query_0_SparseStaticEmbedding"], "document": ["document_0_Transformer"]},
            "types": {"query_0_SparseStaticEmbedding": "sentence_transformers.sparse_encoder.models.SparseStaticEmbedding.SparseStaticEmbedding"},
        })
        tok = token_json()
        tok["truncation"] = {"direction": "Right", "max_length": 3, "stride": 0}
        save_json(sparse / "query_0_SparseStaticEmbedding" / "tokenizer.json", tok)
        safetensors(sparse / "query_0_SparseStaticEmbedding" / "model.safetensors", "weight",
                    [8], [0, 1.5, 2.5, 0, 0, 0, 0, 0])
        sparse_bin = root / "sparse-bin"
        call(compiler, "build", "fixture", "--local", str(sparse), "--output", str(sparse_bin))
        rows = [json.loads(x) for x in call(str(sparse_bin), input_text="hello hello world world\nunknown\n").splitlines()]
        assert rows == [
            {"indices": [1, 2], "values": [1.5, 2.5], "dimension": 8},
            {"indices": [], "values": [], "dimension": 8},
        ], rows
        sparse_raw = subprocess.check_output([str(sparse_bin), "--format", "binary"], input=b"hello world\n")
        assert struct.unpack("<IIfIf", sparse_raw) == (2, 1, 1.5, 2, 2.5)
        sparse_input = "hello hello world world\nunknown\nΣ\nworking hello\n"
        too_long = " ".join(["é" * 101] * 5) + "\n"
        assert json.loads(call(str(sparse_bin), "--tokens", input_text=too_long))["tokens"] == [0, 0, 0]
        assert call(str(sparse_bin), "--threads", "2", "--batch-size", "3", input_text=sparse_input) == \
            call(str(sparse_bin), input_text=sparse_input)
        assert subprocess.check_output([str(sparse_bin), "--threads", "2", "--batch-size", "2",
                                        "--format", "binary"], input=sparse_input.encode()) == \
            subprocess.check_output([str(sparse_bin), "--format", "binary"], input=sparse_input.encode())
        for fmt in ("jsonl", "binary"):
            inputs = (sparse_input * 50).encode()
            assert subprocess.check_output([str(sparse_bin), "--format", fmt, "--threads", "4",
                            "--batch-size", "33", "--pipeline"], input=inputs, timeout=20) == \
                   subprocess.check_output([str(sparse_bin), "--format", fmt], input=inputs)
        invalid = subprocess.run([compiler, "build", "fixture", "--local", str(sparse), "--output", str(compact),
                                  "--quantize", "int8"], capture_output=True)
        assert invalid.returncode and b"dense models only" in invalid.stderr

        bpe = root / "bpe"
        bpe.mkdir()
        save_json(bpe / "modules.json", [
            {"type": "sentence_transformers.models.StaticEmbedding", "path": "0_StaticEmbedding"},
        ])
        bpe_tokenizer = bpe_token_json()
        # Older byte-level BPE tokenizer files omit the model's type field.
        del bpe_tokenizer["model"]["type"]
        save_json(bpe / "0_StaticEmbedding" / "tokenizer.json", bpe_tokenizer)
        size = len(bpe_tokenizer["model"]["vocab"]) + 1
        safetensors(bpe / "0_StaticEmbedding" / "model.safetensors", "embedding.weight",
                    [size, 1], [float(i) for i in range(size)])
        bpe_bin = root / "bpe-bin"
        call(compiler, "build", "fixture", "--local", str(bpe), "--output", str(bpe_bin))
        bpe_text = "hello\n hello\nhello hello\nHi <mask> here\né\n😀\n  hello\nI'm\na  b\na\tb\nhello!\n"
        bpe_expected = [[259], [220, 259], [259, 220, 259],
                        [39, 72, 265, 220, 256, 81, 68], [127, 102],
                        [172, 253, 246, 222], [220, 220, 259],
                        [40, 6, 76], [64, 220, 220, 65], [64, 197, 65], [259, 0]]
        bpe_rows = [json.loads(row)["tokens"] for row in
                    call(str(bpe_bin), "--tokens", input_text=bpe_text).splitlines()]
        assert bpe_rows == bpe_expected, bpe_rows
        assert call(str(bpe_bin), "--threads", "2", "--batch-size", "3", "--tokens",
                    input_text=bpe_text) == call(str(bpe_bin), "--tokens", input_text=bpe_text)
        vector = json.loads(call(str(bpe_bin), input_text="hello\n"))["embedding"]
        assert vector == [259.0], vector
        assert call(str(bpe_bin), "--threads", "2", "--batch-size", "3",
                    input_text=bpe_text) == call(str(bpe_bin), input_text=bpe_text)
        for fmt in ("jsonl", "binary"):
            inputs = (bpe_text * 20).encode()
            assert subprocess.check_output([str(bpe_bin), "--format", fmt, "--threads", "4",
                                "--batch-size", "33", "--pipeline"], input=inputs, timeout=20) == \
                   subprocess.check_output([str(bpe_bin), "--format", fmt], input=inputs)

        bpe_sparse = root / "bpe-sparse"
        bpe_sparse.mkdir()
        save_json(bpe_sparse / "modules.json", [
            {"type": "sentence_transformers.models.Router", "path": ""},
        ])
        save_json(bpe_sparse / "router_config.json", {
            "structure": {"query": ["query_0_SparseStaticEmbedding"]},
            "types": {"query_0_SparseStaticEmbedding":
                      "sentence_transformers.sparse_encoder.models.SparseStaticEmbedding.SparseStaticEmbedding"},
        })
        save_json(bpe_sparse / "query_0_SparseStaticEmbedding" / "tokenizer.json", bpe_tokenizer)
        bpe_weights = [0.0] * size
        bpe_weights[220], bpe_weights[259] = 2.0, 1.5
        safetensors(bpe_sparse / "query_0_SparseStaticEmbedding" / "model.safetensors",
                    "weight", [size], bpe_weights)
        bpe_sparse_bin = root / "bpe-sparse-bin"
        call(compiler, "build", "fixture", "--local", str(bpe_sparse),
             "--route", "query", "--output", str(bpe_sparse_bin))
        result = json.loads(call(str(bpe_sparse_bin), input_text="hello hello\n"))
        assert result == {"indices": [220, 259], "values": [2.0, 1.5],
                          "dimension": size}, result

        # Cross the bounded sparse output buffer, include empty records, and end
        # with a partial batch. Verify the wire format independently of writes.
        safetensors(bpe_sparse / "query_0_SparseStaticEmbedding" / "model.safetensors",
                    "weight", [size], [1.0] * size)
        call(compiler, "build", "fixture", "--local", str(bpe_sparse),
             "--output", str(bpe_sparse_bin))
        text = " ".join(chr(i) for i in range(33, 127))
        ids = sorted(set(json.loads(call(str(bpe_bin), "--tokens", input_text=text + "\n"))["tokens"]))
        record = struct.pack("<I", len(ids)) + b"".join(struct.pack("<If", i, 1.0) for i in ids)
        inputs = ((text + "\n") * 4096 + "\n" + text + "\n").encode()
        expected = record * 4096 + struct.pack("<I", 0) + record
        assert len(expected) > 1024 * 1024
        for flags in ((), ("--threads", "2", "--batch-size", "4096"),
                      ("--threads", "4", "--batch-size", "4096", "--pipeline")):
            assert subprocess.check_output([str(bpe_sparse_bin), "--format", "binary", *flags],
                                           input=inputs) == expected

        # A single long piece exercises heap ties, invalidated candidates and
        # merges on either side of a previous merge. Compare to a simple oracle.
        long_tok = bpe_token_json()
        long_tok["added_tokens"] = []
        long_tok["pre_tokenizer"]["use_regex"] = False
        vocab = long_tok["model"]["vocab"]
        for left, right in [("a", "a"), ("aa", "aa"), ("aaaa", "aaaa"),
                            ("a", "b"), ("aa", "b"), ("b", "a"), ("ba", "aa"),
                            ("c", "a"), ("ca", "b"), ("a", "ca")]:
            vocab[left + right] = len(vocab)
            long_tok["model"]["merges"].append(left + " " + right)
        save_json(bpe / "0_StaticEmbedding" / "tokenizer.json", long_tok)
        safetensors(bpe / "0_StaticEmbedding" / "model.safetensors", "embedding.weight",
                    [len(vocab), 1], [float(i) for i in range(len(vocab))])
        call(compiler, "build", "fixture", "--local", str(bpe), "--output", str(bpe_bin))
        rng = random.Random(71)
        texts = ["a" * n for n in (0, 1, 127, 128, 129, 511, 4096)]
        texts += ["abacaabaa" * 37, "é😀hello" * 32]
        texts += ["".join(rng.choices("aabc helloé😀", k=n)) for n in (127, 128, 129, 256, 1024)]
        expected = [reference_bpe(t, long_tok) for t in texts]
        for flags in ((), ("--threads", "2", "--batch-size", "3")):
            actual = [json.loads(r)["tokens"] for r in call(str(bpe_bin), "--tokens", *flags,
                          input_text="\n".join(texts) + "\n").splitlines()]
            assert actual == expected

        long_tok["truncation"] = {"direction": "Right", "max_length": 7, "stride": 0}
        save_json(bpe / "0_StaticEmbedding" / "tokenizer.json", long_tok)
        call(compiler, "build", "fixture", "--local", str(bpe), "--output", str(bpe_bin))
        actual = [json.loads(r)["tokens"] for r in call(str(bpe_bin), "--tokens",
                       input_text="\n".join(texts) + "\n").splitlines()]
        assert actual == [ids[:7] for ids in expected]
        long_tok["truncation"] = None

        # The byte-pair table must also handle incomplete alphabets: map absent
        # bytes to the unknown token or skip them when no unknown is configured.
        long_tok["model"]["vocab"] = {"a": 0, "b": 1, "ab": 2, "?": 3, "??": 4}
        long_tok["model"]["merges"] = ["a b", "? ?"]
        for unknown in ("?", None):
            long_tok["model"]["unk_token"] = unknown
            save_json(bpe / "0_StaticEmbedding" / "tokenizer.json", long_tok)
            safetensors(bpe / "0_StaticEmbedding" / "model.safetensors", "embedding.weight",
                        [5, 1], [0, 1, 2, 3, 4])
            call(compiler, "build", "fixture", "--local", str(bpe), "--output", str(bpe_bin))
            texts = ["axxb", "a" + "x" * 200 + "b", "xxab" * 100, "é😀ab" * 100]
            actual = [json.loads(r)["tokens"] for r in call(str(bpe_bin), "--tokens",
                           input_text="\n".join(texts) + "\n").splitlines()]
            assert actual == [reference_bpe(t, long_tok) for t in texts]


if __name__ == "__main__":
    main(sys.argv[1])
