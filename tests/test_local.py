"""Offline end-to-end tests for model import, linking, and generated executables."""
import json
import math
import os
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
        assert call(str(sparse_bin), "--threads", "2", "--batch-size", "3", input_text=sparse_input) == \
            call(str(sparse_bin), input_text=sparse_input)
        assert subprocess.check_output([str(sparse_bin), "--threads", "2", "--batch-size", "2",
                                        "--format", "binary"], input=sparse_input.encode()) == \
            subprocess.check_output([str(sparse_bin), "--format", "binary"], input=sparse_input.encode())

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


if __name__ == "__main__":
    main(sys.argv[1])
