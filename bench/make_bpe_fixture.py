"""Make deterministic static-model fixtures around a real tokenizer.json.

The generated weights are synthetic. They let throughput runs use a fixed
vocabulary and embedding shape without requiring a trained BPE static model.
"""

import argparse
from array import array
import json
from pathlib import Path
import struct
import sys


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_weights(path, shape, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_size = shape[0] * (shape[1] if len(shape) == 2 else 1) * 4
    header = json.dumps({
        "embedding.weight" if len(shape) == 2 else "weight": {
            "dtype": "F32", "shape": shape, "data_offsets": [0, payload_size],
        },
    }).encode()
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        for row in rows:
            if sys.byteorder != "little":
                row.byteswap()
            row.tofile(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    if args.dimension <= 0:
        parser.error("--dimension must be positive")
    tokenizer = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    vocab_size = max(
        [*tokenizer["model"]["vocab"].values(),
         *(item["id"] for item in tokenizer.get("added_tokens", []))]
    ) + 1
    module = "query_0_SparseStaticEmbedding" if args.sparse else "0_StaticEmbedding"
    if args.sparse:
        write_json(args.output / "modules.json", [{
            "type": "sentence_transformers.models.Router", "path": "",
        }])
        write_json(args.output / "router_config.json", {
            "structure": {"query": [module]},
            "types": {module: "sentence_transformers.sparse_encoder.models.SparseStaticEmbedding.SparseStaticEmbedding"},
        })
        weights = (array("f", [1.0]) for _ in range(vocab_size))
        shape = [vocab_size]
    else:
        write_json(args.output / "modules.json", [{
            "type": "sentence_transformers.models.StaticEmbedding", "path": module,
        }])
        weights = (array("f", [((i % 257) - 128) / 128.0]) * args.dimension
                   for i in range(vocab_size))
        shape = [vocab_size, args.dimension]
    destination = args.output / module
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())
    write_weights(destination / "model.safetensors", shape, weights)
    print(f"created {args.output} ({vocab_size} tokens, {shape})")


if __name__ == "__main__":
    main()
