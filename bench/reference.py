"""Optional Sentence Transformers comparison for a cached benchmark corpus."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default="sentence-transformers/static-retrieval-mrl-en-v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    queries = args.corpus.read_text(encoding="utf-8").splitlines()
    model = SentenceTransformer(args.model, device="cpu")
    model.encode(queries[:500], batch_size=args.batch_size, show_progress_bar=False)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        embeddings = model.encode(
            queries, batch_size=args.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        times.append(time.perf_counter() - start)
    print(json.dumps({
        "model": args.model,
        "queries": len(queries),
        "output_shape": list(embeddings.shape),
        "batch_size": args.batch_size,
        "torch_threads": 1,
        "runs_seconds": [round(x, 4) for x in times],
        "median_qps": round(len(queries) / statistics.median(times)),
    }, indent=2))


if __name__ == "__main__":
    main()
