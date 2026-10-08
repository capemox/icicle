"""Evaluate dense executables on pinned NanoMSMARCO (50 queries, 5,043 documents).

Requires numpy and pyarrow. Reports retrieval metrics plus vector agreement
against the first executable; this is one NanoBEIR subset, not the full suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
import pyarrow.parquet as pq


DATASET = "zeta-alpha-ai/NanoMSMARCO"
REVISION = "7b8ff22f2771dc65ac5b439f222eb19a1f56abda"


def embeddings(executable, texts, dimension):
    data = "\n".join(t.replace("\r", " ").replace("\n", " ") for t in texts) + "\n"
    raw = subprocess.check_output([str(executable), "--format", "binary", "--threads", "8",
                                   "--batch-size", "256"], input=data.encode())
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != len(texts) * dimension:
        raise RuntimeError("unexpected embedding output size")
    return values.reshape(len(texts), dimension)


def normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", nargs="+", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("build/bench/nanoms"))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    tables, checksums = {}, {}
    for kind in ("corpus", "queries", "qrels"):
        path = args.workdir / (kind + ".parquet")
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{kind}/train-00000-of-00001.parquet"
            urllib.request.urlretrieve(url, path)
        checksums[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[kind] = pq.read_table(path).to_pylist()
    documents, queries = tables["corpus"], tables["queries"]
    positions = {doc["_id"]: i for i, doc in enumerate(documents)}
    relevant = {}
    for row in tables["qrels"]:
        relevant.setdefault(row["query-id"], set()).add(positions[row["corpus-id"]])
    output = []
    reference_vectors = reference_top = None
    for executable in args.executables:
        executable = executable.resolve()
        info = json.loads(subprocess.check_output([str(executable), "--info"]))
        dimension = info["dimension"]
        q = embeddings(executable, [row["text"] for row in queries], dimension)
        d = embeddings(executable, [row["text"] for row in documents], dimension)
        vectors = np.concatenate((q, d))
        scores = normalize(q) @ normalize(d).T
        top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
        ndcg, reciprocal, recall = [], [], []
        for row, order in zip(queries, top):
            truth = relevant[row["_id"]]
            hits = [i + 1 for i, doc in enumerate(order) if doc in truth]
            dcg = sum(1 / np.log2(rank + 1) for rank in hits)
            ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(10, len(truth)) + 1))
            ndcg.append(dcg / ideal)
            reciprocal.append(1 / hits[0] if hits else 0)
            recall.append(len(hits) / len(truth))
        result = dict(executable=str(executable), model=info, ndcg_at_10=float(np.mean(ndcg)),
                      mrr_at_10=float(np.mean(reciprocal)), recall_at_10=float(np.mean(recall)))
        if reference_vectors is None:
            reference_vectors, reference_top = vectors, top
        elif dimension <= reference_vectors.shape[1]:
            ref = reference_vectors[:, :dimension]
            cosine = np.sum(normalize(ref) * normalize(vectors), axis=1)
            result.update(prefix_byte_equal=ref.tobytes() == vectors.tobytes(),
                          mean_cosine_to_reference_prefix=float(np.mean(cosine)),
                          minimum_cosine_to_reference_prefix=float(np.min(cosine)),
                          maximum_absolute_error_to_prefix=float(np.max(np.abs(ref - vectors))),
                          mean_top10_overlap=float(np.mean([len(set(a) & set(b)) / 10
                                                          for a, b in zip(reference_top, top)])))
        output.append(result)
    print(json.dumps(dict(dataset=DATASET, revision=REVISION, file_sha256=checksums,
                         queries=len(queries), documents=len(documents), results=output), indent=2))


if __name__ == "__main__":
    main()
