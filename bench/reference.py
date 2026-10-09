"""Benchmark the standard Hugging Face Sentence Transformers CPU encoder.

Requires sentence-transformers. Defaults leave encode batch size and Torch
threading untouched. Input preparation and loading are excluded; each encode
call returns the complete float32 NumPy matrix. No model code is modified.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time
from importlib.metadata import version

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default="sentence-transformers/static-retrieval-mrl-en-v1")
    parser.add_argument("--revision", default="f60985c706f192d45d218078e49e5a8b6f15283a")
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--check-executable", type=Path)
    args = parser.parse_args()
    if args.repeat <= 0 or args.runs <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("repeat, runs, and batch size must be positive")
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            parser.error("Torch thread count must be positive")
        torch.set_num_threads(args.torch_threads)
    corpus = args.corpus.read_bytes()
    unique = corpus.decode("utf-8").splitlines()
    if not unique or len(unique) * args.repeat > 1000000:
        parser.error("need between 1 and 1,000,000 queries including repetitions")
    queries = unique * args.repeat
    if args.local_model:
        model = SentenceTransformer(str(args.local_model), device="cpu", local_files_only=True)
    else:
        model = SentenceTransformer(args.model, revision=args.revision, device="cpu")
    options = {"show_progress_bar": False}
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    batch_size = args.batch_size or inspect.signature(model.encode).parameters["batch_size"].default
    parity = None
    if args.check_executable:
        import subprocess
        sample = unique[:1000] + ["", "hello hello world", "CAFÉ naïve résumé", "北京你好", "σ Σ ς", "🚀 hello"]
        reference = model.encode(sample, **options)
        raw = subprocess.check_output([str(args.check_executable.resolve()), "--format", "binary"],
                                      input=("\n".join(sample) + "\n").encode())
        native = np.frombuffer(raw, dtype="<f4").reshape(reference.shape)
        np.testing.assert_allclose(native, reference, rtol=1e-5, atol=1e-6)
        parity = dict(queries=len(sample), rtol=1e-5, atol=1e-6,
                      maximum_absolute_error=float(np.max(np.abs(native - reference))),
                      byte_equal=native.tobytes() == reference.tobytes())
    embeddings = model.encode(queries, **options)  # One complete warmup.
    seconds = []
    for _ in range(args.runs):
        embeddings = None  # Release the previous matrix outside timing.
        start = time.perf_counter()
        embeddings = model.encode(queries, **options)
        seconds.append(time.perf_counter() - start)
    if embeddings.dtype != np.float32 or embeddings.shape != (len(queries), model.get_embedding_dimension()):
        raise RuntimeError("unexpected output shape or precision")
    print(json.dumps(dict(
        backend="sentence-transformers", model=args.model, revision=args.revision,
        source="local snapshot" if args.local_model else "Hugging Face Hub",
        corpus_sha256=hashlib.sha256(corpus).hexdigest(), unique_queries=len(unique),
        queries=len(queries), output_shape=list(embeddings.shape), output_dtype=str(embeddings.dtype),
        output_bytes=embeddings.nbytes, batch_size=batch_size,
        batch_size_overridden=args.batch_size is not None,
        torch_threads=torch.get_num_threads(), torch_threads_overridden=args.torch_threads is not None,
        torch_interop_threads=torch.get_num_interop_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        thread_environment={k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM") if k in os.environ},
        platform=platform.platform(),
        versions={name: version(name) for name in ("sentence-transformers", "torch", "tokenizers", "numpy", "safetensors", "huggingface-hub")},
        runs_seconds=seconds, median_qps=len(queries) / statistics.median(seconds),
        checksum=float(embeddings[-1].sum()), parity=parity), indent=2))


if __name__ == "__main__":
    main()
