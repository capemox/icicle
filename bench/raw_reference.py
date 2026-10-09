"""Compare raw text-to-batch encoding in Sentence Transformers and Icicle.

HF raw timing calls the standard preprocess and forward methods under inference
mode, retaining one batch. It includes tokenization, model arithmetic, and output
allocation; excludes encode() sorting/result assembly and all I/O. A separate
stock API measurement keeps normal encode()/encode_query() defaults. Requires
the optional sentence-transformers benchmark dependencies.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["dense", "sparse"])
    parser.add_argument("--local-model", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("build/bench/gooaq-10000.txt"))
    parser.add_argument("--batch-sizes", required=True, help="comma-separated HF batch sizes")
    parser.add_argument("--torch-threads", default="default,1,8")
    parser.add_argument("--sweep-repeat", type=int, default=1)
    parser.add_argument("--final-repeat", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    sizes = [int(v) for v in args.batch_sizes.split(",")]
    threads = args.torch_threads.split(",")
    if min(args.sweep_repeat, args.final_repeat, args.runs, *sizes) <= 0:
        parser.error("batch sizes, repetitions and runs must be positive")
    if args.family == "sparse" and max(sizes) > 4096:
        parser.error("sparse HF batches are limited to 4096 to bound dense intermediate memory")
    corpus = args.corpus.read_bytes()
    queries = corpus.decode().splitlines()
    if not queries or corpus.count(b"\n") != len(queries):
        parser.error("need a nonempty newline-terminated corpus with one query per line")
    executable = str(args.executable.resolve())
    native_info = json.loads(subprocess.check_output([executable, "--info"]))
    expected = "dense" if args.family == "dense" else "sparse-query"
    if native_info["kind"] != expected or native_info["weight_dtype"] != "float32" or native_info["dimension"] != native_info["source_dimension"]:
        parser.error("need a matching full-width float32 executable")
    default_threads = torch.get_num_threads()
    model_class = SentenceTransformer if args.family == "dense" else SparseEncoder
    model = model_class(str(args.local_model), device="cpu", local_files_only=True)
    model.eval()
    route = {} if args.family == "dense" else {"task": "query"}
    prompt = model.prompts.get("query", "") if args.family == "sparse" else (
        model.prompts.get(model.default_prompt_name, "") if model.default_prompt_name else "")
    encoder = model[0] if args.family == "dense" else model[0].sub_modules["query"][0]
    tokenizer = encoder.tokenizer

    def raw(batch):
        features = model.preprocess(batch, prompt=prompt, **route)
        return model.forward(features, **route)["sentence_embedding"]

    # Validate native output against the standard raw pipeline outside timing.
    sample = queries[:1000] + ["", "hello hello world", "CAFÉ naïve résumé", "北京你好", "σ Σ ς", "🚀 hello"]
    with torch.inference_mode():
        reference = raw(sample).numpy()
    output = subprocess.check_output([executable, "--format", "binary"], input=("\n".join(sample)+"\n").encode())
    if args.family == "dense":
        actual = np.frombuffer(output, dtype="<f4").reshape(reference.shape)
    else:
        actual = np.zeros_like(reference)
        cursor = 0
        for row in range(len(sample)):
            count, = struct.unpack_from("<I", output, cursor)
            cursor += 4
            for _ in range(count):
                token, weight = struct.unpack_from("<If", output, cursor)
                cursor += 8
                actual[row, token] = weight
        if cursor != len(output):
            raise RuntimeError("unexpected sparse binary framing")
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    parity = dict(queries=len(sample), maximum_absolute_error=float(np.max(np.abs(actual-reference))),
                  byte_equal=actual.tobytes()==reference.tobytes(), rtol=1e-5, atol=1e-6)
    del actual, reference, output

    def record(result):
        print(json.dumps(result), file=sys.stderr, flush=True)
        return result

    def measure(batch_size, thread_count, repeat, label):
        torch.set_num_threads(thread_count)
        texts = queries * repeat
        batches = [texts[first:first+batch_size] for first in range(0,len(texts),batch_size)]
        seconds = []
        embeddings = None
        with torch.inference_mode():
            for i in range(args.runs+1):
                embeddings = None
                start = time.perf_counter()
                for batch in batches:
                    embeddings = raw(batch)
                elapsed = time.perf_counter()-start
                if i:
                    seconds.append(elapsed)
        return record(dict(label=label, mode="raw-pipeline", backend="sentence-transformers",
                           batch_size=batch_size, torch_threads=thread_count, queries_per_run=len(texts),
                           runs_seconds=seconds, median_qps=len(texts)/statistics.median(seconds),
                           checksum=float(embeddings[-1].sum()), final_output_shape=list(embeddings.shape)))

    # Stock public API: original queries once, default batches and threads.
    torch.set_num_threads(default_threads)
    encode = model.encode if args.family == "dense" else model.encode_query
    seconds = []
    embeddings = None
    for i in range(args.runs+1):
        embeddings = None
        start = time.perf_counter()
        embeddings = encode(queries, show_progress_bar=False)
        elapsed = time.perf_counter()-start
        if i:
            seconds.append(elapsed)
    api = record(dict(label="hf-api-default", mode="public-api", backend="sentence-transformers",
                      batch_size=32, torch_threads=default_threads, queries_per_run=len(queries),
                      runs_seconds=seconds, median_qps=len(queries)/statistics.median(seconds),
                      output_shape=list(embeddings.shape)))
    del embeddings
    sweep = []
    for thread in threads:
        count = default_threads if thread == "default" else int(thread)
        for size in sizes:
            sweep.append(measure(size, count, args.sweep_repeat, "hf-sweep"))
    best = max(sweep, key=lambda r:r["median_qps"])
    final = measure(best["batch_size"], best["torch_threads"], args.final_repeat, "hf-best-raw")
    native_threads = 32 if args.family == "dense" else 24
    memory = json.loads(subprocess.check_output(
        [executable,"--benchmark","--threads",str(native_threads),"--batch-size","65536",
         "--benchmark-repeat",str(args.final_repeat),"--benchmark-runs",str(args.runs)], input=corpus))
    memory.update(label="icicle-best-raw", backend="icicle")
    record(memory)
    cmd = [executable,"--threads",str(native_threads),"--batch-size","16384","--pipeline","--format","binary"]
    seconds = []
    with tempfile.TemporaryFile() as source, open(os.devnull,"wb") as sink:
        for _ in range(args.final_repeat):
            source.write(corpus)
        for i in range(args.runs+1):
            source.seek(0)
            start = time.perf_counter()
            subprocess.run(cmd, stdin=source, stdout=sink, check=True)
            elapsed = time.perf_counter()-start
            if i:
                seconds.append(elapsed)
    count = len(queries)*args.final_repeat
    cli = record(dict(label="icicle-best-cli", backend="icicle", mode="cli", pipeline=True,
                      threads=native_threads, batch_size=16384, queries_per_run=count,
                      runs_seconds=seconds, median_qps=count/statistics.median(seconds)))
    print(json.dumps(dict(date=datetime.now(timezone.utc).date().isoformat(), family=args.family,
                         native_model=native_info, corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                         native_compile_flags="-O3 -march=native", platform=platform.platform(),
                         cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                         versions={p:version(p) for p in ("sentence-transformers","torch","transformers","tokenizers","numpy")},
                         hf_tokenizer_class=type(tokenizer).__name__,
                         hf_tokenizer_is_fast=getattr(tokenizer,"is_fast",None), parity=parity,
                         api_default=api, hf_sweep=sweep, hf_best_raw=final,
                         icicle_best_raw=memory, icicle_best_cli=cli,
                         harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__ == "__main__":
    main()
