"""Compare full-width float32 Icicle with the standard HF CPU encoder.

Configurations run sequentially. Requires the optional dependencies of
reference.py. Loading is excluded; CLI and in-memory timings are separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def native_numpy(executable, corpus, repeat, runs, worker_count, dimension):
    """Time the CLI plus transport and a complete owned NumPy result array."""
    import numpy as np
    payload = corpus * repeat
    queries = corpus.count(b"\n") * repeat
    cmd = [str(executable), "--threads", str(worker_count), "--batch-size", "16384",
           "--pipeline", "--format", "binary"]
    seconds = []
    raw = embeddings = None
    for i in range(runs + 1):
        raw = embeddings = None
        start = time.perf_counter()
        raw = subprocess.check_output(cmd, input=payload)
        embeddings = np.frombuffer(raw, dtype="<f4").reshape(queries, dimension).copy()
        elapsed = time.perf_counter() - start
        if i:
            seconds.append(elapsed)
    return dict(label="icicle-cli-to-numpy", backend="icicle", mode="cli-to-numpy",
                threads=worker_count, batch_size=16384, pipeline=True,
                queries_per_run=queries, output_shape=list(embeddings.shape),
                output_dtype=str(embeddings.dtype), output_bytes=embeddings.nbytes,
                owned_array=bool(embeddings.flags.owndata), runs_seconds=seconds,
                median_qps=queries / statistics.median(seconds),
                checksum=float(embeddings[-1].sum()),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("build/bench/gooaq-10000.txt"))
    parser.add_argument("--model", default="sentence-transformers/static-retrieval-mrl-en-v1")
    parser.add_argument("--revision", default="f60985c706f192d45d218078e49e5a8b6f15283a")
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--hf-batch-sizes", default="256,4096,16384,65536")
    parser.add_argument("--hf-threads", default="default,1,8,32")
    parser.add_argument("--native-threads", type=int, default=32)
    args = parser.parse_args()
    if args.repeat <= 0 or args.runs <= 0 or args.native_threads <= 0:
        parser.error("repeat, runs, and native threads must be positive")
    sizes = [int(n) for n in args.hf_batch_sizes.split(",")]
    threads = args.hf_threads.split(",")
    if any(n <= 0 for n in sizes) or any(n != "default" and int(n) <= 0 for n in threads):
        parser.error("HF batch sizes and thread counts must be positive")
    corpus = args.corpus.read_bytes()
    if not corpus.endswith(b"\n"):
        parser.error("corpus must end with a newline")
    queries = corpus.count(b"\n") * args.repeat
    executable = str(args.executable.resolve())
    info = json.loads(subprocess.check_output([executable, "--info"]))
    if info["kind"] != "dense" or info["weight_dtype"] != "float32" or info["dimension"] != info["source_dimension"]:
        parser.error("need an unquantized, full-width dense executable")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None

    def run(cmd, *, cpu=None, **kwargs):
        def pin():
            os.sched_setaffinity(0, {cpu})
        return subprocess.run(cmd, check=True, preexec_fn=pin if cpu is not None else None, **kwargs)

    results = []

    def record(result):
        results.append(result)
        print(json.dumps({k: result[k] for k in ("label", "median_qps", "batch_size")}), file=sys.stderr, flush=True)

    def hf(label, batch_size=None, torch_threads=None, cpu=None, check=False):
        cmd = [sys.executable, str(Path(__file__).with_name("reference.py")), str(args.corpus),
               "--model", args.model, "--revision", args.revision, "--repeat", str(args.repeat), "--runs", str(args.runs)]
        if args.local_model:
            cmd += ["--local-model", str(args.local_model)]
        if batch_size is not None:
            cmd += ["--batch-size", str(batch_size)]
        if torch_threads is not None:
            cmd += ["--torch-threads", str(torch_threads)]
        if check:
            cmd += ["--check-executable", executable]
        result = json.loads(run(cmd, cpu=cpu, stdout=subprocess.PIPE).stdout)
        if result["queries"] != queries or result["output_shape"][1] != info["dimension"]:
            raise RuntimeError("HF and native input counts or dimensions differ")
        record(dict(result, label=label))

    def native(label, worker_count, batch_size, cpu=None, cli=False):
        cmd = [executable, "--threads", str(worker_count), "--batch-size", str(batch_size)]
        if not cli:
            cmd += ["--benchmark", "--benchmark-repeat", str(args.repeat), "--benchmark-runs", str(args.runs)]
            result = json.loads(run(cmd, cpu=cpu, input=corpus, stdout=subprocess.PIPE).stdout)
            if result["queries_per_run"] != queries:
                raise RuntimeError("native input count differs")
        else:
            cmd += ["--format", "binary", "--pipeline"]
            with tempfile.TemporaryFile() as source, open(os.devnull, "wb") as sink:
                for _ in range(args.repeat):
                    source.write(corpus)
                seconds = []
                for i in range(args.runs + 1):
                    source.seek(0)
                    start = time.perf_counter()
                    run(cmd, cpu=cpu, stdin=source, stdout=sink)
                    elapsed = time.perf_counter() - start
                    if i:
                        seconds.append(elapsed)
            result = dict(mode="cli", threads=worker_count, batch_size=batch_size, pipeline=True,
                          queries_per_run=queries, runs_seconds=seconds,
                          median_qps=queries / statistics.median(seconds))
        record(dict(result, backend="icicle", label=label, cpu_affinity=[cpu] if cpu is not None else affinity))

    hf("hf-default", check=True)
    for thread in threads:
        for size in sizes:
            hf(f"hf-batch-{size}-threads-{thread}", batch_size=size,
               torch_threads=None if thread == "default" else int(thread))
    if affinity:
        hf("hf-single-core-default", cpu=affinity[0])
        hf("hf-single-core-batch-4096", batch_size=4096, cpu=affinity[0])
        native("icicle-single-core-batch-32", 1, 32, cpu=affinity[0])
        native("icicle-single-core-batch-4096", 1, 4096, cpu=affinity[0])
    for size in (16384, 65536):
        native(f"icicle-memory-batch-{size}", args.native_threads, size)
    native("icicle-cli-pipeline", args.native_threads, 16384, cli=True)
    record(native_numpy(executable, corpus, args.repeat, args.runs, args.native_threads, info["dimension"]))
    source_hashes = None
    if args.local_model:
        source_hashes = {str(p.relative_to(args.local_model)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args.local_model.rglob("*")
                         if p.is_file() and p.suffix in (".json", ".safetensors")}
    print(json.dumps(dict(date=datetime.now(timezone.utc).date().isoformat(),
                         model=args.model, revision=args.revision, native_model=info,
                         harness_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (Path(__file__), Path(__file__).with_name("reference.py"))},
                         source_file_sha256=source_hashes, corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                         unique_queries=corpus.count(b"\n"), queries_per_run=queries, runs=args.runs,
                         platform=platform.platform(), cpu_affinity=affinity, results=results), indent=2))


if __name__ == "__main__":
    main()
