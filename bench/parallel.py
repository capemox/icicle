"""Measure aggregate throughput from independent Icicle worker processes.

Each worker reads the same query file and writes binary embeddings to /dev/null.
This measures parallel streaming throughput, including process startup, rather
than an in-memory batched encoder API.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def run_once(exe, input_path, workers, cpus):
    sources = []
    processes = []
    with open(os.devnull, "wb") as sink:
        start = time.perf_counter()
        try:
            for i in range(workers):
                source = input_path.open("rb")
                sources.append(source)
                cpu = cpus[i] if cpus else None

                def affinity(cpu=cpu):
                    os.sched_setaffinity(0, {cpu})

                processes.append(subprocess.Popen(
                    [str(exe), "--format", "binary"], stdin=source,
                    stdout=sink, preexec_fn=affinity if cpu is not None else None,
                ))
            for process in processes:
                if process.wait() != 0:
                    raise RuntimeError(f"worker {process.pid} failed")
            return time.perf_counter() - start
        finally:
            for source in sources:
                source.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("build/bench/gooaq-10000.txt"))
    parser.add_argument("--repeat", type=int, default=20, help="corpus copies per worker")
    parser.add_argument("--workers", default="1,2,4,8", help="comma-separated worker counts")
    parser.add_argument("--cpus", help="comma-separated CPU IDs, in worker order (Linux only)")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    counts = [int(value) for value in args.workers.split(",")]
    cpus = [int(value) for value in args.cpus.split(",")] if args.cpus else []
    if any(count <= 0 for count in counts) or args.repeat <= 0 or args.runs <= 0:
        parser.error("worker counts, repeat, and runs must be positive")
    if cpus and (len(cpus) < max(counts) or not hasattr(os, "sched_setaffinity")):
        parser.error("--cpus requires Linux and one CPU ID per worker")
    exe = args.executable.resolve()
    corpus = args.corpus.read_bytes()
    per_worker = len(corpus.splitlines()) * args.repeat
    input_path = args.corpus.parent / f"{args.corpus.stem}-parallel-x{args.repeat}.txt"
    with input_path.open("wb") as out:
        for _ in range(args.repeat):
            out.write(corpus)
    results = []
    for workers in counts:
        run_once(exe, input_path, workers, cpus)
        seconds = [run_once(exe, input_path, workers, cpus) for _ in range(args.runs)]
        results.append({
            "workers": workers,
            "queries_per_run": per_worker * workers,
            "runs_seconds": [round(value, 4) for value in seconds],
            "median_qps": round(per_worker * workers / statistics.median(seconds)),
        })
    print(json.dumps({
        "model": json.loads(subprocess.check_output([str(exe), "--info"], text=True)),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "queries_per_worker": per_worker,
        "cpu_order": cpus,
        "platform": platform.platform(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
