"""Compare CLI throughput with the generated program's in-memory benchmark."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("build/bench/gooaq-10000.txt"))
    parser.add_argument("--threads", default="8,16,24")
    parser.add_argument("--batch-sizes", default="4096,16384,65536")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", default="memory,cli")
    parser.add_argument("--pipeline", choices=["off", "on", "both"], default="both")
    args = parser.parse_args()
    counts = [int(n) for n in args.threads.split(",")]
    sizes = [int(n) for n in args.batch_sizes.split(",")]
    modes = args.modes.split(",")
    if args.repeat <= 0 or args.runs <= 0 or not set(modes) <= {"memory", "cli"}:
        parser.error("positive repeat/runs and modes memory,cli are required")
    exe = str(args.executable.resolve())
    corpus = args.corpus.read_bytes()
    if corpus and not corpus.endswith(b"\n"):
        corpus += b"\n"
    # Match the executable's getline framing; embedded CR bytes do not end a query.
    queries = corpus.count(b"\n") * args.repeat
    info = json.loads(subprocess.check_output([exe, "--info"]))
    results = []
    for threads in counts:
        for size in sizes:
            flags = ["--threads", str(threads), "--batch-size", str(size)]
            if "memory" in modes:
                result = json.loads(subprocess.check_output(
                    [exe, *flags, "--benchmark", "--benchmark-repeat", str(args.repeat),
                     "--benchmark-runs", str(args.runs)], input=corpus))
                results.append(result)
                print(json.dumps(result), file=sys.stderr, flush=True)
            if "cli" in modes:
                # A file source keeps Python out of the timed input/output path.
                with tempfile.TemporaryFile() as source, open(os.devnull, "wb") as sink:
                    for _ in range(args.repeat):
                        source.write(corpus)
                    for pipeline in ([False, True] if args.pipeline == "both" else [args.pipeline == "on"]):
                        cmd = [exe, *flags, "--format", "binary"] + (["--pipeline"] if pipeline else [])
                        seconds = []
                        for run in range(args.runs + 1):
                            source.seek(0)
                            start = time.perf_counter()
                            subprocess.run(cmd, stdin=source, stdout=sink, check=True)
                            elapsed = time.perf_counter() - start
                            if run:
                                seconds.append(elapsed)
                        result = dict(mode="cli", threads=threads, batch_size=size, pipeline=pipeline,
                                      runs_seconds=seconds, median_qps=queries / statistics.median(seconds))
                        results.append(result)
                        print(json.dumps(result), file=sys.stderr, flush=True)
    print(json.dumps(dict(model=info, corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                         queries_per_run=queries, platform=platform.platform(), schedule="dynamic",
                         cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                         results=results), indent=2))


if __name__ == "__main__":
    main()
