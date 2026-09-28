"""Reproducible streaming throughput benchmark for a generated Icicle executable.

Fetches a fixed-size slice of GooAQ questions through Hugging Face's dataset
viewer API. This is intentionally a CLI benchmark, including text input and
output formatting; it is not an in-memory encoder microbenchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import urllib.parse
import urllib.request


DATASET = "sentence-transformers/gooaq"
CONFIG = "pair"
SPLIT = "train"
PAGE = 100


def fetch_page(offset, count):
    query = urllib.parse.urlencode({
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "offset": offset, "length": count,
    })
    request = urllib.request.Request(
        "https://datasets-server.huggingface.co/rows?" + query,
        headers={"User-Agent": "icicle-benchmark/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return [r["row"]["question"].replace("\r", " ").replace("\n", " ") for r in data["rows"]]


def load_questions(path, count):
    if not path.exists():
        offsets = list(range(0, count, PAGE))
        with ThreadPoolExecutor(max_workers=6) as pool:
            pages = list(pool.map(lambda offset: fetch_page(offset, min(PAGE, count - offset)), offsets))
        questions = [q for page in pages for q in page]
        if len(questions) != count:
            raise RuntimeError(f"expected {count} questions, got {len(questions)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(questions) + "\n", encoding="utf-8")
    questions = path.read_text(encoding="utf-8").splitlines()
    if len(questions) != count:
        raise RuntimeError(f"cached corpus has {len(questions)} questions; expected {count}")
    return questions


def run_once(exe, input_path, fmt, cpu, runtime_args=()):
    def affinity():
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})

    with input_path.open("rb") as stream, open(os.devnull, "wb") as sink:
        start = time.perf_counter()
        subprocess.run(
            [str(exe), "--format", fmt, *runtime_args], stdin=stream, stdout=sink,
            check=True, preexec_fn=affinity if cpu is not None else None,
        )
        return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--questions", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--format", choices=["binary", "jsonl"], default="binary")
    parser.add_argument("--tokens", action="store_true", help="measure token IDs only (JSONL)")
    parser.add_argument("--cpu", type=int, help="pin child to a Linux CPU number")
    parser.add_argument("--threads", type=int, help="runtime worker threads")
    parser.add_argument("--batch-size", type=int, help="runtime queries per batch")
    parser.add_argument("--workdir", type=Path, default=Path("build/bench"))
    args = parser.parse_args()
    if args.questions <= 0 or args.repeat <= 0 or args.runs <= 0:
        parser.error("questions, repeat, and runs must be positive")
    if args.tokens and args.format != "jsonl":
        parser.error("--tokens requires --format jsonl")
    if args.cpu is not None and not hasattr(os, "sched_setaffinity"):
        parser.error("--cpu requires Linux sched_setaffinity")
    exe = args.executable.resolve()
    runtime_args = []
    if args.threads is not None:
        runtime_args += ["--threads", str(args.threads)]
    if args.batch_size is not None:
        runtime_args += ["--batch-size", str(args.batch_size)]
    if args.tokens:
        runtime_args.append("--tokens")
    info = json.loads(subprocess.check_output([str(exe), "--info"], text=True))
    corpus = args.workdir / f"gooaq-{args.questions}.txt"
    questions = load_questions(corpus, args.questions)
    input_path = args.workdir / f"gooaq-{args.questions}-x{args.repeat}.txt"
    with input_path.open("wb") as out:
        for _ in range(args.repeat):
            with corpus.open("rb") as source:
                while block := source.read(1024 * 1024):
                    out.write(block)
    total = len(questions) * args.repeat
    run_once(exe, input_path, args.format, args.cpu, runtime_args)  # warmup
    seconds = [run_once(exe, input_path, args.format, args.cpu, runtime_args) for _ in range(args.runs)]
    result = {
        "model": info,
        "dataset": DATASET,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "unique_queries": len(questions),
        "total_queries_per_run": total,
        "mean_chars_per_query": round(statistics.mean(map(len, questions)), 2),
        "format": args.format,
        "tokens_only": args.tokens,
        "threads": args.threads or 1,
        "batch_size": args.batch_size,
        "cpu_affinity": args.cpu,
        "platform": platform.platform(),
        "runs_seconds": [round(x, 4) for x in seconds],
        "median_qps": round(total / statistics.median(seconds)),
        "best_qps": round(total / min(seconds)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
