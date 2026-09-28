# Batching investigation

The final cleaned-build benchmark for both dense and sparse models is in [FINAL_RESULTS.md](FINAL_RESULTS.md). The measurements below document earlier batch experiments.

Flower Computer [reports](https://www.flowercomputer.com/posts/fast-static-embedding/) 4.5 million batched queries/s and 524,900 queries/s in its single-query test for its static encoder on an M4 Mac mini, an 8.6× throughput ratio. The post does not specify the batch size, thread count, treatment of output vectors, or benchmark harness. The [base M4 Mac mini](https://support.apple.com/en-ie/121555) has a 10-core CPU, so parallel work across cores is a plausible contributor to the gap, but the post does not establish that.

The default Icicle executable reads a stream of lines and encodes each query on one thread. Sending many lines to one process amortizes startup, but does not use more cores. On the existing 10,000-question GooAQ slice, the current native build reaches about 462,000 queries/s with binary output on one pinned performance core of the i9-13900HX.

## Parallel process experiment

`bench/parallel.py` launches independent copies of the current generated executable. Each copy reads the same 10,000 GooAQ questions repeated 20 times (200,000 queries per worker) and writes 1,024-dimensional float32 vectors to `/dev/null`. Workers are pinned first to eight distinct performance cores, then 16 efficiency cores, then the eight sibling hardware threads of the performance cores. Each result is the median of three timed runs after a warmup. Timing includes worker process startup, input reading, and binary output. The corpus SHA-256 is `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.

| Worker processes | Total queries/run | Median aggregate QPS |
| ---: | ---: | ---: |
| 1 | 200,000 | 464,095 |
| 4 | 800,000 | 1,804,456 |
| 8 | 1,600,000 | 3,014,194 |
| 16 | 3,200,000 | 2,954,631 |
| 32 | 6,400,000 | 4,708,434 |

The 32-process result is about 10.1× the one-process result and is in the same broad throughput range as Flower's reported 4.5 million QPS. The hardware and benchmark interfaces differ, so the figures are not a head-to-head comparison. This is a parallel streaming benchmark, not a batch API. On Linux, the model is embedded in the executable's file-backed `.rodata` section, so these processes map the same physical read-only model pages. Each process still has separate small mutable state and its own input and output stream.

```sh
python3 bench/parallel.py build/embed-batch-native \
  --corpus build/bench/gooaq-10000.txt --repeat 20 \
  --workers 1,4,8,16,32 \
  --cpus 8,12,0,2,4,6,10,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,1,3,5,7,9,11,13,15 \
  --runs 3
```

The CPU list is specific to this Linux machine. Omit `--cpus` on macOS.

## Shared-model batch mode

The generated C++ executable now accepts `--threads N --batch-size B`. It reads up to B lines on the main thread, runs a persistent pool of N workers, then writes the results in input order. A batch is also limited to 64 MiB of input text. The compiled weight table and WordPiece trie are one read-only mapping in the process; workers also share one read-only tokenizer object. Token IDs and other query scratch data are separate. Workers write distinct slots in a batch result buffer. The default invocation remains a one-line-at-a-time stream, which avoids waiting for a full batch on an interactive input stream.

Disabling the default C++ stream synchronization was important for throughput on short input lines. Python only launches the process for the benchmark; the executable reads and processes the queries in C++.

The following runs used the same 10,000 GooAQ questions repeated 100 times, binary output to `/dev/null`, a native `-march=native` build, and an i9-13900HX. The median is over three timed runs after one warmup. The single-thread process was pinned to CPU 0; the four- and eight-thread processes could run on CPUs 0, 2, 4, 6, 8, 10, 12, and 14 (eight distinct performance cores). The 16-thread process could run on all 32 logical CPUs. The corpus SHA-256 is `b8a280ef9ffa7b4f7b66b320c847b804bfeaff5b363294911ea7422e55203d87`.

| Mode | Median time for 1M queries | Median QPS |
| --- | ---: | ---: |
| Default streaming, one thread | 2.163 s | 462,328 |
| Batch size 256, four workers | 0.672 s | 1,487,556 |
| Batch size 256, eight workers | 0.706 s | 1,416,247 |
| Batch size 4096, 16 workers | 0.539 s | 1,854,482 |

Four workers gave about 3.2× the throughput of default streaming on this machine. More workers did not scale as well as separate processes: eight processes reached 3.01 million QPS on the eight performance cores, while eight threads allowed on those cores reached 1.42 million QPS. Moving result initialization from the input thread to workers improved batch throughput. Pinning the eight workers individually to performance cores raised one longer run to about 1.8 million QPS, so scheduling explains part of the gap. The remaining scaling gap has not been isolated.

```sh
taskset -c 0 python3 bench/benchmark.py build/embed-batch-native \
  --questions 10000 --repeat 100 --runs 3 --format binary
taskset -c 0,2,4,6,8,10,12,14 python3 bench/benchmark.py build/embed-batch-native \
  --questions 10000 --repeat 100 --runs 3 --format binary \
  --threads 4 --batch-size 256
taskset -c 0,2,4,6,8,10,12,14 python3 bench/benchmark.py build/embed-batch-native \
  --questions 10000 --repeat 100 --runs 3 --format binary \
  --threads 8 --batch-size 256
taskset -c 0-31 python3 bench/benchmark.py build/embed-batch-native \
  --questions 10000 --repeat 100 --runs 3 --format binary \
  --threads 16 --batch-size 4096
```

These numbers include text input and binary output to `/dev/null`; they are not an in-memory encoder microbenchmark. Python starts the executable and times its completion, but the hot loop and input parsing run in C++. The threaded result is below Flower's reported 4.5 million batched QPS, and its workload, output handling, and thread count are unspecified, so no direct speed ratio can be inferred. A direct comparison would need both encoders run on the same machine with the same model, batch size, inputs, and output handling.
