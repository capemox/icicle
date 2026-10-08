#include "tokenizer.hpp"
#include "generated_model.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unicode/utf8.h>

extern "C" const unsigned char icicle_blob[];

static bool valid_utf8(const std::string& s) {
  const auto* p = reinterpret_cast<const uint8_t*>(s.data());
  int32_t i = static_cast<int32_t>(icicle::ascii_prefix_length(s));
  int32_t n = static_cast<int32_t>(s.size());
  while (i < n) { UChar32 c; U8_NEXT(p, i, n, c); if (c < 0) return false; }
  return true;
}
static void number(float x) {
  if (!std::isfinite(x)) { std::cout << "null"; return; }
  std::cout << x;
}
static void json_string(std::string_view s) {
  constexpr char hex[] = "0123456789abcdef";
  std::cout << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { std::cout << '\\' << static_cast<char>(c); }
    else if (c < 0x20) std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
    else std::cout << static_cast<char>(c);
  }
  std::cout << '"';
}

static icicle::Tokenizer make_tokenizer() {
  std::vector<icicle::AddedToken> added;
  for (const auto& t : model::added_tokens) added.push_back({t.text, t.id, t.lstrip, t.rstrip});
  return icicle::Tokenizer(icicle_blob, model::nodes_offset, model::nodes_count,
    model::bpe, model::merges_offset, model::merges_count,
    model::merge_index_offset, model::merge_index_count,
    model::byte_pairs_offset,
    model::bytelevel_prefix_space, model::bytelevel_regex,
    model::unknown_id, model::max_chars, model::max_tokens,
    model::clean_text, model::chinese, model::lowercase, model::strip_accents, std::move(added));
}

static void add_embedding_row(uint32_t id, float* __restrict__ result) {
  if constexpr (model::quantized) {
    const int8_t* row = reinterpret_cast<const int8_t*>(icicle_blob) + static_cast<size_t>(id) * model::dimension;
    const float* scales = reinterpret_cast<const float*>(icicle_blob + model::scales_offset);
    float scale = scales[id];
    for (uint32_t j = 0; j < model::dimension; ++j) result[j] += static_cast<float>(row[j]) * scale;
  } else {
    const float* row = reinterpret_cast<const float*>(icicle_blob) + static_cast<size_t>(id) * model::dimension;
    for (uint32_t j = 0; j < model::dimension; ++j) result[j] += row[j];
  }
}

static void finish_embedding(float* result, size_t count) {
  if (count) {
    float divisor = static_cast<float>(count);
    for (uint32_t j = 0; j < model::dimension; ++j) result[j] /= divisor;
  }
  if (model::normalize) {
    float squared = 0.0f;
    for (uint32_t j = 0; j < model::dimension; ++j) squared += result[j] * result[j];
    float length = std::max(std::sqrt(squared), 1e-12f);
    for (uint32_t j = 0; j < model::dimension; ++j) result[j] /= length;
  }
}

struct Batch {
  std::vector<std::string> lines;
  std::span<const std::string_view> views;
  std::vector<std::vector<uint32_t>> ids;
  std::unique_ptr<float[]> dense_storage;
  size_t dense_capacity = 0;
  std::span<float> dense;
  size_t size() const { return views.empty() ? lines.size() : views.size(); }
  std::string_view line(size_t i) const { return views.empty() ? std::string_view(lines[i]) : views[i]; }
  void resize_dense(size_t count) {
    if (count > dense_capacity) {
      // Workers initialize every output row before using it. Avoid a redundant
      // serial zero-fill (and page faults) when allocating large batch buffers.
      dense_storage.reset(new float[count]);
      dense_capacity = count;
    }
    dense = {dense_storage.get(), count};
  }
};

// Keep frequently modified scratch headers on separate cache lines.
struct alignas(64) WorkerScratch { icicle::TokenizerScratch tokens; };

class BatchWorkers {
public:
  BatchWorkers(size_t count, const std::string& prompt, const float* weights, bool tokens_only)
    : count_(count), prompt_(prompt), weights_(weights), tokens_only_(tokens_only),
      tokenizer_(make_tokenizer()) {
    errors_.resize(count);
    scratch_.resize(count);
    try {
      if (count > 1)
        for (size_t i = 0; i < count; ++i) workers_.emplace_back([this, i] { worker(i); });
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
      }
      start_.notify_all();
      for (auto& worker : workers_) worker.join();
      throw;
    }
  }

  ~BatchWorkers() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    start_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  void start(Batch& batch) {
    if (count_ == 1) {
      for (size_t i = 0; i < batch.size(); ++i) encode_one(batch, i, scratch_[0]);
      return;
    }
    {
      std::lock_guard lock(mutex_);
      active_ = &batch;
      std::fill(errors_.begin(), errors_.end(), std::exception_ptr{});
      remaining_ = count_;
      chunk_ = std::clamp<size_t>(batch.size() / (count_ * 4), 1, 32);
      next_.store(0, std::memory_order_relaxed);
      ++generation_;
    }
    start_.notify_all();
  }

  void finish() {
    if (count_ == 1) return;
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
    for (const auto& error : errors_) if (error) std::rethrow_exception(error);
  }

  void run(Batch& batch) { start(batch); finish(); }

private:
  void encode_one(Batch& batch, size_t index, WorkerScratch& worker_scratch) const {
    auto& scratch = worker_scratch.tokens;
    scratch.input.clear();
    scratch.input.append(prompt_);
    scratch.input.append(batch.line(index));
    tokenizer_.encode_into(scratch.input, scratch);
    auto& ids = scratch.ids;
    if (tokens_only_) {
      // Swap back the previous batch's vector so both buffers keep their capacity.
      batch.ids[index].swap(ids);
      return;
    }
    if (model::kind == 1) {
      float* result = batch.dense.data() + index * model::dimension;
      std::fill_n(result, model::dimension, 0.0f);
      for (uint32_t id : ids) add_embedding_row(id, result);
      finish_embedding(result, ids.size());
    } else {
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      ids.erase(std::remove_if(ids.begin(), ids.end(), [&](uint32_t id) { return weights_[id] == 0.0f; }), ids.end());
      batch.ids[index].swap(ids);
    }
  }

  void worker(size_t worker_index) {
    size_t seen = 0;
    for (;;) {
      Batch* batch;
      {
        std::unique_lock lock(mutex_);
        start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
        batch = active_;
      }
      try {
        // Contiguous chunks reduce atomics and let fast cores take more work.
        size_t chunk = chunk_;
        for (;;) {
          size_t first = next_.fetch_add(chunk, std::memory_order_relaxed);
          if (first >= batch->size()) break;
          size_t last = std::min(first + chunk, batch->size());
          for (size_t i = first; i < last; ++i)
            encode_one(*batch, i, scratch_[worker_index]);
        }
      } catch (...) {
        errors_[worker_index] = std::current_exception();
      }
      {
        std::lock_guard lock(mutex_);
        if (--remaining_ == 0) done_.notify_one();
      }
    }
  }

  size_t count_, generation_ = 0, remaining_ = 0;
  const std::string& prompt_;
  const float* weights_;
  bool tokens_only_, stopping_ = false;
  Batch* active_ = nullptr;
  const icicle::Tokenizer tokenizer_;
  std::vector<WorkerScratch> scratch_;
  std::vector<std::thread> workers_;
  std::vector<std::exception_ptr> errors_;
  std::mutex mutex_;
  std::condition_variable start_, done_;
  size_t chunk_ = 32;
  alignas(64) std::atomic<size_t> next_{0};
};

static void write_batch(const Batch& batch, const float* weights, bool tokens_only, bool binary,
                        std::vector<char>& binary_buffer) {
  if (model::kind == 1 && !tokens_only && binary) {
    std::cout.write(reinterpret_cast<const char*>(batch.dense.data()),
                    static_cast<std::streamsize>(batch.dense.size() * sizeof(float)));
    return;
  }
  if (model::kind == 2 && !tokens_only && binary) {
    // Reuse a bounded buffer across batches. Most batches need a single write;
    // unusually large results are flushed in chunks without changing framing.
    if (binary_buffer.empty()) binary_buffer.resize(1024 * 1024);
    size_t used = 0;
    auto append = [&](const auto& value) {
      if (used + sizeof(value) > binary_buffer.size()) {
        std::cout.write(binary_buffer.data(), static_cast<std::streamsize>(used));
        used = 0;
      }
      std::memcpy(binary_buffer.data() + used, &value, sizeof(value));
      used += sizeof(value);
    };
    struct Pair { uint32_t id; float weight; };
    static_assert(sizeof(Pair) == 8);
    for (const auto& ids : batch.ids) {
      append(static_cast<uint32_t>(ids.size()));
      for (uint32_t id : ids) append(Pair{id, weights[id]});
    }
    if (used) std::cout.write(binary_buffer.data(), static_cast<std::streamsize>(used));
    return;
  }
  for (size_t i = 0; i < batch.size(); ++i) {
    if (tokens_only) {
      std::cout << "{\"tokens\":[";
      for (size_t j = 0; j < batch.ids[i].size(); ++j) {
        if (j) std::cout << ',';
        std::cout << batch.ids[i][j];
      }
      std::cout << "]}\n";
    } else if (model::kind == 1) {
      const float* result = batch.dense.data() + i * model::dimension;
      std::cout << "{\"embedding\":[";
      for (uint32_t j = 0; j < model::dimension; ++j) {
        if (j) std::cout << ',';
        number(result[j]);
      }
      std::cout << "]}\n";
    } else {
      std::cout << "{\"indices\":[";
      for (size_t j = 0; j < batch.ids[i].size(); ++j) {
        if (j) std::cout << ',';
        std::cout << batch.ids[i][j];
      }
      std::cout << "],\"values\":[";
      for (size_t j = 0; j < batch.ids[i].size(); ++j) {
        if (j) std::cout << ',';
        number(weights[batch.ids[i][j]]);
      }
      std::cout << "],\"dimension\":" << model::dimension << "}\n";
    }
  }
}

static void prepare_batch(Batch& batch, bool tokens_only) {
  if (model::kind != 1 || tokens_only) batch.ids.resize(batch.size());
  if (model::kind == 1 && !tokens_only) batch.resize_dense(batch.size() * model::dimension);
}

static size_t bounded_batch_size(size_t requested, bool tokens_only) {
  // Three pipeline slots each hold at most 256 MiB of dense output.
  constexpr size_t max_output = 256 * 1024 * 1024;
  if (model::kind == 1 && !tokens_only) {
    size_t row_bytes = static_cast<size_t>(model::dimension) * sizeof(float);
    if (row_bytes > max_output) throw std::runtime_error("dense output row exceeds 256 MiB");
    return std::min(requested, max_output / row_bytes);
  }
  return requested;
}

static void read_batch(Batch& batch, size_t batch_size, std::optional<std::string>& pending) {
  batch.lines.resize(batch_size);
  size_t count = 0, bytes = 0;
  constexpr size_t max_batch_bytes = 64 * 1024 * 1024;
  while (count < batch_size) {
    std::string& line = batch.lines[count];
    if (pending) { line = std::move(*pending); pending.reset(); }
    else if (!std::getline(std::cin, line)) break;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("input line exceeds 16 MiB");
    if (!valid_utf8(line)) throw std::runtime_error("input is not valid UTF-8");
    if (bytes + line.size() > max_batch_bytes && count != 0) { pending = std::move(line); break; }
    bytes += line.size();
    ++count;
  }
  batch.lines.resize(count);
  if (!std::cin && !std::cin.eof()) throw std::runtime_error("failed to read stdin");
}

static void run_batched(size_t threads, size_t batch_size, const std::string& prompt,
                        const float* weights, bool tokens_only, bool binary, bool pipeline) {
  // Declare batches first: workers must finish before their buffers are freed,
  // including when input validation or output fails while a batch is in flight.
  std::array<Batch, 3> batches;
  BatchWorkers workers(threads, prompt, weights, tokens_only);
  std::vector<char> binary_buffer;
  std::optional<std::string> pending;
  batch_size = bounded_batch_size(batch_size, tokens_only);
  if (!pipeline || threads == 1) {
    for (;;) {
      auto& batch = batches[0];
      read_batch(batch, batch_size, pending);
      if (!batch.size()) break;
      prepare_batch(batch, tokens_only);
      workers.run(batch);
      write_batch(batch, weights, tokens_only, binary, binary_buffer);
      if (!std::cout) throw std::runtime_error("failed to write stdout");
    }
    return;
  }
  size_t current = 0, next = 1, previous = 2;
  bool have_previous = false;
  read_batch(batches[current], batch_size, pending);
  if (!batches[current].size()) return;
  prepare_batch(batches[current], tokens_only);
  workers.start(batches[current]);
  for (;;) {
    // The main thread writes B-1 and reads B+1 while workers encode B. Keeping
    // input on this thread avoids an uncancellable background getline on errors.
    if (have_previous) write_batch(batches[previous], weights, tokens_only, binary, binary_buffer);
    if (!std::cout) throw std::runtime_error("failed to write stdout");
    read_batch(batches[next], batch_size, pending);
    prepare_batch(batches[next], tokens_only);
    workers.finish();
    if (!batches[next].size()) {
      write_batch(batches[current], weights, tokens_only, binary, binary_buffer);
      break;
    }
    workers.start(batches[next]);
    size_t spare = previous;
    previous = current;
    current = next;
    next = spare;
    have_previous = true;
  }
  if (!std::cout) throw std::runtime_error("failed to write stdout");
}

static void run_benchmark(size_t threads, size_t batch_size, const std::string& prompt,
                          const float* weights, bool tokens_only, size_t repeat, size_t runs) {
  std::vector<std::string> corpus;
  std::string line;
  size_t bytes = 0;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("input line exceeds 16 MiB");
    if (!valid_utf8(line)) throw std::runtime_error("input is not valid UTF-8");
    bytes += line.size();
    if (bytes > 256 * 1024 * 1024 || corpus.size() >= 10000000)
      throw std::runtime_error("benchmark corpus exceeds 256 MiB or 10 million lines");
    corpus.push_back(std::move(line));
  }
  if (!std::cin.eof()) throw std::runtime_error("failed to read stdin");
  if (corpus.empty()) throw std::runtime_error("benchmark needs at least one input line");
  if (repeat > 10000000 / corpus.size()) throw std::runtime_error("benchmark exceeds 10 million query references");
  std::vector<std::string_view> inputs;
  inputs.reserve(corpus.size() * repeat);
  for (size_t r = 0; r < repeat; ++r)
    for (const auto& text : corpus) inputs.emplace_back(text);
  batch_size = bounded_batch_size(batch_size, tokens_only);
  Batch batch;
  BatchWorkers workers(threads, prompt, weights, tokens_only);
  std::vector<double> seconds;
  for (size_t run = 0; run <= runs; ++run) {
    auto start = std::chrono::steady_clock::now();
    for (size_t first = 0; first < inputs.size(); first += batch_size) {
      batch.views = std::span(inputs).subspan(first, std::min(batch_size, inputs.size() - first));
      prepare_batch(batch, tokens_only);
      workers.run(batch);
      // Make all output stores observable to the optimizer without timing a
      // checksum or serialization. GCC and Apple Clang support this barrier.
      asm volatile("" : : "g"(batch.dense.data()), "g"(batch.ids.data()) : "memory");
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (run) seconds.push_back(elapsed);
  }
  // Diagnostic checksum of the final batch, outside all timed regions.
  double checksum = 0;
  for (float value : batch.dense) checksum += value;
  for (const auto& ids : batch.ids) for (uint32_t id : ids)
    checksum += tokens_only ? static_cast<double>(id) : static_cast<double>(weights[id]);
  auto sorted = seconds;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[sorted.size() / 2];
  if (sorted.size() % 2 == 0) median = (median + sorted[sorted.size() / 2 - 1]) / 2;
  std::cout << "{\"mode\":\"in-memory\",\"threads\":" << threads
            << ",\"batch_size\":" << batch_size << ",\"unique_queries\":" << corpus.size()
            << ",\"queries_per_run\":" << inputs.size() << ",\"runs_seconds\":[";
  for (size_t i = 0; i < seconds.size(); ++i) { if (i) std::cout << ','; std::cout << seconds[i]; }
  std::cout << "],\"median_qps\":" << inputs.size() / median << ",\"checksum\":";
  number(static_cast<float>(checksum));
  std::cout << "}\n";
}

static size_t positive_option(const std::string& name, const std::string& value, size_t maximum) {
  size_t parsed = 0;
  unsigned long long result;
  try { result = std::stoull(value, &parsed); }
  catch (const std::exception&) { throw std::runtime_error(name + " must be a positive integer"); }
  if (parsed != value.size() || result == 0 || result > maximum)
    throw std::runtime_error(name + " must be between 1 and " + std::to_string(maximum));
  return static_cast<size_t>(result);
}

int main(int argc, char** argv) {
  try {
    // Per-character stdio synchronization dominates short-line batch input.
    std::ios::sync_with_stdio(false);
    static_assert(std::endian::native == std::endian::little, "binary output requires a little-endian target");
    bool tokens_only = false;
    bool binary = false;
    size_t threads = 1, batch_size = 256;
    bool batch_size_set = false, pipeline = false, benchmark = false;
    size_t benchmark_repeat = 100, benchmark_runs = 3;
    std::string prompt = model::default_prompt;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--info") {
        std::cout << "{\"model\":"; json_string(model::repo);
        std::cout << ",\"revision\":"; json_string(model::revision);
        std::cout << ",\"kind\":\"" << (model::kind == 1 ? "dense" : "sparse-query")
                  << "\",\"dimension\":" << model::dimension
                  << ",\"source_dimension\":" << model::source_dimension
                  << ",\"weight_dtype\":\"" << (model::quantized ? "int8" : "float32")
                  << "\",\"default_prompt\":";
        json_string(model::default_prompt); std::cout << "}\n";
        return 0;
      }
      if (arg == "--tokens") tokens_only = true;
      else if (arg == "--format" && i+1 < argc) {
        std::string format = argv[++i];
        if (format == "binary") binary = true;
        else if (format == "jsonl") binary = false;
        else throw std::runtime_error("format must be jsonl or binary");
      }
      else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
      else if (arg == "--threads" && i+1 < argc)
        threads = positive_option(arg, argv[++i], 256);
      else if (arg == "--batch-size" && i+1 < argc) {
        batch_size = positive_option(arg, argv[++i], 65536);
        batch_size_set = true;
      }
      else if (arg == "--pipeline") pipeline = true;
      else if (arg == "--benchmark") benchmark = true;
      else if (arg == "--benchmark-repeat" && i+1 < argc)
        benchmark_repeat = positive_option(arg, argv[++i], 1000000);
      else if (arg == "--benchmark-runs" && i+1 < argc)
        benchmark_runs = positive_option(arg, argv[++i], 100);
      else if (arg == "--help") {
        std::cout << "Read one UTF-8 text per line; write one JSON embedding per line.\n"
                     "Options: --info, --tokens, --prompt TEXT, --format jsonl|binary, "
                     "--threads N, --batch-size N, --pipeline,\n"
                     "--benchmark, --benchmark-repeat N, --benchmark-runs N\n";
        return 0;
      } else throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (tokens_only && binary) throw std::runtime_error("--tokens requires JSON output");
    const float* weights = reinterpret_cast<const float*>(icicle_blob);
    std::cin.tie(nullptr);
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (benchmark) {
      if (pipeline) throw std::runtime_error("--pipeline applies to CLI batching, not the in-memory benchmark");
      run_benchmark(threads, batch_size, prompt, weights, tokens_only, benchmark_repeat, benchmark_runs);
      return 0;
    }
    if (threads > 1 || batch_size_set || pipeline) {
      run_batched(threads, batch_size, prompt, weights, tokens_only, binary, pipeline);
      if (!std::cin.eof()) throw std::runtime_error("failed to read stdin");
      return 0;
    }
    auto tokenizer = make_tokenizer();
    icicle::TokenizerScratch scratch;
    std::string line;
    std::vector<float> result;
    if (model::kind == 1 && !tokens_only) result.resize(model::dimension);
    while (true) {
      if (!std::getline(std::cin, line)) break;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("input line exceeds 16 MiB");
      if (!valid_utf8(line)) throw std::runtime_error("input is not valid UTF-8");
      scratch.input.clear();
      scratch.input.append(prompt);
      scratch.input.append(line);
      tokenizer.encode_into(scratch.input, scratch);
      auto& ids = scratch.ids;
      if (tokens_only) {
        std::cout << "{\"tokens\":[";
        for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
        std::cout << "]}\n";
      } else if (model::kind == 1) {
        std::fill(result.begin(), result.end(), 0.0f);
        for (uint32_t id : ids) add_embedding_row(id, result.data());
        finish_embedding(result.data(), ids.size());
        if (binary) {
          std::cout.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size() * sizeof(float)));
          continue;
        }
        std::cout << "{\"embedding\":[";
        for (size_t j = 0; j < result.size(); ++j) { if (j) std::cout << ','; number(result[j]); }
        std::cout << "]}\n";
      } else {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](uint32_t id) { return weights[id] == 0.0f; }), ids.end());
        if (binary) {
          uint32_t count = static_cast<uint32_t>(ids.size());
          std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
          for (uint32_t id : ids) {
            std::cout.write(reinterpret_cast<const char*>(&id), sizeof(id));
            std::cout.write(reinterpret_cast<const char*>(&weights[id]), sizeof(float));
          }
          continue;
        }
        std::cout << "{\"indices\":[";
        for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
        std::cout << "],\"values\":[";
        for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; number(weights[ids[i]]); }
        std::cout << "],\"dimension\":" << model::dimension << "}\n";
      }
    }
    if (!std::cin.eof()) throw std::runtime_error("failed to read stdin");
  } catch (const std::exception& e) {
    std::cerr << "icicle runtime: " << e.what() << '\n';
    return 1;
  }
}
