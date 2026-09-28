#include "tokenizer.hpp"
#include "generated_model.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
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
    model::bytelevel_prefix_space, model::bytelevel_regex,
    model::unknown_id, model::max_chars, model::max_tokens,
    model::clean_text, model::chinese, model::lowercase, model::strip_accents, std::move(added));
}

struct Batch {
  std::vector<std::string> lines;
  std::vector<std::vector<uint32_t>> ids;
  std::vector<float> dense;
};

class BatchWorkers {
public:
  BatchWorkers(size_t count, const std::string& prompt, const float* weights, bool tokens_only)
    : count_(count), prompt_(prompt), weights_(weights), tokens_only_(tokens_only),
      tokenizer_(make_tokenizer()) {
    errors_.resize(count);
    scratch_.resize(count);
    try {
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

  void run(Batch& batch) {
    {
      std::lock_guard lock(mutex_);
      active_ = &batch;
      std::fill(errors_.begin(), errors_.end(), std::exception_ptr{});
      remaining_ = count_;
      ++generation_;
    }
    start_.notify_all();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
    for (const auto& error : errors_) if (error) std::rethrow_exception(error);
  }

private:
  void encode_one(Batch& batch, size_t index, icicle::TokenizerScratch& scratch) const {
    scratch.input.clear();
    scratch.input.append(prompt_);
    scratch.input.append(batch.lines[index]);
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
      for (uint32_t id : ids) {
        const float* row = weights_ + static_cast<size_t>(id) * model::dimension;
        for (uint32_t j = 0; j < model::dimension; ++j) result[j] += row[j];
      }
      if (!ids.empty()) {
        float divisor = static_cast<float>(ids.size());
        for (uint32_t j = 0; j < model::dimension; ++j) result[j] /= divisor;
      }
      if (model::normalize) {
        float squared = 0.0f;
        for (uint32_t j = 0; j < model::dimension; ++j) squared += result[j] * result[j];
        float length = std::max(std::sqrt(squared), 1e-12f);
        for (uint32_t j = 0; j < model::dimension; ++j) result[j] /= length;
      }
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
        for (size_t i = worker_index; i < batch->lines.size(); i += count_)
          encode_one(*batch, i, scratch_[worker_index]);
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
  std::vector<icicle::TokenizerScratch> scratch_;
  std::vector<std::thread> workers_;
  std::vector<std::exception_ptr> errors_;
  std::mutex mutex_;
  std::condition_variable start_, done_;
};

static void write_batch(const Batch& batch, const float* weights, bool tokens_only, bool binary) {
  if (model::kind == 1 && !tokens_only && binary) {
    std::cout.write(reinterpret_cast<const char*>(batch.dense.data()),
                    static_cast<std::streamsize>(batch.dense.size() * sizeof(float)));
    return;
  }
  for (size_t i = 0; i < batch.lines.size(); ++i) {
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
    } else if (binary) {
      uint32_t count = static_cast<uint32_t>(batch.ids[i].size());
      std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
      for (uint32_t id : batch.ids[i]) {
        std::cout.write(reinterpret_cast<const char*>(&id), sizeof(id));
        std::cout.write(reinterpret_cast<const char*>(&weights[id]), sizeof(float));
      }
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

static void run_batched(size_t threads, size_t batch_size, const std::string& prompt,
                        const float* weights, bool tokens_only, bool binary) {
  BatchWorkers workers(threads, prompt, weights, tokens_only);
  Batch batch;
  std::optional<std::string> pending;
  constexpr size_t max_batch_bytes = 64 * 1024 * 1024;
  for (;;) {
    batch.lines.resize(batch_size);
    size_t count = 0;
    size_t bytes = 0;
    while (count < batch_size) {
      std::string& line = batch.lines[count];
      if (pending) { line = std::move(*pending); pending.reset(); }
      else if (!std::getline(std::cin, line)) break;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("input line exceeds 16 MiB");
      if (!valid_utf8(line)) throw std::runtime_error("input is not valid UTF-8");
      if (bytes + line.size() > max_batch_bytes && count != 0) {
        pending = std::move(line);
        break;
      }
      bytes += line.size();
      ++count;
    }
    batch.lines.resize(count);
    if (batch.lines.empty()) break;
    if (model::kind != 1 || tokens_only)
      batch.ids.resize(batch.lines.size());
    if (model::kind == 1 && !tokens_only)
      batch.dense.resize(batch.lines.size() * model::dimension);
    workers.run(batch);
    write_batch(batch, weights, tokens_only, binary);
  }
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
    bool batch_size_set = false;
    std::string prompt = model::default_prompt;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--info") {
        std::cout << "{\"model\":"; json_string(model::repo);
        std::cout << ",\"revision\":"; json_string(model::revision);
        std::cout << ",\"kind\":\"" << (model::kind == 1 ? "dense" : "sparse-query")
                  << "\",\"dimension\":" << model::dimension << ",\"default_prompt\":";
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
        batch_size = positive_option(arg, argv[++i], 4096);
        batch_size_set = true;
      }
      else if (arg == "--help") {
        std::cout << "Read one UTF-8 text per line; write one JSON embedding per line.\n"
                     "Options: --info, --tokens, --prompt TEXT, --format jsonl|binary, "
                     "--threads N, --batch-size N\n";
        return 0;
      } else throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (tokens_only && binary) throw std::runtime_error("--tokens requires JSON output");
    const float* weights = reinterpret_cast<const float*>(icicle_blob);
    std::cin.tie(nullptr);
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (threads > 1 || batch_size_set) {
      run_batched(threads, batch_size, prompt, weights, tokens_only, binary);
      if (!std::cin.eof()) throw std::runtime_error("failed to read stdin");
      return 0;
    }
    auto tokenizer = make_tokenizer();
    icicle::TokenizerScratch scratch;
    std::string line;
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
        std::vector<float> result(model::dimension, 0.0f);
        for (uint32_t id : ids) {
          const float* row = weights + static_cast<size_t>(id) * model::dimension;
          for (uint32_t j = 0; j < model::dimension; ++j) result[j] += row[j];
        }
        if (!ids.empty()) {
          float divisor = static_cast<float>(ids.size());
          for (auto& v : result) v /= divisor;
        }
        if (model::normalize) {
          float squared = 0.0f;
          for (float v : result) squared += v * v;
          float length = std::max(std::sqrt(squared), 1e-12f);
          for (float& v : result) v /= length;
        }
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
