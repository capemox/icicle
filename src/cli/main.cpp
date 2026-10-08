#include <nlohmann/json.hpp>
#include "tokenizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
static_assert(sizeof(icicle::TrieNode) == 12);

static void run(const std::vector<std::string>& args) {
  std::vector<char*> av;
  for (const auto& s : args) av.push_back(const_cast<char*>(s.c_str()));
  av.push_back(nullptr);
  pid_t p = fork();
  if (p < 0) throw std::runtime_error("fork failed");
  if (p == 0) { execvp(av[0], av.data()); _exit(127); }
  int status;
  if (waitpid(p, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("command failed: " + args[0]);
}
static std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(in), {});
}
static std::vector<char> read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  auto size = in.tellg();
  if (size < 0 || size > static_cast<std::streamoff>(2ull * 1024 * 1024 * 1024)) throw std::runtime_error("file too large");
  std::vector<char> data(static_cast<size_t>(size));
  in.seekg(0); in.read(data.data(), size);
  if (!in && size) throw std::runtime_error("short read");
  return data;
}
static void write_text(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary);
  if (!out || !(out << s)) throw std::runtime_error("cannot write " + p.string());
}
static std::string cpp_string(std::string_view s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    out += '\\';
    out += static_cast<char>('0' + ((c >> 6) & 7));
    out += static_cast<char>('0' + ((c >> 3) & 7));
    out += static_cast<char>('0' + (c & 7));
  }
  return out + '"';
}
static std::string url_path(std::string_view s) {
  std::string out;
  constexpr char hex[] = "0123456789ABCDEF";
  for (unsigned char c : s) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.') out += c;
    else { out += '%'; out += hex[c >> 4]; out += hex[c & 15]; }
  }
  return out;
}
static bool safe_repo(const std::string& s) {
  if (s.empty() || s.front() == '/' || s.find("..") != std::string::npos) return false;
  for (char c : s) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.')) return false;
  return true;
}
static void download(const std::string& url, const fs::path& out, bool optional = false) {
  std::vector<std::string> args{"curl", "--fail", "--location", "--silent"};
  if (!optional) args.push_back("--show-error");
  args.insert(args.end(), {"--retry", "2", "--output", out.string(), url});
  run(args);
}
struct Source {
  std::string name, revision;
  fs::path local, temp, cache;
  fs::path get(const std::string& rel, bool optional = false) const {
    if (rel.empty() || rel.front() == '/' || rel.find("..") != std::string::npos) throw std::runtime_error("unsafe model path");
    if (!local.empty()) {
      fs::path p = local / rel;
      if (optional && !fs::exists(p)) return {};
      if (!fs::is_regular_file(p)) throw std::runtime_error("missing model file: " + p.string());
      return p;
    }
    fs::path p = cache / name / revision / rel;
    fs::create_directories(p.parent_path());
    if (fs::is_regular_file(p)) return p;
    fs::path part = p.string() + ".part-" + std::to_string(getpid());
    try {
      download("https://huggingface.co/" + url_path(name) + "/resolve/" + url_path(revision) + "/" + url_path(rel), part, optional);
      fs::rename(part, p);
    } catch (...) {
      std::error_code ec; fs::remove(part, ec);
      if (optional) return {};
      throw;
    }
    return p;
  }
};
static fs::path cache_root() {
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) return fs::path(xdg) / "icicle";
  if (const char* home = std::getenv("HOME")) return fs::path(home) / ".cache/icicle";
  return fs::temp_directory_path() / "icicle-cache";
}
struct Imported {
  bool sparse = false;
  bool normalize = false;
  bool bpe = false;
  bool bytelevel_prefix_space = false;
  bool bytelevel_regex = true;
  std::string path;
  json tokenizer;
  json tokenizer_config;
  uint32_t vocab_size = 0, max_tokens = 0;
};
static std::string module_file(const Imported& m, std::string_view name) {
  return (m.path == "." ? "" : m.path + "/") + std::string(name);
}
static Imported inspect(const Source& src) {
  json modules = json::parse(read_text(src.get("modules.json")));
  if (!modules.is_array() || modules.empty()) throw std::runtime_error("missing Sentence Transformers modules");
  auto type = modules.at(0).at("type").get<std::string>();
  Imported m;
  if (type.find("StaticEmbedding") != std::string::npos) {
    m.path = modules.at(0).at("path").get<std::string>();
    if (modules.size() == 2 && modules.at(1).at("type").get<std::string>().find("Normalize") != std::string::npos)
      m.normalize = true;
    else if (modules.size() != 1) throw std::runtime_error("unsupported module after StaticEmbedding");
  } else if (type.find("Router") != std::string::npos) {
    if (modules.size() != 1) throw std::runtime_error("unsupported module after Router");
    json router = json::parse(read_text(src.get("router_config.json")));
    auto query = router.at("structure").at("query");
    if (!query.is_array() || query.size() != 1) throw std::runtime_error("query route must contain exactly one module");
    m.path = query.at(0).get<std::string>();
    std::string qtype = router.at("types").at(m.path).get<std::string>();
    if (qtype.find("SparseStaticEmbedding") == std::string::npos) throw std::runtime_error("query route is not SparseStaticEmbedding");
    m.sparse = true;
  } else throw std::runtime_error("unsupported model module: " + type);
  if (m.path.empty() || m.path.find("..") != std::string::npos) throw std::runtime_error("invalid module path");
  m.tokenizer = json::parse(read_text(src.get(module_file(m, "tokenizer.json"))));
  fs::path cfg = src.get(module_file(m, "tokenizer_config.json"), true);
  if (!cfg.empty()) m.tokenizer_config = json::parse(read_text(cfg));
  auto& tok = m.tokenizer;
  const auto& tokenizer_model = tok.at("model");
  // Older tokenizer.json files omit the BPE type but still carry merges.
  std::string tokenizer_type = tokenizer_model.value("type", tokenizer_model.contains("merges") ? "BPE" : "");
  if (tokenizer_type == "WordPiece") {
    if (tokenizer_model.value("continuing_subword_prefix", std::string("##")) != "##")
      throw std::runtime_error("unsupported WordPiece continuation prefix");
    if (tok.at("pre_tokenizer").at("type") != "BertPreTokenizer") throw std::runtime_error("only BertPreTokenizer is supported for WordPiece");
    if (tok.at("normalizer").at("type") != "BertNormalizer") throw std::runtime_error("only BertNormalizer is supported for WordPiece");
  } else if (tokenizer_type == "BPE") {
    m.bpe = true;
    if (!tok.contains("normalizer") || !tok["normalizer"].is_null())
      throw std::runtime_error("BPE currently requires a null normalizer");
    const auto& pre = tok.at("pre_tokenizer");
    if (pre.at("type") != "ByteLevel") throw std::runtime_error("BPE currently requires a ByteLevel pre-tokenizer");
    m.bytelevel_prefix_space = pre.value("add_prefix_space", true);
    m.bytelevel_regex = pre.value("use_regex", true);
    if (!tokenizer_model.contains("merges") || !tokenizer_model["merges"].is_array())
      throw std::runtime_error("BPE merges are missing");
    if (tokenizer_model.contains("dropout") && !tokenizer_model["dropout"].is_null())
      throw std::runtime_error("BPE dropout is not supported");
    auto nonempty_option = [&](const char* name) {
      return tokenizer_model.contains(name) && !tokenizer_model[name].is_null() &&
             !tokenizer_model[name].get<std::string>().empty();
    };
    if (tokenizer_model.value("byte_fallback", false) || tokenizer_model.value("ignore_merges", false) ||
        tokenizer_model.value("fuse_unk", false) ||
        nonempty_option("continuing_subword_prefix") || nonempty_option("end_of_word_suffix"))
      throw std::runtime_error("unsupported BPE model option");
  } else throw std::runtime_error("unsupported tokenizer model: " + tokenizer_type);
  if (tok.contains("truncation") && !tok["truncation"].is_null()) {
    auto tr = tok["truncation"];
    if (tr.at("direction") != "Right" || tr.at("stride") != 0) throw std::runtime_error("unsupported truncation policy");
    m.max_tokens = tr.at("max_length").get<uint32_t>();
  }
  if (m.sparse && m.tokenizer_config.contains("model_max_length") && m.tokenizer_config["model_max_length"].is_number_unsigned()) {
    uint64_t length = m.tokenizer_config["model_max_length"].get<uint64_t>();
    if (length < UINT32_MAX) m.max_tokens = m.max_tokens ? std::min<uint32_t>(m.max_tokens, length) : length;
  }
  m.vocab_size = static_cast<uint32_t>(tok.at("model").at("vocab").size());
  for (const auto& a : tok.value("added_tokens", json::array())) {
    if (a.value("single_word", false) || (!m.bpe && (a.value("lstrip", false) || a.value("rstrip", false))) ||
        (!m.bpe && a.value("normalized", true)))
      throw std::runtime_error("unsupported added-token matching rule");
    uint64_t id = a.at("id").get<uint64_t>();
    if (a.at("content").get<std::string>().empty() || id >= 10000000)
      throw std::runtime_error("invalid added token");
    m.vocab_size = std::max<uint32_t>(m.vocab_size, static_cast<uint32_t>(id + 1));
  }
  if (m.vocab_size == 0 || m.vocab_size > 10000000) throw std::runtime_error("invalid vocabulary size");
  return m;
}
static std::string default_prompt(const Source& src) {
  fs::path path = src.get("config_sentence_transformers.json", true);
  if (path.empty()) return {};
  auto cfg = json::parse(read_text(path));
  if (!cfg.contains("prompts") || !cfg.contains("default_prompt_name") || !cfg["default_prompt_name"].is_string()) return {};
  auto key = cfg["default_prompt_name"].get<std::string>();
  if (!cfg["prompts"].contains(key)) return {};
  return cfg["prompts"][key].get<std::string>();
}
static uint64_t u64le(const char* p) {
  uint64_t x = 0; for (int i = 7; i >= 0; --i) x = (x << 8) | static_cast<unsigned char>(p[i]); return x;
}
struct Weights { std::vector<char> data; uint32_t dimension = 0; };
struct BuildOptions { uint32_t dimension = 0; bool quantize = false; };
static Weights weights(const fs::path& p, const Imported& m) {
  auto bytes = read_bytes(p);
  if (bytes.size() < 9) throw std::runtime_error("truncated safetensors file");
  uint64_t hlen = u64le(bytes.data());
  if (hlen > 16 * 1024 * 1024 || hlen > bytes.size() - 8) throw std::runtime_error("invalid safetensors header size");
  json header = json::parse(std::string(bytes.data()+8, bytes.data()+8+hlen));
  std::vector<std::string> tensor_names;
  for (auto it = header.begin(); it != header.end(); ++it) if (it.key() != "__metadata__") tensor_names.push_back(it.key());
  std::string name = m.sparse ? "weight" : "embedding.weight";
  if (!header.contains(name)) {
    if (tensor_names.size() == 1) name = tensor_names[0];
    else throw std::runtime_error("expected tensor " + name);
  }
  if (tensor_names.size() != 1) throw std::runtime_error("unsupported additional tensors");
  auto t = header.at(name);
  if (t.at("dtype") != "F32") throw std::runtime_error("only F32 safetensors weights are currently supported");
  auto shape = t.at("shape");
  if (m.sparse ? shape.size() != 1 : shape.size() != 2) throw std::runtime_error("unexpected weight tensor rank");
  uint64_t vocab = shape.at(0).get<uint64_t>();
  uint64_t dim = m.sparse ? vocab : shape.at(1).get<uint64_t>();
  if (vocab != m.vocab_size || dim == 0 || dim > UINT32_MAX) throw std::runtime_error("weight dimensions do not match tokenizer");
  auto offsets = t.at("data_offsets");
  uint64_t start = offsets.at(0).get<uint64_t>(), end = offsets.at(1).get<uint64_t>();
  uint64_t size = (m.sparse ? vocab : vocab * dim) * sizeof(float);
  if (end < start || end - start != size || start > bytes.size() - 8 - hlen || end > bytes.size() - 8 - hlen)
    throw std::runtime_error("invalid safetensors tensor offsets");
  return {std::vector<char>(bytes.begin()+8+hlen+start, bytes.begin()+8+hlen+end), static_cast<uint32_t>(dim)};
}
static void align8(std::vector<char>& v) { while (v.size() & 7) v.push_back(0); }
static_assert(sizeof(icicle::BpeMerge) == 16);
static_assert(sizeof(icicle::BpePair) == 8);
template<class T> static void append(std::vector<char>& v, const T& x) {
  const char* p = reinterpret_cast<const char*>(&x); v.insert(v.end(), p, p + sizeof(T));
}
static std::vector<float> compact_weights(Weights& weights, const Imported& m, const BuildOptions& options) {
  if (m.sparse && (options.dimension || options.quantize))
    throw std::runtime_error("--dimension and --quantize apply to dense models only");
  if (options.dimension > weights.dimension) throw std::runtime_error("requested dimension exceeds source dimension");
  if (!options.dimension && !options.quantize) return {};
  uint32_t source_dimension = weights.dimension;
  uint32_t dimension = options.dimension ? options.dimension : source_dimension;
  auto source = std::move(weights.data);
  std::vector<float> scales;
  if (options.quantize) scales.reserve(m.vocab_size);
  weights.data.resize(static_cast<size_t>(m.vocab_size) * dimension * (options.quantize ? 1 : sizeof(float)));
  for (uint32_t id = 0; id < m.vocab_size; ++id) {
    const char* row = source.data() + static_cast<size_t>(id) * source_dimension * sizeof(float);
    if (!options.quantize) {
      std::memcpy(weights.data.data() + static_cast<size_t>(id) * dimension * sizeof(float), row, dimension * sizeof(float));
      continue;
    }
    float maximum = 0;
    for (uint32_t j = 0; j < dimension; ++j) {
      float value; std::memcpy(&value, row + j * sizeof(float), sizeof(value));
      if (!std::isfinite(value)) throw std::runtime_error("int8 quantization requires finite weights");
      maximum = std::max(maximum, std::abs(value));
    }
    float scale = maximum ? maximum / 127.0f : 1.0f;
    if (scale == 0) scale = maximum; // Avoid underflow for subnormal rows.
    scales.push_back(scale);
    for (uint32_t j = 0; j < dimension; ++j) {
      float value; std::memcpy(&value, row + j * sizeof(float), sizeof(value));
      int8_t q = static_cast<int8_t>(std::lround(std::clamp(value / scale, -127.0f, 127.0f)));
      std::memcpy(weights.data.data() + static_cast<size_t>(id) * dimension + j, &q, sizeof(q));
    }
  }
  weights.dimension = dimension;
  return scales;
}
static fs::path resources(const char* argv0) {
  fs::path executable;
#ifdef __APPLE__
  uint32_t length = 4096;
  std::string buffer(length, '\0');
  if (_NSGetExecutablePath(buffer.data(), &length) != 0) {
    buffer.resize(length);
    if (_NSGetExecutablePath(buffer.data(), &length) != 0) buffer.clear();
  }
  if (!buffer.empty()) executable = fs::weakly_canonical(buffer.c_str());
#else
  std::error_code ec;
  executable = fs::read_symlink("/proc/self/exe", ec);
#endif
  if (executable.empty() && fs::exists(argv0)) executable = fs::weakly_canonical(argv0);
  if (!executable.empty()) {
    fs::path installed = executable.parent_path().parent_path() / "share/icicle/runtime";
    if (fs::exists(installed / "main.cpp")) return installed;
  }
  fs::path configured = ICICLE_RESOURCE_DIR;
  if (fs::exists(configured / "main.cpp")) return configured;
  throw std::runtime_error("runtime source files not found");
}
static void build(const Source& src, const Imported& m, const fs::path& output, const std::string& cxx,
                  const fs::path& temp, const char* argv0, const BuildOptions& options) {
  auto w = weights(src.get(module_file(m, "model.safetensors")), m);
  uint32_t source_dimension = w.dimension;
  auto scales = compact_weights(w, m, options);
  const auto& tok = m.tokenizer;
  const auto& model = tok.at("model");
  auto vocab = model.at("vocab");
  for (const auto& a : tok.value("added_tokens", json::array())) {
    std::string content = a.at("content").get<std::string>();
    uint32_t id = a.at("id").get<uint32_t>();
    if (vocab.contains(content) && vocab.at(content).get<uint32_t>() != id)
      throw std::runtime_error("added token conflicts with model vocabulary");
    vocab[content] = id;
  }
  std::vector<std::string> by_id(m.vocab_size);
  std::vector<bool> seen(m.vocab_size);
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    uint32_t id = it.value().get<uint32_t>();
    if (id >= m.vocab_size || seen[id]) throw std::runtime_error("noncontiguous tokenizer vocabulary");
    by_id[id] = it.key(); seen[id] = true;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) throw std::runtime_error("missing tokenizer vocabulary ID");
  uint32_t unknown_id = UINT32_MAX;
  if (model.contains("unk_token") && !model["unk_token"].is_null()) {
    std::string unk = model.at("unk_token").get<std::string>();
    if (!vocab.contains(unk)) throw std::runtime_error("unknown token missing from vocabulary");
    unknown_id = vocab.at(unk).get<uint32_t>();
  } else if (!m.bpe) throw std::runtime_error("unknown token missing from vocabulary");
  struct BuildNode {
    std::map<uint8_t, uint32_t> children;
    uint32_t token_id = UINT32_MAX;
  };
  std::vector<BuildNode> trie(1);
  for (uint32_t id = 0; id < m.vocab_size; ++id) {
    uint32_t node = 0;
    for (unsigned char byte : by_id[id]) {
      auto it = trie[node].children.find(byte);
      if (it == trie[node].children.end()) {
        if (trie.size() == UINT32_MAX) throw std::runtime_error("token trie exceeds 32-bit offsets");
        uint32_t next = static_cast<uint32_t>(trie.size());
        trie.emplace_back();
        trie[node].children.emplace(byte, next);
        node = next;
      } else node = it->second;
    }
    trie[node].token_id = id;
  }
  // A double-array trie uses base[parent] + byte as the child's slot. The
  // parent field distinguishes an occupied slot from an unrelated transition.
  // This gives each byte a direct offset lookup with no hash or edge search.
  std::vector<icicle::TrieNode> nodes(1, {0, UINT32_MAX, UINT32_MAX});
  nodes[0] = {0, 0, trie[0].token_id};
  std::vector<std::pair<uint32_t, uint32_t>> pending{{0, 0}};
  uint32_t next_base = 1;
  for (size_t head = 0; head < pending.size(); ++head) {
    auto [source, target] = pending[head];
    const auto& children = trie[source].children;
    if (children.empty()) continue;
    uint32_t base = next_base;
    for (;;) {
      if (base > UINT32_MAX - children.rbegin()->first) throw std::runtime_error("token trie exceeds 32-bit offsets");
      bool free = true;
      for (const auto& [byte, unused] : children) {
        uint32_t slot = base + byte;
        if (slot < nodes.size() && nodes[slot].parent != UINT32_MAX) { free = false; break; }
      }
      if (free) break;
      ++base;
    }
    next_base = base + 1;
    nodes[target].base = base;
    uint32_t last = base + children.rbegin()->first;
    if (nodes.size() <= last) nodes.resize(static_cast<size_t>(last) + 1, {0, UINT32_MAX, UINT32_MAX});
    for (const auto& [byte, child] : children) {
      uint32_t slot = base + byte;
      nodes[slot] = {0, target, trie[child].token_id};
      pending.emplace_back(child, slot);
    }
  }
  std::vector<char> blob = std::move(w.data);
  align8(blob);
  uint64_t scales_offset = blob.size();
  for (float scale : scales) append(blob, scale);
  align8(blob);
  uint64_t nodes_offset = blob.size();
  for (const auto& node : nodes) append(blob, node);
  align8(blob);
  uint64_t merges_offset = blob.size();
  std::vector<icicle::BpeMerge> merges;
  if (m.bpe) {
    const auto& source_merges = model.at("merges");
    if (source_merges.size() > UINT32_MAX) throw std::runtime_error("too many BPE merges");
    for (size_t rank = 0; rank < source_merges.size(); ++rank) {
      std::string left, right;
      const auto& entry = source_merges.at(rank);
      if (entry.is_string()) {
        std::string value = entry.get<std::string>();
        size_t split = value.find(' ');
        if (split == std::string::npos || split == 0 || split + 1 == value.size())
          throw std::runtime_error("invalid BPE merge");
        left = value.substr(0, split); right = value.substr(split + 1);
      } else if (entry.is_array() && entry.size() == 2) {
        left = entry.at(0).get<std::string>(); right = entry.at(1).get<std::string>();
      } else throw std::runtime_error("invalid BPE merge");
      if (!vocab.contains(left) || !vocab.contains(right) || !vocab.contains(left + right))
        throw std::runtime_error("BPE merge token missing from vocabulary");
      uint64_t key = (static_cast<uint64_t>(vocab.at(left).get<uint32_t>()) << 32) |
                      vocab.at(right).get<uint32_t>();
      merges.push_back({key, static_cast<uint32_t>(rank), vocab.at(left + right).get<uint32_t>()});
    }
    std::sort(merges.begin(), merges.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    for (size_t i = 1; i < merges.size(); ++i)
      if (merges[i - 1].key == merges[i].key) throw std::runtime_error("duplicate BPE merge");
  }
  for (const auto& merge : merges) append(blob, merge);
  uint64_t merge_index_offset = blob.size();
  if (m.bpe) {
    size_t cursor = 0;
    for (uint32_t left = 0; left <= m.vocab_size; ++left) {
      while (cursor < merges.size() && (merges[cursor].key >> 32) < left) ++cursor;
      append(blob, static_cast<uint32_t>(cursor));
    }
  }
  align8(blob);
  uint64_t byte_pairs_offset = blob.size();
  if (m.bpe) {
    std::array<uint32_t, 256> byte_ids;
    const auto& alphabet = icicle::bytelevel_alphabet();
    for (size_t b = 0; b < byte_ids.size(); ++b)
      byte_ids[b] = vocab.contains(alphabet[b]) ? vocab.at(alphabet[b]).get<uint32_t>() : unknown_id;
    // Precompute the first generation of merges. Later merges still use the
    // per-left offset index. This 512 KiB table is shared read-only by workers.
    for (uint32_t left : byte_ids)
      for (uint32_t right : byte_ids) {
        icicle::BpePair pair{UINT32_MAX, UINT32_MAX};
        if (left != UINT32_MAX && right != UINT32_MAX) {
          uint64_t key = (static_cast<uint64_t>(left) << 32) | right;
          auto rule = std::lower_bound(merges.begin(), merges.end(), key,
            [](const auto& rule, uint64_t value) { return rule.key < value; });
          if (rule != merges.end() && rule->key == key) pair = {rule->rank, rule->token_id};
        }
        append(blob, pair);
      }
  }
  fs::path blobfile = temp / "model.bin";
  std::ofstream bin(blobfile, std::ios::binary); bin.write(blob.data(), blob.size()); bin.close();
  if (!bin) throw std::runtime_error("cannot write model blob");
  json norm = m.bpe ? json::object() : tok.at("normalizer");
  bool lower = norm.value("lowercase", false);
  bool strip = norm.contains("strip_accents") && !norm["strip_accents"].is_null() ? norm["strip_accents"].get<bool>() : lower;
  std::string prompt = default_prompt(src);
  std::string header = "#pragma once\n#include <cstdint>\n#include <string_view>\n#include <utility>\n#include <vector>\nnamespace model {\n";
  auto constant = [&](const std::string& key, auto value) { header += "inline constexpr auto " + key + " = " + std::to_string(value) + ";\n"; };
  constant("kind", m.sparse ? 2 : 1); constant("dimension", w.dimension);
  constant("source_dimension", source_dimension);
  constant("quantized", options.quantize ? 1 : 0);
  constant("scales_offset", scales_offset);
  constant("bpe", m.bpe ? 1 : 0);
  constant("bytelevel_prefix_space", m.bytelevel_prefix_space ? 1 : 0);
  constant("bytelevel_regex", m.bytelevel_regex ? 1 : 0);
  constant("merges_offset", merges_offset); constant("merges_count", merges.size());
  constant("merge_index_offset", merge_index_offset);
  constant("merge_index_count", m.bpe ? static_cast<uint64_t>(m.vocab_size) + 1 : 0);
  constant("byte_pairs_offset", byte_pairs_offset);
  constant("normalize", m.normalize ? 1 : 0);
  constant("nodes_offset", nodes_offset); constant("nodes_count", nodes.size());
  constant("unknown_id", unknown_id);
  constant("max_chars", model.value("max_input_chars_per_word", 100)); constant("max_tokens", m.max_tokens);
  constant("clean_text", norm.value("clean_text", true) ? 1 : 0);
  constant("chinese", norm.value("handle_chinese_chars", true) ? 1 : 0);
  constant("lowercase", lower ? 1 : 0); constant("strip_accents", strip ? 1 : 0);
  header += "inline constexpr const char* repo = " + cpp_string(src.name) + ";\n";
  header += "inline constexpr const char* revision = " + cpp_string(src.revision) + ";\n";
  header += "inline constexpr const char* default_prompt = " + cpp_string(prompt) + ";\n";
  header += "struct AddedToken { std::string_view text; uint32_t id; bool lstrip, rstrip; };\n";
  header += "inline const std::vector<AddedToken> added_tokens = {";
  for (const auto& a : tok.value("added_tokens", json::array()))
    header += "{" + cpp_string(a.at("content").get<std::string>()) + "," + std::to_string(a.at("id").get<uint32_t>()) +
              "," + (a.value("lstrip", false) ? "true" : "false") + "," +
              (a.value("rstrip", false) ? "true" : "false") + "},";
  header += "};\n}\n";
  write_text(temp / "generated_model.hpp", header);
  std::string symbol;
#ifdef __APPLE__
  symbol = "_icicle_blob";
  std::string assembly = ".section __TEXT,__const\n.p2align 4\n.globl " + symbol + "\n" + symbol + ":\n.incbin \"" + blobfile.string() + "\"\n";
#else
  symbol = "icicle_blob";
  std::string assembly = ".section .rodata\n.p2align 4\n.globl " + symbol + "\n" + symbol + ":\n.incbin \"" + blobfile.string() + "\"\n.section .note.GNU-stack,\"\",@progbits\n";
#endif
  write_text(temp / "model.S", assembly);
  fs::path resource = resources(argv0);
  std::vector<std::string> cmd{cxx, "-std=c++20", "-O3", "-DNDEBUG", "-pthread", "-I"+resource.string(),
    "-I"+temp.string(), "-I"+std::string(ICICLE_ICU_INCLUDE),
    (resource / "main.cpp").string(), (resource / "tokenizer.cpp").string(), (temp / "model.S").string(),
    "-o", output.string(), ICICLE_ICU_LIBRARY, ICICLE_ICU_I18N_LIBRARY};
  run(cmd);
  std::cout << "built " << output << " (" << (m.sparse ? "sparse query" : "dense") << ", " << w.dimension << " dimensions)\n";
}

int main(int argc, char** argv) {
  try {
    if (argc == 1 || std::string_view(argv[1]) == "--help") {
      std::cout << "icicle inspect MODEL [--revision REV] [--local DIR]\n"
                   "icicle build MODEL --output FILE [--revision REV] [--local DIR] [--route query] [--cxx COMPILER]\n"
                   "             [--dimension N] [--quantize int8]\n";
      return argc == 1 ? 1 : 0;
    }
    if (argc < 3) throw std::runtime_error("usage: icicle inspect|build MODEL [--revision SHA] [--local DIR] [--output PATH] [--cxx COMPILER]");
    std::string action = argv[1], model = argv[2], rev = "main", cxx = "c++";
    fs::path local, output;
    BuildOptions options;
    for (int i = 3; i < argc; ++i) {
      std::string opt = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + opt);
      std::string val = argv[++i];
      if (opt == "--revision") rev = val;
      else if (opt == "--local") local = val;
      else if (opt == "--output") output = val;
      else if (opt == "--cxx") cxx = val;
      else if (opt == "--quantize") {
        if (val != "int8") throw std::runtime_error("--quantize supports int8 only");
        options.quantize = true;
      }
      else if (opt == "--dimension") {
        size_t parsed = 0;
        auto dimension = std::stoull(val, &parsed);
        if (parsed != val.size() || dimension == 0 || dimension > UINT32_MAX)
          throw std::runtime_error("--dimension must be a positive 32-bit integer");
        options.dimension = static_cast<uint32_t>(dimension);
      }
      else if (opt == "--route") { if (val != "query") throw std::runtime_error("only --route query is supported"); }
      else throw std::runtime_error("unknown option " + opt);
    }
    if (action != "inspect" && action != "build") throw std::runtime_error("unknown command " + action);
    if (action == "inspect" && (options.dimension || options.quantize))
      throw std::runtime_error("--dimension and --quantize are build options");
    if (action == "build" && output.empty()) throw std::runtime_error("--output is required");
    fs::path temp = fs::temp_directory_path() / ("icicle-" + std::to_string(getpid()));
    fs::create_directories(temp);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{temp};
    Source src{model, rev, local, temp, cache_root()};
    if (!local.empty()) {
      if (!fs::is_directory(local)) throw std::runtime_error("local snapshot directory not found");
      src.name = local.string(); src.revision = "local";
    } else {
      if (!safe_repo(model) || !safe_repo(rev)) throw std::runtime_error("invalid Hub model ID or revision");
      fs::path api = temp / "model-api.json";
      download("https://huggingface.co/api/models/" + url_path(model) + "/revision/" + url_path(rev), api);
      src.revision = json::parse(read_text(api)).at("sha").get<std::string>();
    }
    Imported m = inspect(src);
    if (action == "inspect") {
      std::cout << "model: " << src.name << "\nrevision: " << src.revision
                << "\nkind: " << (m.sparse ? "sparse query" : "dense")
                << "\ntokenizer: " << (m.bpe ? "ByteLevel BPE" : "BERT WordPiece") << "\nvocabulary: " << m.vocab_size
                << "\nmax tokens: " << (m.max_tokens ? std::to_string(m.max_tokens) : "unlimited")
                << "\nnormalize: " << (m.normalize ? "yes" : "no")
                << "\ndefault prompt: " << std::quoted(default_prompt(src)) << '\n';
    } else build(src, m, fs::absolute(output), cxx, temp, argv[0], options);
  } catch (const std::exception& e) {
    std::cerr << "icicle: " << e.what() << '\n'; return 1;
  }
}
