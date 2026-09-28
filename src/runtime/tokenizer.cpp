#include "tokenizer.hpp"
#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/locid.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unicode/utf8.h>

namespace icicle {
static const std::array<std::string, 256>& bytelevel_alphabet();
Tokenizer::Tokenizer(const unsigned char* blob, uint64_t nodes_offset, uint32_t nodes_count,
                     bool bpe, uint64_t merges_offset, uint32_t merges_count,
                     uint64_t merge_index_offset, uint32_t merge_index_count,
                     bool bytelevel_prefix_space, bool bytelevel_regex,
                     uint32_t unknown_id, uint32_t max_chars,
                     uint32_t max_tokens, bool clean_text, bool chinese, bool lowercase,
                     bool strip_accents, std::vector<AddedToken> added)
  : nodes_(reinterpret_cast<const TrieNode*>(blob + nodes_offset)),
    merges_(reinterpret_cast<const BpeMerge*>(blob + merges_offset)),
    merge_index_(reinterpret_cast<const uint32_t*>(blob + merge_index_offset)),
    nodes_count_(nodes_count), merges_count_(merges_count), merge_index_count_(merge_index_count),
    bpe_(bpe), bytelevel_prefix_space_(bytelevel_prefix_space), bytelevel_regex_(bytelevel_regex),
    unknown_id_(unknown_id), max_chars_(max_chars),
    max_tokens_(max_tokens), clean_text_(clean_text), chinese_(chinese),
    lowercase_(lowercase), strip_accents_(strip_accents), added_(std::move(added)) {
  if (bpe_) {
    const auto& alphabet = bytelevel_alphabet();
    for (size_t byte = 0; byte < byte_ids_.size(); ++byte)
      byte_ids_[byte] = token_id(alphabet[byte]);
  }
}

uint32_t Tokenizer::child(uint32_t node, uint8_t byte) const {
  if (node == UINT32_MAX) return UINT32_MAX;
  uint32_t base = nodes_[node].base;
  if (!base) return UINT32_MAX;
  uint32_t slot = base + byte;
  return slot < nodes_count_ && nodes_[slot].parent == node ? slot : UINT32_MAX;
}

uint32_t Tokenizer::token_id(std::string_view value) const {
  uint32_t node = 0;
  for (unsigned char byte : value) {
    node = child(node, byte);
    if (node == UINT32_MAX) return UINT32_MAX;
  }
  return nodes_[node].token_id;
}

const BpeMerge* Tokenizer::merge(uint32_t left, uint32_t right) const {
  if (merge_index_count_ < 2 || left >= merge_index_count_ - 1) return nullptr;
  const BpeMerge* first = merges_ + merge_index_[left];
  const BpeMerge* last = merges_ + merge_index_[left + 1];
  uint64_t key = (static_cast<uint64_t>(left) << 32) | right;
  const BpeMerge* end = last;
  while (first < last) {
    const BpeMerge* middle = first + (last - first) / 2;
    if (middle->key < key) first = middle + 1;
    else last = middle;
  }
  return first < end && first->key == key ? first : nullptr;
}

static const std::array<std::string, 256>& bytelevel_alphabet() {
  static const auto alphabet = [] {
    std::array<std::string, 256> result;
    std::array<bool, 256> direct{};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      UChar32 cp = direct[b] ? b : 256 + extra++;
      icu::UnicodeString one; one.append(cp);
      one.toUTF8String(result[b]);
    }
    return result;
  }();
  return alphabet;
}

static const icu::RegexPattern& bytelevel_pattern() {
  static const std::unique_ptr<icu::RegexPattern> pattern = [] {
    UErrorCode status = U_ZERO_ERROR;
    auto result = std::unique_ptr<icu::RegexPattern>(icu::RegexPattern::compile(
      icu::UnicodeString::fromUTF8("'(?:s|t|re|ve|m|ll|d)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    if (U_FAILURE(status)) throw std::runtime_error("cannot compile ByteLevel pattern");
    return result;
  }();
  return *pattern;
}

static bool ascii_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
static bool ascii_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static bool ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }

// Match one alternative of the GPT-2 ByteLevel pre-tokenizer regex. The
// Unicode path below still uses ICU for non-ASCII text.
static size_t ascii_bytelevel_piece_end(std::string_view input, size_t start) {
  const size_t n = input.size();
  unsigned char c = static_cast<unsigned char>(input[start]);
  if (c == '\'') {
    constexpr std::string_view contractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (auto suffix : contractions)
      if (input.substr(start + 1).starts_with(suffix)) return start + 1 + suffix.size();
  }
  auto match_class = [&](auto predicate) -> size_t {
    size_t pos = start;
    if (c == ' ' && pos + 1 < n && predicate(static_cast<unsigned char>(input[pos + 1]))) ++pos;
    if (!predicate(static_cast<unsigned char>(input[pos]))) return start;
    do { ++pos; } while (pos < n && predicate(static_cast<unsigned char>(input[pos])));
    return pos;
  };
  if (size_t end = match_class(ascii_letter); end != start) return end;
  if (size_t end = match_class(ascii_digit); end != start) return end;
  if (size_t end = match_class([](unsigned char value) {
        return !ascii_space(value) && !ascii_letter(value) && !ascii_digit(value);
      }); end != start) return end;
  size_t end = start;
  while (end < n && ascii_space(static_cast<unsigned char>(input[end]))) ++end;
  // The first whitespace regex alternative leaves the final whitespace
  // character for a following token unless the run ends the input.
  return end < n && end - start > 1 ? end - 1 : end;
}

void Tokenizer::encode_bpe_segment(std::string_view text, std::vector<uint32_t>& out,
                                   std::vector<BpeSymbol>& symbols, std::string& bytes,
                                   std::string& prefix_input) const {
  auto encode_piece = [&](std::string_view piece) {
    symbols.clear();
    symbols.reserve(piece.size());
    for (unsigned char byte : piece) {
      uint32_t id = byte_ids_[byte];
      if (id == UINT32_MAX) id = unknown_id_;
      if (id != UINT32_MAX) symbols.push_back({id, UINT32_MAX, UINT32_MAX});
    }
    auto update_pair = [&](size_t i) {
      if (i + 1 == symbols.size()) {
        symbols[i].pair_rank = UINT32_MAX;
        return;
      }
      const BpeMerge* rule = merge(symbols[i].id, symbols[i + 1].id);
      symbols[i].pair_rank = rule ? rule->rank : UINT32_MAX;
      symbols[i].merged_id = rule ? rule->token_id : UINT32_MAX;
    };
    for (size_t i = 0; i + 1 < symbols.size(); ++i) update_pair(i);
    for (;;) {
      uint32_t best_rank = UINT32_MAX;
      size_t best_index = 0;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        if (symbols[i].pair_rank < best_rank) {
          best_rank = symbols[i].pair_rank;
          best_index = i;
        }
      }
      if (best_rank == UINT32_MAX) break;
      symbols[best_index].id = symbols[best_index].merged_id;
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
      if (best_index) update_pair(best_index - 1);
      update_pair(best_index);
    }
    for (const auto& symbol : symbols) out.push_back(symbol.id);
    if (max_tokens_ && out.size() > max_tokens_) out.resize(max_tokens_);
  };
  if (ascii_prefix_length(text) == text.size()) {
    std::string_view input = text;
    if (bytelevel_prefix_space_ && !input.empty() && input.front() != ' ') {
      prefix_input.clear();
      prefix_input.push_back(' ');
      prefix_input.append(input);
      input = prefix_input;
    }
    if (!bytelevel_regex_) { encode_piece(input); return; }
    for (size_t pos = 0; pos < input.size();) {
      size_t end = ascii_bytelevel_piece_end(input, pos);
      encode_piece(input.substr(pos, end - pos));
      pos = end;
      if (max_tokens_ && out.size() >= max_tokens_) break;
    }
    return;
  }
  icu::UnicodeString input = icu::UnicodeString::fromUTF8(
    icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  if (bytelevel_prefix_space_ && !input.isEmpty() && input.charAt(0) != ' ')
    input.insert(0, static_cast<UChar32>(' '));
  auto encode_unicode_piece = [&](const icu::UnicodeString& piece) {
    bytes.clear();
    piece.toUTF8String(bytes);
    encode_piece(bytes);
  };
  if (!bytelevel_regex_) { encode_unicode_piece(input); return; }
  UErrorCode status = U_ZERO_ERROR;
  std::unique_ptr<icu::RegexMatcher> matcher(bytelevel_pattern().matcher(input, status));
  if (U_FAILURE(status)) throw std::runtime_error("cannot run ByteLevel pattern");
  while (matcher->find(status)) {
    encode_unicode_piece(input.tempSubStringBetween(matcher->start(status), matcher->end(status)));
    if (max_tokens_ && out.size() >= max_tokens_) break;
  }
  if (U_FAILURE(status)) throw std::runtime_error("ByteLevel pattern failed");
}

static bool space(UChar32 c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || u_charType(c) == U_SPACE_SEPARATOR; }
static bool chinese(UChar32 c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
         (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) ||
         (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}
static bool punct(UChar32 c) {
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
      (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
  auto t = u_charType(c);
  return t == U_DASH_PUNCTUATION || t == U_START_PUNCTUATION ||
         t == U_END_PUNCTUATION || t == U_CONNECTOR_PUNCTUATION ||
         t == U_OTHER_PUNCTUATION || t == U_INITIAL_PUNCTUATION ||
         t == U_FINAL_PUNCTUATION;
}
static std::string utf8(const icu::UnicodeString& s) { std::string out; s.toUTF8String(out); return out; }

void Tokenizer::encode_ascii_segment(std::string_view text, std::vector<uint32_t>& out,
                                      std::string& cleaned) const {
  cleaned.clear();
  cleaned.reserve(text.size());
  for (unsigned char c : text) {
    if (clean_text_) {
      if (c == 0 || c == 0x7f) continue;
      if (c == '\t' || c == '\n' || c == '\r') c = ' ';
      else if (c < 0x20) continue;
    }
    if (lowercase_ && c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + ('a' - 'A'));
    cleaned.push_back(static_cast<char>(c));
  }
  auto emit = [&](std::string_view word) {
    if (word.size() > max_chars_) {
      out.push_back(unknown_id_);
    } else {
      size_t saved = out.size();
      bool failed = false;
      for (size_t start = 0; start < word.size();) {
        uint32_t node = start ? child(child(0, '#'), '#') : 0;
        size_t best_end = start;
        uint32_t found_id = UINT32_MAX;
        for (size_t end = start; end < word.size() && node != UINT32_MAX; ++end) {
          node = child(node, static_cast<uint8_t>(word[end]));
          if (node != UINT32_MAX && nodes_[node].token_id != UINT32_MAX) {
            found_id = nodes_[node].token_id;
            best_end = end + 1;
          }
        }
        if (found_id == UINT32_MAX) { failed = true; break; }
        out.push_back(found_id);
        start = best_end;
      }
      if (failed) { out.resize(saved); out.push_back(unknown_id_); }
    }
    if (max_tokens_ && out.size() >= max_tokens_) { out.resize(max_tokens_); return true; }
    return false;
  };
  size_t start = 0;
  for (size_t i = 0; i < cleaned.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(cleaned[i]);
    bool is_space = c == ' ' || c == '\t' || c == '\n' || c == '\r';
    bool is_punct = (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
                    (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    if (!is_space && !is_punct) continue;
    if (i > start && emit(std::string_view(cleaned.data() + start, i - start))) return;
    if (is_punct && emit(std::string_view(cleaned.data() + i, 1))) return;
    start = i + 1;
  }
  if (start < cleaned.size()) emit(std::string_view(cleaned.data() + start, cleaned.size() - start));
}

void Tokenizer::encode_segment(std::string_view text, std::vector<uint32_t>& out,
                               std::string& ascii) const {
  bool is_ascii = ascii_prefix_length(text) == text.size();
  if (is_ascii) {
    encode_ascii_segment(text, out, ascii);
    return;
  }
  icu::UnicodeString input = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  icu::UnicodeString cleaned;
  for (int32_t i = 0; i < input.length();) {
    UChar32 c = input.char32At(i); i += U16_LENGTH(c);
    if (clean_text_) {
      if (c == 0 || c == 0xFFFD) continue;
      if (space(c)) { cleaned.append(static_cast<UChar32>(' ')); continue; }
      auto t = u_charType(c);
      if (t == U_CONTROL_CHAR || t == U_FORMAT_CHAR) continue;
    }
    if (chinese_ && chinese(c)) cleaned.append(static_cast<UChar32>(' '));
    cleaned.append(c);
    if (chinese_ && chinese(c)) cleaned.append(static_cast<UChar32>(' '));
  }
  if (lowercase_) {
    // Tokenizers lowercases each code point independently. ICU's whole-string
    // lowercasing applies contextual Greek sigma rules and changes token IDs.
    icu::UnicodeString lowered;
    for (int32_t i = 0; i < cleaned.length();) {
      UChar32 c = cleaned.char32At(i); i += U16_LENGTH(c);
      icu::UnicodeString one; one.append(c);
      one.toLower(icu::Locale::getRoot());
      lowered.append(one);
    }
    cleaned = std::move(lowered);
  }
  if (strip_accents_) {
    UErrorCode err = U_ZERO_ERROR;
    auto* nfd = icu::Normalizer2::getNFDInstance(err);
    if (U_FAILURE(err)) throw std::runtime_error("ICU NFD normalization failed");
    icu::UnicodeString decomposed; nfd->normalize(cleaned, decomposed, err);
    if (U_FAILURE(err)) throw std::runtime_error("ICU NFD normalization failed");
    cleaned.remove();
    for (int32_t i = 0; i < decomposed.length();) {
      UChar32 c = decomposed.char32At(i); i += U16_LENGTH(c);
      if (u_charType(c) != U_NON_SPACING_MARK) cleaned.append(c);
    }
  }
  std::vector<icu::UnicodeString> words;
  icu::UnicodeString word;
  for (int32_t i = 0; i < cleaned.length();) {
    UChar32 c = cleaned.char32At(i); i += U16_LENGTH(c);
    if (space(c) || punct(c)) {
      if (!word.isEmpty()) { words.push_back(word); word.remove(); }
      if (punct(c)) { icu::UnicodeString p; p.append(c); words.push_back(std::move(p)); }
    } else word.append(c);
  }
  if (!word.isEmpty()) words.push_back(std::move(word));
  for (const auto& w : words) {
    std::string bytes = utf8(w);
    std::vector<int32_t> boundaries{0};
    int32_t cursor = 0, byte_count = static_cast<int32_t>(bytes.size());
    while (cursor < byte_count) {
      UChar32 c;
      U8_NEXT(reinterpret_cast<const uint8_t*>(bytes.data()), cursor, byte_count, c);
      boundaries.push_back(cursor);
    }
    uint32_t count = static_cast<uint32_t>(boundaries.size() - 1);
    if (count > max_chars_) { out.push_back(unknown_id_); continue; }
    size_t saved = out.size();
    bool failed = false;
    for (uint32_t start = 0; start < count;) {
      uint32_t node = 0, best_end = start, found_id = UINT32_MAX;
      if (start) node = child(child(0, '#'), '#');
      for (uint32_t end = start + 1; end <= count && node != UINT32_MAX; ++end) {
        for (int32_t i = boundaries[end - 1]; i < boundaries[end]; ++i) {
          node = child(node, static_cast<uint8_t>(bytes[i]));
          if (node == UINT32_MAX) break;
        }
        if (node == UINT32_MAX) break;
        if (nodes_[node].token_id != UINT32_MAX) {
          found_id = nodes_[node].token_id;
          best_end = end;
        }
      }
      if (found_id == UINT32_MAX) { failed = true; break; }
      out.push_back(found_id); start = best_end;
    }
    if (failed) { out.resize(saved); out.push_back(unknown_id_); }
    if (max_tokens_ && out.size() >= max_tokens_) {
      out.resize(max_tokens_);
      return;
    }
  }
}

std::vector<uint32_t> Tokenizer::encode(std::string_view input) const {
  TokenizerScratch scratch;
  encode_into(input, scratch);
  return std::move(scratch.ids);
}

void Tokenizer::encode_into(std::string_view input, TokenizerScratch& scratch) const {
  auto& out = scratch.ids;
  out.clear();
  size_t pos = 0;
  while (pos < input.size()) {
    size_t best = std::string_view::npos; const AddedToken* chosen = nullptr;
    for (const auto& token : added_) {
      size_t at = input.find(token.text, pos);
      if (at < best) { best = at; chosen = &token; }
    }
    if (!chosen) {
      if (bpe_) encode_bpe_segment(input.substr(pos), out, scratch.bpe_symbols, scratch.ascii, scratch.bpe_input);
      else encode_segment(input.substr(pos), out, scratch.ascii);
      break;
    }
    size_t segment_end = best;
    const auto* bytes = reinterpret_cast<const uint8_t*>(input.data());
    if (chosen->lstrip) {
      while (segment_end > pos) {
        int32_t cursor = static_cast<int32_t>(segment_end);
        UChar32 c;
        U8_PREV(bytes, static_cast<int32_t>(pos), cursor, c);
        if (c < 0 || !u_isUWhiteSpace(c)) break;
        segment_end = static_cast<size_t>(cursor);
      }
    }
    if (segment_end > pos) {
      if (bpe_) encode_bpe_segment(input.substr(pos, segment_end-pos), out, scratch.bpe_symbols, scratch.ascii, scratch.bpe_input);
      else encode_segment(input.substr(pos, segment_end-pos), out, scratch.ascii);
    }
    if (max_tokens_ && out.size() >= max_tokens_) break;
    out.push_back(chosen->id); pos = best + chosen->text.size();
    if (chosen->rstrip) {
      while (pos < input.size()) {
        int32_t cursor = static_cast<int32_t>(pos);
        UChar32 c;
        U8_NEXT(bytes, cursor, static_cast<int32_t>(input.size()), c);
        if (c < 0 || !u_isUWhiteSpace(c)) break;
        pos = static_cast<size_t>(cursor);
      }
    }
    if (max_tokens_ && out.size() >= max_tokens_) break;
  }
}
}
