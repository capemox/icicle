#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace icicle {
// Find the first non-ASCII byte with SIMD where available, then handle the tail.
inline size_t ascii_prefix_length(std::string_view text) {
  size_t i = 0;
#if defined(__SSE2__)
  for (; i + 16 <= text.size(); i += 16) {
    auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(text.data() + i));
    if (_mm_movemask_epi8(bytes)) break;
  }
#elif defined(__aarch64__) && defined(__ARM_NEON)
  for (; i + 16 <= text.size(); i += 16) {
    auto bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(text.data() + i));
    if (vmaxvq_u8(bytes) & 0x80) break;
  }
#endif
  for (; i + 8 <= text.size(); i += 8) {
    uint64_t bytes;
    std::memcpy(&bytes, text.data() + i, sizeof(bytes));
    if (bytes & UINT64_C(0x8080808080808080)) break;
  }
  while (i < text.size() && static_cast<uint8_t>(text[i]) < 0x80) ++i;
  return i;
}

struct TrieNode { uint32_t base, parent, token_id; };
struct BpeMerge { uint64_t key; uint32_t rank, token_id; };
struct BpeSymbol { uint32_t id, pair_rank, merged_id; };
struct AddedToken { std::string_view text; uint32_t id; bool lstrip, rstrip; };
struct TokenizerScratch {
  std::string input;
  std::string ascii;
  std::vector<uint32_t> ids;
  std::vector<BpeSymbol> bpe_symbols;
  std::string bpe_input;
};
class Tokenizer {
public:
  Tokenizer(const unsigned char* blob, uint64_t nodes_offset, uint32_t nodes_count,
            bool bpe, uint64_t merges_offset, uint32_t merges_count,
            uint64_t merge_index_offset, uint32_t merge_index_count,
            bool bytelevel_prefix_space, bool bytelevel_regex,
            uint32_t unknown_id, uint32_t max_chars,
            uint32_t max_tokens, bool clean_text, bool chinese, bool lowercase,
            bool strip_accents, std::vector<AddedToken> added);
  std::vector<uint32_t> encode(std::string_view input) const;
  void encode_into(std::string_view input, TokenizerScratch& scratch) const;
private:
  uint32_t child(uint32_t node, uint8_t byte) const;
  uint32_t token_id(std::string_view value) const;
  const BpeMerge* merge(uint32_t left, uint32_t right) const;
  void encode_bpe_segment(std::string_view text, std::vector<uint32_t>& out,
                          std::vector<BpeSymbol>& symbols, std::string& bytes,
                          std::string& prefix_input) const;
  void encode_ascii_segment(std::string_view text, std::vector<uint32_t>& out,
                            std::string& cleaned) const;
  void encode_segment(std::string_view text, std::vector<uint32_t>& out,
                      std::string& ascii) const;
  const TrieNode* nodes_;
  const BpeMerge* merges_;
  const uint32_t* merge_index_;
  uint32_t nodes_count_;
  uint32_t merges_count_;
  uint32_t merge_index_count_;
  std::array<uint32_t, 256> byte_ids_{};
  bool bpe_, bytelevel_prefix_space_, bytelevel_regex_;
  uint32_t unknown_id_, max_chars_, max_tokens_;
  bool clean_text_, chinese_, lowercase_, strip_accents_;
  std::vector<AddedToken> added_;
};
}
