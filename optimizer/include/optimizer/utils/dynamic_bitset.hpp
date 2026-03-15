#ifndef OPTIMIZER_UTILS_DYNAMIC_BITSET_HPP_DEFINED
#define OPTIMIZER_UTILS_DYNAMIC_BITSET_HPP_DEFINED
#include <algorithm>
#include <cstdint>
#include <utility/assumptions.hpp>
#include <vector>
namespace optimizer::utils
{

class DynamicBitset
{
  public:
    static constexpr std::size_t BITS_PER_WORD = 64U;

  public:
    DynamicBitset() = default;
    explicit DynamicBitset(std::size_t num_bits) { resize(num_bits); }

    void resize(std::size_t num_bits)
    {
        num_bits_ = num_bits;
        words_.assign(words_for_bits(num_bits), 0ULL);
    }

    void reset() { std::fill(words_.begin(), words_.end(), 0ULL); }

    void reset(std::size_t bit) { words_[word_index(bit)] &= ~bit_mask(bit); }

    void set(std::size_t bit) { words_[word_index(bit)] |= bit_mask(bit); }

    bool test(std::size_t bit) const { return (words_[word_index(bit)] & bit_mask(bit)) != 0ULL; }

    void and_with(const DynamicBitset& other)
    {
        ASSUMPTION(words_.size() == other.words_.size());
        for (std::size_t i = 0; i < words_.size(); ++i)
        {
            words_[i] &= other.words_[i];
        }
    }

    void subtract(const DynamicBitset& other)
    {
        ASSUMPTION(words_.size() == other.words_.size());
        for (std::size_t i = 0; i < words_.size(); ++i)
        {
            words_[i] &= ~other.words_[i];
        }
    }

    bool operator==(const DynamicBitset& other) const { return words_ == other.words_; }
    bool operator!=(const DynamicBitset& other) const { return !(*this == other); }

    [[nodiscard]] const std::vector<std::uint64_t>& words() const { return words_; }
    std::vector<std::uint64_t>&                     words() { return words_; }

  private:
    static std::size_t words_for_bits(std::size_t bits)
    {
        return (bits + BITS_PER_WORD - 1U) / BITS_PER_WORD;
    }

    static std::size_t   word_index(std::size_t bit) { return bit / BITS_PER_WORD; }
    static std::uint64_t bit_mask(std::size_t bit) { return 1ULL << (bit % BITS_PER_WORD); }

  private:
    std::size_t                num_bits_{0};
    std::vector<std::uint64_t> words_{};
};
} // namespace optimizer::utils
#endif
