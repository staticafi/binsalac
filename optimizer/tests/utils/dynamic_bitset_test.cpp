#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include <optimizer/utils/dynamic_bitset.hpp>

namespace
{
using optimizer::utils::DynamicBitset;
}

TEST_CASE("DynamicBitset starts empty by default", "[DynamicBitset]")
{
    DynamicBitset bitset;

    REQUIRE(bitset.words().empty());
}

TEST_CASE("DynamicBitset sized constructor creates zeroed storage", "[DynamicBitset]")
{
    DynamicBitset bitset(0);
    REQUIRE(bitset.words().empty());

    {
        DynamicBitset one_word(1);
        REQUIRE(one_word.words().size() == 1);
        REQUIRE(one_word.words()[0] == 0ULL);
    }
    {
        DynamicBitset one_word(64);
        REQUIRE(one_word.words().size() == 1);
        REQUIRE(one_word.words()[0] == 0ULL);
    }

    {
        DynamicBitset two_words(65);
        REQUIRE(two_words.words().size() == 2);
        REQUIRE(two_words.words()[0] == 0ULL);
        REQUIRE(two_words.words()[1] == 0ULL);
    }
}

TEST_CASE("DynamicBitset resize resets contents and adjusts number of words", "[DynamicBitset]")
{
    DynamicBitset bitset(64);
    bitset.set(1);
    bitset.set(63);

    REQUIRE(bitset.test(1));
    REQUIRE(bitset.test(63));

    bitset.resize(128);

    REQUIRE(bitset.words().size() == 2);
    REQUIRE(bitset.words()[0] == 0ULL);
    REQUIRE(bitset.words()[1] == 0ULL);

    bitset.resize(32);
    REQUIRE(bitset.words().size() == 1);
    REQUIRE(bitset.words()[0] == 0ULL);
}

TEST_CASE("DynamicBitset set and test work for bits in same word", "[DynamicBitset]")
{
    DynamicBitset bitset(64);

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE_FALSE(bitset.test(1));
    REQUIRE_FALSE(bitset.test(63));

    bitset.set(0);
    bitset.set(1);
    bitset.set(63);

    REQUIRE(bitset.test(0));
    REQUIRE(bitset.test(1));
    REQUIRE(bitset.test(63));

    REQUIRE(bitset.words().size() == 1);
    REQUIRE(bitset.words()[0] == ((1ULL << 0) | (1ULL << 1) | (1ULL << 63)));
}

TEST_CASE("DynamicBitset set and test work across multiple words", "[DynamicBitset]")
{
    DynamicBitset bitset(130);

    bitset.set(0);
    bitset.set(63);
    bitset.set(64);
    bitset.set(65);
    bitset.set(129);

    REQUIRE(bitset.test(0));
    REQUIRE(bitset.test(63));
    REQUIRE(bitset.test(64));
    REQUIRE(bitset.test(65));
    REQUIRE(bitset.test(129));

    REQUIRE_FALSE(bitset.test(1));
    REQUIRE_FALSE(bitset.test(62));
    REQUIRE_FALSE(bitset.test(66));
    REQUIRE_FALSE(bitset.test(128));

    REQUIRE(bitset.words().size() == 3);
    REQUIRE(bitset.words()[0] == ((1ULL << 0) | (1ULL << 63)));
    REQUIRE(bitset.words()[1] == ((1ULL << 0) | (1ULL << 1)));
    REQUIRE(bitset.words()[2] == (1ULL << 1));
}

TEST_CASE("DynamicBitset reset without index clears all bits", "[DynamicBitset]")
{
    DynamicBitset bitset(130);

    bitset.set(0);
    bitset.set(64);
    bitset.set(129);

    REQUIRE(bitset.test(0));
    REQUIRE(bitset.test(64));
    REQUIRE(bitset.test(129));

    bitset.reset();

    for (std::uint64_t word : bitset.words())
    {
        REQUIRE(word == 0ULL);
    }

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE_FALSE(bitset.test(64));
    REQUIRE_FALSE(bitset.test(129));
}

TEST_CASE("DynamicBitset reset by index clears only requested bit", "[DynamicBitset]")
{
    DynamicBitset bitset(130);

    bitset.set(1);
    bitset.set(64);
    bitset.set(65);
    bitset.set(129);

    bitset.reset(64);

    REQUIRE(bitset.test(1));
    REQUIRE_FALSE(bitset.test(64));
    REQUIRE(bitset.test(65));
    REQUIRE(bitset.test(129));

    bitset.reset(1);
    bitset.reset(129);

    REQUIRE_FALSE(bitset.test(1));
    REQUIRE_FALSE(bitset.test(64));
    REQUIRE(bitset.test(65));
    REQUIRE_FALSE(bitset.test(129));
}

TEST_CASE("DynamicBitset and_with keeps only common bits", "[DynamicBitset]")
{
    DynamicBitset lhs(130);
    DynamicBitset rhs(130);

    lhs.set(0);
    lhs.set(1);
    lhs.set(64);
    lhs.set(129);

    rhs.set(1);
    rhs.set(64);
    rhs.set(65);

    lhs.and_with(rhs);

    REQUIRE_FALSE(lhs.test(0));
    REQUIRE(lhs.test(1));
    REQUIRE(lhs.test(64));
    REQUIRE_FALSE(lhs.test(65));
    REQUIRE_FALSE(lhs.test(129));

    REQUIRE(lhs.words().size() == 3);
    REQUIRE(lhs.words()[0] == (1ULL << 1));
    REQUIRE(lhs.words()[1] == (1ULL << 0));
    REQUIRE(lhs.words()[2] == 0ULL);
}

TEST_CASE("DynamicBitset subtract removes bits present in other", "[DynamicBitset]")
{
    DynamicBitset lhs(130);
    DynamicBitset rhs(130);

    lhs.set(0);
    lhs.set(1);
    lhs.set(64);
    lhs.set(65);
    lhs.set(129);

    rhs.set(1);
    rhs.set(64);
    rhs.set(128);

    lhs.subtract(rhs);

    REQUIRE(lhs.test(0));
    REQUIRE_FALSE(lhs.test(1));
    REQUIRE_FALSE(lhs.test(64));
    REQUIRE(lhs.test(65));
    REQUIRE(lhs.test(129));

    REQUIRE(lhs.words().size() == 3);
    REQUIRE(lhs.words()[0] == (1ULL << 0));
    REQUIRE(lhs.words()[1] == (1ULL << 1));
    REQUIRE(lhs.words()[2] == (1ULL << 1));
}

TEST_CASE("DynamicBitset equality compares underlying words", "[DynamicBitset]")
{
    DynamicBitset a(130);
    DynamicBitset b(130);
    DynamicBitset c(130);

    a.set(0);
    a.set(65);

    b.set(0);
    b.set(65);

    c.set(0);
    c.set(64);

    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);

    REQUIRE(a != c);
    REQUIRE_FALSE(a == c);
}

TEST_CASE("DynamicBitset mutable words accessor can modify storage", "[DynamicBitset]")
{
    DynamicBitset bitset(130);

    auto& words = bitset.words();
    REQUIRE(words.size() == 3);

    words[0] = (1ULL << 3);
    words[1] = (1ULL << 5);
    words[2] = (1ULL << 2);

    REQUIRE(bitset.test(3));
    REQUIRE(bitset.test(64 + 5));
    REQUIRE(bitset.test(128 + 2));

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE_FALSE(bitset.test(64));
    REQUIRE_FALSE(bitset.test(128));
}
