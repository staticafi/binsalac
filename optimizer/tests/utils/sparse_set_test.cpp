#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <optimizer/utils/sparse_set.hpp>

namespace
{
using optimizer::utils::SparseSet;
}

TEST_CASE("SparseSet starts empty", "[SparseSet]")
{
    SparseSet<int> set;

    REQUIRE(set.empty());
    REQUIRE(set.size() == 0);
    REQUIRE(set.begin() == set.end());
    REQUIRE_FALSE(set.contains(1));
    REQUIRE(set.find(1) == set.end());
}

TEST_CASE("SparseSet initializer_list constructor inserts unique sorted values", "[SparseSet]")
{
    SparseSet<int> set{5, 1, 3, 3, 2, 1};

    REQUIRE_FALSE(set.empty());
    REQUIRE(set.size() == 4);

    std::vector<int> values(set.begin(), set.end());
    REQUIRE(values == std::vector<int>{1, 2, 3, 5});
}

TEST_CASE("SparseSet range constructor inserts unique sorted values", "[SparseSet]")
{
    std::vector<int> input{4, 2, 4, 1, 3, 2};

    SparseSet<int> set(input.begin(), input.end());

    REQUIRE(set.size() == 4);
    REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 2, 3, 4});
}

TEST_CASE("SparseSet insert adds new elements and ignores duplicates", "[SparseSet]")
{
    SparseSet<int> set;

    auto [it1, inserted1] = set.insert(3);
    REQUIRE(inserted1);
    REQUIRE(*it1 == 3);

    auto [it2, inserted2] = set.insert(1);
    REQUIRE(inserted2);
    REQUIRE(*it2 == 1);

    auto [it3, inserted3] = set.insert(3);
    REQUIRE_FALSE(inserted3);
    REQUIRE(*it3 == 3);

    REQUIRE(set.size() == 2);
    REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 3});
}

TEST_CASE("SparseSet insert rvalue works", "[SparseSet]")
{
    SparseSet<std::string> set;

    std::string value   = "hello";
    auto [it, inserted] = set.insert(std::move(value));

    REQUIRE(inserted);
    REQUIRE(*it == "hello");
    REQUIRE(set.contains("hello"));
    REQUIRE(set.size() == 1);
}

TEST_CASE("SparseSet find and contains work for mutable and const access", "[SparseSet]")
{
    SparseSet<int> set{1, 3, 5};

    REQUIRE(set.contains(1));
    REQUIRE(set.contains(3));
    REQUIRE_FALSE(set.contains(2));

    auto it = set.find(3);
    REQUIRE(it != set.end());
    REQUIRE(*it == 3);

    REQUIRE(set.find(4) == set.end());

    const auto& constSet = set;
    auto        cit      = constSet.find(5);
    REQUIRE(cit != constSet.end());
    REQUIRE(*cit == 5);
    REQUIRE(constSet.find(6) == constSet.end());
}

TEST_CASE("SparseSet emplace and emplace_hint insert correctly", "[SparseSet]")
{
    SparseSet<std::string> set;

    auto [it1, inserted1] = set.emplace("gamma");
    REQUIRE(inserted1);
    REQUIRE(*it1 == "gamma");

    auto it2 = set.emplace_hint(set.begin(), "alpha");
    REQUIRE(*it2 == "alpha");

    auto it3 = set.emplace_hint(set.end(), "omega");
    REQUIRE(*it3 == "omega");

    REQUIRE(set.size() == 3);
    REQUIRE(std::vector<std::string>(set.begin(), set.end()) ==
            std::vector<std::string>{"alpha", "gamma", "omega"});
}

TEST_CASE("SparseSet insert with hint handles insertion and duplicate lookup", "[SparseSet]")
{
    SparseSet<int> set{2, 4};

    SECTION("insert before hint")
    {
        auto it = set.insert(set.begin(), 1);
        REQUIRE(*it == 1);
        REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 2, 4});
    }

    SECTION("insert at end with end hint")
    {
        auto it = set.insert(set.end(), 5);
        REQUIRE(*it == 5);
        REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{2, 4, 5});
    }

    SECTION("duplicate at hint")
    {
        auto hint = set.find(4);
        REQUIRE(hint != set.end());

        auto it = set.insert(hint, 4);
        REQUIRE(*it == 4);
        REQUIRE(set.size() == 2);
    }

    SECTION("duplicate before hint")
    {
        auto hint = set.find(4);
        REQUIRE(hint != set.end());

        auto it = set.insert(hint, 2);
        REQUIRE(*it == 2);
        REQUIRE(set.size() == 2);
    }
}

TEST_CASE("SparseSet insert range and initializer_list work", "[SparseSet]")
{
    SparseSet<int> set;

    std::vector<int> values{4, 1, 4, 2};
    set.insert(values.begin(), values.end());
    set.insert({3, 2, 5});

    REQUIRE(set.size() == 5);
    REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE("SparseSet erase by value removes element and reports result", "[SparseSet]")
{
    SparseSet<int> set{1, 2, 3};

    REQUIRE(set.erase(2) == 1);
    REQUIRE_FALSE(set.contains(2));
    REQUIRE(set.size() == 2);

    REQUIRE(set.erase(2) == 0);
    REQUIRE(set.size() == 2);
    REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 3});
}

TEST_CASE("SparseSet erase by iterator removes element and returns next iterator", "[SparseSet]")
{
    SparseSet<int> set{1, 2, 3};

    auto it = set.find(2);
    REQUIRE(it != set.end());

    auto next = set.erase(it);
    REQUIRE(next != set.end());
    REQUIRE(*next == 3);

    REQUIRE(set.size() == 2);
    REQUIRE(std::vector<int>(set.begin(), set.end()) == std::vector<int>{1, 3});
}

TEST_CASE("SparseSet clear removes all elements", "[SparseSet]")
{
    SparseSet<int> set{1, 2, 3};

    REQUIRE_FALSE(set.empty());

    set.clear();

    REQUIRE(set.empty());
    REQUIRE(set.size() == 0);
    REQUIRE(set.begin() == set.end());
}

TEST_CASE("SparseSet equality and ordering compare contents", "[SparseSet]")
{
    SparseSet<int> a{1, 2, 3};
    SparseSet<int> b{1, 2, 3};
    SparseSet<int> c{1, 2, 4};

    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
    REQUIRE((a < c));
    REQUIRE((c > a));
}

TEST_CASE("SparseSet union_with handles disjoint, overlapping, empty, and identical sets",
          "[SparseSet]")
{
    SECTION("union with empty other leaves set unchanged")
    {
        SparseSet<int> a{1, 3};
        SparseSet<int> b;

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 3});
    }

    SECTION("union into empty copies other")
    {
        SparseSet<int> a;
        SparseSet<int> b{2, 4};

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{2, 4});
    }

    SECTION("all current elements before other")
    {
        SparseSet<int> a{1, 2};
        SparseSet<int> b{4, 5};

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 4, 5});
    }

    SECTION("all other elements before current")
    {
        SparseSet<int> a{4, 5};
        SparseSet<int> b{1, 2};

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 4, 5});
    }

    SECTION("overlapping sets")
    {
        SparseSet<int> a{1, 3, 5};
        SparseSet<int> b{2, 3, 4};

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 3, 4, 5});
    }

    SECTION("identical sets")
    {
        SparseSet<int> a{1, 2, 3};
        SparseSet<int> b{1, 2, 3};

        a.union_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 3});
    }

    SECTION("self union")
    {
        SparseSet<int> a{1, 2, 3};

        a.union_with(a);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 3});
    }
}

TEST_CASE("SparseSet join_with is alias for union_with", "[SparseSet]")
{
    SparseSet<int> a{1, 3};
    SparseSet<int> b{2, 3, 4};

    a.join_with(b);

    REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 3, 4});
}

TEST_CASE("SparseSet intersect_with keeps only common elements", "[SparseSet]")
{
    SECTION("intersection with empty self stays empty")
    {
        SparseSet<int> a;
        SparseSet<int> b{1, 2, 3};

        a.intersect_with(b);

        REQUIRE(a.empty());
    }

    SECTION("intersection with empty other clears self")
    {
        SparseSet<int> a{1, 2, 3};
        SparseSet<int> b;

        a.intersect_with(b);

        REQUIRE(a.empty());
    }

    SECTION("overlapping sets")
    {
        SparseSet<int> a{1, 2, 3, 4};
        SparseSet<int> b{2, 4, 6};

        a.intersect_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{2, 4});
    }

    SECTION("disjoint sets")
    {
        SparseSet<int> a{1, 2};
        SparseSet<int> b{3, 4};

        a.intersect_with(b);

        REQUIRE(a.empty());
    }

    SECTION("identical sets")
    {
        SparseSet<int> a{1, 2, 3};
        SparseSet<int> b{1, 2, 3};

        a.intersect_with(b);

        REQUIRE(std::vector<int>(a.begin(), a.end()) == std::vector<int>{1, 2, 3});
    }
}
