#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include <optimizer/utils/sparse_map.hpp>

namespace
{
using optimizer::utils::SparseMap;
}

TEST_CASE("SparseMap starts empty", "[SparseMap]")
{
    SparseMap<int, int> map;

    REQUIRE(map.empty());
    REQUIRE(map.size() == 0);
    REQUIRE(map.begin() == map.end());
    REQUIRE_FALSE(map.contains(1));
    REQUIRE(map.find(1) == map.end());
}

TEST_CASE("SparseMap operator[] inserts default values and keeps keys sorted", "[SparseMap]")
{
    SparseMap<int, int> map;

    map[5] = 50;
    map[1] = 10;
    map[3] = 30;

    REQUIRE(map.size() == 3);
    REQUIRE_FALSE(map.empty());

    REQUIRE(map.at(1) == 10);
    REQUIRE(map.at(3) == 30);
    REQUIRE(map.at(5) == 50);

    std::vector<int> keys;
    for (const auto& [key, _] : map)
    {
        keys.push_back(key);
    }

    REQUIRE(keys == std::vector<int>{1, 3, 5});
}

TEST_CASE("SparseMap at throws for missing key", "[SparseMap]")
{
    SparseMap<int, std::string> map;
    map.insert({1, "one"});

    REQUIRE_NOTHROW(map.at(1));
    REQUIRE_THROWS_AS(map.at(2), std::out_of_range);

    const auto& constMap = map;
    REQUIRE(constMap.at(1) == "one");
    REQUIRE_THROWS_AS(constMap.at(2), std::out_of_range);
}

TEST_CASE("SparseMap insert does not overwrite existing key", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it1, inserted1] = map.insert({2, "first"});
    REQUIRE(inserted1);
    REQUIRE(it1->first == 2);
    REQUIRE(it1->second == "first");

    auto [it2, inserted2] = map.insert({2, "second"});
    REQUIRE_FALSE(inserted2);
    REQUIRE(it2 == it1);
    REQUIRE(map.size() == 1);
    REQUIRE(map.at(2) == "first");
}

TEST_CASE("SparseMap insert rvalue works", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    std::pair<int, std::string> value{4, "four"};
    auto [it, inserted] = map.insert(std::move(value));

    REQUIRE(inserted);
    REQUIRE(it->first == 4);
    REQUIRE(it->second == "four");
    REQUIRE(map.at(4) == "four");
}

TEST_CASE("SparseMap emplace and emplace_hint insert elements correctly", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it1, inserted1] = map.emplace(3, "three");
    REQUIRE(inserted1);
    REQUIRE(it1->first == 3);
    REQUIRE(it1->second == "three");

    auto hint = map.begin();
    auto it2  = map.emplace_hint(hint, 1, "one");
    REQUIRE(it2->first == 1);
    REQUIRE(it2->second == "one");

    auto it3 = map.emplace_hint(map.end(), 5, "five");
    REQUIRE(it3->first == 5);
    REQUIRE(it3->second == "five");

    REQUIRE(map.size() == 3);
    REQUIRE(map.at(1) == "one");
    REQUIRE(map.at(3) == "three");
    REQUIRE(map.at(5) == "five");
}

TEST_CASE("SparseMap try_emplace inserts new key and does not overwrite existing key",
          "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it1, inserted1] = map.try_emplace(2, "two");
    REQUIRE(inserted1);
    REQUIRE(it1 != map.end());
    REQUIRE(it1->first == 2);
    REQUIRE(it1->second == "two");

    auto [it2, inserted2] = map.try_emplace(2, "updated");
    REQUIRE_FALSE(inserted2);
    REQUIRE(it2 == it1);
    REQUIRE(map.size() == 1);
    REQUIRE(map.at(2) == "two");
}

TEST_CASE("SparseMap try_emplace keeps keys sorted", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it5, inserted5] = map.try_emplace(5, "five");
    REQUIRE(inserted5);
    REQUIRE(it5->first == 5);
    REQUIRE(it5->second == "five");

    auto [it1, inserted1] = map.try_emplace(1, "one");
    REQUIRE(inserted1);
    REQUIRE(it1->first == 1);
    REQUIRE(it1->second == "one");

    auto [it3, inserted3] = map.try_emplace(3, "three");
    REQUIRE(inserted3);
    REQUIRE(it3->first == 3);
    REQUIRE(it3->second == "three");

    REQUIRE(map.size() == 3);

    std::vector<int> keys;
    for (const auto& [key, _] : map)
    {
        keys.push_back(key);
    }

    REQUIRE(keys == std::vector<int>{1, 3, 5});
}

TEST_CASE("SparseMap try_emplace with hint handles insert and duplicate key", "[SparseMap]")
{
    SparseMap<int, std::string> map;
    map.insert({2, "two"});
    map.insert({4, "four"});

    SECTION("insert before begin hint")
    {
        auto [it, inserted] = map.try_emplace(map.begin(), 1, "one");

        REQUIRE(inserted);
        REQUIRE(it != map.end());
        REQUIRE(it->first == 1);
        REQUIRE(it->second == "one");
        REQUIRE(map.size() == 3);
    }

    SECTION("insert after end hint")
    {
        auto [it, inserted] = map.try_emplace(map.end(), 5, "five");

        REQUIRE(inserted);
        REQUIRE(it != map.end());
        REQUIRE(it->first == 5);
        REQUIRE(it->second == "five");
        REQUIRE(map.size() == 3);
    }

    SECTION("duplicate at hint is not overwritten")
    {
        auto hint = map.find(2);
        REQUIRE(hint != map.end());

        auto [it, inserted] = map.try_emplace(hint, 2, "updated");

        REQUIRE_FALSE(inserted);
        REQUIRE(it == hint);
        REQUIRE(it->first == 2);
        REQUIRE(it->second == "two");
        REQUIRE(map.size() == 2);
    }

    SECTION("duplicate before hint is not overwritten")
    {
        auto hint = map.find(4);
        REQUIRE(hint != map.end());

        auto [it, inserted] = map.try_emplace(hint, 2, "updated");

        REQUIRE_FALSE(inserted);
        REQUIRE(it != map.end());
        REQUIRE(it->first == 2);
        REQUIRE(it->second == "two");
        REQUIRE(map.size() == 2);
    }

    SECTION("poor hint falls back and inserts in sorted order")
    {
        auto [it, inserted] = map.try_emplace(map.begin(), 3, "three");

        REQUIRE(inserted);
        REQUIRE(it != map.end());
        REQUIRE(it->first == 3);
        REQUIRE(it->second == "three");
        REQUIRE(map.size() == 3);

        std::vector<int> keys;
        for (const auto& [key, _] : map)
        {
            keys.push_back(key);
        }

        REQUIRE(keys == std::vector<int>{2, 3, 4});
    }
}

TEST_CASE("SparseMap try_emplace constructs mapped value from arguments", "[SparseMap]")
{
    SparseMap<int, std::vector<int>> map;

    auto [it, inserted] = map.try_emplace(1, 3, 7);

    REQUIRE(inserted);
    REQUIRE(it != map.end());
    REQUIRE(it->first == 1);
    REQUIRE(it->second == std::vector<int>{7, 7, 7});
}

TEST_CASE("SparseMap try_emplace with hint constructs mapped value from arguments", "[SparseMap]")
{
    SparseMap<int, std::vector<int>> map;

    auto [it, inserted] = map.try_emplace(map.end(), 1, 3, 7);

    REQUIRE(inserted);
    REQUIRE(it != map.end());
    REQUIRE(it->first == 1);
    REQUIRE(it->second == std::vector<int>{7, 7, 7});
}

TEST_CASE("SparseMap insert with hint returns existing iterator for duplicate key", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({3, 30});

    auto it = map.insert(map.end(), {3, 999});

    REQUIRE(it != map.end());
    REQUIRE(it->first == 3);
    REQUIRE(it->second == 30);
    REQUIRE(map.size() == 2);
}

TEST_CASE("SparseMap insert_or_assign inserts new keys and overwrites existing values",
          "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it1, inserted1] = map.insert_or_assign(2, "two");
    REQUIRE(inserted1);
    REQUIRE(it1->first == 2);
    REQUIRE(it1->second == "two");

    auto [it2, inserted2] = map.insert_or_assign(2, "updated");
    REQUIRE_FALSE(inserted2);
    REQUIRE(it2->first == 2);
    REQUIRE(it2->second == "updated");

    auto [it3, inserted3] = map.insert_or_assign(1, "one");
    REQUIRE(inserted3);
    REQUIRE(it3->first == 1);
    REQUIRE(it3->second == "one");

    REQUIRE(map.size() == 2);
    REQUIRE(map.at(1) == "one");
    REQUIRE(map.at(2) == "updated");
}

TEST_CASE("SparseMap insert_or_assign with hint handles insert and update", "[SparseMap]")
{
    SparseMap<int, std::string> map;
    map.insert({2, "two"});
    map.insert({4, "four"});

    SECTION("insert using begin hint")
    {
        auto it = map.insert_or_assign(map.begin(), 1, "one");
        REQUIRE(it->first == 1);
        REQUIRE(it->second == "one");
        REQUIRE(map.size() == 3);
    }

    SECTION("insert using end hint")
    {
        auto it = map.insert_or_assign(map.end(), 5, "five");
        REQUIRE(it->first == 5);
        REQUIRE(it->second == "five");
        REQUIRE(map.size() == 3);
    }

    SECTION("update existing element at hint")
    {
        auto it = map.find(2);
        REQUIRE(it != map.end());

        auto updated = map.insert_or_assign(it, 2, "updated");
        REQUIRE(updated->first == 2);
        REQUIRE(updated->second == "updated");
        REQUIRE(map.size() == 2);
    }

    SECTION("update previous element when hint points after it")
    {
        auto it = map.find(4);
        REQUIRE(it != map.end());

        auto updated = map.insert_or_assign(it, 2, "updated");
        REQUIRE(updated->first == 2);
        REQUIRE(updated->second == "updated");
        REQUIRE(map.size() == 2);
    }
}

TEST_CASE("SparseMap erase by key and iterator removes elements", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({2, 20});
    map.insert({3, 30});

    map.erase(2);
    REQUIRE(map.size() == 2);
    REQUIRE_FALSE(map.contains(2));

    map.erase(999);
    REQUIRE(map.size() == 2);

    auto it = map.find(1);
    REQUIRE(it != map.end());

    auto next = map.erase(it);
    REQUIRE(next != map.end());
    REQUIRE(next->first == 3);
    REQUIRE(map.size() == 1);
    REQUIRE(map.contains(3));
    REQUIRE(map.contains(3));
    REQUIRE_FALSE(map.contains(1));
}

TEST_CASE("SparseMap clear removes all elements", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({2, 20});

    REQUIRE_FALSE(map.empty());

    map.clear();

    REQUIRE(map.empty());
    REQUIRE(map.size() == 0);
    REQUIRE(map.begin() == map.end());
}

TEST_CASE("SparseMap equality compares contents", "[SparseMap]")
{
    SparseMap<int, std::string> a;
    SparseMap<int, std::string> b;
    SparseMap<int, std::string> c;

    a.insert({1, "one"});
    a.insert({2, "two"});

    b.insert({1, "one"});
    b.insert({2, "two"});

    c.insert({1, "one"});
    c.insert({2, "different"});

    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
}

TEST_CASE("SparseMap operator[] default constructs missing value", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto& value = map[7];

    REQUIRE(map.size() == 1);
    REQUIRE(map.contains(7));
    REQUIRE(value.empty());

    value = "seven";
    REQUIRE(map.at(7) == "seven");
}

TEST_CASE("SparseMap find works for const and non-const maps", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({3, 30});

    auto it = map.find(3);
    REQUIRE(it != map.end());
    REQUIRE(it->first == 3);
    REQUIRE(it->second == 30);

    auto missing = map.find(2);
    REQUIRE(missing == map.end());

    const auto& constMap = map;
    auto        cit      = constMap.find(1);
    REQUIRE(cit != constMap.end());
    REQUIRE(cit->first == 1);
    REQUIRE(cit->second == 10);

    auto cmissing = constMap.find(99);
    REQUIRE(cmissing == constMap.end());
}

TEST_CASE("SparseMap reserve does not change logical contents", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.reserve(32);

    REQUIRE(map.empty());
    REQUIRE(map.size() == 0);

    map.insert({2, 20});
    map.insert({1, 10});

    REQUIRE(map.size() == 2);
    REQUIRE(map.at(1) == 10);
    REQUIRE(map.at(2) == 20);
}

TEST_CASE("SparseMap insert with hint inserts in the middle", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({5, 50});

    auto hint = map.find(5);
    REQUIRE(hint != map.end());

    auto it = map.insert(hint, {3, 30});

    REQUIRE(it != map.end());
    REQUIRE(it->first == 3);
    REQUIRE(it->second == 30);
    REQUIRE(map.size() == 3);

    std::vector<int> keys;
    for (const auto& [key, _] : map)
    {
        keys.push_back(key);
    }

    REQUIRE(keys == std::vector<int>{1, 3, 5});
}

TEST_CASE("SparseMap insert with begin hint returns existing previous duplicate", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({3, 30});
    map.insert({5, 50});

    auto hint = map.find(3);
    REQUIRE(hint != map.end());

    auto it = map.insert(hint, {1, 999});

    REQUIRE(it != map.end());
    REQUIRE(it->first == 1);
    REQUIRE(it->second == 10);
    REQUIRE(map.size() == 3);
}

TEST_CASE("SparseMap insert_or_assign with hint falls back when hint is poor", "[SparseMap]")
{
    SparseMap<int, std::string> map;
    map.insert({2, "two"});
    map.insert({6, "six"});

    auto poorHint = map.begin();
    auto it       = map.insert_or_assign(poorHint, 4, "four");

    REQUIRE(it != map.end());
    REQUIRE(it->first == 4);
    REQUIRE(it->second == "four");
    REQUIRE(map.size() == 3);

    std::vector<int> keys;
    for (const auto& [key, _] : map)
    {
        keys.push_back(key);
    }

    REQUIRE(keys == std::vector<int>{2, 4, 6});
}

TEST_CASE("SparseMap emplace does not overwrite existing key", "[SparseMap]")
{
    SparseMap<int, std::string> map;

    auto [it1, inserted1] = map.emplace(1, "one");
    REQUIRE(inserted1);
    REQUIRE(it1->second == "one");

    auto [it2, inserted2] = map.emplace(1, "different");
    REQUIRE_FALSE(inserted2);
    REQUIRE(it2 == it1);
    REQUIRE(map.size() == 1);
    REQUIRE(map.at(1) == "one");
}

TEST_CASE("SparseMap erase end iterator after removing last element", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});

    auto it = map.find(1);
    REQUIRE(it != map.end());

    auto next = map.erase(it);

    REQUIRE(next == map.end());
    REQUIRE(map.empty());
}

TEST_CASE("SparseMap supports mutable iteration over values", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({1, 10});
    map.insert({2, 20});

    for (auto& [key, value] : map)
    {
        value += key;
    }

    REQUIRE(map.at(1) == 11);
    REQUIRE(map.at(2) == 22);
}

TEST_CASE("SparseMap const iteration preserves sorted order", "[SparseMap]")
{
    SparseMap<int, int> map;
    map.insert({4, 40});
    map.insert({2, 20});
    map.insert({3, 30});

    const auto& constMap = map;

    std::vector<int> keys;
    for (const auto& [key, _] : constMap)
    {
        keys.push_back(key);
    }

    REQUIRE(keys == std::vector<int>{2, 3, 4});
}
