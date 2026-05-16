#ifndef OPTIMIZER_UTILS_POINTS_TO_MAY_VALUE_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_MAY_VALUE_HPP_INCLUDED

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <optimizer/utils/points_to/nodes.hpp>
#include <optimizer/utils/sparse_set.hpp>

namespace optimizer::utils::points_to
{
using MayTargetSet = optimizer::utils::SparseSet<Target>;

struct MayValue
{
    bool         is_top{false};
    MayTargetSet targets{};

    static MayValue top() noexcept;
    static MayValue singleton(Target target);

    [[nodiscard]] bool        empty() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    void clear_to_bottom() noexcept;
    void make_top() noexcept;

    void insert(const Target& target);
    void insert(Target&& target);

    void join_with(const MayValue& other);

    [[nodiscard]] MayTargetSet::iterator       begin() noexcept;
    [[nodiscard]] MayTargetSet::iterator       end() noexcept;
    [[nodiscard]] MayTargetSet::const_iterator begin() const noexcept;
    [[nodiscard]] MayTargetSet::const_iterator end() const noexcept;
    [[nodiscard]] MayTargetSet::const_iterator cbegin() const noexcept;
    [[nodiscard]] MayTargetSet::const_iterator cend() const noexcept;

    friend bool          operator==(const MayValue& a, const MayValue& b);
    friend std::ostream& operator<<(std::ostream& os, const MayValue& obj);
};

[[nodiscard]] bool contains_objectId(const MayTargetSet& set, objectId id) noexcept;
[[nodiscard]] bool contains_only_objectId(const MayTargetSet& set, objectId id) noexcept;

[[nodiscard]] bool contains_objectId(const MayValue& value, objectId id) noexcept;
[[nodiscard]] bool contains_only_objectId(const MayValue& value, objectId id) noexcept;

[[nodiscard]] inline MayValue make_singleton(objectId id, bool offset_flag = false)
{
    return MayValue::singleton(Target{.id = id, .offset_flag = offset_flag});
}
} // namespace optimizer::utils::points_to

namespace std
{
template <>
struct hash<optimizer::utils::points_to::MayValue>
{
    std::size_t operator()(const optimizer::utils::points_to::MayValue& value) const noexcept
    {
        constexpr auto top_hash = 0x9d4e12c1ULL;

        std::size_t seed = 0U;
        hash_combine(seed, std::hash<bool>{}(value.is_top));

        if (value.is_top)
        {
            hash_combine(seed, top_hash);
            return seed;
        }

        for (const auto& target : value)
        {
            hash_combine(seed, std::hash<optimizer::utils::points_to::Target>{}(target));
        }

        return seed;
    }
};
} // namespace std

#endif
