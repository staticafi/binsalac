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

enum class MayLossFlag : std::uint8_t
{
    None                 = 0U,
    MergeUnknown         = 1U << 0U,
    CallOrderDiscrepancy = 1U << 1U,
};

struct MayValue
{
    bool         is_top{false};
    std::uint8_t loss_flags{0};
    MayTargetSet targets{};

    static MayValue top() noexcept;
    static MayValue singleton(Target target);

    [[nodiscard]] bool        empty() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] bool has_loss() const noexcept;
    [[nodiscard]] bool has_merge_unknown() const noexcept;
    [[nodiscard]] bool has_call_order_discrepancy() const noexcept;

    [[nodiscard]] bool is_singleton_precise_target() const noexcept;
    [[nodiscard]] bool is_singleton_dereferenceable_target() const noexcept;
    [[nodiscard]] bool is_must_fact() const noexcept;

    [[nodiscard]] std::optional<Target> singleton_target() const noexcept;
    [[nodiscard]] std::optional<Target> singleton_dereferenceable_target() const noexcept;
    [[nodiscard]] std::optional<Target> must_fact() const noexcept;

    void clear_to_bottom() noexcept;
    void make_top() noexcept;

    void add_merge_unknown() noexcept;
    void add_call_order_discrepancy() noexcept;

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
        std::size_t    seed     = std::hash<std::uint8_t>{}(value.loss_flags);
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
