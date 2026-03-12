#ifndef OPTIMIZER_UTILS_POINTS_TO_NODES_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_NODES_HPP_INCLUDED
#include <array>
#include <cstdint>
#include <ostream>

namespace optimizer::utils::points_to
{
using objectId   = std::int32_t;
using offsetFlag = bool;

namespace grouped_objects
{
constexpr objectId OUT_OF_LOCAL_SCOPE  = -3;
constexpr objectId OUT_OF_GLOBAL_SCOPE = -4;

constexpr objectId FUNCTION      = -5;
constexpr objectId VARGARG_BLOCK = -6;
constexpr objectId ALLOCA        = -7;
constexpr objectId HEAP          = -8;

constexpr objectId MERGE_UNKNOWN          = -9;
constexpr objectId CALL_ORDER_DISCREPANCY = -10;

constexpr std::size_t ABSTRACT_TARGET_NODE_COUNT = 6;

static constexpr std::array<objectId, ABSTRACT_TARGET_NODE_COUNT> ABSTRACT_TARGET_NODES = {
        OUT_OF_LOCAL_SCOPE, OUT_OF_GLOBAL_SCOPE, FUNCTION, VARGARG_BLOCK, ALLOCA, HEAP};
} // namespace grouped_objects

constexpr inline bool is_concrete_object(const objectId id) noexcept
{
    return id >= 0;
}

constexpr inline bool is_precision_marker(const objectId id) noexcept
{
    return id == grouped_objects::MERGE_UNKNOWN || id == grouped_objects::CALL_ORDER_DISCREPANCY;
}

constexpr inline bool is_summary_target(const objectId id) noexcept
{
    return id == grouped_objects::OUT_OF_LOCAL_SCOPE ||
           id == grouped_objects::OUT_OF_GLOBAL_SCOPE || id == grouped_objects::FUNCTION ||
           id == grouped_objects::VARGARG_BLOCK || id == grouped_objects::ALLOCA ||
           id == grouped_objects::HEAP;
}

constexpr inline bool is_dereferenceable_summary_target(const objectId id) noexcept
{
    return id == grouped_objects::OUT_OF_LOCAL_SCOPE ||
           id == grouped_objects::OUT_OF_GLOBAL_SCOPE || id == grouped_objects::VARGARG_BLOCK ||
           id == grouped_objects::ALLOCA || id == grouped_objects::HEAP;
}

constexpr inline bool can_have_state_cell(const objectId id) noexcept
{
    return is_concrete_object(id) || is_dereferenceable_summary_target(id);
}

enum class RegionTag : std::uint8_t
{
    Local,
    Parameter,
    Static,
    Constant,
};

static inline std::ostream& operator<<(std::ostream& os, const RegionTag region)
{
    switch (region)
    {
    case RegionTag::Local:
        return os << "local";
    case RegionTag::Parameter:
        return os << "param";
    case RegionTag::Static:
        return os << "static";
    case RegionTag::Constant:
        return os << "const";
    default:
        return os << static_cast<int>(region);
    }
}

struct Object
{
    objectId  id{};
    RegionTag region{};
};

struct Target
{
    objectId id;
    bool     offset_flag{false};

    auto operator<=>(const Target&) const = default;
};

static inline std::ostream& operator<<(std::ostream& os, const Target& object)
{
    os << object.id;
    if (object.offset_flag)
    {
        os << "off";
    }
    return os;
}

struct TargetHash
{
    std::size_t operator()(const Target& target) const noexcept
    {
        const auto h1 = std::hash<objectId>{}(target.id);
        const auto h2 = std::hash<bool>{}(target.offset_flag);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};
} // namespace optimizer::utils::points_to
#endif
