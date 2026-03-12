#include <optimizer/utils/points_to/may_value.hpp>

namespace optimizer::utils::points_to
{

MayValue MayValue::top() noexcept
{
    MayValue value;
    value.is_top = true;
    return value;
}

MayValue MayValue::singleton(Target target)
{
    MayValue value;
    value.targets.insert(std::move(target));
    return value;
}

bool MayValue::empty() const noexcept
{
    return !is_top && loss_flags == 0 && targets.empty();
}

std::size_t MayValue::size() const noexcept
{
    return is_top ? 0U : targets.size();
}

bool MayValue::has_loss() const noexcept
{
    return !is_top && loss_flags != 0;
}

bool MayValue::has_merge_unknown() const noexcept
{
    return !is_top && (loss_flags & static_cast<std::uint8_t>(MayLossFlag::MergeUnknown)) != 0;
}

bool MayValue::has_call_order_discrepancy() const noexcept
{
    return !is_top &&
           (loss_flags & static_cast<std::uint8_t>(MayLossFlag::CallOrderDiscrepancy)) != 0;
}

bool MayValue::is_singleton_precise_target() const noexcept
{
    return !is_top && loss_flags == 0 && targets.size() == 1;
}

bool MayValue::is_singleton_dereferenceable_target() const noexcept
{
    return is_singleton_precise_target() && can_have_state_cell(targets.begin()->id);
}

bool MayValue::is_must_fact() const noexcept
{
    return is_singleton_precise_target() && is_concrete_object(targets.begin()->id);
}

std::optional<Target> MayValue::singleton_target() const noexcept
{
    if (!is_singleton_precise_target())
    {
        return std::nullopt;
    }
    return *targets.begin();
}

std::optional<Target> MayValue::singleton_dereferenceable_target() const noexcept
{
    if (!is_singleton_dereferenceable_target())
    {
        return std::nullopt;
    }
    return *targets.begin();
}

void MayValue::clear_to_bottom() noexcept
{
    is_top     = false;
    loss_flags = 0;
    targets.clear();
}

void MayValue::make_top() noexcept
{
    is_top     = true;
    loss_flags = 0;
    targets.clear();
}

void MayValue::add_merge_unknown() noexcept
{
    if (!is_top)
    {
        loss_flags |= static_cast<std::uint8_t>(MayLossFlag::MergeUnknown);
    }
}

void MayValue::add_call_order_discrepancy() noexcept
{
    if (!is_top)
    {
        loss_flags |= static_cast<std::uint8_t>(MayLossFlag::CallOrderDiscrepancy);
    }
}

void MayValue::insert(const Target& target)
{
    if (!is_top)
    {
        targets.insert(target);
    }
}

void MayValue::insert(Target&& target)
{
    if (!is_top)
    {
        targets.insert(std::move(target));
    }
}

void MayValue::join_with(const MayValue& other)
{
    if (is_top || other.is_top)
    {
        make_top();
        return;
    }

    loss_flags |= other.loss_flags;
    targets.join_with(other.targets);
}

MayTargetSet::iterator MayValue::begin() noexcept
{
    return targets.begin();
}
MayTargetSet::iterator MayValue::end() noexcept
{
    return targets.end();
}
MayTargetSet::const_iterator MayValue::begin() const noexcept
{
    return targets.begin();
}
MayTargetSet::const_iterator MayValue::end() const noexcept
{
    return targets.end();
}
MayTargetSet::const_iterator MayValue::cbegin() const noexcept
{
    return targets.cbegin();
}
MayTargetSet::const_iterator MayValue::cend() const noexcept
{
    return targets.cend();
}

bool contains_objectId(const MayTargetSet& set, const objectId id) noexcept
{
    for (const auto& target : set)
    {
        if (target.id == id)
        {
            return true;
        }
    }
    return false;
}

bool contains_only_objectId(const MayTargetSet& set, const objectId id) noexcept
{
    if (set.empty())
    {
        return false;
    }

    for (const auto& target : set)
    {
        if (target.id != id)
        {
            return false;
        }
    }
    return true;
}

bool contains_objectId(const MayValue& value, const objectId id) noexcept
{
    if (value.is_top)
    {
        return true;
    }
    return contains_objectId(value.targets, id);
}

bool contains_only_objectId(const MayValue& value, const objectId id) noexcept
{
    if (value.is_top)
    {
        return false;
    }
    return value.loss_flags == 0 && contains_only_objectId(value.targets, id);
}

std::ostream& operator<<(std::ostream& os, const MayValue& may_value)
{
    if (may_value.is_top)
    {
        return os << "{ TOP }";
    }

    os << "{ ";
    bool first = true;

    for (const auto& target : may_value.targets)
    {
        if (!first)
            os << ", ";
        os << target;
        first = false;
    }

    if (may_value.has_merge_unknown())
    {
        if (!first)
            os << ", ";
        os << "MERGE_UNKNOWN";
        first = false;
    }

    if (may_value.has_call_order_discrepancy())
    {
        if (!first)
            os << ", ";
        os << "CALL_ORDER_DISCREPANCY";
    }

    return os << " }";
}

bool operator==(const MayValue& a, const MayValue& b)
{
    if (a.is_top != b.is_top)
    {
        return false;
    }
    if (a.is_top)
    {
        return true;
    }
    return a.loss_flags == b.loss_flags && a.targets == b.targets;
}

} // namespace optimizer::utils::points_to
