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
    return !is_top && targets.empty();
}

std::size_t MayValue::size() const noexcept
{
    return is_top ? 0U : targets.size();
}

void MayValue::clear_to_bottom() noexcept
{
    is_top = false;
    targets.clear();
}

void MayValue::make_top() noexcept
{
    is_top = true;
    targets.clear();
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

    return contains_only_objectId(value.targets, id);
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
        {
            os << ", ";
        }

        os << target;
        first = false;
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

    return a.targets == b.targets;
}
} // namespace optimizer::utils::points_to
