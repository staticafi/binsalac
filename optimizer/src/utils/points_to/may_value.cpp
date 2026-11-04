#include <optimizer/utils/points_to/may_value.hpp>

#include <algorithm>

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
    value.insert(target);
    return value;
}

bool MayValue::empty() const noexcept
{
    return !is_top && targets().empty();
}

std::size_t MayValue::size() const noexcept
{
    return is_top ? 0U : targets().size();
}

void MayValue::clear_to_bottom() noexcept
{
    is_top = false;
    targets_.reset();
}

void MayValue::make_top() noexcept
{
    is_top = true;
    targets_.reset();
}

void MayValue::insert(const Target& target)
{
    if (!is_top && !targets().contains(target))
        writable_targets().insert(target);
}

const MayTargetSet& MayValue::targets() const noexcept
{
    static const MayTargetSet empty;
    return targets_ ? *targets_ : empty;
}

MayTargetSet& MayValue::writable_targets()
{
    if (!targets_)
        targets_ = std::make_shared<MayTargetSet>();
    else if (!targets_.unique())
        targets_ = std::make_shared<MayTargetSet>(*targets_);

    return *targets_;
}

void MayValue::join_with(const MayValue& other)
{
    if (is_top || other.is_top)
    {
        make_top();
        return;
    }

    if (targets_ == other.targets_ || other.empty())
        return;

    if (empty())
    {
        targets_ = other.targets_;
        return;
    }

    // A join that contributes no new targets must not detach a shared payload.
    if (!std::includes(begin(), end(), other.begin(), other.end()))
        writable_targets().join_with(other.targets());
}

MayTargetSet::const_iterator MayValue::begin() const noexcept
{
    return targets().begin();
}
MayTargetSet::const_iterator MayValue::end() const noexcept
{
    return targets().end();
}
MayTargetSet::const_iterator MayValue::cbegin() const noexcept
{
    return targets().cbegin();
}
MayTargetSet::const_iterator MayValue::cend() const noexcept
{
    return targets().cend();
}

bool contains_objectId(const MayTargetSet& set, const objectId id) noexcept
{
    return std::any_of(set.begin(), set.end(),
        [id](const Target& target) { return target.id == id; });
}

bool contains_only_objectId(const MayTargetSet& set, const objectId id) noexcept
{
    return !set.empty() && std::all_of(set.begin(), set.end(),
        [id](const Target& target) { return target.id == id; });
}

bool contains_objectId(const MayValue& value, const objectId id) noexcept
{
    if (value.is_top)
        return true;

    return contains_objectId(value.targets(), id);
}

bool contains_only_objectId(const MayValue& value, const objectId id) noexcept
{
    if (value.is_top)
        return false;

    return contains_only_objectId(value.targets(), id);
}

std::ostream& operator<<(std::ostream& os, const MayValue& may_value)
{
    if (may_value.is_top)
    {
        return os << "{ TOP }";
    }

    os << "{ ";
    bool first = true;

    for (const auto& target : may_value)
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

    return a.targets_ == b.targets_ || a.targets() == b.targets();
}
} // namespace optimizer::utils::points_to
