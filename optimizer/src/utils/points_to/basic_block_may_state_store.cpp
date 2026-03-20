#include <optimizer/utils/points_to/basic_block_may_state_store.hpp>

#include <limits>
#include <stdexcept>
#include <utility/hash_combine.hpp>

namespace optimizer::utils::points_to
{
namespace
{
inline std::size_t hash_may_map(const MayState& state) noexcept
{
    std::size_t seed = std::hash<std::size_t>{}(state.size());
    for (const auto& [object_id, may_value] : state)
    {
        hash_combine(seed, std::hash<objectId>{}(object_id));
        hash_combine(seed, std::hash<MayValue>{}(may_value));
    }
    return seed;
}

inline std::size_t hash_may_state(const MayAnalysisState& state) noexcept
{
    std::size_t seed = std::hash<bool>{}(state.poisoned);
    if (!state.poisoned)
    {
        hash_combine(seed, hash_may_map(state.may));
    }
    return seed;
}
} // namespace

BasicBlockMayStateStore::BasicBlockMayStateStore()
{
    states_.emplace_back();
    buckets_[hash_may_state(states_.front())].push_back(0);
}

StateId BasicBlockMayStateStore::empty_id() const noexcept
{
    return 0;
}

StateId BasicBlockMayStateStore::intern(const MayAnalysisState& state)
{
    const auto hash   = hash_may_state(state);
    auto&      bucket = buckets_[hash];

    for (const StateId id : bucket)
    {
        if (states_[id] == state)
        {
            return id;
        }
    }

    const StateId new_id = checked_next_id_();
    states_.push_back(state);
    bucket.push_back(new_id);
    return new_id;
}

StateId BasicBlockMayStateStore::intern(MayAnalysisState&& state)
{
    const auto hash   = hash_may_state(state);
    auto&      bucket = buckets_[hash];

    for (const StateId id : bucket)
    {
        if (states_[id] == state)
        {
            return id;
        }
    }

    const StateId new_id = checked_next_id_();
    states_.push_back(std::move(state));
    bucket.push_back(new_id);
    return new_id;
}

const MayAnalysisState& BasicBlockMayStateStore::get(const StateId id) const
{
    if (id >= states_.size())
    {
        throw std::out_of_range("BasicBlockMayStateStore::get: invalid StateId");
    }

    return states_[id];
}

bool BasicBlockMayStateStore::equals(const StateId id, const MayAnalysisState& state) const
{
    return get(id) == state;
}

std::size_t BasicBlockMayStateStore::unique_state_count() const noexcept
{
    return states_.size();
}

StateId BasicBlockMayStateStore::checked_next_id_() const
{
    const auto next = states_.size();
    if (next > static_cast<std::size_t>(std::numeric_limits<StateId>::max()))
    {
        throw std::overflow_error("BasicBlockMayStateStore: too many interned states");
    }
    return static_cast<StateId>(next);
}

BasicBlockMayStateSlots::BasicBlockMayStateSlots()
    : store_{std::make_shared<BasicBlockMayStateStore>()}
{
}

BasicBlockMayStateSlots::BasicBlockMayStateSlots(std::shared_ptr<BasicBlockMayStateStore> store)
    : store_{std::move(store)}
{
    if (!store_)
    {
        throw std::invalid_argument("BasicBlockMayStateSlots: store must not be null");
    }
}

void BasicBlockMayStateSlots::reset(const std::size_t n)
{
    ids_.assign(n, store_->empty_id());
    has_state_.assign(n, false);
}

std::size_t BasicBlockMayStateSlots::size() const noexcept
{
    return ids_.size();
}

bool BasicBlockMayStateSlots::has(const std::size_t bb_index) const
{
    return has_state_.at(bb_index);
}

StateId BasicBlockMayStateSlots::id(const std::size_t bb_index) const
{
    return ids_.at(bb_index);
}

const MayAnalysisState& BasicBlockMayStateSlots::get(const std::size_t bb_index) const
{
    return store_->get(ids_.at(bb_index));
}

bool BasicBlockMayStateSlots::equals(const std::size_t       bb_index,
                                     const MayAnalysisState& state) const
{
    return has(bb_index) && store_->equals(ids_.at(bb_index), state);
}

bool BasicBlockMayStateSlots::set(const std::size_t bb_index, const MayAnalysisState& state)
{
    return assign_id_(bb_index, store_->intern(state));
}

bool BasicBlockMayStateSlots::set(const std::size_t bb_index, MayAnalysisState&& state)
{
    return assign_id_(bb_index, store_->intern(std::move(state)));
}

std::shared_ptr<BasicBlockMayStateStore> BasicBlockMayStateSlots::store() const noexcept
{
    return store_;
}

bool BasicBlockMayStateSlots::assign_id_(const std::size_t bb_index, const StateId new_id)
{
    const bool changed      = !has_state_.at(bb_index) || ids_.at(bb_index) != new_id;
    ids_.at(bb_index)       = new_id;
    has_state_.at(bb_index) = true;
    return changed;
}

} // namespace optimizer::utils::points_to
