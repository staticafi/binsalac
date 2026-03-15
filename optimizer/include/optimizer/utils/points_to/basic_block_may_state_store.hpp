#ifndef OPTIMIZER_UTILS_POINTS_TO_BASIC_BLOCK_MAY_STATE_STORE_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_BASIC_BLOCK_MAY_STATE_STORE_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <optimizer/utils/points_to/defines.hpp>

namespace optimizer::utils::points_to
{
using StateId = std::uint32_t;

class BasicBlockMayStateStore
{
  public:
    BasicBlockMayStateStore();

    [[nodiscard]] StateId empty_id() const noexcept;

    [[nodiscard]] StateId intern(const MayState& state);
    [[nodiscard]] StateId intern(MayState&& state);

    [[nodiscard]] const MayState& get(StateId id) const;
    [[nodiscard]] bool            equals(StateId id, const MayState& state) const;

    [[nodiscard]] std::size_t unique_state_count() const noexcept;

  private:
    [[nodiscard]] StateId checked_next_id_() const;

  private:
    std::vector<MayState>                                 states_{};
    std::unordered_map<std::size_t, std::vector<StateId>> buckets_{};
};

class BasicBlockMayStateSlots
{
  public:
    BasicBlockMayStateSlots();
    explicit BasicBlockMayStateSlots(std::shared_ptr<BasicBlockMayStateStore> store);

    void reset(std::size_t n);

    [[nodiscard]] std::size_t     size() const noexcept;
    [[nodiscard]] bool            has(std::size_t bb_index) const;
    [[nodiscard]] StateId         id(std::size_t bb_index) const;
    [[nodiscard]] const MayState& get(std::size_t bb_index) const;
    [[nodiscard]] bool            equals(std::size_t bb_index, const MayState& state) const;

    bool set(std::size_t bb_index, const MayState& state);
    bool set(std::size_t bb_index, MayState&& state);

    [[nodiscard]] std::shared_ptr<BasicBlockMayStateStore> store() const noexcept;

  private:
    bool assign_id_(std::size_t bb_index, StateId new_id);

  private:
    std::shared_ptr<BasicBlockMayStateStore> store_;
    std::vector<StateId>                     ids_{};
    std::vector<bool>                        has_state_{};
};

} // namespace optimizer::utils::points_to

#endif
