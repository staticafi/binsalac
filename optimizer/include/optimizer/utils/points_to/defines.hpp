#ifndef OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED

#include <limits>
#include <optimizer/programIR/instruction_ir.hpp>

#include <optimizer/utils/points_to/may_value.hpp>
#include <optimizer/utils/points_to/nodes.hpp>

#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/sparse_set.hpp>

#include <optimizer/utils/common.hpp>
#include <span>
#include <utility/invariants.hpp>

namespace optimizer::utils::points_to
{
using ObjectPool = utils::SparseMap<objectId, Object>;

using MayState = SparseMap<objectId, MayValue>;

using UniqueState = SparseMap<objectId, Target>;

struct MayAnalysisState
{
    MayState    may{};
    UniqueState unique{};
    bool        poisoned{false};

    void clear() noexcept
    {
        may.clear();
        unique.clear();
        poisoned = false;
    }

    bool operator==(const MayAnalysisState& other) const
    {
        if (poisoned || other.poisoned)
        {
            return poisoned == other.poisoned;
        }

        return unique == other.unique && may == other.may;
    }
};

struct ProgramPoint
{
    std::size_t function;
    std::size_t bb;
    std::size_t instr;
};

struct MayTransferContextBundle
{
    ProgramPoint                 pp;
    sala::Instruction::Opcode    opcode;
    MayAnalysisState&            state;
    const ObjectPool&            global_objects;
    const ObjectPool&            local_objects;
    const std::vector<objectId>& operands_id;
    const std::size_t            operands_count;
    const objectId               last_local_id;
};

void dump_may_set(const MayState& may_in);
bool is_constant_object(const ObjectPool& globals, const objectId id) noexcept;

bool is_reachable_state_target(const objectId id) noexcept;

bool is_objectId_reachable(const MayTransferContextBundle& context, objectId source,
                           objectId    target,
                           std::size_t max_depth = std::numeric_limits<std::size_t>::max());

void state_join_cfg(MayAnalysisState& A, const MayAnalysisState& B);

void merge_state(MayAnalysisState& target, const MayAnalysisState& source, bool& initialized);

void dump_may_set(const MayAnalysisState& state);

void dump_unique_set(const UniqueState& must_in);
void dump_must_set(const MayAnalysisState& state);

void handle_call_boundary(std::span<const objectId>       escaped_args,
                          const MayTransferContextBundle& context, bool unmodifiable_constants);

std::size_t get_relevant_operands_count(const program::InstructionIR& instruction);

bool is_poisoned(const MayAnalysisState& state) noexcept;

void poison_may_state(MayAnalysisState& state) noexcept;

void poison_may(const MayTransferContextBundle& context);

bool is_globally_visible_target(const ObjectPool& global_objects_, objectId id);

constexpr static inline bool is_abstract(objectId node)
{
    return node < 0;
}

} // namespace optimizer::utils::points_to

#endif
