#ifndef OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED

#include <limits>
#include <optimizer/programIR/instruction_ir.hpp>

#include <optimizer/utils/points_to/may_value.hpp>
#include <optimizer/utils/points_to/nodes.hpp>

#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/sparse_set.hpp>

#include <optimizer/utils/common.hpp>
#include <unordered_map>
#include <utility/invariants.hpp>

namespace optimizer::utils::points_to
{
using ObjectPool = std::unordered_map<objectId, Object>;

// Raw sparse points-to map.
// Missing key == bottom for that particular cell.
using MayState = SparseMap<objectId, MayValue>;

// Solver / stored abstract state.
// `poisoned == true` means: whole points-to state is invalidated.
struct MayAnalysisState
{
    MayState may{};
    bool     poisoned{false};

    void clear() noexcept
    {
        may.clear();
        poisoned = false;
    }

    bool operator==(const MayAnalysisState& other) const
    {
        return std::tie(poisoned, may) == std::tie(other.poisoned, other.may);
    };
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

void state_join_or_relaxed(MayAnalysisState& A, const MayAnalysisState& B);

void state_join_or_strict(MayAnalysisState& A, const MayAnalysisState& B,
                          objectId extension_node = grouped_objects::MERGE_UNKNOWN);

void dump_may_set(const MayAnalysisState& state);

void dump_must_set(const MayState& must_in);
void dump_must_set(const MayAnalysisState& state);

void handle_call_boundary(objectId source, const MayTransferContextBundle& context,
                          bool unmodifiable_constants);

std::size_t get_relevant_operands_count(const program::InstructionIR& instruction);

constexpr static inline bool is_abstract(objectId node)
{
    return node < 0;
}

[[nodiscard]] inline bool is_poisoned(const MayAnalysisState& state) noexcept
{
    return state.poisoned;
}

inline void poison_may_state(MayAnalysisState& state) noexcept
{
    state.may.clear();
    state.poisoned = true;
}

static inline void poison_may(const MayTransferContextBundle& context)
{
    std::cout << "POISINING MAY " << instruction_opcode_to_string(context.opcode) << " ";
    for (size_t op = 0; op < context.operands_count; ++op)
    {
        std::cout << context.operands_id[op] << " ";
    }
    std::cout << "\n";
    dump_may_set(context.state);
    poison_may_state(context.state);
}

} // namespace optimizer::utils::points_to

#endif
