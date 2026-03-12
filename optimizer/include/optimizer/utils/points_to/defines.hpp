#ifndef OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED

#include <iostream>
#include <limits>
#include <optimizer/programIR/instruction_ir.hpp>

#include <optimizer/utils/points_to/may_value.hpp>
#include <optimizer/utils/points_to/nodes.hpp>

#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/sparse_set.hpp>

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility/invariants.hpp>

namespace optimizer::utils::points_to
{
// Pool of concrete objects
using ObjectPool = std::unordered_map<objectId, Object>;

// May state
using MayState = SparseMap<objectId, MayValue>;
// Must state
using MustState = SparseMap<objectId, Target>;

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
    MayState&                    may_in;
    const MustState&             must_in;
    const ObjectPool&            global_objects;
    const ObjectPool&            local_objects;
    const std::vector<objectId>& operands_id;
    const std::size_t            operands_count;
    const objectId               last_local_id;
};

struct MustTransferContextBundle
{
    sala::Instruction::Opcode    opcode;
    MustState&                   must_in;
    const MayState&              may_in;
    const ObjectPool&            global_objects;
    const ObjectPool&            local_objects;
    const std::vector<objectId>& operands_id;
    const std::size_t            operands_count;
    const objectId               last_local_id;
};

constexpr static inline bool is_abstract(objectId node)
{
    return node < 0;
}

static inline void nuke_must(MustState& must_state)
{
    must_state.clear();
}

static inline void nuke_may(const MayTransferContextBundle& context)
{
    for (auto& kvp_iter : context.may_in)
    {
        kvp_iter.second.make_top();
    }
}

bool is_constant_object(const ObjectPool& globals, const objectId id) noexcept;

bool is_reachable_state_target(const objectId id) noexcept;

// Reachability over the memory graph induced by may-state cells.
//
// Important:
// - We only traverse nodes that can actually have state cells.
// - Precision-loss flags are not graph nodes anymore and therefore are not traversed.
// - Summary memory objects such as OUT_OF_LOCAL_SCOPE / OUT_OF_GLOBAL_SCOPE / HEAP / ALLOCA
//   may still be traversed if can_have_state_cell(id) says yes.

bool is_objectId_reachable(const MayTransferContextBundle& context, objectId source,
                           objectId    target,
                           std::size_t max_depth = std::numeric_limits<std::size_t>::max());

void state_join_or_relaxed(MayState& A, const MayState& B);

// Strict may join used at CFG/function joins where
// "present on one path, absent on another path" must explicitly lose precision.
void state_join_or_strict(MayState& A, const MayState& B,
                          objectId extension_node = grouped_objects::MERGE_UNKNOWN);

void state_join_and(MustState& A, const MustState& B);

// Kill exact must information transitively through dereferenceable may edges.
void transitive_kill(MustState& must_in, const objectId id, const MayState& may);

void dump_may_set(const MayState& may_in);

void dump_must_set(const MustState& must_in);

// Kill must information reachable through dereferenceable may edges.
// Non-state-cell targets are ignored.
void nuke_reachable_must(objectId source, const MustTransferContextBundle& context);

// Make may information top for all reachable dereferenceable state cells.
//
// Important differences from the old version:
// - we do not fabricate cells for non-dereferenceable nodes
// - we do not traverse through non-state-cell nodes
// - constants may be skipped if requested
void nuke_reachable_may(objectId source, const MayTransferContextBundle& context,
                        bool unmodifiable_constants);

std::size_t get_relevant_operands_count(const program::InstructionIR& instruction);

} // namespace optimizer::utils::points_to

#endif
