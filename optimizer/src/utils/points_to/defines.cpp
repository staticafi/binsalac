#include <optimizer/utils/points_to/defines.hpp>
#include <optimizer/utils/points_to/nodes.hpp>

#include <iostream>
#include <queue>
#include <set>
#include <span>
#include <vector>

namespace optimizer::utils::points_to
{

namespace
{
bool is_modifiable_target_for_call_boundary(const objectId                  id,
                                            const MayTransferContextBundle& context,
                                            const bool                      unmodifiable_constants)
{
    if (id == grouped_objects::OUT_OF_LOCAL_SCOPE)
    {
        return false;
    }

    if (!is_reachable_state_target(id))
    {
        return false;
    }

    if (unmodifiable_constants && is_constant_object(context.global_objects, id))
    {
        return false;
    }

    return true;
}

bool is_observable_pointer_for_call_boundary(const objectId id)
{
    return id != grouped_objects::OUT_OF_LOCAL_SCOPE;
}

struct ObservablePayloadResult
{
    MayValue payload{};
    bool     saw_top{false};
};

struct ModifiableCellsResult
{
    std::vector<objectId> cells{};
    bool                  saw_top{false};
};

ObservablePayloadResult
build_observable_payload_for_call_boundary(const std::span<const objectId> escaped_args,
                                           const MayTransferContextBundle& context)
{
    ObservablePayloadResult result{};
    result.payload = make_singleton(grouped_objects::OUT_OF_LOCAL_SCOPE, false);

    utils::SparseSet<objectId> observable_seen;
    std::queue<objectId>       observable_wl;

    for (const objectId escapee : escaped_args)
    {
        if (observable_seen.insert(escapee).second)
        {
            observable_wl.push(escapee);
        }
    }

    while (!observable_wl.empty())
    {
        const objectId current = observable_wl.front();
        observable_wl.pop();

        if (is_observable_pointer_for_call_boundary(current))
        {
            result.payload.insert(Target{current, false});
        }

        const auto it = context.state.may.find(current);
        if (it == context.state.may.end())
        {
            continue;
        }

        if (it->second.is_top)
        {
            result.payload.make_top();
            result.saw_top = true;
            return result;
        }

        for (const auto& next : it->second)
        {
            if (!is_observable_pointer_for_call_boundary(next.id))
            {
                continue;
            }

            result.payload.insert(next);

            if (observable_seen.insert(next.id).second)
            {
                observable_wl.push(next.id);
            }
        }
    }

    return result;
}

std::vector<objectId>
collect_all_modifiable_cells_for_call_boundary(const MayTransferContextBundle& context,
                                               const bool unmodifiable_constants)
{
    std::vector<objectId>      result;
    utils::SparseSet<objectId> seen;

    const auto add_if_modifiable = [&](objectId id)
    {
        if (seen.contains(id))
        {
            return;
        }

        if (is_modifiable_target_for_call_boundary(id, context, unmodifiable_constants))
        {
            seen.insert(id);
            result.push_back(id);
        }
    };

    for (const auto& [id, _] : context.global_objects)
    {
        add_if_modifiable(id);
    }

    for (const auto& [id, _] : context.local_objects)
    {
        add_if_modifiable(id);
    }

    add_if_modifiable(grouped_objects::OUT_OF_GLOBAL_SCOPE);
    add_if_modifiable(grouped_objects::VARGARG_BLOCK);
    add_if_modifiable(grouped_objects::ALLOCA);
    add_if_modifiable(grouped_objects::HEAP);

    return result;
}

ModifiableCellsResult
collect_modifiable_cells_for_call_boundary(const std::span<const objectId> escaped_args,
                                           const MayTransferContextBundle& context,
                                           const bool                      unmodifiable_constants)
{
    ModifiableCellsResult result{};

    utils::SparseSet<objectId> mod_seen;
    std::queue<objectId>       mod_wl;

    for (const objectId escapee : escaped_args)
    {
        const auto it = context.state.may.find(escapee);
        if (it == context.state.may.end())
        {
            continue;
        }

        if (it->second.is_top)
        {
            result.saw_top = true;
            result.cells =
                    collect_all_modifiable_cells_for_call_boundary(context, unmodifiable_constants);
            return result;
        }

        for (const auto& tgt : it->second)
        {
            if (is_modifiable_target_for_call_boundary(tgt.id, context, unmodifiable_constants) &&
                mod_seen.insert(tgt.id).second)
            {
                mod_wl.push(tgt.id);
            }
        }
    }

    while (!mod_wl.empty())
    {
        const objectId current = mod_wl.front();
        mod_wl.pop();

        result.cells.push_back(current);

        const auto it = context.state.may.find(current);
        if (it == context.state.may.end())
        {
            continue;
        }

        if (it->second.is_top)
        {
            result.saw_top = true;
            result.cells =
                    collect_all_modifiable_cells_for_call_boundary(context, unmodifiable_constants);
            return result;
        }

        for (const auto& next : it->second)
        {
            if (is_modifiable_target_for_call_boundary(next.id, context, unmodifiable_constants) &&
                mod_seen.insert(next.id).second)
            {
                mod_wl.push(next.id);
            }
        }
    }

    return result;
}

void apply_observable_payload_to_modifiable_cells(const std::vector<objectId>& mod_cells,
                                                  const MayValue&              observable_payload,
                                                  const MayTransferContextBundle& context)
{
    for (const objectId id : mod_cells)
    {
        auto it = context.state.may.find(id);
        if (it == context.state.may.end())
        {
            context.state.may.emplace(id, observable_payload);
        }
        else
        {
            it->second.join_with(observable_payload);
        }

        // Calls are weak unknown external effects. Exactness of modified cells is lost.
        context.state.must.erase(id);
    }
}
} // namespace

bool is_constant_object(const ObjectPool& globals, const objectId id) noexcept
{
    const auto it = globals.find(id);
    return it != globals.end() && it->second.region == RegionTag::Constant;
}

bool is_reachable_state_target(const objectId id) noexcept
{
    return can_have_state_cell(id);
}

bool is_objectId_reachable(const MayTransferContextBundle& context, objectId source,
                           const objectId target, std::size_t const max_depth)
{
    if (context.state.poisoned)
    {
        return true;
    }

    if (source == target)
    {
        return true;
    }

    std::set<objectId>                           seen;
    std::queue<std::pair<objectId, std::size_t>> wl;

    seen.insert(source);
    wl.emplace(source, 0);

    while (!wl.empty())
    {
        const auto [current, depth] = wl.front();
        wl.pop();

        if (depth >= max_depth)
        {
            continue;
        }

        const auto it = context.state.may.find(current);
        if (it == context.state.may.end() || it->second.is_top)
        {
            continue;
        }

        for (const auto& next : it->second)
        {
            const objectId nid = next.id;

            if (nid == target)
            {
                return true;
            }

            if (!is_reachable_state_target(nid))
            {
                continue;
            }

            if (seen.insert(nid).second)
            {
                wl.emplace(nid, depth + 1);
            }
        }
    }

    return false;
}

void must_join_and(MustState& A, const MustState& B)
{
    A.intersect_keep_equal_with(B, [](const Target& lhs, const Target& rhs) { return lhs == rhs; });
}

void state_join_cfg(MayAnalysisState& A, const MayAnalysisState& B)
{
    if (A.poisoned || B.poisoned)
    {
        poison_may_state(A);
        return;
    }

    // May component: ordinary union/top join.
    A.may.union_join_with(B.may, [](MayValue& lhs, const MayValue& rhs) { lhs.join_with(rhs); });

    // Must component: strict agreement.
    // A must fact survives only when both incoming states contain the same fact.
    must_join_and(A.must, B.must);
}

void dump_may_set(const MayState& may_in)
{
    std::cout << "========== DUMPING MAY =========== \n";
    for (const auto& kvp : may_in)
    {
        std::cout << kvp.first << " = " << kvp.second;
        std::cout << ";\n ";
    }
    std::cout << "<<\n";
    std::cout << "^^^^^^^^^^^^^ END ^^^^^^^^^^^^^ \n";
}

void dump_may_set(const MayAnalysisState& state)
{
    if (state.poisoned)
    {
        std::cout << "========== DUMPING MAY ===========" << "\n";
        std::cout << "POISONED\n";
        std::cout << "<<\n";
        std::cout << "^^^^^^^^^^^^^ END ^^^^^^^^^^^^^ \n";
        return;
    }

    dump_may_set(state.may);
}

void dump_must_set(const MustState& must_in)
{
    std::cout << "========== DUMPING MUST =========== \n";
    for (const auto& [cell, target] : must_in)
    {
        std::cout << cell << " = " << target << ";\n ";
    }
    std::cout << "<<\n";
    std::cout << "^^^^^^^^^^^^^ END ^^^^^^^^^^^^^ \n";
}

void dump_must_set(const MayAnalysisState& state)
{
    if (state.poisoned)
    {
        std::cout << "========== DUMPING MAY =========== \n";
        std::cout << "POISONED\n";
        std::cout << "<<\n";
        std::cout << "^^^^^^^^^^^^^ END ^^^^^^^^^^^^^ \n";
        return;
    }

    dump_must_set(state.must);
}

void merge_state(MayAnalysisState& target, const MayAnalysisState& source, bool& initialized)
{
    if (!initialized)
    {
        target      = source;
        initialized = true;
        return;
    }

    state_join_cfg(target, source);
}

void handle_call_boundary(const std::span<const objectId> escaped_args,
                          const MayTransferContextBundle& context,
                          const bool                      unmodifiable_constants)
{
    if (context.state.poisoned)
    {
        return;
    }

    auto observable = build_observable_payload_for_call_boundary(escaped_args, context);

    auto modifiable = collect_modifiable_cells_for_call_boundary(escaped_args, context,
                                                                 unmodifiable_constants);

    if (observable.saw_top || modifiable.saw_top)
    {
        observable.payload.make_top();
        modifiable.cells =
                collect_all_modifiable_cells_for_call_boundary(context, unmodifiable_constants);
    }

    apply_observable_payload_to_modifiable_cells(modifiable.cells, observable.payload, context);
}

std::size_t get_relevant_operands_count(const program::InstructionIR& instruction)
{
    switch (instruction.get_opcode())
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::RET:
        return 0;

    case sala::Instruction::Opcode::ADDRESS:
    case sala::Instruction::Opcode::LOAD:
    case sala::Instruction::Opcode::STORE:
    case sala::Instruction::Opcode::COPY:
    case sala::Instruction::Opcode::P2I:
    case sala::Instruction::Opcode::I2P:
    case sala::Instruction::Opcode::MEMCPY:
    case sala::Instruction::Opcode::MEMMOVE:
    case sala::Instruction::Opcode::MOVEPTR:
        return 2;

    case sala::Instruction::Opcode::VA_ARG:
    case sala::Instruction::Opcode::VA_COPY:
        return 2;

    case sala::Instruction::Opcode::VA_START:
    case sala::Instruction::Opcode::VA_END:
    case sala::Instruction::Opcode::MEMSET:
    case sala::Instruction::Opcode::ALLOCA:
    case sala::Instruction::Opcode::MALLOC:
    case sala::Instruction::Opcode::STACKRESTORE:
    case sala::Instruction::Opcode::ADD:
    case sala::Instruction::Opcode::SUB:
    case sala::Instruction::Opcode::MUL:
    case sala::Instruction::Opcode::DIV:
    case sala::Instruction::Opcode::REM:
    case sala::Instruction::Opcode::AND:
    case sala::Instruction::Opcode::OR:
    case sala::Instruction::Opcode::XOR:
    case sala::Instruction::Opcode::SHL:
    case sala::Instruction::Opcode::SHR:
    case sala::Instruction::Opcode::NEG:
    case sala::Instruction::Opcode::EXTEND:
    case sala::Instruction::Opcode::TRUNCATE:
    case sala::Instruction::Opcode::F2I:
    case sala::Instruction::Opcode::I2F:
    case sala::Instruction::Opcode::LESS:
    case sala::Instruction::Opcode::LESS_EQUAL:
    case sala::Instruction::Opcode::GREATER:
    case sala::Instruction::Opcode::GREATER_EQUAL:
    case sala::Instruction::Opcode::EQUAL:
    case sala::Instruction::Opcode::UNEQUAL:
    case sala::Instruction::Opcode::ISNAN:
        return 1;

    case sala::Instruction::Opcode::STACKSAVE:
    case sala::Instruction::Opcode::CALL:
        return instruction.get_operands().size();

    case sala::Instruction::Opcode::FREE:
        return 0;

    default:
        UNREACHABLE();
    }
}
} // namespace optimizer::utils::points_to
