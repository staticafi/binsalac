#include "optimizer/utils/points_to/nodes.hpp"
#include <optimizer/utils/points_to/defines.hpp>

#include <iostream>
#include <queue>
#include <set>
#include <span>

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

MayValue build_observable_payload_for_call_boundary(const std::span<const objectId> escaped_args,
                                                    const MayTransferContextBundle& context)
{
    MayValue observable_payload = make_singleton(grouped_objects::OUT_OF_LOCAL_SCOPE, false);

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
            observable_payload.insert(Target{current, false});
        }

        const auto it = context.state.may.find(current);
        if (it == context.state.may.end() || it->second.is_top)
        {
            continue;
        }

        for (const auto& next : it->second)
        {
            if (!is_observable_pointer_for_call_boundary(next.id))
            {
                continue;
            }

            observable_payload.insert(next);

            if (observable_seen.insert(next.id).second)
            {
                observable_wl.push(next.id);
            }
        }
    }

    return observable_payload;
}

std::vector<objectId>
collect_modifiable_cells_for_call_boundary(const std::span<const objectId> escaped_args,
                                           const MayTransferContextBundle& context,
                                           const bool                      unmodifiable_constants)
{
    utils::SparseSet<objectId> mod_seen;
    std::queue<objectId>       mod_wl;
    std::vector<objectId>      mod_cells;

    for (const objectId escapee : escaped_args)
    {
        const auto it = context.state.may.find(escapee);
        if (it == context.state.may.end() || it->second.is_top)
        {
            continue;
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

        mod_cells.push_back(current);

        const auto it = context.state.may.find(current);
        if (it == context.state.may.end() || it->second.is_top)
        {
            continue;
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

    return mod_cells;
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
        else if (!it->second.is_top)
        {
            it->second.join_with(observable_payload);
        }
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

void state_join_or_relaxed(MayAnalysisState& A, const MayAnalysisState& B)
{
    if (A.poisoned || B.poisoned)
    {
        poison_may_state(A);
        return;
    }

    for (const auto& [source, transfer] : B.may)
    {
        auto A_source_iter = A.may.find(source);
        if (A_source_iter == A.may.end())
        {
            A.may.emplace(source, transfer);
        }
        else
        {
            A_source_iter->second.join_with(transfer);
        }
    }
}

void state_join_or_strict(MayAnalysisState& A, const MayAnalysisState& B,
                          const objectId extension_node)
{
    if (A.poisoned || B.poisoned)
    {
        poison_may_state(A);
        return;
    }

    const bool use_call_order_discrepancy =
            extension_node == grouped_objects::CALL_ORDER_DISCREPANCY;

    // Keys present in A but absent in B lose precision.
    for (auto& [target, pointees] : A.may)
    {
        if (!B.may.contains(target) && !pointees.is_top)
        {
            if (use_call_order_discrepancy)
            {
                pointees.add_call_order_discrepancy();
            }
            else
            {
                pointees.add_merge_unknown();
            }
        }
    }

    for (const auto& [source, transfer] : B.may)
    {
        auto source_iter = A.may.find(source);
        if (source_iter == A.may.end())
        {
            if (transfer.is_top)
            {
                A.may.emplace(source, MayValue::top());
            }
            else
            {
                MayValue seeded = transfer;
                if (use_call_order_discrepancy)
                {
                    seeded.add_call_order_discrepancy();
                }
                else
                {
                    seeded.add_merge_unknown();
                }
                A.may.insert_or_assign(source, std::move(seeded));
            }
        }
        else
        {
            source_iter->second.join_with(transfer);
        }
    }
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

void dump_must_set(const MayState& may_in)
{
    std::cout << "========== DUMPING MAY =========== \n";
    for (const auto& kvp : may_in)
    {
        const auto must_fact = kvp.second.must_fact();
        if (must_fact.has_value())
        {
            std::cout << kvp.first << " = " << must_fact.value();
        }
        else
        {
            std::cout << kvp.first << " = nan";
        }
        std::cout << ";\n ";
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

    dump_must_set(state.may);
}

void handle_call_boundary(const std::span<const objectId> escaped_args,
                          const MayTransferContextBundle& context,
                          const bool                      unmodifiable_constants)
{
    if (context.state.poisoned)
    {
        return;
    }

    const MayValue observable_payload =
            build_observable_payload_for_call_boundary(escaped_args, context);

    const std::vector<objectId> mod_cells = collect_modifiable_cells_for_call_boundary(
            escaped_args, context, unmodifiable_constants);

    apply_observable_payload_to_modifiable_cells(mod_cells, observable_payload, context);
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
