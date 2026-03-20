#include "optimizer/utils/points_to/nodes.hpp"
#include <optimizer/utils/points_to/defines.hpp>

#include <iostream>
#include <queue>
#include <set>

namespace optimizer::utils::points_to
{

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
void handle_call_boundary(const objectId escapee, const MayTransferContextBundle& context,
                          const bool unmodifiable_constants)
{
    if (context.state.poisoned)
    {
        return;
    }

    const auto escapee_may_iter = context.state.may.find(escapee);
    if (escapee_may_iter == context.state.may.end())
    {
        return;
    }

    if (escapee_may_iter->second.is_top)
    {
        return;
    }

    utils::SparseSet<objectId> reachable;
    utils::SparseSet<objectId> seen;
    std::queue<objectId>       wl;

    // first indirection (we cannot modify values of these)
    for (const auto& target : escapee_may_iter->second)
    {
        if (!is_reachable_state_target(target.id))
        {
            continue;
        }

        if (unmodifiable_constants && is_constant_object(context.global_objects, target.id))
        {
            continue;
        }

        if (seen.insert(target.id).second)
        {
            wl.push(target.id);
        }
    }

    while (wl.empty())
    {
        const auto reachable_current = wl.front();
        wl.pop();

        if (is_reachable_state_target(reachable_current))
        {
            reachable.insert(reachable_current);
        }
    }
}

void handle_call_boundary_s(const objectId source, const MayTransferContextBundle& context,
                            const bool unmodifiable_constants)
{
    if (context.state.poisoned)
    {
        return;
    }

    const auto source_may_iter = context.state.may.find(source);
    if (source_may_iter == context.state.may.end())
    {
        return;
    }

    if (source_may_iter->second.is_top)
    {
        return;
    }

    std::set<objectId>   seen;
    std::queue<objectId> wl;

    // first indirection
    for (const auto& target : source_may_iter->second)
    {
        if (!is_reachable_state_target(target.id))
        {
            continue;
        }

        if (unmodifiable_constants && is_constant_object(context.global_objects, target.id))
        {
            continue;
        }

        if (seen.insert(target.id).second)
        {
            wl.push(target.id);
        }
    }

    // TODO: Improve precision with OUT_OF_LOCAL_SCOPE (and all the other reachables )
    while (!wl.empty())
    {
        const auto current_id = wl.front();
        wl.pop();

        auto current_may_iter = context.state.may.find(current_id);

        if (current_may_iter != context.state.may.end() && !current_may_iter->second.is_top)
        {
            const auto current_copy = current_may_iter->second;

            for (const auto& target : current_copy)
            {
                if (!is_reachable_state_target(target.id))
                {
                    continue;
                }

                if (unmodifiable_constants && is_constant_object(context.global_objects, target.id))
                {
                    continue;
                }

                if (seen.insert(target.id).second)
                {
                    wl.push(target.id);
                }
            }

            // TODO : set to
            current_may_iter->second.insert(
                    Target{.id = grouped_objects::OUT_OF_LOCAL_SCOPE, .offset_flag = false});
        }
        else if (current_may_iter != context.state.may.end())
        {
            current_may_iter->second.make_top();
        }
    }
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
