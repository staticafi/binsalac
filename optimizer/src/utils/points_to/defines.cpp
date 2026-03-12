#include <optimizer/utils/points_to/defines.hpp>

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

// Reachability over the memory graph induced by may-state cells.
//
// Important:
// - We only traverse nodes that can actually have state cells.
// - Precision-loss flags are not graph nodes anymore and therefore are not traversed.
// - Summary memory objects such as OUT_OF_LOCAL_SCOPE / OUT_OF_GLOBAL_SCOPE / HEAP / ALLOCA
//   may still be traversed if can_have_state_cell(id) says yes.
bool is_objectId_reachable(const MayTransferContextBundle& context, objectId source,
                           const objectId target, std::size_t const max_depth)
{
    if (source == target)
    {
        return true;
    }

    std::unordered_set<objectId>                 seen;
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

        const auto it = context.may_in.find(current);
        if (it == context.may_in.end() || it->second.is_top)
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

void state_join_or_relaxed(MayState& A, const MayState& B)
{
    for (const auto& [source, transfer] : B)
    {
        auto A_source_iter = A.find(source);
        if (A_source_iter == A.end())
        {
            A.emplace(source, transfer);
        }
        else
        {
            A_source_iter->second.join_with(transfer);
        }
    }
}

// Strict may join used at CFG/function joins where
// "present on one path, absent on another path" must explicitly lose precision.
//
// Old behavior injected a fake target node such as MERGE_UNKNOWN.
// New behavior records that loss in MayValue loss flags.
void state_join_or_strict(MayState& A, const MayState& B, const objectId extension_node)
{
    const bool use_call_order_discrepancy =
            extension_node == grouped_objects::CALL_ORDER_DISCREPANCY;

    // Keys present in A but absent in B lose precision.
    for (auto& [target, pointees] : A)
    {
        if (!B.contains(target) && !pointees.is_top)
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

    // Keys present in B:
    // - if absent in A, seed from B and mark precision loss due to absence on A side
    // - if present in both, just join the MayValues
    for (const auto& [source, transfer] : B)
    {
        auto source_iter = A.find(source);
        if (source_iter == A.end())
        {
            if (transfer.is_top)
            {
                A.emplace(source, MayValue::top());
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
                A.emplace(source, std::move(seeded));
            }
        }
        else
        {
            source_iter->second.join_with(transfer);
        }
    }
}

void state_join_and(MustState& A, const MustState& B)
{
    std::vector<MustState::key_type> to_erase;
    to_erase.reserve(A.size());

    for (auto A_kvp_iter = A.begin(); A_kvp_iter != A.end(); ++A_kvp_iter)
    {
        const auto B_kvp_iter = B.find(A_kvp_iter->first);
        if (B_kvp_iter == B.end() || B_kvp_iter->second != A_kvp_iter->second)
        {
            to_erase.push_back(A_kvp_iter->first);
        }
    }

    for (const auto& elem : to_erase)
    {
        A.erase(elem);
    }
}

void transitive_kill(MustState& must_in, const objectId id, const MayState& may)
{
    std::unordered_set<objectId> seen;
    std::queue<objectId>         wl;

    if (seen.insert(id).second)
    {
        wl.push(id);
    }

    while (!wl.empty())
    {
        const auto to_kill = wl.front();
        wl.pop();

        must_in.erase(to_kill);

        const auto to_kill_may_iter = may.find(to_kill);
        if (to_kill_may_iter == may.end() || to_kill_may_iter->second.is_top)
        {
            continue;
        }

        for (const auto& child : to_kill_may_iter->second)
        {
            if (!is_reachable_state_target(child.id))
            {
                continue;
            }

            if (seen.insert(child.id).second)
            {
                wl.push(child.id);
            }
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

void dump_must_set(const MustState& must_in)
{
    std::cout << "========== DUMPING MUST =========== \n";
    for (const auto& kvp : must_in)
    {
        std::cout << kvp.first << " = " << kvp.second;
        std::cout << ";\n ";
    }
    std::cout << "<<\n";
    std::cout << "^^^^^^^^^^^^^ END ^^^^^^^^^^^^^ \n";
}

// Kill must information reachable through dereferenceable may edges.
// Non-state-cell targets are ignored.
void nuke_reachable_must(const objectId source, const MustTransferContextBundle& context)
{
    std::unordered_set<objectId> seen;
    std::queue<objectId>         wl;

    const auto source_may_iter = context.may_in.find(source);
    if (source_may_iter == context.may_in.end() || source_may_iter->second.is_top)
    {
        return;
    }

    for (const auto& target : source_may_iter->second)
    {
        if (!is_reachable_state_target(target.id))
        {
            continue;
        }

        if (seen.insert(target.id).second)
        {
            wl.push(target.id);
        }
    }

    while (!wl.empty())
    {
        const auto current = wl.front();
        wl.pop();

        context.must_in.erase(current);

        const auto current_may_iter = context.may_in.find(current);
        if (current_may_iter == context.may_in.end() || current_may_iter->second.is_top)
        {
            continue;
        }

        for (const auto& target : current_may_iter->second)
        {
            if (!is_reachable_state_target(target.id))
            {
                continue;
            }

            if (seen.insert(target.id).second)
            {
                wl.push(target.id);
            }
        }
    }
}

// Make may information top for all reachable dereferenceable state cells.
//
// Important differences from the old version:
// - we do not fabricate cells for non-dereferenceable nodes
// - we do not traverse through non-state-cell nodes
// - constants may be skipped if requested
void nuke_reachable_may(const objectId source, const MayTransferContextBundle& context,
                        const bool unmodifiable_constants)
{
    std::unordered_set<objectId> seen;
    std::queue<objectId>         wl;

    const auto source_may_iter = context.may_in.find(source);
    if (source_may_iter == context.may_in.end() || source_may_iter->second.is_top)
    {
        return;
    }

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

    while (!wl.empty())
    {
        const auto current_id = wl.front();
        wl.pop();

        auto current_may_iter = context.may_in.find(current_id);
        if (current_may_iter != context.may_in.end() && !current_may_iter->second.is_top)
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

            current_may_iter->second.make_top();
        }
        else if (current_may_iter != context.may_in.end())
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
