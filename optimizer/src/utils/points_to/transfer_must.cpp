#include <optimizer/utils/points_to/transfer_must.hpp>

#include <optimizer/utils/points_to/defines.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::utils::points_to
{
namespace
{

// Return true iff this may value contains at least one target that can denote a memory cell.
inline bool may_has_dereferenceable_target(const MayValue& value) noexcept
{
    if (value.is_top)
    {
        return false;
    }

    for (const auto& target : value)
    {
        if (can_have_state_cell(target.id))
        {
            return true;
        }
    }

    return false;
}

// Remove exact must information for every dereferenceable object that may be pointed to by
// `may_value`.
//
// This is used for weak updates: if an instruction may write through a pointer that can
// reference multiple objects, we cannot preserve exact must points-to information for any
// of those possible targets.
//
// Non-dereferenceable targets (for example FUNCTION) are ignored because they cannot denote
// memory cells.
inline void erase_must_for_all_may_pointees(MustState& must_state, const MayValue& may_value)
{
    ASSUMPTION(!may_value.is_top);

    for (const auto& target : may_value)
    {
        if (!can_have_state_cell(target.id))
        {
            continue;
        }

        must_state.erase(target.id);
    }
}

// Return true iff using `id` as a memory access base is invalid for must reasoning.
//
// In the must analysis, exact memory reasoning is invalid when the corresponding may
// information is either:
// - absent,
// - top, or
// - finite but contains no dereferenceable target.
//
// The last case matters in the new design because a finite may value may still contain only
// non-memory symbolic targets such as FUNCTION.
inline bool may_access_is_invalid_for_must(const MustTransferContextBundle& context,
                                           const objectId                   id) noexcept
{
    const auto it = context.may_in.find(id);
    return it == context.may_in.end() || it->second.is_top ||
           !may_has_dereferenceable_target(it->second);
}

// Return true iff exact must dereference through `target` is trusted.
//
// In the new design we keep summary targets and concrete objects separate. A summary target
// may be dereferenceable for may reasoning, but it still does not denote a unique concrete
// memory cell. Therefore exact must strong memory semantics are trusted only for concrete
// objects.
inline bool is_exact_must_memory_cell(const Target& target) noexcept
{
    return is_concrete_object(target.id);
}

} // namespace

void apply_transfer_must_free(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

void apply_transfer_must_address(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id       = context.operands_id[0];
    const auto vM_id       = context.operands_id[1];
    context.must_in[vN_id] = {.id = vM_id, .offset_flag = false};
}

void apply_transfer_must_copy(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto xM_id = context.operands_id[1];

    const auto xM_id_must_iter = context.must_in.find(xM_id);
    if (xM_id_must_iter == context.must_in.end())
    {
        context.must_in.erase(vN_id);
        return;
    }

    if (vN_id != xM_id)
    {
        context.must_in[vN_id] = xM_id_must_iter->second;
    }
}

void apply_transfer_must_load(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_must_iter = context.must_in.find(vM_id);

    if (vM_id_must_iter != context.must_in.end() &&
        is_exact_must_memory_cell(vM_id_must_iter->second))
    {
        const auto pointee_object_id = vM_id_must_iter->second.id;
        const auto target_must_iter  = context.must_in.find(pointee_object_id);

        if (target_must_iter != context.must_in.end())
        {
            context.must_in[vN_id] = target_must_iter->second;
        }
        else
        {
            context.must_in.erase(vN_id);
        }
        return;
    }

    if (may_access_is_invalid_for_must(context, vM_id))
    {
        nuke_must(context.must_in);
        return;
    }

    context.must_in.erase(vN_id);
}

void apply_transfer_must_store(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_must_iter = context.must_in.find(vM_id);
    const auto vN_id_must_iter = context.must_in.find(vN_id);

    if (vN_id_must_iter != context.must_in.end() &&
        is_exact_must_memory_cell(vN_id_must_iter->second))
    {
        const auto dst_id = vN_id_must_iter->second.id;

        if (vM_id_must_iter != context.must_in.end())
        {
            context.must_in[dst_id] = vM_id_must_iter->second;
        }
        else
        {
            context.must_in.erase(dst_id);
        }
        return;
    }

    if (may_access_is_invalid_for_must(context, vN_id))
    {
        nuke_must(context.must_in);
        return;
    }

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    INVARIANT(vN_id_may_iter != context.may_in.end());
    erase_must_for_all_may_pointees(context.must_in, vN_id_may_iter->second);
}

void apply_transfer_must_alloca(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

void apply_transfer_must_malloc(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

void apply_transfer_must_i2p_p2i(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id           = context.operands_id[0];
    const auto vM_id           = context.operands_id[1];
    const auto vM_id_must_iter = context.must_in.find(vM_id);

    if (vM_id_must_iter != context.must_in.end())
    {
        context.must_in[vN_id] = vM_id_must_iter->second;
        return;
    }

    context.must_in.erase(vN_id);
}

void apply_transfer_must_memcpy_memmove(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    if (may_access_is_invalid_for_must(context, vN_id) ||
        may_access_is_invalid_for_must(context, vM_id))
    {
        nuke_must(context.must_in);
        return;
    }

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    INVARIANT(vN_id_may_iter != context.may_in.end());
    erase_must_for_all_may_pointees(context.must_in, vN_id_may_iter->second);
}

void apply_transfer_must_moveptr(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

void apply_transfer_must_memset(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];

    if (may_access_is_invalid_for_must(context, vN_id))
    {
        nuke_must(context.must_in);
        return;
    }

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    INVARIANT(vN_id_may_iter != context.may_in.end());
    erase_must_for_all_may_pointees(context.must_in, vN_id_may_iter->second);
}

void apply_transfer_must_stackrestore(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id           = context.operands_id[0];
    const auto vN_id_must_iter = context.must_in.find(vN_id);

    if (vN_id_must_iter == context.must_in.end() ||
        vN_id_must_iter->second.id != context.last_local_id)
    {
        nuke_must(context.must_in);
        return;
    }
}

void apply_transfer_must_stacksave(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto lN_id       = context.operands_id[0];
    context.must_in[lN_id] = {context.last_local_id, false};
}

void apply_transfer_must_va_start(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

void apply_transfer_must_va_end(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];

    context.must_in.erase(vN_id);
}

void apply_transfer_must_va_arg(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    context.must_in.erase(vN_id);
}

void apply_transfer_must_va_copy(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];

    context.must_in.erase(vN_id);
}

void apply_transfer_must_call(const MustTransferContextBundle& context)
{
    for (std::size_t i = 1; i < context.operands_count; ++i)
    {
        nuke_reachable_must(context.operands_id[i], context);
    }
}

void apply_transfer_must(const MustTransferContextBundle& context)
{
    switch (context.opcode)
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::RET:
        return;

    case sala::Instruction::Opcode::FREE:
        return apply_transfer_must_free(context);

    case sala::Instruction::Opcode::ADDRESS:
        return apply_transfer_must_address(context);

    case sala::Instruction::Opcode::COPY:
        return apply_transfer_must_copy(context);

    case sala::Instruction::Opcode::LOAD:
        return apply_transfer_must_load(context);

    case sala::Instruction::Opcode::STORE:
        return apply_transfer_must_store(context);

    case sala::Instruction::Opcode::ALLOCA:
        return apply_transfer_must_alloca(context);

    case sala::Instruction::Opcode::MALLOC:
        return apply_transfer_must_malloc(context);

    case sala::Instruction::Opcode::I2P:
    case sala::Instruction::Opcode::P2I:
        return apply_transfer_must_i2p_p2i(context);

    case sala::Instruction::Opcode::MEMMOVE:
    case sala::Instruction::Opcode::MEMCPY:
        return apply_transfer_must_memcpy_memmove(context);

    case sala::Instruction::Opcode::MOVEPTR:
        return apply_transfer_must_moveptr(context);

    case sala::Instruction::Opcode::MEMSET:
        return apply_transfer_must_memset(context);

    case sala::Instruction::Opcode::STACKRESTORE:
        return apply_transfer_must_stackrestore(context);

    case sala::Instruction::Opcode::STACKSAVE:
        return apply_transfer_must_stacksave(context);

    case sala::Instruction::Opcode::VA_START:
        return apply_transfer_must_va_start(context);

    case sala::Instruction::Opcode::VA_END:
        return apply_transfer_must_va_end(context);

    case sala::Instruction::Opcode::VA_ARG:
        return apply_transfer_must_va_arg(context);

    case sala::Instruction::Opcode::VA_COPY:
        return apply_transfer_must_va_copy(context);

    case sala::Instruction::Opcode::CALL:
        return apply_transfer_must_call(context);

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
    {
        ASSUMPTION(context.operands_id.size() >= 1);
        const auto vN_id = context.operands_id[0];
        context.must_in.erase(vN_id);
        return;
    }
    }

    UNREACHABLE();
}

} // namespace optimizer::utils::points_to
