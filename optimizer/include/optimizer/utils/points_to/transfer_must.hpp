#ifndef OPTIMIZER_UTILS_TRANSFER_MUST_HPP_INCLUDED
#define OPTIMIZER_UTILS_TRANSFER_MUST_HPP_INCLUDED
#include <optimizer/utils/points_to/defines.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::utils::points_to
{

static inline void apply_transfer_must_free(const MustTransferContextBundle& context)
{
    // FREE n vN; free(vN)
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    context.must_in.erase(vN_id);
}

static inline void apply_transfer_must_address(const MustTransferContextBundle& context)
{
    // ADDRESS n vN vM; vN := &vM (although defined as rM in program.hpp there is an exception
    // for static initializer so we interpret rM as vM)
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id       = context.operands_id[0];
    const auto vM_id       = context.operands_id[1];
    context.must_in[vN_id] = {vM_id, false};
}

static inline void apply_transfer_must_copy(const MustTransferContextBundle& context)
{
    // ADDRESS n vN vM; vN := &vM (although defined as rM in program.hpp there is an exception
    // for static initializer so we interpret rM as vM)

    // COPY n vN xM; vN := xM
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

static inline void apply_transfer_must_load(const MustTransferContextBundle& context)
{
    // LOAD vN, vM; vN := *vM;
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_must_iter = context.must_in.find(vM_id);

    if (vM_id_must_iter != context.must_in.end())
    {
        const auto target_must_iter = context.must_in.find(vM_id_must_iter->second.id);
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

    if (const auto vM_id_may_iter = context.may_in.find(vM_id);
        vM_id_may_iter == context.may_in.end() ||
        contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // undefined access
        nuke_must(context.must_in);
    }
    else
    {
        // no must information
        context.must_in.erase(vN_id);
    }
}

static inline void apply_transfer_must_store(const MustTransferContextBundle& context)
{
    // STORE n vN vM; *vN := vM
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_must_iter = context.must_in.find(vM_id);
    const auto vN_id_must_iter = context.must_in.find(vN_id);

    if (vN_id_must_iter != context.must_in.end())
    {
        if (vM_id_must_iter != context.must_in.end() && vN_id_must_iter->second.id >= 0)
        {
            // target has must points to and is not a grouped object -> points to transfer
            context.must_in[vN_id_must_iter->second.id] = vM_id_must_iter->second;
        }
        else
        {
            context.must_in.erase(vN_id_must_iter->second.id);
        }
        return;
    }

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end() ||
        contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        nuke_must(context.must_in);
        return;
    }

    INVARIANT(vN_id_may_iter != context.may_in.end());
    for (const auto target : vN_id_may_iter->second)
    {
        context.must_in.erase(target.id);
    }
}

static inline void apply_transfer_must_alloca(const MustTransferContextBundle& context)
{
    // ALLOCA n vN nM cH
    // Corresponds to a C code:
    //     case: nM > 1                            case: nM == 1
    //     ----------------------------------------------------------
    //     void* vN;                               void* vN;
    //     T a[nM];                                T a;
    //     vN = (void*)&a[0];                      vN = (void*)&a;
    // We do not distinguish nM in points to analysis
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id       = context.operands_id[0];
    context.must_in[vN_id] = {grouped_objects::ALLOCA, false};
}

static inline void apply_transfer_must_malloc(const MustTransferContextBundle& context)
{
    // MALLOC n vN nM; vN := malloc(nM)
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id       = context.operands_id[0];
    context.must_in[vN_id] = {grouped_objects::HEAP, false};
}
static inline void apply_transfer_must_i2p_p2i(const MustTransferContextBundle& context)
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

static inline void apply_transfer_must_memcpy_memmove(const MustTransferContextBundle& context)
{
    // MEMMOVE n vN vM nH, MEMCPY n vN vM nH
    // memmove(vN, vM, nH), memcpy(vN, vM, nH)
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end() ||
        contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        nuke_must(context.must_in);
        return;
    }

    if (const auto vM_id_may_iter = context.may_in.find(vM_id);
        vM_id_may_iter == context.may_in.end() ||
        contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        nuke_must(context.must_in);
        return;
    }

    INVARIANT(vN_id_may_iter != context.may_in.end());
    // We cannot track this reliably
    for (const auto& target : vN_id_may_iter->second)
    {
        context.must_in.erase(target.id);
    }
}

static inline void apply_transfer_must_moveptr(const MustTransferContextBundle& context)
{
    // MOVEPTR n vN vM nH cG; void* vN := (void*)((char*)vM + (nH * cG))
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];

    // we can't proivde reliable must points to information about the offset
    // (no tracking of value that the pointer is moved by)
    context.must_in.erase(vN_id);
}

static inline void apply_transfer_must_memset(const MustTransferContextBundle& context)
{
    // MEMSET n vN nM nH; memset(vN, nM, nH)
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end() ||
        contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        nuke_must(context.must_in);
        return;
    }

    for (const auto target : vN_id_may_iter->second)
    {
        context.must_in.erase(target.id);
    }
}

static inline void apply_transfer_must_stackrestore(const MustTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id           = context.operands_id[0];
    const auto vN_id_must_iter = context.must_in.find(vN_id);

    // If this does not hold we encountered undefined behaviour or
    // analysis went wrong, meaning we nuke it
    // For more details see definition of this instruction
    if (vN_id_must_iter == context.must_in.end() ||
        vN_id_must_iter->second.id != context.last_local_id)
    {
        nuke_must(context.must_in);
        return;
    }
}

static inline void apply_transfer_must_stacksave(const MustTransferContextBundle& context)
{
    // STACKSAVE n lN
    // Saves the current stack pointer to bytes of the local variable lN.
    // Assumptions:
    //     lN.num_bytes() == sizeof(void*).
    //     The variable lN must be available since the start of the function.
    //         So it may not be allocated by ALLOCA instruction.
    //     An execution of each STACKSAVE instruction must be followed by
    //         the corresponding execution of STACKRESTORE instruction,
    //         just like brackets work. Moreover, the corresponding
    //         STACKRESTORE instruction obtains the saved stack pointer.
    //         Examples:
    //             ...STACKSAVE lN...STACKRESTORE lN...                                    <-- OK
    //             ...STACKSAVE lN...STACKSAVE lM...STACKRESTORE lM...STACKRESTORE lN...   <-- OK
    //             ...STACKSAVE lN...STACKSAVE lM...STACKRESTORE lN...STACKRESTORE lM...   <--
    //             WRONG!
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto lN_id       = context.operands_id[0];
    context.must_in[lN_id] = {context.last_local_id, false};
}

static inline void apply_transfer_must_va_start(const MustTransferContextBundle& context)
{
    // VA_START n vN
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id       = context.operands_id[0];
    context.must_in[vN_id] = {grouped_objects::VARGARG_BLOCK, false};
}

static inline void apply_transfer_must_va_end(const MustTransferContextBundle& context)
{
    // VA_END n vN
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    if (const auto vN_id_may_iter = context.may_in.find(vN_id);
        vN_id_may_iter == context.may_in.end() ||
        !contains_objectId(vN_id_may_iter->second, grouped_objects::VARGARG_BLOCK))
    {
        nuke_must(context.must_in);
    }
    else
    {
        context.must_in.erase(vN_id);
    }
}

static inline void apply_transfer_must_va_arg(const MustTransferContextBundle& context)
{
    // VA_ARG n vN vM
    // Let 'p' be the address (pointer) stored in bytes of vM. The instruction first copies
    // bytes at addresses [p,p+vN.num_bytes()) to the bytes of vN. Then the address p in the
    // bytes of vN is moved (shifted) to point to the next variable function parameter (packed
    // into the '...' C parameter). The moved address will be stored in bytes of vM.
    // Assumptions: vN.num_bytes() == sizeof(void*).
    //              There must be a preceding execution of the corresponding VA_START instruction.
    //              There must follow an execution of the corresponding VA_END instruction.
    //              The address 'p' must be equal to one returned either be the corresponding
    //              VA_START instruction or a preceding VA_ARG instruction.
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[0];

    const auto vM_id_may_iter = context.may_in.find(vN_id);
    if (vM_id_may_iter == context.may_in.end())
    {
        nuke_must(context.must_in);
        return;
    }
    if (contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        nuke_must(context.must_in);
        return;
    }
    if (!contains_objectId(vM_id_may_iter->second, grouped_objects::VARGARG_BLOCK))
    {
        nuke_must(context.must_in);
        return;
    }

    if (const auto vM_id_must_iter = context.must_in.find(vM_id);
        vM_id_must_iter != context.must_in.end())
    {
        context.must_in[vN_id] = {vM_id_must_iter->second.id, true};
    }
    else
    {
        context.must_in.erase(vN_id);
    }
}

static inline void apply_transfer_must_va_copy(const MustTransferContextBundle& context)
{
    // TODO: ?
}

static inline void apply_transfer_must_call(const MustTransferContextBundle& context)
{
    // CALL n rN [p0 xH1 ... xHm]
    //     rN([p0, xH1, ..., xHm])    // rN is actually fN
    //     (*rN)([p0, xH1, ..., xHm]) // rN is actually vN holding the pointer to the called
    //     function
    for (std::size_t i = 1; i < context.operands_count; ++i)
    {
        nuke_reachable_must(context.operands_id[i], context);
    }
}

static inline void apply_transfer_must(const MustTransferContextBundle& context)
{
    switch (context.opcode)
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::RET: // No effect
        return;
    case sala::Instruction::Opcode::FREE:
        return apply_transfer_must_free(context);
    case sala::Instruction::Opcode::ADDRESS: // vN = &vM; strong update
    {
        return apply_transfer_must_address(context);
    }
    case sala::Instruction::Opcode::COPY: // vN = vM; strong update
    {
        return apply_transfer_must_copy(context);
    }

    case sala::Instruction::Opcode::LOAD: // vN := *vM; strong update if pointee singleton
    {
        return apply_transfer_must_load(context);
    }
    case sala::Instruction::Opcode::STORE: // *vN := vM;
    {
        return apply_transfer_must_store(context);
    }
    case sala::Instruction::Opcode::ALLOCA: // vN := alloca(...); strong update
    {
        return apply_transfer_must_alloca(context);
    }
    case sala::Instruction::Opcode::MALLOC: // vN := malloc(...); strong update
    {
        return apply_transfer_must_malloc(context);
    }
    case sala::Instruction::Opcode::I2P: // int* vN = (unsigned int*)(vM); strong update
    case sala::Instruction::Opcode::P2I: // vN := (unsigned int)(vM); strong update
    {
        return apply_transfer_must_i2p_p2i(context);
    }
    case sala::Instruction::Opcode::MEMMOVE: // memove(vN, vM, nH)
    case sala::Instruction::Opcode::MEMCPY:  // memcpy(vN, vM, nH)
    {
        return apply_transfer_must_memcpy_memmove(context);
    }
    case sala::Instruction::Opcode::MOVEPTR:
    {
        return apply_transfer_must_moveptr(context);
    }
    case sala::Instruction::Opcode::MEMSET: // memeset(vN, nM, nH)
    {
        return apply_transfer_must_memset(context);
    }
    case sala::Instruction::Opcode::STACKRESTORE:
    {
        return apply_transfer_must_stackrestore(context);
    }
    case sala::Instruction::Opcode::STACKSAVE:
    {
        return apply_transfer_must_stacksave(context);
    }
    case sala::Instruction::Opcode::VA_START: // VA_START vN; *vN = VA_START
    {
        return apply_transfer_must_va_start(context);
    }
    case sala::Instruction::Opcode::VA_END: // VA_END vN
    {
        return apply_transfer_must_va_end(context);
    }
    case sala::Instruction::Opcode::VA_ARG:
    {
        return apply_transfer_must_va_arg(context);
    }
    case sala::Instruction::Opcode::VA_COPY:
    {
        return apply_transfer_must_va_copy(context);
    }
    case sala::Instruction::Opcode::CALL:
    {
        return apply_transfer_must_call(context);
    }

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
        // leaves the pointer set if present
        context.must_in.erase(vN_id);
        return;
    }
    }
    UNREACHABLE();
}

} // namespace optimizer::utils::points_to
#endif
