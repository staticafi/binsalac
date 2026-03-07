#ifndef OPTIMIZER_UTILS_TRANSFER_MAY_HPP_INCLUDED
#define OPTIMIZER_UTILS_TRANSFER_MAY_HPP_INCLUDED
#include <optimizer/utils/points_to/defines.hpp>

#include <iostream>
#include <utility/assumptions.hpp>
namespace optimizer::utils::points_to
{
static inline std::string serialize_program_point(const ProgramPoint& pp)
{
    std::ostringstream builder;
    builder << "PP: [" << pp.function << ", " << pp.bb << ", " << pp.instr << " ]";
    return builder.str();
}

static inline void apply_transfer_may_load(const MayTransferContextBundle& context)
{
    // LOAD vN, vM; vN := *vM;
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_may_iter = context.may_in.find(vM_id);

    // undefined check
    if (vM_id_may_iter == context.may_in.end())
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING LOAD: not tracked: " << vM_id
                  << std::endl;
        nuke_may(context);
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING LOAD: contains UNDEFINED "
                  << vM_id << std::endl;
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED, false}};
        nuke_may(context);
        return;
    }

    // FIXME: failing assumption somewhere we generate empty set
    ASSUMPTION(!vM_id_may_iter->second.empty());
    // singleton
    if (vM_id_may_iter->second.size() == 1)
    {
        const auto target_may_iter = context.may_in.find((vM_id_may_iter->second.begin())->id);
        if (target_may_iter != context.may_in.end())
        {
            // target was a pointer
            ASSUMPTION(!target_may_iter->second.empty());
            context.may_in.emplace(vM_id, target_may_iter->second);
        }
        else if (target_may_iter->second.empty())
        {
            // leaves pointers set; assigned a non pointer
            context.may_in.erase(vN_id);
        }
        return;
    }

    // TODO: non singleton (maybe do not sepearte )
    MayState::iterator vN_id_may_iter;
    auto               untracked_encountered = false;
    auto               tracked_encountered   = false;

    const auto vM_targets_copy = vM_id_may_iter->second;
    for (const auto source : vM_targets_copy)
    {
        const auto source_may_iter = context.may_in.find(source.id);
        if (source_may_iter == context.may_in.end())
        {
            // not a pointer
            untracked_encountered = true;
        }
        else
        {
            // is a pointer
            const auto copy = source_may_iter->second;
            for (const auto transfer : copy)
            {
                if (tracked_encountered)
                {
                    vN_id_may_iter->second.insert(transfer);
                }
                else
                {
                    vN_id_may_iter = context.may_in.insert_or_assign(vN_id, MaySet{transfer}).first;
                    tracked_encountered = true;
                }
            }
        }

        if (tracked_encountered && untracked_encountered)
        {
            vN_id_may_iter->second.insert({grouped_objects::MERGE_UNKNOWN, false});
        }
    }
}

static inline void apply_transfer_may_store(const MayTransferContextBundle& context)
{
    // STORE n vN vM; *vN := vM
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vN_id_may_iter = context.may_in.find(vN_id);

    // not recognized as a valid pointer
    if (vN_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp) << " NUKING STORE: not tracked " << vN_id
                  << std::endl;
        dump_may_set(context.may_in);
        nuke_may(context);
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING STORE: contains UNDEFINED "
                  << vN_id << std::endl;
        nuke_may(context);
        return;
    }

    const auto vM_id_may_iter        = context.may_in.find(vM_id);
    const auto is_vM_tracked_pointer = vM_id_may_iter != context.may_in.end();

    std::optional<MaySet> vM_targets_copy;
    if (is_vM_tracked_pointer)
    {
        vM_targets_copy = vM_id_may_iter->second;
    }

    const auto vN_targets_copy = vN_id_may_iter->second;
    for (const auto& target : vN_targets_copy)
    {
        const auto target_may_iter = context.may_in.find(target.id);
        // target recognized as a pointer
        if (target_may_iter != context.may_in.end())
        {
            if (is_vM_tracked_pointer)
            {
                for (const auto& transfer : vM_targets_copy.value())
                {
                    target_may_iter->second.insert(transfer);
                }
            }
            else
            {
                target_may_iter->second.insert({grouped_objects::MERGE_UNKNOWN, false});
            }
        }
        // target not recognized as a pointer and source is a pointer
        // as such we transfer the information cleanly
        else if (is_vM_tracked_pointer)
        {
            ASSUMPTION(!vM_targets_copy->empty());
            context.may_in.emplace(target.id, vM_targets_copy.value());
        }
        // target not recognized as pointer and source not recognized as a pointer no need for
        // points to information transfer
    }
}

static inline void apply_transfer_may_memcpy_memmove(const MayTransferContextBundle& context)
{
    // MEMMOVE n vN vM nH, MEMCPY n vN vM nH
    // memmove(vN, vM, nH), memcpy(vN, vM, nH)
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    const auto vM_id_may_iter = context.may_in.find(vN_id);
    if (vM_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp) << " NUKING MEMOVE MEMCPY: not tracked"
                  << std::endl;
        nuke_may(context);
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING MEMOVE MEMCPY: not tracked"
                  << std::endl;
        nuke_may(context);
        return;
    }

    INVARIANT(vM_id_may_iter != context.may_in.end());
    if (vN_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp) << " NUKING MEMOVE MEMCPY: not tracked"
                  << std::endl;
        nuke_may(context);
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING MEMOVE MEMCPY: undefined"
                  << std::endl;
        nuke_may(context);
        return;
    }

    auto       untracked_encountered = false;
    auto       tracked_encountered   = false;
    const auto vM_target_copy        = vM_id_may_iter->second;
    const auto vN_target_copy        = vN_id_may_iter->second;

    for (const auto source : vM_target_copy)
    {
        const auto source_may_iter = context.may_in.find(source.id);
        if (!untracked_encountered && source_may_iter == context.may_in.end())
        {
            untracked_encountered = true;
        }
        else if (!tracked_encountered)
        {
            tracked_encountered = true;
        }

        for (const auto target : vN_target_copy)
        {
            auto target_may_iter = context.may_in.find(target.id);

            // source does hold points-to information
            if (source_may_iter != context.may_in.end())
            {
                // transfer the information
                const auto source_target_copy = source_may_iter->second;
                for (const auto transfer : source_target_copy)
                {
                    if (target_may_iter == context.may_in.end())
                    {
                        target_may_iter =
                                context.may_in.insert_or_assign(target.id, MaySet{transfer}).first;
                    }
                    else
                    {
                        target_may_iter->second.insert(transfer);
                    }
                }
            }
            // source not recognized as pointer, target holds points to information
            // we extend source by non ptr
            else if (target_may_iter != context.may_in.end())
            {
                target_may_iter->second.insert({grouped_objects::MERGE_UNKNOWN, false});
            }
            // target and source not recognized as pointers no points-to information to transfer
        }
    }

    // the source points to a tracked pointer and atleast one more object which does not have a
    // may points to information, as such we insert to all targets UNDEFINED
    if (tracked_encountered && untracked_encountered)
    {
        for (const auto target : vN_target_copy)
        {
            context.may_in[target.id].insert({{grouped_objects::MERGE_UNKNOWN, false}});
        }
    }
}

static inline void apply_transfer_may_address(const MayTransferContextBundle& context)
{
    // ADDRESS n vN vM; vN := &vM (although defined as rM in program.hpp there is an exception
    // for static initializer so we interpret rM as vM)
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id      = context.operands_id[0];
    const auto vM_id      = context.operands_id[1];
    context.may_in[vN_id] = {{vM_id, false}};
}

static inline void apply_transfer_may_copy(const MayTransferContextBundle& context)
{
    // COPY n vN xM; vN := xM
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id = context.operands_id[0];
    const auto xM_id = context.operands_id[1];

    const auto xM_id_iter = context.may_in.find(xM_id);
    // source not recognized as pointer
    if (xM_id_iter == context.may_in.end())
    {
        // vN leaves the tracked pointers if assigned non pointer
        context.may_in.erase(vN_id);
        return;
    }

    // We do not create empty MaySet-s
    ASSUMPTION(!xM_id_iter->second.empty());
    if (vN_id != xM_id)
    {
        context.may_in.emplace(vN_id, xM_id_iter->second);
    }
}

static inline void apply_transfer_may_alloca(const MayTransferContextBundle& context)
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
    const auto vN_id      = context.operands_id[0];
    context.may_in[vN_id] = {{grouped_objects::ALLOCA, false}};
}

static inline void apply_transfer_may_malloc(const MayTransferContextBundle& context)
{
    // MALLOC n vN nM; vN := malloc(nM)
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id      = context.operands_id[0];
    context.may_in[vN_id] = {{grouped_objects::HEAP, false}};
}

static inline void apply_transfer_may_i2p_p2i(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id          = context.operands_id[0];
    const auto vM_id          = context.operands_id[1];
    const auto vM_id_may_iter = context.may_in.find(vM_id);
    if (vM_id_may_iter == context.may_in.end())
    {
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    context.may_in.emplace(vN_id, vM_id_may_iter->second);
    return;
}

static inline void apply_transfer_may_moveptr(const MayTransferContextBundle& context)
{
    // MOVEPTR n vN vM nH cG; void* vN := (void*)((char*)vM + (nH * cG))
    ASSUMPTION(context.operands_id.size() >= 2);
    const auto vN_id          = context.operands_id[0];
    const auto vM_id          = context.operands_id[1];
    const auto vM_id_may_iter = context.may_in.find(vM_id);
    if (vM_id_may_iter == context.may_in.end())
    {
        std::cout << "APPENDING UNDEFINED MOVEPTR: " << vN_id << " " << vM_id << std::endl;
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED}};
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED}};
        return;
    }

    ASSUMPTION(!vM_id_may_iter->second.empty());
    context.may_in.emplace(vN_id, vM_id_may_iter->second);
}

static inline void apply_transfer_may_memset(const MayTransferContextBundle& context)
{
    // MEMSET n vN nM nH; memset(vN, nM, nH)
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp) << " NUKING MEMSET: untracked"
                  << std::endl;
        nuke_may(context);
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING MEMSET: contains undefined"
                  << std::endl;
        nuke_may(context);
        return;
    }

    for (const auto target : vN_id_may_iter->second)
    {
        if (const auto target_may_iter = context.may_in.find(target.id);
            target_may_iter != context.may_in.end())
        {
            target_may_iter->second.insert({grouped_objects::UNDEFINED, false});
            return;
        }
    }
}

static inline void apply_transfer_may_call(const MayTransferContextBundle& context)
{
    // CALL n rN [p0 xH1 ... xHm]
    //     rN([p0, xH1, ..., xHm])    // rN is actually fN
    //     (*rN)([p0, xH1, ..., xHm]) // rN is actually vN holding the pointer to the called
    //     function
    for (std::size_t i = 1; i < context.operands_count; ++i)
    {
        nuke_reachable_may(context.operands_id[i], context, grouped_objects::OUT_OF_LOCAL_SCOPE);
    }
}

static inline void apply_transfer_may_stacksave(const MayTransferContextBundle& context)
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
    const auto lN_id      = context.operands_id[0];
    context.may_in[lN_id] = {{context.last_local_id, false}};
}

static inline void apply_transfer_may_stackrestore(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = context.may_in.find(vN_id);

    if (vN_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp) << " NUKING STACKRESTORE: untracked"
                  << std::endl;
        nuke_may(context);
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (!contains_only_objectId(vN_id_may_iter->second, context.last_local_id))
    {
        // this would be considered UNDEFINED BEHAVIOUR or our analysis went wrong
        std::cout << serialize_program_point(context.pp)
                  << " NUKING STACKRESTORE: we dont pop until last local id" << std::endl;
        nuke_may(context);
        return;
    }
}

static inline void apply_transfer_may_va_start(const MayTransferContextBundle& context)
{
    // VA_START n vN
    ASSUMPTION(context.operands_id.size() >= 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = context.may_in.find(vN_id);

    if (vN_id_may_iter == context.may_in.end())
    {
        // access to untracked
        std::cout << serialize_program_point(context.pp)
                  << " NUKING VA_START: not tracked: " << vN_id << std::endl;
        nuke_may(context);
        context.may_in[vN_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }

    if (contains_objectId(vN_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        // access to undefined
        std::cout << serialize_program_point(context.pp) << " NUKING VA_START: contains UNDEFINED "
                  << vN_id << std::endl;
        nuke_may(context);
        return;
    }
    const auto vN_targets_copy = vN_id_may_iter->second;
    for (const auto& target : vN_targets_copy)
    {
        const auto target_may_iter = context.may_in.find(target.id);
        if (target_may_iter != context.may_in.end())
        {
            // target recognized as a pointer
            target_may_iter->second.insert({grouped_objects::VARGARG_BLOCK, false});
        }
        else
        {
            context.may_in[target.id] = {{grouped_objects::VARGARG_BLOCK, false}};
        }
    }
}

static inline void apply_transfer_may_va_end(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    constexpr auto max_vargarg_block_depth = 2;

    const auto vN_id = context.operands_id[0];
    if (!is_objectId_reachable(context, vN_id, grouped_objects::VARGARG_BLOCK,
                               max_vargarg_block_depth))
    {
        std::cout << serialize_program_point(context.pp)
                  << " NUKING VA_END: does not contain VARGARG_BLOCK" << std::endl;
        nuke_may(context);
    }
}

static inline void apply_transfer_may_va_arg(const MayTransferContextBundle& context)
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
    const auto vM_id = context.operands_id[1];

    auto vM_id_may_iter = context.may_in.find(vM_id);
    if (vM_id_may_iter == context.may_in.end())
    {
        std::cout << serialize_program_point(context.pp) << " NUKING VA_ARG: access to untracked"
                  << std::endl;
        nuke_may(context);
        context.may_in[vM_id] = {{grouped_objects::UNDEFINED, false}};
        return;
    }
    if (contains_objectId(vM_id_may_iter->second, grouped_objects::UNDEFINED))
    {
        std::cout << serialize_program_point(context.pp) << " NUKING VA_ARG: access to undefined"
                  << std::endl;
        nuke_may(context);
        return;
    }
    if (!contains_objectId(vM_id_may_iter->second, grouped_objects::VARGARG_BLOCK))
    {
        std::cout << serialize_program_point(context.pp)
                  << " NUKING VA_ARG: does not contain VARGARG_BLOCK" << std::endl;
        nuke_may(context);
        return;
    }

    auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end())
    {
        ASSUMPTION(!vM_id_may_iter->second.empty());
        vN_id_may_iter = context.may_in.insert_or_assign(vN_id, vM_id_may_iter->second).first;
        vM_id_may_iter = context.may_in.find(vM_id);
    }
    else
    {
        for (const auto transfer : vM_id_may_iter->second)
        {
            vN_id_may_iter->second.insert(transfer);
        }
    }
    vN_id_may_iter->second.insert({grouped_objects::VARGARG_BLOCK, true});
}

static inline void apply_transfer_may_va_copy(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    constexpr auto max_vargarg_block_depth = 2;
    const auto     vN_id                   = context.operands_id[0];
    const auto     vM_id                   = context.operands_id[1];

    const auto vN_id_may_iter = context.may_in.find(vN_id);
    if (vN_id_may_iter == context.may_in.end())
    {
        std::cout << serialize_program_point(context.pp) << " NUKING VA_COPY: vN untracked"
                  << std::endl;
        nuke_may(context);
        return;
    }

    if (!is_objectId_reachable(context, vM_id, grouped_objects::VARGARG_BLOCK,
                               max_vargarg_block_depth))
    {
        std::cout << serialize_program_point(context.pp)
                  << " NUKING VA_COPY: vM does not contain VARGARG_BLOCK" << std::endl;
        nuke_may(context);
        return;
    }

    const auto vN_targets_copy = vN_id_may_iter->second;
    for (const auto& target : vN_targets_copy)
    {
        if (const auto target_id_may_iter = context.may_in.find(target.id);
            target_id_may_iter != context.may_in.end())
        {
            target_id_may_iter->second.insert({grouped_objects::VARGARG_BLOCK, false});
        }
        else
        {
            context.may_in[target.id] = {{grouped_objects::VARGARG_BLOCK, false}};
        }
    }
}

static inline void apply_transfer_may(const MayTransferContextBundle& context)
{
    // std::cout << serialize_program_point(context.pp) << std::endl;
    // dump_may_set(context.may_in);
    switch (context.opcode)
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::FREE:
    case sala::Instruction::Opcode::RET: // No effect
        return;
    case sala::Instruction::Opcode::ADDRESS: // vN = &vM; strong update
    {
        return apply_transfer_may_address(context);
    }
    case sala::Instruction::Opcode::COPY: // vN = vM; strong update
    {
        return apply_transfer_may_copy(context);
    }

    case sala::Instruction::Opcode::LOAD: // vN := *vM; strong update if pointee singleton
    {
        return apply_transfer_may_load(context);
    }
    case sala::Instruction::Opcode::STORE: // *vN := vM;
    {
        return apply_transfer_may_store(context);
    }
    case sala::Instruction::Opcode::ALLOCA: // vN := alloca(...); strong update
    {
        return apply_transfer_may_alloca(context);
    }
    case sala::Instruction::Opcode::MALLOC: // vN := malloc(...); strong update
    {
        return apply_transfer_may_malloc(context);
    }
    case sala::Instruction::Opcode::I2P: // int* vN = (unsigned int*)(vM); strong update
    case sala::Instruction::Opcode::P2I: // vN := (unsigned int)(vM); strong update
    {
        return apply_transfer_may_i2p_p2i(context);
    }

    case sala::Instruction::Opcode::MEMMOVE: // memove(vN, vM, nH)
    case sala::Instruction::Opcode::MEMCPY:  // memcpy(vN, vM, nH)
    {
        return apply_transfer_may_memcpy_memmove(context);
    }
    case sala::Instruction::Opcode::MOVEPTR:
    {
        return apply_transfer_may_moveptr(context);
    }
    case sala::Instruction::Opcode::MEMSET: // memeset(vN, nM, nH)
    {
        return apply_transfer_may_memset(context);
    }
    case sala::Instruction::Opcode::STACKRESTORE:
    {
        return apply_transfer_may_stackrestore(context);
    }
    case sala::Instruction::Opcode::STACKSAVE:
    {
        return apply_transfer_may_stacksave(context);
    }
    case sala::Instruction::Opcode::VA_START: // VA_START vN; *vN = VA_START
    {
        return apply_transfer_may_va_start(context);
    }
    case sala::Instruction::Opcode::VA_END: // VA_END vN
    {
        return apply_transfer_may_va_end(context);
    }
    case sala::Instruction::Opcode::VA_ARG: // VA_ARG n vN vM
    {
        return apply_transfer_may_va_arg(context);
    }
    case sala::Instruction::Opcode::VA_COPY:
    {
        return apply_transfer_may_va_copy(context);
    }
    case sala::Instruction::Opcode::CALL:
    {
        return apply_transfer_may_call(context);
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
        context.may_in.erase(vN_id);
        return;
    }
    }
    UNREACHABLE();
}

} // namespace optimizer::utils::points_to
#endif
