#include <optimizer/utils/points_to/transfer_may.hpp>

#include <iostream>
#include <utility/assumptions.hpp>

namespace optimizer::utils::points_to
{
namespace
{

inline std::string serialize_program_point(const ProgramPoint& pp)
{
    std::ostringstream builder;
    builder << "PP: [" << pp.function << ", " << pp.bb << ", " << pp.instr << " ]";
    return builder.str();
}

inline void set_top_if_present(MayState& state, const objectId id)
{
    if (const auto id_iter = state.find(id); id_iter != state.end())
    {
        id_iter->second.make_top();
    }
}

inline void set_top(MayState& state, const objectId id)
{
    state.insert_or_assign(id, MayValue::top());
}

inline void set_singleton(MayState& state, const objectId id, const Target target)
{
    state.insert_or_assign(id, MayValue::singleton(target));
}

inline bool is_top_value(const MayValue& value) noexcept
{
    return value.is_top;
}

inline bool is_tracked(const MayState& state, const objectId id) noexcept
{
    return state.find(id) != state.end();
}

inline bool is_dereferenceable_target(const Target& target) noexcept
{
    return can_have_state_cell(target.id);
}

inline void insert_merge_unknown(MayValue& value)
{
    if (!value.is_top)
    {
        value.add_merge_unknown();
    }
}

inline bool may_value_has_dereferenceable_target(const MayValue& value) noexcept
{
    if (value.is_top)
    {
        return false;
    }

    for (const auto& target : value)
    {
        if (is_dereferenceable_target(target))
        {
            return true;
        }
    }

    return false;
}

inline void strong_transfer(const objectId& copy_dest, const objectId& copy_source,
                            const MayTransferContextBundle& context)
{
    if (copy_dest == copy_source)
    {
        return;
    }

    if (const auto points_to_transfer_iter = context.state.may.find(copy_source);
        points_to_transfer_iter != context.state.may.end())
    {
        // copies points to information
        context.state.may.insert_or_assign(copy_dest, points_to_transfer_iter->second);
    }
    else
    {
        // leaves points to set assigned an non pointer or bottom
        context.state.may.erase(copy_dest);
    }
};

inline MayState::iterator require_tracked_pointer(const MayTransferContextBundle& context,
                                                  const objectId id, const std::string& what)
{
    ASSUMPTION(!context.state.poisoned);
    auto it = context.state.may.find(id);
    if (it == context.state.may.end())
    {
        // std::cout << serialize_program_point(context.pp) << " NUKING " << what << ": not tracked
        // "
        //           << id << std::endl;
        poison_may(context);
        return context.state.may.end();
    }

    return it;
}

// Require that `id` currently holds a tracked finite pointer value.
//
// Current policy:
// - missing may information => invalid / too imprecise access, nuke may, then materialize `id` as
// top
// - top may information     => invalid / too imprecise access, nuke may
//
// Return value:
// - iterator to valid finite tracked may value
// - end() if the caller must stop
inline MayState::iterator require_tracked_non_top_pointer(const MayTransferContextBundle& context,
                                                          const objectId                  id,
                                                          const std::string&              what)
{
    const auto it = require_tracked_pointer(context, id, what);
    if (it == context.state.may.end())
    {
        ASSUMPTION(context.state.poisoned);
        return it;
    }

    if (it->second.is_top)
    {
        // std::cout << serialize_program_point(context.pp) << " NUKING " << what << ": top " << id
        //           << std::endl;
        poison_may(context);
        return context.state.may.end();
    }

    return it;
}

// Merge points-to information stored in dereferenceable object `source_id`
// into dereferenceable object `target_id`.
//
// Return value:
// - true  => source object was tracked
// - false => source object was untracked
//
// Semantics:
// - untracked source: destination loses precision via merge_unknown
// - top source:      destination becomes top
// - finite source:   destination joins with source
bool merge_object_points_to_into_object(MayState& state, const objectId source_id,
                                        const objectId target_id)
{
    ASSUMPTION(can_have_state_cell(target_id));

    const auto source_may_iter = state.find(source_id);
    auto       target_may_iter = state.find(target_id);

    if (source_may_iter == state.end())
    {
        if (target_may_iter != state.end())
        {
            insert_merge_unknown(target_may_iter->second);
        }
        return false;
    }

    if (is_top_value(source_may_iter->second))
    {
        state.insert_or_assign(target_id, MayValue::top());
        return true;
    }

    const auto source_copy = source_may_iter->second;
    if (target_may_iter == state.end())
    {
        state.insert_or_assign(target_id, source_copy);
    }
    else
    {
        target_may_iter->second.join_with(source_copy);
    }

    return true;
}

// Merge pointee contents from all dereferenceable source targets into all
// dereferenceable destination targets.
//
// Non-dereferenceable targets are ignored here because they cannot denote state cells.
// If tracked and untracked dereferenceable sources are mixed, destination cells lose
// precision via merge_unknown.
void merge_pointees_into_pointees(MayState& state, const MayValue& dst_ptr_value,
                                  const MayValue& src_ptr_value)
{
    ASSUMPTION(!is_top_value(dst_ptr_value));
    ASSUMPTION(!is_top_value(src_ptr_value));

    bool tracked_encountered   = false;
    bool untracked_encountered = false;

    const auto src_targets_copy = src_ptr_value;
    const auto dst_targets_copy = dst_ptr_value;

    for (const auto& source : src_targets_copy)
    {
        if (!is_dereferenceable_target(source))
        {
            continue;
        }

        const bool source_tracked = is_tracked(state, source.id);
        tracked_encountered |= source_tracked;
        untracked_encountered |= !source_tracked;

        for (const auto& target : dst_targets_copy)
        {
            if (!is_dereferenceable_target(target))
            {
                continue;
            }

            const bool merged = merge_object_points_to_into_object(state, source.id, target.id);
            tracked_encountered |= merged;
            untracked_encountered |= !merged;
        }
    }

    if (tracked_encountered && untracked_encountered)
    {
        for (const auto& target : dst_targets_copy)
        {
            if (!is_dereferenceable_target(target))
            {
                continue;
            }

            if (auto target_iter = state.find(target.id); target_iter != state.end())
            {
                insert_merge_unknown(target_iter->second);
            }
        }
    }
}

// Store a source may value through a finite destination pointer value.
//
// For each dereferenceable destination pointee:
// - tracked top source    => destination cell becomes top
// - tracked finite source => destination cell joins with source
// - untracked source      => destination cell loses precision via merge_unknown
//
// Non-dereferenceable targets are ignored.
void store_through_pointer(MayState& state, MayValue& dst_ptr_value, objectId src_id)
{
    ASSUMPTION(!is_top_value(dst_ptr_value));

    const auto              dst_targets_copy = dst_ptr_value;
    const auto              src_may_iter     = state.find(src_id);
    std::optional<MayValue> transfer;
    if (const auto src_may_iter = state.find(src_id); src_may_iter != state.end())
    {
        transfer = src_may_iter->second;
    }
    else if (is_dereferenceable_summary_target(src_id))
    {
        transfer = make_singleton(src_id);
    }

    for (const auto& target : dst_targets_copy)
    {
        if (!is_dereferenceable_target(target))
        {
            continue;
        }
        const auto target_may_iter = state.find(target.id);

        if (target_may_iter != state.end())
        {
            if (transfer.has_value())
            {
                target_may_iter->second.join_with(transfer.value());
            }
            else
            {
                target_may_iter->second.add_merge_unknown();
            }
        }
        else if (transfer.has_value())

        {
            state.insert_or_assign(target.id, transfer.value());
        }
    }
}

// Compute the may result of dereferencing a finite pointer value.
//
// Only dereferenceable targets are used as state cells.
// If at least one dereferenceable pointee is tracked and another is untracked,
// the result loses precision via merge_unknown.
//
// Out-parameters:
// - any_tracked   => at least one dereferenceable pointee had a tracked cell
// - any_untracked => at least one dereferenceable pointee had no tracked cell
// TODO: look into
MayValue load_through_pointer(const MayState& state, const MayValue& ptr_value, bool& any_tracked,
                              bool& any_untracked)
{
    if (is_top_value(ptr_value))
    {
        any_tracked = true;
        return MayValue::top();
    }

    MayValue result{};

    any_tracked   = false;
    any_untracked = false;

    for (const auto& source : ptr_value)
    {
        if (!is_dereferenceable_target(source))
        {
            any_untracked = true;
            continue;
        }

        if (is_summary_target(source.id))
        {
            result.insert({.id = source.id, .offset_flag = false});
            any_tracked = true;
        }

        const auto source_may_iter = state.find(source.id);
        if (source_may_iter == state.end())
        {
            any_untracked = !is_summary_target(source.id);
            continue;
        }

        any_tracked = true;
        result.join_with(source_may_iter->second);

        if (result.is_top)
        {
            return result;
        }
    }

    if (!result.is_top && any_tracked && any_untracked)
    {
        result.add_merge_unknown();
    }

    return result;
}

} // namespace

// LOAD n vN vM; vN = *vN
// - strong if must fact
// - weak else
void apply_transfer_may_load(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_may_iter = require_tracked_pointer(context, vM_id, "LOAD");
    if (vM_id_may_iter == context.state.may.end())
    {
        return;
    }

    if (const auto must_fact = vM_id_may_iter->second.must_fact(); must_fact.has_value())
    {
        strong_transfer(vN_id, must_fact.value().id, context);
        return;
    }
    bool any_tracked   = false;
    bool any_untracked = false;

    MayValue loaded = load_through_pointer(context.state.may, vM_id_may_iter->second, any_tracked,
                                           any_untracked);
    if (any_tracked)
    {
        context.state.may.insert_or_assign(vN_id, std::move(loaded));
    }
    else if (const auto vN_id_may_iter = context.state.may.find(vN_id);
             vN_id_may_iter != context.state.may.end())
    {
        vN_id_may_iter->second.add_merge_unknown();
    }
}

// STORE n vN vM; *vN = vM;
void apply_transfer_may_store(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vN_id_may_iter = require_tracked_non_top_pointer(context, vN_id, "STORE");
    if (vN_id_may_iter == context.state.may.end())
    {
        return;
    }

    if (const auto must_fact = vN_id_may_iter->second.must_fact(); must_fact.has_value())
    {
        strong_transfer(must_fact.value().id, vM_id, context);
        return;
    }

    store_through_pointer(context.state.may, vN_id_may_iter->second, vM_id);
}

void apply_transfer_may_memcpy_memmove(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_may_iter = require_tracked_non_top_pointer(context, vM_id, "MEMMOVE MEMCPY");
    if (vM_id_may_iter == context.state.may.end())
    {
        return;
    }

    const auto vN_id_may_iter = require_tracked_non_top_pointer(context, vN_id, "MEMMOVE MEMCPY");
    if (vN_id_may_iter == context.state.may.end())
    {
        return;
    }

    merge_pointees_into_pointees(context.state.may, vN_id_may_iter->second, vM_id_may_iter->second);
}

// ADDRESS n vN rM; vN = &rM
void apply_transfer_may_address(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];
    // strong update
    set_singleton(context.state.may, vN_id, {.id = vM_id, .offset_flag = false});
}

// COPY n vN xM; vN = xM
void apply_transfer_may_copy(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto xM_id = context.operands_id[1];

    strong_transfer(vN_id, xM_id, context);
}

void apply_transfer_may_alloca(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id = context.operands_id[0];
    set_singleton(context.state.may, vN_id, {.id = grouped_objects::ALLOCA, .offset_flag = false});
}

void apply_transfer_may_malloc(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id = context.operands_id[0];
    set_singleton(context.state.may, vN_id, {.id = grouped_objects::HEAP, .offset_flag = false});
}

void apply_transfer_may_i2p_p2i(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id          = context.operands_id[0];
    const auto vM_id          = context.operands_id[1];
    const auto vM_id_may_iter = context.state.may.find(vM_id);

    if (vM_id_may_iter == context.state.may.end())
    {
        set_top(context.state.may, vN_id);
        set_top(context.state.may, vM_id);
        return;
    }

    context.state.may.insert_or_assign(vN_id, vM_id_may_iter->second);
}

void apply_transfer_may_moveptr(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id          = context.operands_id[0];
    const auto vM_id          = context.operands_id[1];
    const auto vM_id_may_iter = context.state.may.find(vM_id);

    if (vM_id_may_iter == context.state.may.end())
    {
        set_top(context.state.may, vN_id);
        set_top(context.state.may, vM_id);
        return;
    }

    if (vM_id_may_iter->second.is_top)
    {
        set_top(context.state.may, vN_id);
        return;
    }

    MayValue moved_value;
    moved_value.loss_flags = vM_id_may_iter->second.loss_flags;

    const auto src_copy = vM_id_may_iter->second;
    for (const auto& target : src_copy)
    {
        moved_value.insert({.id = target.id, .offset_flag = true});
        moved_value.insert({.id = target.id, .offset_flag = false});
    }

    context.state.may.insert_or_assign(vN_id, std::move(moved_value));
}

void apply_transfer_may_memset(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id = context.operands_id[0];

    const auto vN_id_may_iter = require_tracked_non_top_pointer(context, vN_id, "MEMSET");
    if (vN_id_may_iter == context.state.may.end())
    {
        return;
    }

    const auto vN_id_may_copy = vN_id_may_iter->second;
    for (const auto& target : vN_id_may_copy)
    {
        if (!is_dereferenceable_target(target))
        {
            continue;
        }

        set_top(context.state.may, target.id);
    }
}

void apply_transfer_may_call(const MayTransferContextBundle& context)
{
    for (std::size_t i = 1; i < context.operands_count; ++i)
    {
        // TODO:
        // set_reachable_to_top(context.operands_id[i], context,
        // grouped_objects::OUT_OF_LOCAL_SCOPE);
        handle_call_boundary(context.operands_id[i], context, true);
    }
}

void apply_transfer_may_stacksave(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto lN_id = context.operands_id[0];
    set_singleton(context.state.may, lN_id, {.id = context.last_local_id, .offset_flag = false});
}

void apply_transfer_may_stackrestore(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = context.state.may.find(vN_id);

    if (vN_id_may_iter == context.state.may.end())
    {
        // std::cout << serialize_program_point(context.pp) << " NUKING STACKRESTORE: untracked"
        //           << std::endl;
        poison_may(context);
        set_top(context.state.may, vN_id);
        return;
    }

    if (!contains_only_objectId(vN_id_may_iter->second, context.last_local_id))
    {
        // std::cout << serialize_program_point(context.pp)
        //           << " NUKING STACKRESTORE: invalid saved stack marker" << std::endl;
        poison_may(context);
        return;
    }
}

void apply_transfer_may_va_start(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id          = context.operands_id[0];
    const auto vN_id_may_iter = require_tracked_non_top_pointer(context, vN_id, "VA_START");
    if (vN_id_may_iter == context.state.may.end())
    {
        return;
    }

    const auto vN_targets_copy = vN_id_may_iter->second;
    for (const auto& target : vN_targets_copy)
    {
        if (!is_dereferenceable_target(target))
        {
            continue;
        }

        const auto target_may_iter = context.state.may.find(target.id);
        if (target_may_iter != context.state.may.end())
        {
            target_may_iter->second.insert(
                    {.id = grouped_objects::VARGARG_BLOCK, .offset_flag = false});
        }
        else
        {
            set_singleton(context.state.may, target.id,
                          {.id = grouped_objects::VARGARG_BLOCK, .offset_flag = false});
        }
    }
}

void apply_transfer_may_va_end(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    constexpr auto max_vargarg_block_depth = 2;

    const auto vN_id = context.operands_id[0];
    if (!is_objectId_reachable(context, vN_id, grouped_objects::VARGARG_BLOCK,
                               max_vargarg_block_depth))
    {
        // std::cout << serialize_program_point(context.pp)
        //           << " NUKING VA_END: does not contain VARGARG_BLOCK" << std::endl;
        poison_may(context);
    }
}

void apply_transfer_may_va_arg(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    auto vM_id_may_iter = require_tracked_non_top_pointer(context, vM_id, "VA_ARG");
    if (vM_id_may_iter == context.state.may.end())
    {
        return;
    }

    if (!contains_objectId(vM_id_may_iter->second, grouped_objects::VARGARG_BLOCK))
    {
        // std::cout << serialize_program_point(context.pp)
        //           << " NUKING VA_ARG: does not contain VARGARG_BLOCK" << std::endl;
        poison_may(context);
        return;
    }

    auto vN_id_may_iter = context.state.may.find(vN_id);
    if (vN_id_may_iter == context.state.may.end())
    {
        vN_id_may_iter = context.state.may.insert_or_assign(vN_id, vM_id_may_iter->second).first;
    }
    else
    {
        vN_id_may_iter->second.join_with(vM_id_may_iter->second);
    }

    vN_id_may_iter->second.insert({grouped_objects::VARGARG_BLOCK, true});
}

void apply_transfer_may_va_copy(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    constexpr auto max_vargarg_block_depth = 2;
    const auto     vN_id                   = context.operands_id[0];
    const auto     vM_id                   = context.operands_id[1];

    const auto vN_id_may_iter = context.state.may.find(vN_id);
    if (vN_id_may_iter == context.state.may.end())
    {
        // std::cout << serialize_program_point(context.pp) << " NUKING VA_COPY: vN untracked"
        //           << std::endl;
        poison_may(context);
        return;
    }

    if (!is_objectId_reachable(context, vM_id, grouped_objects::VARGARG_BLOCK,
                               max_vargarg_block_depth))
    {
        // std::cout << serialize_program_point(context.pp)
        //           << " NUKING VA_COPY: vM does not contain VARGARG_BLOCK" << std::endl;
        poison_may(context);
        return;
    }

    const auto vN_targets_copy = vN_id_may_iter->second;
    for (const auto& target : vN_targets_copy)
    {
        if (!is_dereferenceable_target(target))
        {
            continue;
        }

        if (auto target_id_may_iter = context.state.may.find(target.id);
            target_id_may_iter != context.state.may.end())
        {
            target_id_may_iter->second.insert({grouped_objects::VARGARG_BLOCK, false});
        }
        else
        {
            set_singleton(context.state.may, target.id,
                          {.id = grouped_objects::VARGARG_BLOCK, .offset_flag = false});
        }
    }
}

void apply_transfer_may(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= context.operands_count);
    if (context.state.poisoned)
    {
        return;
    }

    switch (context.opcode)
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::FREE:
    case sala::Instruction::Opcode::RET:
        return;

    case sala::Instruction::Opcode::ADDRESS:
        return apply_transfer_may_address(context); // correct

    case sala::Instruction::Opcode::COPY:
        return apply_transfer_may_copy(context); // correct

    case sala::Instruction::Opcode::LOAD:
        return apply_transfer_may_load(context);

    case sala::Instruction::Opcode::STORE:
        return apply_transfer_may_store(context);

    case sala::Instruction::Opcode::ALLOCA:
        return apply_transfer_may_alloca(context); // correct

    case sala::Instruction::Opcode::MALLOC:
        return apply_transfer_may_malloc(context); // correct

    case sala::Instruction::Opcode::I2P:
    case sala::Instruction::Opcode::P2I:
        return apply_transfer_may_i2p_p2i(context); // correct

    case sala::Instruction::Opcode::MEMMOVE:
    case sala::Instruction::Opcode::MEMCPY:
        return apply_transfer_may_memcpy_memmove(context);

    case sala::Instruction::Opcode::MOVEPTR:
        return apply_transfer_may_moveptr(context); // correct

    case sala::Instruction::Opcode::MEMSET:
        return apply_transfer_may_memset(context); // correct

    case sala::Instruction::Opcode::STACKRESTORE:
        return apply_transfer_may_stackrestore(context);

    case sala::Instruction::Opcode::STACKSAVE:
        return apply_transfer_may_stacksave(context);

    case sala::Instruction::Opcode::VA_START:
        return apply_transfer_may_va_start(context);

    case sala::Instruction::Opcode::VA_END:
        return apply_transfer_may_va_end(context);

    case sala::Instruction::Opcode::VA_ARG:
        return apply_transfer_may_va_arg(context);

    case sala::Instruction::Opcode::VA_COPY:
        return apply_transfer_may_va_copy(context);

    case sala::Instruction::Opcode::CALL:
        return apply_transfer_may_call(context);

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
        context.state.may.erase(vN_id);
        return;
    }
    }

    UNREACHABLE();
}

} // namespace optimizer::utils::points_to
