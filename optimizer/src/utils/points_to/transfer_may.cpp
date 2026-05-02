#include <optimizer/utils/points_to/transfer_may.hpp>

#include <iostream>
#include <span>
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

inline bool is_exact_target(const Target& target) noexcept
{
    return is_concrete_object(target.id) && !target.offset_flag;
}

inline void erase_cell(MayAnalysisState& state, objectId cell)
{
    state.may.erase(cell);
    state.must.erase(cell);
}

inline void set_top_cell(MayAnalysisState& state, objectId cell)
{
    state.may.insert_or_assign(cell, MayValue::top());
    state.must.erase(cell);
}

inline void set_singleton_cell(MayAnalysisState& state, objectId cell, Target target)
{
    state.may.insert_or_assign(cell, MayValue::singleton(target));

    if (is_exact_target(target))
    {
        state.must.insert_or_assign(cell, target);
    }
    else
    {
        state.must.erase(cell);
    }
}

inline void copy_cell(MayAnalysisState& state, objectId dst, objectId src)
{
    if (dst == src)
    {
        return;
    }

    if (const auto may_it = state.may.find(src); may_it != state.may.end())
    {
        state.may.insert_or_assign(dst, may_it->second);
    }
    else
    {
        state.may.erase(dst);
    }

    if (const auto must_it = state.must.find(src); must_it != state.must.end())
    {
        state.must.insert_or_assign(dst, must_it->second);
    }
    else
    {
        state.must.erase(dst);
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

inline void strong_transfer(objectId copy_dest, objectId copy_source,
                            const MayTransferContextBundle& context)
{
    copy_cell(context.state, copy_dest, copy_source);
}

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

bool merge_object_points_to_into_object(MayAnalysisState& state, const objectId source_id,
                                        const objectId target_id)
{
    ASSUMPTION(can_have_state_cell(target_id));

    const auto source_may_iter = state.may.find(source_id);
    auto       target_may_iter = state.may.find(target_id);

    if (source_may_iter == state.may.end())
    {
        // Missing source contains no may information to join.
        // The destination may component is unchanged, but exactness is lost.
        state.must.erase(target_id);
        return false;
    }

    if (is_top_value(source_may_iter->second))
    {
        set_top_cell(state, target_id);
        return true;
    }

    const auto source_copy = source_may_iter->second;
    if (target_may_iter == state.may.end())
    {
        state.may.insert_or_assign(target_id, source_copy);
    }
    else
    {
        target_may_iter->second.join_with(source_copy);
    }

    const auto source_must_iter = state.must.find(source_id);
    const auto target_must_iter = state.must.find(target_id);

    if (source_must_iter == state.must.end() || target_must_iter == state.must.end() ||
        !(source_must_iter->second == target_must_iter->second))
    {
        state.must.erase(target_id);
    }

    return true;
}
// Merge pointee contents from all dereferenceable source targets into all
// dereferenceable destination targets.
//
// Non-dereferenceable targets are ignored here because they cannot denote state cells.
// If tracked and untracked dereferenceable sources are mixed, destination cells lose
// precision via merge_unknown.
void merge_pointees_into_pointees(MayAnalysisState& state, const MayValue& dst_ptr_value,
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

        const bool source_tracked = is_tracked(state.may, source.id);
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

            state.must.erase(target.id);
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
void store_through_pointer(MayAnalysisState& state, MayValue& dst_ptr_value, objectId src_id)
{
    ASSUMPTION(!is_top_value(dst_ptr_value));

    const auto              dst_targets_copy = dst_ptr_value;
    std::optional<MayValue> transfer;

    if (const auto src_may_iter = state.may.find(src_id); src_may_iter != state.may.end())
    {
        transfer = src_may_iter->second;
    }
    else if (is_dereferenceable_summary_target(src_id))
    {
        transfer = make_singleton(src_id);
    }

    const auto src_must_iter   = state.must.find(src_id);
    const bool source_has_must = src_must_iter != state.must.end();

    for (const auto& target : dst_targets_copy)
    {
        if (!is_dereferenceable_target(target))
        {
            continue;
        }

        const auto target_may_iter = state.may.find(target.id);

        if (target_may_iter != state.may.end())
        {
            if (transfer.has_value())
            {
                target_may_iter->second.join_with(transfer.value());
            }
            else
            {
                // No may information to join from the source.
                // Exactness is handled below.
            }
        }
        else if (transfer.has_value())
        {
            state.may.insert_or_assign(target.id, transfer.value());
        }

        if (!source_has_must)
        {
            state.must.erase(target.id);
            continue;
        }

        const auto target_must_iter = state.must.find(target.id);
        if (target_must_iter == state.must.end() ||
            !(target_must_iter->second == src_must_iter->second))
        {
            state.must.erase(target.id);
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
    return result;
}

std::optional<Target> load_must_through_pointer(const MustState& must, const MayValue& ptr_value,
                                                const bool any_untracked_or_ambiguous)
{
    if (ptr_value.is_top || any_untracked_or_ambiguous)
    {
        return std::nullopt;
    }

    std::optional<Target> result;

    for (const auto& pointee : ptr_value)
    {
        if (!can_have_state_cell(pointee.id))
        {
            return std::nullopt;
        }

        const auto it = must.find(pointee.id);
        if (it == must.end())
        {
            return std::nullopt;
        }

        if (!result.has_value())
        {
            result = it->second;
        }
        else if (!(result.value() == it->second))
        {
            return std::nullopt;
        }
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

    if (const auto must_it = context.state.must.find(vM_id); must_it != context.state.must.end())
    {
        strong_transfer(vN_id, must_it->second.id, context);
        return;
    }
    bool any_tracked   = false;
    bool any_untracked = false;

    MayValue loaded = load_through_pointer(context.state.may, vM_id_may_iter->second, any_tracked,
                                           any_untracked);
    if (any_tracked)
    {
        context.state.may.insert_or_assign(vN_id, std::move(loaded));

        if (auto loaded_must = load_must_through_pointer(context.state.must, vM_id_may_iter->second,
                                                         any_untracked);
            loaded_must.has_value())
        {
            context.state.must.insert_or_assign(vN_id, loaded_must.value());
        }
        else
        {
            context.state.must.erase(vN_id);
        }
    }
    else
    {
        context.state.must.erase(vN_id);
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

    if (const auto must_it = context.state.must.find(vN_id); must_it != context.state.must.end())
    {
        strong_transfer(must_it->second.id, vM_id, context);
        return;
    }

    store_through_pointer(context.state, vN_id_may_iter->second, vM_id);
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

    merge_pointees_into_pointees(context.state, vN_id_may_iter->second, vM_id_may_iter->second);
}

// ADDRESS n vN rM; vN = &rM
void apply_transfer_may_address(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];
    // strong update
    set_singleton_cell(context.state, vN_id, Target{.id = vM_id, .offset_flag = false});
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
    set_singleton_cell(context.state, vN_id,
                       Target{.id = grouped_objects::ALLOCA, .offset_flag = false});
}

void apply_transfer_may_malloc(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto vN_id = context.operands_id[0];
    set_singleton_cell(context.state, vN_id,
                       Target{.id = grouped_objects::HEAP, .offset_flag = false});
}

void apply_transfer_may_p2i(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);

    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    if (context.state.may.find(vM_id) == context.state.may.end())
    {
        erase_cell(context.state, vN_id);
        return;
    }

    copy_cell(context.state, vN_id, vM_id);
}

void apply_transfer_may_i2p(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);

    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_may_iter = require_tracked_pointer(context, vM_id, "I2P");
    if (vM_id_may_iter == context.state.may.end())
    {
        return;
    }

    copy_cell(context.state, vN_id, vM_id);
}

void apply_transfer_may_moveptr(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);

    const auto vN_id = context.operands_id[0];
    const auto vM_id = context.operands_id[1];

    const auto vM_id_may_iter = require_tracked_pointer(context, vM_id, "MOVEPTR");
    if (vM_id_may_iter == context.state.may.end())
    {
        return;
    }

    if (vM_id_may_iter->second.is_top)
    {
        set_top_cell(context.state, vN_id);
        return;
    }

    MayValue moved_value;

    const auto src_copy = vM_id_may_iter->second;
    for (const auto& target : src_copy)
    {
        moved_value.insert(Target{.id = target.id, .offset_flag = false});
        moved_value.insert(Target{.id = target.id, .offset_flag = true});
    }

    context.state.may.insert_or_assign(vN_id, std::move(moved_value));
    context.state.must.erase(vN_id);
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

        set_top_cell(context.state, target.id);
    }
}

void apply_transfer_may_call(const MayTransferContextBundle& context)
{
    std::span<const objectId> escapees =
            context.operands_id.size() > 1
                    ? std::span<const objectId>(context.operands_id).subspan(1)
                    : std::span<const objectId>{};
    handle_call_boundary(escapees, context, false);
}

void apply_transfer_may_stacksave(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 1);
    ASSUMPTION(context.operands_count == 1);
    const auto lN_id = context.operands_id[0];
    set_singleton_cell(context.state, lN_id,
                       Target{.id = context.last_local_id, .offset_flag = false});
}

void apply_transfer_may_stackrestore(const MayTransferContextBundle& context)
{
    {
        ASSUMPTION(context.operands_id.size() >= 1);
        ASSUMPTION(context.operands_count == 1);

        const auto vN_id = context.operands_id[0];

        const auto must_it = context.state.must.find(vN_id);
        if (must_it == context.state.must.end() || must_it->second.id != context.last_local_id ||
            must_it->second.offset_flag)
        {
            poison_may(context);
            return;
        }
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
            set_singleton_cell(context.state, target.id,
                               Target{.id = grouped_objects::VARGARG_BLOCK, .offset_flag = false});
        }

        context.state.must.erase(target.id);
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
    context.state.must.erase(vN_id);
}

void apply_transfer_may_va_copy(const MayTransferContextBundle& context)
{
    ASSUMPTION(context.operands_id.size() >= 2);
    ASSUMPTION(context.operands_count == 2);
    constexpr auto max_vargarg_block_depth = 2;
    const auto     vN_id                   = context.operands_id[0];
    const auto     vM_id                   = context.operands_id[1];

    const auto vN_id_may_iter = require_tracked_non_top_pointer(context, vN_id, "VA_COPY");
    if (vN_id_may_iter == context.state.may.end())
    {
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
            set_singleton_cell(context.state, target.id,
                               Target{.id = grouped_objects::VARGARG_BLOCK, .offset_flag = false});
        }
        context.state.must.erase(target.id);
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
        return apply_transfer_may_i2p(context);
    case sala::Instruction::Opcode::P2I:
        return apply_transfer_may_p2i(context);

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
        erase_cell(context.state, vN_id);
        return;
    }
    }

    UNREACHABLE();
}

} // namespace optimizer::utils::points_to
