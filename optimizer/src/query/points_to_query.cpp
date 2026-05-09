#include <optimizer/query/points_to_query.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <optimizer/utils/points_to/import.hpp>

#include <utility/assumptions.hpp>

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace optimizer::query
{

namespace
{
using MayValue         = utils::MayValue;
using MayAnalysisState = utils::MayAnalysisState;

objectId get_variable_id(const program::VariableIR& variable)
{
    return variable.get_metadata().get<metadata::points_to::VariableMeta>().id;
}

objectId get_constant_id(const program::ConstantIR& constant)
{
    return constant.get_metadata().get<metadata::points_to::ConstantMeta>().id;
}

objectId get_operand_id(const program::InstructionIR_sptr& instruction, const std::size_t position)
{
    const auto operand_raw = instruction->get_operands().at(position);

    if (std::holds_alternative<program::FunctionIR_raw>(operand_raw))
    {
        return utils::abstract_nodes::FUNCTION;
    }
    else if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand_raw))
    {
        ASSUMPTION(*variable_raw != nullptr);
        return (*variable_raw)->get_metadata().get<metadata::points_to::VariableMeta>().id;
    }
    else if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand_raw))
    {
        ASSUMPTION(*constant_raw != nullptr);
        return (*constant_raw)->get_metadata().get<metadata::points_to::ConstantMeta>().id;
    }

    ASSUMPTION(false);
}

objectId compute_last_local_id(const metadata::points_to::ObjectPool& local_objects)
{
    objectId result = 0;
    bool     first  = true;

    for (const auto& [id, object] : local_objects)
    {
        if (first || id > result)
        {
            result = id;
            first  = false;
        }
    }

    return result;
}

PointsToResult make_poisoned_result()
{
    PointsToResult result{};
    result.poisoned = true;
    result.unique   = std::nullopt;
    result.may.clear();
    return result;
}

PointsToResult make_result_from_value(const MayValue* value, std::optional<Target> unqiue)
{
    PointsToResult result{};

    if (value == nullptr)
    {
        result.poisoned = false;
        result.unique   = std::nullopt;
        result.may.clear();
        return result;
    }

    result.poisoned = false;
    result.unique   = std::move(unqiue);
    result.may      = value->targets;
    return result;
}

PointsToResult make_result_from_state(const MayAnalysisState& state, const objectId queried_id)
{
    if (state.poisoned)
    {
        return make_poisoned_result();
    }

    const auto may_it    = state.may.find(queried_id);
    const auto unqiue_it = state.unique.find(queried_id);

    return make_result_from_value(may_it == state.may.end() ? nullptr : &may_it->second,
                                  unqiue_it == state.unique.end()
                                          ? std::nullopt
                                          : std::optional<Target>{unqiue_it->second});
}

} // namespace

void PointsToQueryFunction::build_object_cache()
{
    object_cache_.clear();
    object_cache_.reserve(program_keepalive_->get_constants().size() +
                          program_keepalive_->get_static_vars().size() +
                          function_->get_parameters().size() +
                          function_->get_local_variables().size());

    const auto add_variable = [this](const auto& variable_sptr)
    {
        ASSUMPTION(variable_sptr != nullptr);

        const auto* meta =
                variable_sptr->get_metadata().template get_raw<metadata::points_to::VariableMeta>();

        if (meta != nullptr)
        {
            object_cache_.emplace_hint(object_cache_.end(), meta->id,
                                       program::OperandIR_sptr{variable_sptr});
        }
    };

    const auto add_constant = [this](const auto& constant_sptr)
    {
        ASSUMPTION(constant_sptr != nullptr);

        const auto* meta =
                constant_sptr->get_metadata().template get_raw<metadata::points_to::ConstantMeta>();

        if (meta != nullptr)
        {
            object_cache_.emplace_hint(object_cache_.end(), meta->id,
                                       program::OperandIR_sptr{constant_sptr});
        }
    };

    for (const auto& constant : program_keepalive_->get_constants())
    {
        add_constant(constant);
    }

    for (const auto& variable : program_keepalive_->get_static_vars())
    {
        add_variable(variable);
    }

    for (const auto& parameter : function_->get_parameters())
    {
        add_variable(parameter);
    }

    for (const auto& variable : function_->get_local_variables())
    {
        add_variable(variable);
    }
}

PointsToQueryFunction::PointsToQueryFunction(program::FunctionIR_csptr function)
    : program_keepalive_{}, function_{std::move(function)}
{
    ASSUMPTION(function_ != nullptr);

    program_keepalive_ = function_->get_program();
    ASSUMPTION(program_keepalive_ != nullptr);

    program_meta_ = program_keepalive_->get_metadata().get_raw<metadata::points_to::ProgramMeta>();
    ASSUMPTION(program_meta_ != nullptr);

    function_meta_ = function_->get_metadata().get_raw<metadata::points_to::FunctionMeta>();
    ASSUMPTION(function_meta_ != nullptr);

    global_objects_ = &program_meta_->global_objects;
    local_objects_  = &function_meta_->local_objects;

    transfer_may_ = program_meta_->transfer_may_;
    ASSUMPTION(static_cast<bool>(transfer_may_));

    last_local_id_ = compute_last_local_id(*local_objects_);
    build_object_cache();

    cache_.bb          = nullptr;
    cache_.instr       = nullptr;
    cache_.bb_index    = 0;
    cache_.instr_index = 0;
    cache_.state.clear();

    cache_.after_valid = false;
    cache_.after_state.clear();
}

std::size_t
PointsToQueryFunction::get_basic_block_index(const program::BasicBlockIR_raw bb_raw) const
{
    ASSUMPTION(bb_raw != nullptr);

    std::size_t index = 0;

    for (const auto& bb : function_->get_basic_blocks())
    {
        ASSUMPTION(bb != nullptr);

        if (bb.get() == bb_raw)
        {
            return index;
        }

        ++index;
    }

    ASSUMPTION(false);
}

const metadata::points_to::ObjectPool*
PointsToQueryFunction::get_object_pool(const program::InstructionIR_sptr& instruction,
                                       const program::VariableIR&         x)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    switch (x.get_context())
    {
    case program::VariableIR::Context::UNDEFINED:
        ASSUMPTION(false);
        break;

    case program::VariableIR::Context::PARAMETER:
    case program::VariableIR::Context::LOCAL:
        ASSUMPTION(x.get_function() == function_);
        return local_objects_;

    case program::VariableIR::Context::STATIC:
        return global_objects_;
    }

    ASSUMPTION(false);
}

void PointsToQueryFunction::apply_transfer_to_state(const program::InstructionIR_sptr& instruction,
                                                    MayAnalysisState&                  state,
                                                    const std::size_t                  bb_index,
                                                    const std::size_t                  instr_index)
{
    ASSUMPTION(instruction != nullptr);

    if (state.poisoned)
    {
        return;
    }

    const auto            relevant_ops_count = utils::get_relevant_operands_count(*instruction);
    std::vector<objectId> operands_id(relevant_ops_count);

    for (std::size_t i = 0; i < relevant_ops_count; ++i)
    {
        operands_id[i] = get_operand_id(instruction, i);
    }

    utils::MayTransferContextBundle context{
            .pp =
                    {
                            .function = 0,
                            .bb       = bb_index,
                            .instr    = instr_index,
                    },
            .opcode         = instruction->get_opcode(),
            .state          = state,
            .global_objects = *global_objects_,
            .local_objects  = *local_objects_,
            .operands_id    = operands_id,
            .operands_count = relevant_ops_count,
            .last_local_id  = last_local_id_,
    };

    transfer_may_(context);
}

void PointsToQueryFunction::reset_cached_after_state()
{
    cache_.after_valid = false;
    cache_.after_state.clear();
}

bool PointsToQueryFunction::try_advance_cache_to(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto target_bb = instruction->get_basic_block_raw();
    ASSUMPTION(target_bb != nullptr);

    if (cache_.bb != target_bb)
    {
        return false;
    }

    if (cache_.instr == nullptr)
    {
        return false;
    }

    auto self_it = cache_.instr->get_self_it();
    ASSUMPTION(self_it.has_value());

    auto it = *self_it;

    while (it != cache_.bb->get_instructions().end())
    {
        ASSUMPTION(*it != nullptr);
        ASSUMPTION(cache_.instr == it->get());

        if (it->get() == instruction.get())
        {
            return true;
        }

        if (cache_.after_valid)
        {
            cache_.state = cache_.after_state;
        }
        else
        {
            apply_transfer_to_state(*it, cache_.state, cache_.bb_index, cache_.instr_index);
        }

        ++it;
        ++cache_.instr_index;

        if (it == cache_.bb->get_instructions().end())
        {
            cache_.instr = nullptr;
        }
        else
        {
            ASSUMPTION(*it != nullptr);
            cache_.instr = it->get();
        }

        reset_cached_after_state();
    }

    return false;
}

void PointsToQueryFunction::rebuild_cache_before(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    const auto* bb_meta = bb_raw->get_metadata().get_raw<metadata::points_to::BasicBlockMeta>();
    ASSUMPTION(bb_meta != nullptr);
    ASSUMPTION(function_meta_ != nullptr);
    ASSUMPTION(function_meta_->bb_may_state_store != nullptr);

    cache_.bb          = bb_raw;
    cache_.instr       = instruction.get();
    cache_.bb_index    = get_basic_block_index(bb_raw);
    cache_.instr_index = 0;
    cache_.state       = function_meta_->bb_may_state_store->get(bb_meta->may_in_id);

    reset_cached_after_state();

    bool found = false;

    for (const auto& current_instruction : bb_raw->get_instructions())
    {
        ASSUMPTION(current_instruction != nullptr);

        if (current_instruction.get() == instruction.get())
        {
            found = true;
            break;
        }

        apply_transfer_to_state(current_instruction, cache_.state, cache_.bb_index,
                                cache_.instr_index);

        ++cache_.instr_index;
    }

    ASSUMPTION(found);
}

void PointsToQueryFunction::populate_cache_before(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    if (cache_.bb == bb_raw && cache_.instr == instruction.get())
    {
        return;
    }

    if (try_advance_cache_to(instruction))
    {
        return;
    }

    rebuild_cache_before(instruction);
}

MayAnalysisState PointsToQueryFunction::compute_after_state_from_cache(
        const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);
    ASSUMPTION(cache_.bb == instruction->get_basic_block_raw());
    ASSUMPTION(cache_.instr == instruction.get());

    if (!cache_.after_valid)
    {
        cache_.after_state = cache_.state;

        if (!cache_.after_state.poisoned)
        {
            apply_transfer_to_state(instruction, cache_.after_state, cache_.bb_index,
                                    cache_.instr_index);
        }

        cache_.after_valid = true;
    }

    return cache_.after_state;
}

MayAnalysisState
PointsToQueryFunction::handle_state_request(const program::InstructionIR_sptr& instruction,
                                            bool                               before)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    populate_cache_before(instruction);

    if (before)
    {
        return cache_.state;
    }

    return compute_after_state_from_cache(instruction);
}

PointsToResult
PointsToQueryFunction::handle_cache_hit(const program::InstructionIR_sptr& instruction,
                                        const program::VariableIR& x, bool before)
{
    ASSUMPTION(instruction != nullptr);
    ASSUMPTION(cache_.bb == instruction->get_basic_block_raw());
    ASSUMPTION(cache_.instr == instruction.get());

    const auto object_pool = get_object_pool(instruction, x);
    ASSUMPTION(object_pool != nullptr);

    const auto queried_id = get_variable_id(x);

    if (before)
    {
        return make_result_from_state(cache_.state, queried_id);
    }

    const auto after_state = compute_after_state_from_cache(instruction);
    return make_result_from_state(after_state, queried_id);
}

PointsToResult PointsToQueryFunction::handle_request(const program::InstructionIR_sptr& instruction,
                                                     const program::VariableIR& x, bool before)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    populate_cache_before(instruction);
    return handle_cache_hit(instruction, x, before);
}

MayAnalysisState PointsToQueryFunction::before_state(const program::InstructionIR_sptr& instruction)
{
    return handle_state_request(instruction, true);
}

MayAnalysisState PointsToQueryFunction::after_state(const program::InstructionIR_sptr& instruction)
{
    return handle_state_request(instruction, false);
}

PointsToResult PointsToQueryFunction::before(const program::InstructionIR_sptr& instruction,
                                             const program::VariableIR&         x)
{
    return handle_request(instruction, x, true);
}

PointsToResult PointsToQueryFunction::after(const program::InstructionIR_sptr& instruction,
                                            const program::VariableIR&         x)
{
    return handle_request(instruction, x, false);
}

PointsToResult PointsToQueryFunction::before(const program::InstructionIR_sptr& instruction,
                                             const program::ConstantIR&         x)
{
    const auto state = before_state(instruction);
    return make_result_from_state(state, get_constant_id(x));
}

PointsToResult PointsToQueryFunction::after(const program::InstructionIR_sptr& instruction,
                                            const program::ConstantIR&         x)
{
    const auto state = after_state(instruction);
    return make_result_from_state(state, get_constant_id(x));
}

std::optional<program::OperandIR_sptr> PointsToQueryFunction::get_object(objectId id) const
{
    const auto object_cache_iter = object_cache_.find(id);
    if (object_cache_iter == object_cache_.end())
    {
        return std::nullopt;
    }

    return object_cache_iter->second;
}

} // namespace optimizer::query
