#include <optimizer/analysis/points_to_query.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <optimizer/utils/points_to/import.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::analysis
{

namespace
{
using MayState         = utils::MayState;
using MayValue         = utils::MayValue;
using MayAnalysisState = utils::MayAnalysisState;

inline objectId get_variable_id(const program::VariableIR& variable)
{
    return variable.get_metadata().get<metadata::points_to::VariableMeta>().id;
}

inline objectId get_constant_id(const program::ConstantIR& constant)
{
    return constant.get_metadata().get<metadata::points_to::ConstantMeta>().id;
}

inline objectId get_operand_id(const program::InstructionIR_sptr& instruction,
                               const std::size_t                  position)
{
    const auto operand_raw = instruction->get_operands().at(position);

    if (std::holds_alternative<program::FunctionIR_raw>(operand_raw))
    {
        return utils::grouped_objects::FUNCTION;
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

inline std::size_t get_basic_block_index(const program::FunctionIR_csptr& function,
                                         const program::BasicBlockIR_raw  bb_raw)
{
    ASSUMPTION(function != nullptr);
    ASSUMPTION(bb_raw != nullptr);

    std::size_t idx = 0;
    for (const auto& bb : function->get_basic_blocks())
    {
        if (bb.get() == bb_raw)
        {
            return idx;
        }
        ++idx;
    }

    ASSUMPTION(false);
}

inline std::size_t get_instruction_index(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb = instruction->get_basic_block();
    ASSUMPTION(bb != nullptr);

    std::size_t idx = 0;
    for (const auto& current : bb->get_instructions())
    {
        if (current.get() == instruction.get())
        {
            return idx;
        }
        ++idx;
    }

    ASSUMPTION(false);
}

inline objectId compute_last_local_id(const metadata::points_to::ObjectPool& local_objects)
{
    objectId result = 0;
    bool     first  = true;

    for (const auto& [id, object] : local_objects)
    {
        (void)object;
        if (first || id > result)
        {
            result = id;
            first  = false;
        }
    }

    return result;
}

inline PointsToResult make_poisoned_result()
{
    PointsToResult result{};
    result.poisoned = true;
    result.must     = std::nullopt;
    result.may.clear();
    return result;
}

inline PointsToResult make_result_from_value(const MayValue* value)
{
    PointsToResult result{};

    if (value == nullptr)
    {
        result.poisoned = false;
        result.must     = std::nullopt;
        result.may.clear();
        return result;
    }

    result.poisoned = false;
    result.must     = value->must_fact();
    result.may      = value->targets;
    return result;
}

inline PointsToResult make_result_from_state(const MayAnalysisState& state,
                                             const objectId          queried_id)
{
    if (state.poisoned)
    {
        return make_poisoned_result();
    }

    const auto it = state.may.find(queried_id);
    return make_result_from_value(it == state.may.end() ? nullptr : &it->second);
}

} // namespace

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

    cache_.bb    = nullptr;
    cache_.instr = nullptr;
    cache_.state.clear();
}

[[nodiscard]] const metadata::points_to::ObjectPool*
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

    const auto* bb_meta = bb_raw->get_metadata().get_raw<metadata::points_to::BasicBlockMeta>();
    ASSUMPTION(bb_meta != nullptr);
    ASSUMPTION(function_meta_ != nullptr);
    ASSUMPTION(function_meta_->bb_may_state_store != nullptr);

    cache_.bb    = bb_raw;
    cache_.instr = instruction.get();
    cache_.state = function_meta_->bb_may_state_store->get(bb_meta->may_in_id);

    const auto bb_index = get_basic_block_index(function_, bb_raw);

    std::size_t current_instr_index = 0;
    for (const auto& current_instruction : bb_raw->get_instructions())
    {
        if (current_instruction.get() == instruction.get())
        {
            break;
        }

        apply_transfer_to_state(current_instruction, cache_.state, bb_index, current_instr_index);
        ++current_instr_index;

        if (cache_.state.poisoned)
        {
            break;
        }
    }
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

    auto after_state = cache_.state;
    if (!after_state.poisoned)
    {
        const auto bb_index = get_basic_block_index(function_, instruction->get_basic_block_raw());
        const auto instr_index = get_instruction_index(instruction);
        apply_transfer_to_state(instruction, after_state, bb_index, instr_index);
    }

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
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    populate_cache_before(instruction);

    const auto queried_id = get_constant_id(x);
    return make_result_from_state(cache_.state, queried_id);
}

PointsToResult PointsToQueryFunction::after(const program::InstructionIR_sptr& instruction,
                                            const program::ConstantIR&         x)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    populate_cache_before(instruction);

    auto after_state = cache_.state;
    if (!after_state.poisoned)
    {
        const auto bb_index    = get_basic_block_index(function_, bb_raw);
        const auto instr_index = get_instruction_index(instruction);
        apply_transfer_to_state(instruction, after_state, bb_index, instr_index);
    }

    const auto queried_id = get_constant_id(x);
    return make_result_from_state(after_state, queried_id);
}

std::optional<program::OperandIR_sptr> PointsToQueryFunction::get_object(objectId id) const
{
    for (const auto& parameter : function_->get_parameters())
    {
        const auto* meta = parameter->get_metadata().get_raw<metadata::points_to::VariableMeta>();
        if (meta != nullptr && meta->id == id)
        {
            return program::OperandIR_sptr{parameter};
        }
    }

    for (const auto& variable : function_->get_local_variables())
    {
        const auto* meta = variable->get_metadata().get_raw<metadata::points_to::VariableMeta>();
        if (meta != nullptr && meta->id == id)
        {
            return program::OperandIR_sptr{variable};
        }
    }

    for (const auto& variable : program_keepalive_->get_static_vars())
    {
        const auto* meta = variable->get_metadata().get_raw<metadata::points_to::VariableMeta>();
        if (meta != nullptr && meta->id == id)
        {
            return program::OperandIR_sptr{variable};
        }
    }

    for (const auto& constant : program_keepalive_->get_constants())
    {
        const auto* meta = constant->get_metadata().get_raw<metadata::points_to::ConstantMeta>();
        if (meta != nullptr && meta->id == id)
        {
            return program::OperandIR_sptr{constant};
        }
    }

    return std::nullopt;
}

} // namespace optimizer::analysis
