#include <optimizer/translation/ir_to_sala.hpp>

#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::translation
{
using MetaKey = metadata::MetaKey;

std::shared_ptr<sala::Program> IRToSala::translate(const program::ProgramIR_sptr& source)
{
    ir_program    = source;
    sala_program_ = std::make_unique<sala::Program>();
    sala_program_->set_num_cpu_bits(source->get_num_cpu_bits());

    const auto& metadata = source->get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::ProgramMeta>());
    const auto& translation_meta = metadata.get<metadata::translation::ProgramMeta>();

    sala_program_->set_name(translation_meta.name);
    sala_program_->set_system(translation_meta.system);

    for (const auto& constant_ir : source->get_constants())
    {
        const auto id            = static_cast<uint32_t>(sala_program_->constants().size());
        auto       constant_sala = translate_part(constant_ir);
        constant_sala.set_index(id);
        sala_program_->push_back_constant() = std::move(constant_sala);

        const auto [_, succes] = constant_map_.try_emplace(constant_ir, id);
        ASSUMPTION(succes);
    };

    for (const auto& static_variable_ir : source->get_static_vars())
    {
        ASSUMPTION(static_variable_ir->get_context() == program::VariableIR::Context::STATIC);
        const auto id = static_cast<uint32_t>(sala_program_->static_variables().size());

        auto static_var_sala = translate_part(static_variable_ir);
        static_var_sala.set_index(id);
        static_var_sala.set_region(sala::Variable::Region::STATIC);

        const auto [_, succes] = static_variables_map_.try_emplace(static_variable_ir, id);
        ASSUMPTION(succes);

        sala_program_->push_back_static_variable() = static_var_sala;

        if (static_variable_ir->get_external_flag())
        {
            const auto& var_metadata = static_variable_ir->get_metadata();
            ASSUMPTION(var_metadata.has<metadata::translation::VariableMeta>());
            const auto& translation_meta = var_metadata.get<metadata::translation::VariableMeta>();
            sala_program_->push_back_external_variable(id, translation_meta.external_name.value());
        }
    }

    prefill_function_map();
    for (const auto& [function_ir, id] : function_map_)
    {
        sala_program_->function_ref(id) = translate_part(function_ir);

        if (function_ir == source->get_entry_func())
        {
            sala_program_->set_entry_function(id);
        }

        if (function_ir == source->get_static_initializer_func())
        {
            ASSUMPTION(id == 0);
        }

        if (function_ir->get_external_flag())
        {
            sala_program_->push_back_external_function(id);
        }
    }

    auto extracted_program = std::move(sala_program_);
    clear_program_context();
    return extracted_program;
}

sala::Function IRToSala::translate_part(const program::FunctionIR_sptr& source)
{
    const auto func_id       = function_map_.at(source);
    auto       function_sala = sala::Function();

    const auto& metadata = source->get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::FunctionMeta>());
    const auto& translation_meta = metadata.get<metadata::translation::FunctionMeta>();

    function_sala.set_program(sala_program_.get());
    function_sala.set_name(translation_meta.name);
    function_sala.set_external(source->get_external_flag());
    function_sala.source_back_mapping() = translation_meta.source_back_mapping;
    function_sala.set_index(func_id);

    for (const auto& parameter_ir : source->get_parameters())
    {
        ASSUMPTION(parameter_ir->get_context() == program::VariableIR::Context::PARAMETER);
        const auto id          = function_sala.parameters().size();
        const auto [_, succes] = parameters_map_.try_emplace(parameter_ir, id);
        ASSUMPTION(succes);

        auto parameter_sala = translate_part(parameter_ir);
        parameter_sala.set_program(sala_program_.get());
        parameter_sala.set_index(id);
        parameter_sala.set_function_index(func_id);
        parameter_sala.set_region(sala::Variable::Region::STACK);

        function_sala.push_back_parameter() = parameter_sala;
    }

    for (const auto& local_var_ir : source->get_local_variables())
    {
        ASSUMPTION(local_var_ir->get_context() == program::VariableIR::Context::LOCAL);
        const auto id          = function_sala.local_variables().size();
        const auto [_, succes] = local_variables_map_.try_emplace(local_var_ir, id);
        ASSUMPTION(succes);

        auto local_var_sala = translate_part(local_var_ir);
        local_var_sala.set_function_index(func_id);
        local_var_sala.set_index(id);
        local_var_sala.set_region(sala::Variable::Region::STACK);
        local_var_sala.set_program(sala_program_.get());
        function_sala.push_back_local_variable() = local_var_sala;
    }

    prefill_b_block_map(source, function_sala);
    for (const auto& [basic_block_ir, id] : b_block_map_)
    {
        auto basic_block_sala = translate_part(basic_block_ir);
        basic_block_sala.set_function_index(func_id);
        function_sala.basic_block_ref(id) = std::move(basic_block_sala);
    }

    clear_function_context();
    return function_sala;
}

sala::BasicBlock IRToSala::translate_part(const program::BasicBlockIR_sptr& source)
{
    const auto basic_block_id   = b_block_map_.at(source);
    auto       basic_block_sala = sala::BasicBlock();
    basic_block_sala.set_index(basic_block_id);
    for (const auto& successor_ir : source->get_successors())
    {
        basic_block_sala.push_back_successor(b_block_map_.at(successor_ir.lock()));
    }

    for (const auto& instruction_ir : source->get_instructions())
    {

        auto instruction_sala = translate_part(instruction_ir);
        instruction_sala.set_basic_block_index(basic_block_id);
        instruction_sala.set_index(
                static_cast<std::uint32_t>(basic_block_sala.instructions().size()));
        basic_block_sala.push_back_instruction() = std::move(instruction_sala);
    }

    return basic_block_sala;
};

sala::Instruction IRToSala::translate_part(const program::InstructionIR_sptr& source)
{
    auto        instruction_sala = sala::Instruction();
    const auto& metadata         = source->get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::InstructionMeta>());
    const auto& translation_meta = metadata.get<metadata::translation::InstructionMeta>();

    instruction_sala.source_back_mapping() = translation_meta.source_back_mapping;
    instruction_sala.set_opcode(source->get_opcode());
    instruction_sala.set_modifier(source->get_modifier());

    for (const auto& operand_ir : source->get_operands())
    {
        if (std::holds_alternative<program::VariableIR_wptr>(operand_ir))
        {
            std::optional<std::uint32_t>                 id;
            std::optional<sala::Instruction::Descriptor> descriptor;
            const auto variable_operand_ir = std::get<program::VariableIR_wptr>(operand_ir).lock();
            const auto context             = variable_operand_ir->get_context();
            switch (context)
            {
            case (program::VariableIR::Context::LOCAL):
            {

                descriptor         = sala::Instruction::Descriptor::LOCAL;
                const auto id_iter = local_variables_map_.find(variable_operand_ir);
                ASSUMPTION(id_iter != local_variables_map_.end());
                id = id_iter->second;
                break;
            }
            case (program::VariableIR::Context::STATIC):
            {

                descriptor         = sala::Instruction::Descriptor::STATIC;
                const auto id_iter = static_variables_map_.find(variable_operand_ir);
                ASSUMPTION(id_iter != static_variables_map_.end());
                id = id_iter->second;
                break;
            }
            case (program::VariableIR::Context::PARAMETER):
            {

                descriptor         = sala::Instruction::Descriptor::PARAMETER;
                const auto id_iter = parameters_map_.find(variable_operand_ir);
                ASSUMPTION(id_iter != parameters_map_.end());
                id = id_iter->second;
                break;
            }
            case (program::VariableIR::Context::UNDEFINED):
                ASSUMPTION(false);
                throw std::runtime_error("Invalid operand");
            }
            INVARIANT(id.has_value());
            INVARIANT(descriptor.has_value());
            instruction_sala.push_back_operand(id.value(), descriptor.value());
        }

        else if (std::holds_alternative<program::ConstantIR_wptr>(operand_ir))
        {
            const auto constant_operand_ir = std::get<program::ConstantIR_wptr>(operand_ir).lock();
            const auto id_iter             = constant_map_.find(constant_operand_ir);
            ASSUMPTION(id_iter != constant_map_.end());
            instruction_sala.push_back_operand(id_iter->second,
                                               sala::Instruction::Descriptor::CONSTANT);
        }

        else if (std::holds_alternative<program::FunctionIR_wptr>(operand_ir))
        {
            const auto function_operand_ir = std::get<program::FunctionIR_wptr>(operand_ir).lock();
            const auto id_iter             = function_map_.find(function_operand_ir);
            ASSUMPTION(id_iter != function_map_.end());
            instruction_sala.push_back_operand(id_iter->second,
                                               sala::Instruction::Descriptor::FUNCTION);
        }
    }

    return instruction_sala;
}

sala::Variable IRToSala::translate_part(const program::VariableIR_sptr& source)
{
    auto variable_sala = sala::Variable();

    const auto& metadata = source->get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::VariableMeta>());
    const auto& translation_meta = metadata.get<metadata::translation::VariableMeta>();

    variable_sala.set_program(sala_program_.get());
    variable_sala.set_external(source->get_external_flag());
    variable_sala.set_num_bytes(source->get_num_bytes());
    variable_sala.source_back_mapping() = translation_meta.source_back_mapping;

    return variable_sala;
}

sala::Constant IRToSala::translate_part(const program::ConstantIR_sptr& source)
{
    auto constant_sala = sala::Constant();
    constant_sala.set_program(sala_program_.get());
    for (const auto& byte : source->get_bytes())
    {
        constant_sala.push_back_byte(byte);
    }

    return constant_sala;
}

void IRToSala::clear_program_context()
{
    sala_program_ = nullptr;
    function_map_.clear();
    constant_map_.clear();
    static_variables_map_.clear();

    clear_function_context();
}

void IRToSala::clear_function_context()
{
    b_block_map_.clear();
    parameters_map_.clear();
    local_variables_map_.clear();
}

void IRToSala::prefill_b_block_map(const program::FunctionIR_sptr& ir_function_context,
                                   sala::Function&                 sala_function_context)
{

    for (const auto& ir_b_block : ir_function_context->get_basic_blocks())
    {
        const auto b_block_id =
                static_cast<std::uint32_t>(sala_function_context.basic_blocks().size());
        sala_function_context.push_back_basic_block();
        const auto [_, succes] = b_block_map_.try_emplace(ir_b_block, b_block_id);
        ASSUMPTION(succes);
    }
}

void IRToSala::prefill_function_map()
{
    const auto static_init_func_ir = ir_program->get_static_initializer_func();
    const auto func_id = static_cast<std::uint32_t>(sala_program_->functions().size()); // 0
    sala_program_->push_back_function("") = {};
    const auto [_, succes]                = function_map_.emplace(static_init_func_ir, func_id);
    ASSUMPTION(succes);

    for (const auto& ir_func : ir_program->get_functions())
    {
        if (ir_func == static_init_func_ir)
        {
            continue;
        }

        const auto func_id = static_cast<std::uint32_t>(sala_program_->functions().size());
        sala_program_->push_back_function("") = {};
        const auto [_, succes]                = function_map_.emplace(ir_func, func_id);
        ASSUMPTION(succes);
    }
}
} // namespace optimizer::translation
