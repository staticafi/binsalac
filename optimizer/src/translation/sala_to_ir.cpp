#include <optimizer/translation/sala_to_ir.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <optimizer/metadata/translation.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::translation
{
using MetaKey = metadata::MetaKey;

program::ProgramIR_sptr SalaToIR::translate(const std::shared_ptr<sala::Program> source)
{
    for (const auto& [extern_var_id, name] : source->external_variables())
    {
        const auto [_, success] = external_variables_names_map_.try_emplace(extern_var_id, name);
        ASSUMPTION(success);
    }

    const auto program          = std::make_shared<program::ProgramIR>();
    program->get_num_cpu_bits() = source->num_cpu_bits();

    // FIXME: remove cpu bits from translation meta
    auto& metadata            = program->get_metadata();
    auto  translation_meta    = std::make_unique<metadata::translation::ProgramMeta>();
    translation_meta->name    = source->name();
    translation_meta->system  = source->system();
    translation_meta->version = source->version();

    metadata.set(std::move(translation_meta));

    for (std::size_t index = 0; index < source->constants().size(); ++index)
    {
        const auto& source_const = source->constants().at(index);
        const auto  constant     = translate_part(source_const);
        program->acquire_constant(constant);

        const auto [_, succes] = constant_map_.try_emplace(index, constant);
        ASSUMPTION(succes);
    }

    for (std::size_t index = 0; index < source->static_variables().size(); ++index)
    {
        const auto& source_static_variable = source->static_variables().at(index);
        const auto  static_variable        = translate_part(source_static_variable);
        static_variable->assign_as_static(program);

        const auto [_, succes] = static_variables_map_.try_emplace(index, static_variable);
        ASSUMPTION(succes);
    }

    const auto entry_function_id     = source->entry_function();
    const auto static_initializer_id = source->static_initializer();
    // prefill functions
    for (const auto& _ : source->functions())
    {
        functions_holder.emplace_back(std::make_shared<program::FunctionIR>());
    }

    for (const auto& source_function : source->functions())
    {
        const auto& index    = source_function.index();
        const auto  function = functions_holder.at(index);

        translate_part(source_function, functions_holder.at(index));
        function->assign_to_program(program);

        if (entry_function_id == index)
        {
            function->get_entry_flag() = true;
            program->get_entry_func()  = function;
        }

        if (static_initializer_id == index)
        {
            program->get_static_initializer_func() = function;
            function->get_initializer_flag()       = true;
        }
    }

    clear_program_context();
    return program;
}

void SalaToIR::translate_part(const sala::Function&           source,
                              const program::FunctionIR_sptr& destination)
{
    destination->get_external_flag()       = source.is_external();
    destination->get_initial_stack_bytes() = source.initial_stack_bytes();

    auto& metadata                        = destination->get_metadata();
    auto  translation_meta                = std::make_unique<metadata::translation::FunctionMeta>();
    translation_meta->name                = source.name();
    translation_meta->source_back_mapping = source.source_back_mapping();

    metadata.set(std::move(translation_meta));

    for (std::size_t index = 0; index < source.parameters().size(); ++index)
    {
        const auto& source_parameter   = source.parameters().at(index);
        const auto  parameter          = translate_part(source_parameter);
        parameter->get_external_flag() = source_parameter.is_external();
        parameter->assign_as_parameter(destination);

        const auto [_, succes] = parameters_map_.try_emplace(index, parameter);
        ASSUMPTION(succes);
    }

    for (std::size_t index = 0; index < source.local_variables().size(); ++index)
    {
        const auto& source_local_variable = source.local_variables().at(index);
        const auto  local_variable        = translate_part(source_local_variable);
        local_variable->assign_as_local(destination);

        const auto [_, success] = local_variables_map_.try_emplace(index, local_variable);
        ASSUMPTION(success);
    }

    constexpr std::uint32_t entry_basic_block_id = 0; // Implicit by the sala program representation
    for (const auto& source_b_block : source.basic_blocks())
    {
        const auto basic_block = translate_part(source_b_block);
        ASSUMPTION(basic_block != nullptr);
        basic_block->assign_to_function(destination);

        {
            const auto [_, succes] = b_block_map_.try_emplace(source_b_block.index(), basic_block);
            ASSUMPTION(succes);
        }
        {
            const auto [_, success] = b_block_succesors_map_.try_emplace(
                    source_b_block.index(), source_b_block.successors());
            ASSUMPTION(success);
        }
        if (entry_basic_block_id == source_b_block.index())
        {
            destination->get_entry_basic_block() = basic_block;
        }
    }

    resolve_b_blocks();
    clear_function_context();
}

program::BasicBlockIR_sptr SalaToIR::translate_part(const sala::BasicBlock& source)
{
    const auto basic_block = std::make_shared<program::BasicBlockIR>();
    for (const auto& source_instruction : source.instructions())
    {
        const auto instruction = translate_part(source_instruction);
        ASSUMPTION(instruction != nullptr);
        basic_block->acquire_instruction(instruction);
    }

    return basic_block;
}

program::InstructionIR_sptr SalaToIR::translate_part(const sala::Instruction& source)
{
    const auto instruction = std::make_shared<program::InstructionIR>();

    auto& metadata         = instruction->get_metadata();
    auto  translation_meta = std::make_unique<metadata::translation::InstructionMeta>();
    translation_meta->source_back_mapping = source.source_back_mapping();
    metadata.set(std::move(translation_meta));

    instruction->get_opcode()   = source.opcode();
    instruction->get_modifier() = source.modifier();
    resolve_instruction_operands(instruction, source);

    return instruction;
};

program::ConstantIR_sptr SalaToIR::translate_part(const sala::Constant& source)
{
    const auto constant   = std::make_shared<program::ConstantIR>();
    constant->get_bytes() = source.bytes();

    return constant;
}

program::VariableIR_sptr SalaToIR::translate_part(const sala::Variable& source)
{
    const auto variable                   = std::make_shared<program::VariableIR>();
    auto&      metadata                   = variable->get_metadata();
    auto       translation_meta           = std::make_unique<metadata::translation::VariableMeta>();
    translation_meta->source_back_mapping = source.source_back_mapping();
    if (source.is_external())
    {
        const auto it = external_variables_names_map_.find(source.index());
        ASSUMPTION(it != external_variables_names_map_.end());
        translation_meta->external_name = it->second;
    }

    metadata.set(std::move(translation_meta));
    variable->get_external_flag() = source.is_external();
    variable->get_num_bytes()     = source.num_bytes();

    return variable;
}

void SalaToIR::resolve_instruction_operands(const program::InstructionIR_sptr& instruction,
                                            const sala::Instruction&           source)
{
    for (std::size_t i = 0; i < source.operands().size(); ++i)
    {
        ASSUMPTION(i < source.descriptors().size());
        const auto source_id = source.operands().at(i);
        switch (source.descriptors().at(i))
        {
        case (sala::Instruction::Descriptor::CONSTANT):
        {
            const auto it = constant_map_.find(source_id);
            ASSUMPTION(it != constant_map_.end());
            instruction->push_back_operand(it->second);
            break;
        }
        case (sala::Instruction::Descriptor::FUNCTION):
        {
            instruction->push_back_operand(functions_holder.at(source_id));
            break;
        }
        case (sala::Instruction::Descriptor::STATIC):
        {
            const auto it = static_variables_map_.find(source_id);
            ASSUMPTION(it != static_variables_map_.end());
            instruction->push_back_operand(it->second);
            break;
        }
        case (sala::Instruction::Descriptor::PARAMETER):
        {
            const auto it = parameters_map_.find(source_id);
            ASSUMPTION(it != parameters_map_.end());
            instruction->push_back_operand(it->second);
            break;
        }
        case (sala::Instruction::Descriptor::LOCAL):
        {
            const auto it = local_variables_map_.find(source_id);
            ASSUMPTION(it != local_variables_map_.end());
            instruction->push_back_operand(it->second);
            break;
        }
        default:
            ASSUMPTION(false);
        }
    }
};

void SalaToIR::resolve_b_blocks() const
{
    for (const auto& [source_id, basic_block] : b_block_map_)
    {
        const auto iter = b_block_succesors_map_.find(source_id);
        ASSUMPTION(iter != b_block_succesors_map_.end());

        for (const auto succ_id : iter->second)
        {
            const auto successor = b_block_map_.at(succ_id);
            basic_block->add_successor(successor);
            successor->add_predecessor(basic_block);
        }
    }
}

void SalaToIR::clear_program_context()
{
    b_block_map_.clear();
    b_block_succesors_map_.clear();
    constant_map_.clear();
    functions_holder.clear();
    external_variables_names_map_.clear();

    clear_function_context();
}

void SalaToIR::clear_function_context()
{
    b_block_map_.clear();
    b_block_succesors_map_.clear();
    local_variables_map_.clear();
    parameters_map_.clear();
}
} // namespace optimizer::translation
