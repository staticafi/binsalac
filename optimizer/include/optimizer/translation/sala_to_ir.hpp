#ifndef OPTIMIZER_SALA_TO_IR_HPP_INCLUDED
#define OPTIMIZER_SALA_TO_IR_HPP_INCLUDED

#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/program_ir.hpp>

#include <unordered_map>

namespace optimizer::translation
{

class SalaToIR
{
  public:
    program::ProgramIR_sptr translate(const std::shared_ptr<sala::Program> source);

  private:
    void translate_part(const sala::Function& source, const program::FunctionIR_sptr& destination);
    program::BasicBlockIR_sptr  translate_part(const sala::BasicBlock& source);
    program::InstructionIR_sptr translate_part(const sala::Instruction& source);
    program::VariableIR_sptr    translate_part(const sala::Variable& source);
    program::ConstantIR_sptr    translate_part(const sala::Constant& source);

    void resolve_b_blocks() const;
    void resolve_instruction_operands(const program::InstructionIR_sptr& instruction,
                                      const sala::Instruction&           source);
    void use_dummy_function_operand(const program::InstructionIR_sptr& instruction,
                                    const sala::Instruction&           source);

    void clear_program_context();
    void clear_function_context();

  private:
    // Program context
    std::vector<program::FunctionIR_sptr>                       functions_holder;
    std::unordered_map<std::uint32_t, program::ConstantIR_sptr> constant_map_;
    std::unordered_map<std::uint32_t, program::VariableIR_sptr> static_variables_map_;
    std::unordered_map<std::uint32_t, std::string>              external_variables_names_map_;

    // Function context
    std::unordered_map<std::uint32_t, program::BasicBlockIR_sptr> b_block_map_;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> b_block_succesors_map_;
    std::unordered_map<std::uint32_t, program::VariableIR_sptr>   local_variables_map_;
    std::unordered_map<std::uint32_t, program::VariableIR_sptr>   parameters_map_;
};

} // namespace optimizer::translation

#endif
