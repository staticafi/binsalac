#ifndef OPTIMIZER_IR_TO_SALA_HPP_INCLUDED
#define OPTIMIZER_IR_TO_SALA_HPP_INCLUDED

#include <optimizer/programIR/program_repr.hpp>
#include <unordered_map>

namespace optimizer::translation
{

class IRToSala
{
  public:
    std::shared_ptr<sala::Program> translate(const program::ProgramIR_sptr& source);

  private:
    sala::Function    translate_part(const program::FunctionIR_sptr& source);
    sala::BasicBlock  translate_part(const program::BasicBlockIR_sptr& source);
    sala::Instruction translate_part(const program::InstructionIR_sptr& source);
    sala::Variable    translate_part(const program::VariableIR_sptr& source);
    sala::Constant    translate_part(const program::ConstantIR_sptr& source);

    void clear_program_context();
    void clear_function_context();

    void prefill_function_map();
    void prefill_b_block_map(const program::FunctionIR_sptr& ir_function_context,
                             sala::Function&                 sala_function_context);

  private:
    // Program context
    program::ProgramIR_sptr                                     ir_program;
    std::unique_ptr<sala::Program>                              sala_program_;
    std::unordered_map<program::FunctionIR_sptr, std::uint32_t> function_map_;
    std::unordered_map<program::ConstantIR_sptr, std::uint32_t> constant_map_;
    std::unordered_map<program::VariableIR_sptr, std::uint32_t> static_variables_map_;

    // Function context
    std::unordered_map<program::BasicBlockIR_sptr, std::uint32_t> b_block_map_;
    std::unordered_map<program::VariableIR_sptr, std::uint32_t>   parameters_map_;
    std::unordered_map<program::VariableIR_sptr, std::uint32_t>   local_variables_map_;
};

} // namespace optimizer::translation

#endif
