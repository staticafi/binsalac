
#ifndef OPTIMIZER_UTILS_COMMON_HPP_INCLUDED
#define OPTIMIZER_UTILS_COMMON_HPP_INCLUDED

#include <optimizer/programIR/ir_types.hpp>
#include <sala/program.hpp>
#include <string>
namespace optimizer::utils
{
std::string instruction_opcode_to_string(sala::Instruction::Opcode const opcode);

std::string get_offset(int offset);

std::string_view get_function_name(const program::FunctionIR& function);

std::string sanitize_filename(std::string name);

std::string get_program_name(const optimizer::program::ProgramIR& program);

} // namespace optimizer::utils

#endif
