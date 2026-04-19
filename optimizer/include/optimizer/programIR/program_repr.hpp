#ifndef OPTIMIZER_PROGRAM_REPR_HPP
#define OPTIMIZER_PROGRAM_REPR_HPP
#include <optimizer/programIR/program_ir.hpp>

#include <sala/program.hpp>

#include <variant>

namespace optimizer::program
{

using ProgramRepr = std::variant<ProgramIR_sptr, std::shared_ptr<sala::Program>>;

}

#endif
