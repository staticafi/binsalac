#ifndef OPTIMIZER_DEAD_INSTRUCTION_ELIMINATION_HPP_INCLUDED
#define OPTIMIZER_DEAD_INSTRUCTION_ELIMINATION_HPP_INCLUDED

#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class DeadInstructionElimination
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes

#endif
