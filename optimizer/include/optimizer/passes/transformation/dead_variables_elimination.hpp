#ifndef OPTIMIZER_PASSES_TRANSFORMATION_DEAD_VARIABLES_ELIMINATION_HPP_INCLUDED
#define OPTIMIZER_PASSES_TRANSFORMATION_DEAD_VARIABLES_ELIMINATION_HPP_INCLUDED

#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class DeadVariablesElimination
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes

#endif
