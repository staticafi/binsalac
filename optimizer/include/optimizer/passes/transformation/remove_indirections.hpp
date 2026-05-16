#ifndef OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#define OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#include <memory>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class RemoveIndirections
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes
#endif
