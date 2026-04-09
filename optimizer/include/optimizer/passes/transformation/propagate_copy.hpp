#ifndef OPTIMIZER_PROPAGATE_COPY_HPP_INCLUDED
#define OPTIMIZER_PROPAGATE_COPY_HPP_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class PropagateCopy
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes
#endif
