
#ifndef OPTIMIZER_PASSES_DEBUG_DUMP_LIVENESS_HPP_INCLUDED
#define OPTIMIZER_PASSES_DEBUG_DUMP_LIVENESS_HPP_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class DumpLiveness
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes
#endif
