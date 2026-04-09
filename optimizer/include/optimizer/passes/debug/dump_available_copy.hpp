#ifndef OPTIMIZER_PASSES_DEBUG_DUMP_AVAIL_COPY_HPP_INCLUDED
#define OPTIMIZER_PASSES_DEBUG_DUMP_AVAIL_COPY_HPP_INCLUDED

#include <optimizer/programIR/program_ir.hpp>

namespace optimizer::passes
{
class DumpAvailCopy
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes

#endif
