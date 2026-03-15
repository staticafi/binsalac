#ifndef OPTIMIZER_PASSES_DEBUG_DUMP_AVAIL_COPY_HPP_INCLUDED
#define OPTIMIZER_PASSES_DEBUG_DUMP_AVAIL_COPY_HPP_INCLUDED

#include <optimizer/programIR/program_ir.hpp>

#include <memory>

namespace optimizer::passes
{

class DumpAvailCopy
{
  public:
    DumpAvailCopy();
    ~DumpAvailCopy();

    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace optimizer::passes

#endif
