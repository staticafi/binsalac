#ifndef OPTIMIZER_PASSES_ANALYSIS_AVAILABLE_COPY_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_PASSES_ANALYSIS_AVAILABLE_COPY_ANALYSIS_HPP_INCLUDED

#include <optimizer/programIR/ir_types.hpp>

#include <memory>

namespace optimizer::passes
{
class AvailableCopyAnalysis
{
  public:
    AvailableCopyAnalysis();
    ~AvailableCopyAnalysis();

    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace optimizer::passes

#endif
