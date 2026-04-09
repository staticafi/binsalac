#ifndef OPTIMIZER_PASSES_ANALYSIS_LIVE_VARIABLE_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_PASSES_ANALYSIS_LIVE_VARIABLE_ANALYSIS_HPP_INCLUDED

#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class LivenessAnalysis
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes

#endif
