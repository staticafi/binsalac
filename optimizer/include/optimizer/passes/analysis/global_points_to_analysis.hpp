#ifndef OPTIMIZER_PASSES_ANALYSIS_GLOBAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_PASSES_ANALYSIS_GLOBAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#include <memory>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class GlobalPointsToAnalysis
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};

} // namespace optimizer::passes
#endif
