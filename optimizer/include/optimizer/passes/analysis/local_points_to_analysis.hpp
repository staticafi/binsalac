#ifndef OPTIMIZER_PASSES_ANALYSIS_LOCAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_PASSES_ANALYSIS_LOCAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class LocalPointsToAnalysis
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};

} // namespace optimizer::passes
#endif
