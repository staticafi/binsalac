#ifndef OPTIMIZER_PIPELINE_PIPELINES_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PIPELINES_HPP_INCLUDED

#include <optimizer/pipeline/pass_catalog.hpp>
#include <optimizer/pipeline/typed_pipeline.hpp>

namespace optimizer::pipeline
{

using DebugPipeline =
        TypedPipeline<Repr::Sala, LowerSalaToIRPass, MergeConstantsPass, GlobalPointsToPass,
                      LocalPointsToPass, DumpPointsToPass, RemoveIndirectionsPass,
                      AvailableCopyPass, DumpAvailCopyPass, PropagateCopyPass, LivenessPass,
                      DumpLivenessPass, DeadInstructionEliminationPass,
                      DeadVariablesEliminationPass, BumpIrToSalaPass>;

using DefaultPipeline =
        TypedPipeline<Repr::Sala, LowerSalaToIRPass, MergeConstantsPass, GlobalPointsToPass,
                      LocalPointsToPass, RemoveIndirectionsPass, AvailableCopyPass,
                      PropagateCopyPass, LivenessPass, DeadInstructionEliminationPass,
                      DeadVariablesEliminationPass, BumpIrToSalaPass>;

} // namespace optimizer::pipeline

#endif
