#ifndef OPTIMIZER_PIPELINE_PIPELINES_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PIPELINES_HPP_INCLUDED

#include <optimizer/pipeline/pass_catalog.hpp>
#include <optimizer/pipeline/typed_pipeline.hpp>

namespace optimizer::pipeline
{

// using TestingPipeline =
//         TypedPipeline<Repr::Sala, LowerSalaToIRPass, GlobalPointsToPass, LocalPointsToPass,
//                       DumpPointsToPass, RemoveIndirectionsPass, BumpIrToSalaPass>;

using TestingPipeline = TypedPipeline<Repr::Sala, LowerSalaToIRPass, GlobalPointsToPass,
                                      LocalPointsToPass, DumpPointsToPass, RemoveIndirectionsPass,
                                      AvailableCopyPass, DumpAvailCopyPass, BumpIrToSalaPass>;

} // namespace optimizer::pipeline

#endif
