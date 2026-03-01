#ifndef OPTIMIZER_PIPELINE_DEFINITIONS_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_DEFINITIONS_HPP_INCLUDED
#include <optimizer/pipeline/typed_pipeline.hpp>

#include <optimizer/passes/analysis/global_points_to_analysis.hpp>
#include <optimizer/passes/analysis/local_points_to_analysis.hpp>
#include <optimizer/passes/debug/serialize_points_to.hpp>
#include <optimizer/passes/translation/bump_ir_to_sala.hpp>
#include <optimizer/passes/translation/lower_sala_to_ir.hpp>

#include <optimizer/passes/transformation/merge_constants.hpp>

#include <sala/program.hpp>

namespace optimizer::pipeline
{
using TestingPipeline = TypedPipeline<passes::Repr::Sala, passes::LowerSalaToIR,
                                      passes::GlobalPointsToAnalysis, passes::LocalPointsToAnalysis,
                                      passes::SerializePointsTo, passes::BumpIrToSala>;
// using TestingPipeline = TypedPipeline<passes::Repr::Sala,
// passes::LowerSalaToIR,passes::MergeConstants,
//                                       passes::LocalPointsToAnalysis, passes::SerializePointsTo,
//                                       passes::BumpIrToSala>;
} // namespace optimizer::pipeline
#endif
