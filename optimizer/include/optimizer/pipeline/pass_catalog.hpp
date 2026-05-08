#ifndef OPTIMIZER_PIPELINE_PASS_CATALOG_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PASS_CATALOG_HPP_INCLUDED

// analysis
#include <optimizer/passes/analysis/available_copy_analysis.hpp>
#include <optimizer/passes/analysis/global_points_to_analysis.hpp>
#include <optimizer/passes/analysis/liveness_analysis.hpp>
#include <optimizer/passes/analysis/local_points_to_analysis.hpp>

// debug
#include <optimizer/passes/debug/dump_available_copy.hpp>
#include <optimizer/passes/debug/dump_liveness.hpp>
#include <optimizer/passes/debug/dump_points_to.hpp>

// transformation
#include <optimizer/passes/transformation/dead_instruction_elimination.hpp>
#include <optimizer/passes/transformation/dead_variables_elimination.hpp>
#include <optimizer/passes/transformation/merge_constants.hpp>
#include <optimizer/passes/transformation/propagate_copy.hpp>
#include <optimizer/passes/transformation/remove_indirections.hpp>

// translation
#include <optimizer/passes/translation/bump_ir_to_sala.hpp>
#include <optimizer/passes/translation/lower_sala_to_ir.hpp>

#include <optimizer/pipeline/pass_contracts.hpp>
#include <optimizer/pipeline/pass_names.hpp>

namespace optimizer::pipeline
{
// Translation
using LowerSalaToIRPass =
        PassDef<passes::LowerSalaToIR,
                TranslationSpec<Repr::Sala, Repr::IR, ProductList<>, products::TranslationAll,
                                ProductList<>, names::lower_sala_to_ir>>;
using BumpIrToSalaPass =
        PassDef<passes::BumpIrToSala,
                TranslationSpec<Repr::IR, Repr::Sala, products::TranslationAll, ProductList<>,
                                ProductList<>, names::bump_ir_to_sala>>;
// Analysis

using GlobalPointsToPass = PassDef<
        passes::GlobalPointsToAnalysis,
        AnalysisSpec<Repr::IR, ProductList<>, products::GlobalPointsToSet,
                     utils::set_union_many_t<products::TranslationAll, products::AvailableCopyAll,
                                             products::LivenessAll>,
                     names::global_points_to>>;

using LocalPointsToPass = PassDef<
        passes::LocalPointsToAnalysis,
        AnalysisSpec<Repr::IR, products::GlobalPointsToSet, products::LocalPointsToSet,
                     utils::set_union_many_t<products::TranslationAll, products::GlobalPointsToSet,
                                             products::AvailableCopyAll, products::LivenessAll>,
                     names::local_points_to>>;

using AvailableCopyPass = PassDef<
        passes::AvailableCopyAnalysis,
        AnalysisSpec<Repr::IR, ProductList<>, products::AvailableCopyAll,
                     utils::set_union_many_t<products::TranslationAll, products::PointsToSetAll,
                                             products::LivenessAll>,
                     names::available_copy>>;

using LivenessPass = PassDef<
        passes::LivenessAnalysis,
        AnalysisSpec<Repr::IR, ProductList<>, products::LivenessAll,
                     utils::set_union_many_t<products::TranslationAll, products::PointsToSetAll,
                                             products::AvailableCopyAll>,
                     names::liveness>>;

// Transformation
using RemoveIndirectionsPass =
        PassDef<passes::RemoveIndirections,
                TransformSpec<Repr::IR, ProductList<products::PointsToBasicBlock>, ProductList<>,
                              products::TranslationAll, names::remove_indirections>>;

using MergeConstantsPass = PassDef<passes::MergeConstants,
                                   TransformSpec<Repr::IR, ProductList<>, ProductList<>,
                                                 products::TranslationAll, names::merge_constants>>;

using PropagateCopyPass = PassDef<passes::PropagateCopy,
                                  TransformSpec<Repr::IR, products::AvailableCopyAll, ProductList<>,
                                                products::TranslationAll, names::propagate_copy>>;

using DeadInstructionEliminationPass =
        PassDef<passes::DeadInstructionElimination,
                TransformSpec<Repr::IR, products::LivenessAll, ProductList<>,
                              products::TranslationAll, names::dead_instruction_elimination>>;

using DeadVariablesEliminationPass =
        PassDef<passes::DeadVariablesElimination,
                TransformSpec<Repr::IR, ProductList<>, ProductList<>, products::TranslationAll,
                              names::dead_variables_eliminitation>>;

// Debug
using DumpAvailCopyPass = PassDef<passes::DumpAvailCopy,
                                  DebugSpec<Repr::IR, products::AvailableCopyAll, ProductList<>,
                                            ProductList<>, names::dump_avail_copy>>;

using DumpPointsToPass =
        PassDef<passes::DumpPointsTo, DebugSpec<Repr::IR, products::PointsToSetAll, ProductList<>,
                                                ProductList<>, names::dump_points_to>>;

using DumpLivenessPass =
        PassDef<passes::DumpLiveness, DebugSpec<Repr::IR, products::LivenessAll, ProductList<>,
                                                ProductList<>, names::dump_avail_copy>>;

} // namespace optimizer::pipeline

#endif
