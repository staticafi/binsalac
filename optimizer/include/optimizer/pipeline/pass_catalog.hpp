#ifndef OPTIMIZER_PIPELINE_PASS_CATALOG_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PASS_CATALOG_HPP_INCLUDED

#include <optimizer/passes/analysis/available_copy_analysis.hpp>
#include <optimizer/passes/analysis/global_points_to_analysis.hpp>
#include <optimizer/passes/analysis/local_points_to_analysis.hpp>
#include <optimizer/passes/debug/dump_available_copy.hpp>
#include <optimizer/passes/debug/dump_points_to.hpp>
#include <optimizer/passes/transformation/merge_constants.hpp>
#include <optimizer/passes/transformation/remove_indirections.hpp>
#include <optimizer/passes/translation/bump_ir_to_sala.hpp>
#include <optimizer/passes/translation/lower_sala_to_ir.hpp>
#include <optimizer/pipeline/pass_contracts.hpp>

namespace optimizer::pipeline
{
namespace names
{
inline constexpr char lower_sala_to_ir[]    = "LowerSalaToIR";
inline constexpr char global_points_to[]    = "GlobalPointsToAnalysis";
inline constexpr char local_points_to[]     = "LocalPointsToAnalysis";
inline constexpr char remove_indirections[] = "RemoveIndirections";
inline constexpr char merge_constants[]     = "MergeConstants";
inline constexpr char bump_ir_to_sala[]     = "BumpIrToSala";
inline constexpr char dump_points_to[]      = "DumpPointsTo";
inline constexpr char available_copy[]      = "AvailableCopy";
inline constexpr char dump_avail_copy[]     = "DumpAvailCopy";
} // namespace names

using LowerSalaToIRPass =
        PassDef<passes::LowerSalaToIR,
                TranslationSpec<Repr::Sala, Repr::IR, ProductList<>, products::TranslationAll,
                                ProductList<>, names::lower_sala_to_ir>>;

using GlobalPointsToPass = PassDef<
        passes::GlobalPointsToAnalysis,
        AnalysisSpec<Repr::IR, ProductList<>, products::GlobalPointsToSet,
                     utils::set_union_many_t<products::TranslationAll, products::AvailableCopyAll>,
                     names::global_points_to>>;

using LocalPointsToPass = PassDef<
        passes::LocalPointsToAnalysis,
        AnalysisSpec<Repr::IR, products::GlobalPointsToSet, products::LocalPointsToSet,
                     utils::set_union_many_t<products::TranslationAll, products::GlobalPointsToSet,
                                             products::AvailableCopyAll>,
                     names::local_points_to>>;

using RemoveIndirectionsPass =
        PassDef<passes::RemoveIndirections,
                TransformSpec<Repr::IR, ProductList<products::PointsToBasicBlock>, ProductList<>,
                              products::TranslationAll, names::remove_indirections>>;

using MergeConstantsPass = PassDef<passes::MergeConstants,
                                   TransformSpec<Repr::IR, ProductList<>, ProductList<>,
                                                 products::TranslationAll, names::merge_constants>>;

using BumpIrToSalaPass =
        PassDef<passes::BumpIrToSala,
                TranslationSpec<Repr::IR, Repr::Sala, products::TranslationAll, ProductList<>,
                                ProductList<>, names::bump_ir_to_sala>>;

using DumpPointsToPass =
        PassDef<passes::DumpPointsTo, DebugSpec<Repr::IR, products::PointsToSetAll, ProductList<>,
                                                ProductList<>, names::dump_points_to>>;

using AvailableCopyPass = PassDef<
        passes::AvailableCopyAnalysis,
        AnalysisSpec<Repr::IR, ProductList<>, products::AvailableCopyAll,
                     utils::set_union_many_t<products::TranslationAll, products::PointsToSetAll>,
                     names::available_copy>>;

using DumpAvailCopyPass = PassDef<passes::DumpAvailCopy,
                                  DebugSpec<Repr::IR, products::AvailableCopyAll, ProductList<>,
                                            ProductList<>, names::dump_avail_copy>>;
} // namespace optimizer::pipeline

#endif
