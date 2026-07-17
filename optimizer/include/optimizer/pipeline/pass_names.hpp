#ifndef OPTIMIZER_PIPELINE_PASS_NAMES_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PASS_NAMES_HPP_INCLUDED
namespace optimizer::pipeline::names
{
inline constexpr char lower_sala_to_ir[]             = "LowerSalaToIR";
inline constexpr char global_points_to[]             = "GlobalPointsToAnalysis";
inline constexpr char local_points_to[]              = "LocalPointsToAnalysis";
inline constexpr char remove_indirections[]          = "RemoveIndirections";
inline constexpr char merge_constants[]              = "MergeConstants";
inline constexpr char bump_ir_to_sala[]              = "BumpIrToSala";
inline constexpr char dump_points_to[]               = "DumpPointsTo";
inline constexpr char available_copy[]               = "AvailableCopy";
inline constexpr char dump_avail_copy[]              = "DumpAvailCopy";
inline constexpr char propagate_copy[]               = "PropagateCopy";
inline constexpr char liveness[]                     = "LivenessAnalysis";
inline constexpr char dead_instruction_elimination[] = "DeadInstructionElimination";
inline constexpr char dead_variables_elimination[]   = "DeadVariablesElimination";
} // namespace optimizer::pipeline::names

#endif
