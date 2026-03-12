#ifndef OPTIMIZER_LOCAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_LOCAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
/**
 * Local intraprocedural points-to analysis over the CFG of each non-initializer function.
 *
 * High-level goal
 * ----------------
 * The analysis computes, for every basic block entry, an approximation of pointer values and
 * pointer-stored memory contents inside the current function. It also computes a function-exit
 * summary for global objects, which is then iterated to a program-level fixpoint across functions.
 *
 * The analysis is flow-sensitive and intraprocedural:
 * - flow-sensitive: transfer is applied in CFG order and join is performed at merge points
 * - intraprocedural: each function is solved locally; calls are handled conservatively through
 *   summary effects rather than by inlining callees
 *
 * Two coupled abstractions
 * ------------------------
 * The analysis maintains two dataflow states:
 *
 *   1) MayState
 *      Maps an object id x to a may-value M(x).
 *      M(x) describes the set of abstract targets that x may point to.
 *
 *   2) MustState
 *      Maps an object id x to a single target T(x).
 *      T(x) exists only when x is known to point to exactly one abstract target.
 *
 * The two states are computed together because:
 * - may information is needed to validate memory accesses and weak updates
 * - must information is used when an operation has an exact singleton interpretation
 *
 * Abstract objects
 * ----------------
 * Concrete variables and constants are assigned object ids. In addition, the analysis uses
 * grouped abstract objects for memory regions that are intentionally summarized, such as:
 *
 * - HEAP
 * - ALLOCA
 * - OUT_OF_LOCAL_SCOPE
 * - OUT_OF_GLOBAL_SCOPE
 * - MERGE_UNKNOWN
 * - VARGARG_BLOCK
 * - CALL_ORDER_DISCREPENCY
 * - FUNCTION
 *
 * These grouped objects represent regions or effects that are relevant for optimization, but are
 * not distinguished object-by-object in the abstract domain.
 *
 * Mathematical view of the may domain
 * -----------------------------------
 * For each tracked object x, the may domain stores a value of the form:
 *
 *   M(x) in { TOP } union P_finite(Target)
 *
 * where:
 * - TOP means "x may point to anything representable by the analysis"
 * - P_finite(Target) is a finite set of abstract targets
 *
 * Bottom is represented implicitly:
 * - if x is absent from MayState, then there is no tracked may points-to fact for x
 *
 * Thus a present may entry is always either:
 * - TOP, or
 * - a non-empty finite target set
 *
 * The partial order is the usual may-information order:
 *
 *   A <= B    iff    targets(A) subseteq targets(B)
 *
 * with finite sets below TOP. Join is union-like, with TOP absorbing:
 *
 *   finite U finite  = union
 *   TOP U anything   = TOP
 *
 * At CFG joins, the analysis uses a strict may join that also inserts MERGE_UNKNOWN when one
 * predecessor provides points-to information for an object and another predecessor does not.
 * This distinguishes "known finite alternatives" from "finite alternatives plus missing branch
 * information".
 *
 * Mathematical view of the must domain
 * ------------------------------------
 * For each tracked object x, the must domain stores either:
 *
 *   no entry
 *   or
 *   one exact target T(x)
 *
 * Bottom / lack of exactness is represented implicitly by absence from MustState.
 * There is no explicit must-TOP element. Whenever exact singleton information is lost, the entry
 * is erased.
 *
 * The meet used at CFG joins is intersection-like:
 * - a must fact is preserved only if all predecessors agree on the same exact target
 * - otherwise the fact is dropped
 *
 * Concretely, for a given object x:
 * - if all incoming must states contain the same target t, keep x -> t
 * - otherwise erase x from the joined must state
 *
 * Transfer model
 * --------------
 * Each instruction is interpreted as a transfer function over the pair:
 *
 *   (MustState, MayState)
 *
 * The transfer functions are designed so that:
 * - exact singleton cases update MustState
 * - weak or ambiguous cases fall back to MayState
 * - invalid or maximally imprecise memory accesses may nuke the current may/must information
 *   conservatively
 *
 * Examples:
 * - ADDRESS, COPY, ALLOCA, MALLOC perform strong updates on the destination
 * - LOAD uses exact singleton dereference when possible, otherwise joins may-information from all
 *   possible pointees
 * - STORE performs weak updates through all possible pointees of the destination
 * - MEMCPY/MEMMOVE/MEMSET conservatively invalidate or merge stored pointer information in memory
 * - CALL conservatively propagates escape effects through reachable arguments
 *
 * Meaning of "nuke"
 * -----------------
 * Some operations, such as dereferencing an untracked or TOP pointer under the current model,
 * are treated as invalid or too imprecise for the local abstraction. In such cases, the transfer
 * may conservatively replace tracked information by TOP-like summary effects. This intentionally
 * sacrifices precision to preserve sound over-approximation for optimization clients.
 *
 * CFG solution
 * ------------
 * For each function, the analysis solves a forward dataflow problem over the basic-block CFG.
 *
 * Let OUT[b] be the pair of abstract states after block b.
 * Let IN[b] be the pair of states before block b.
 *
 * Entry initialization:
 * - IN[entry] starts from the current global seed states
 * - assumed pointer parameters are initialized to OUT_OF_LOCAL_SCOPE
 * - grouped abstract nodes are seeded in the may state as self-pointing singleton facts
 *
 * Transfer:
 * - IN[b] is propagated instruction-by-instruction through block b
 * - the resulting state becomes OUT[b]
 *
 * Join at merge points:
 * - may component uses strict may-join
 * - must component uses agreement/intersection meet
 *
 * The block equations are solved by a standard worklist iteration until block OUT states stabilize.
 *
 * Interprocedural summary iteration
 * ---------------------------------
 * Although the analysis is intraprocedural, functions communicate through global object summaries.
 *
 * After each function is solved:
 * - the exit-block OUT states are projected to global objects
 * - local targets are remapped or discarded when exporting summaries
 * - the resulting global may/must summaries are joined across all functions
 *
 * This process is iterated until the program-level global summaries reach a fixpoint.
 *
 * Export policy
 * -------------
 * When exporting function-exit information to the global summary:
 * - may facts targeting non-global objects are remapped to OUT_OF_GLOBAL_SCOPE
 * - must facts are preserved only when the exact target is globally meaningful
 *   (for example a global object or HEAP); otherwise exactness is dropped
 *
 * Precision profile
 * -----------------
 * The analysis is:
 * - flow-sensitive
 * - field-insensitive / offset-summary-based
 * - intraprocedural with iterative global summary propagation
 * - conservative around calls, memory intrinsics, variadic state, and invalid accesses
 *
 */
class LocalPointsToAnalysis
{
  public:
    void run(program::ProgramIR_sptr sala_ir);
};

} // namespace optimizer::passes
#endif
