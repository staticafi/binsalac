#ifndef OPTIMIZER_GLOBAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#define OPTIMIZER_GLOBAL_POINTS_TO_ANALYSIS_HPP_INCLUDED
#include <memory>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
/*
Points-to Analysis (Global)
=========================

Scope & Sensitivity
-------------------
- Global scope: static initializer, we track statics and constants
- Context-insensitive: we do not propagate points-to facts across calls (see “Calls” below).

Pointer Model
-------------
Pointer is an object whose value represents the memory address of another object.
Since we do not distinguish a pointer type in our language, pointers are identified by their
semantic use: operands that participate in instructions operating on memory addresses are treated as
pointers defined above.

Tracked vs. Untracked Pointers
------------------------------

Let t be an instruction operand descriptor such t ∈ { local, parameter, static, constant}
Then tracked pointer variables are defined as follows:
    1. variable tN defined via ADRESS tN tM
    2. variable tN defined via COPY tN tM; where tM is a tracked pointer variable
    3. variable tN defined via LOAD tN tM; where tM holds the address of a tracked pointer variable
    4. variable at address stored in tN defined via STORE tN tM; where tM and tN are tracked pointer
        variables
    5. variable tN defined via P2I tN tM; where tM is a tracked pointer variable
    6. variable tN defined via I2P tN tM; where tM was defined by 5. and not modified.

Any other pointer is untracked.

Untracked Memory Nodes
----------------------
For untracked pointers we conservatively abstract memory with the following nodes
    - Undefined     : unknown / top location (may point to anything)
    - Function      : function addresses
    - Heap          : memory from MALLOC
    - Stack         : memory from ALLOCA
    - StackPointer  : memory from STACKSAVE

Scope sanitization
------------------
The resulting must-, may-points to set hold for whole-program context, therefore
any local-scoped (locals, parameters) pointer variables must be excluded from the resulting points
to sets

Access Policy & Conservatism
----------------------------
- If a pointer is not present in the (may ⊇ must) points-to set at a use site,
  the access is treated as going through the **Undefined** untracked node.

- Any access through **Undefined** kills (invalidates) all must-points-to information
  at that program point.

- “Kill must” means: remove the affected locations from the must-set (the may-set
  remains a conservative over-approximation).

Updates (Strong vs. Weak)
-------------------------
- A strong update applies when the write is to a single cell
  (the base pointer is a must-singleton and the location is not a summary).
  Strong update overwrites both may and must of the target cell.

- Otherwise perform a weak update: union into may; drop/clear must for that cell
  unless you can prove the written value is the only possible content.

Function Calls (Escapes)
------------------------
- Since our static initializer is generated during translation from C to SALA
  and static variables have to be initialized with a compile-time constant,
  we do not expect functions to be invoked inside the initializer. Nevertheless
  the language as is does not forbid it so we must treat it in this regard as
  an any other function

- Passing a tracked pointer as an argument is treated as an escape.
  We conservatively kill must for all locations reachable from that pointer
  (transitively) at the call site. May is preserved/expanded as needed.

- No interprocedural propagation: callees do not refine caller facts. Unknown calls
  may additionally invalidate must for any location they may modify.

Joins
-----
- At CFG merges: May = union of predecessors; Must = intersection of predecessors.

Soundness Invariants
--------------------
- If a concrete execution can make x point to l at a program point, then l ∈ May(x).
- If l ∈ Must(x) at a point, then in all executions reaching that point, x points to l.
*/

class GlobalPointsToAnalysis
{
  public:
    GlobalPointsToAnalysis();
    ~GlobalPointsToAnalysis();
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace optimizer::passes
#endif
