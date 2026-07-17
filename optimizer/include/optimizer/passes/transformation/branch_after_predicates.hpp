#ifndef OPTIMIZER_PASSES_TRANSFORMATION_BRANCH_AFTER_PREDICATES_HPP_INCLUDED
#   define OPTIMIZER_PASSES_TRANSFORMATION_BRANCH_AFTER_PREDICATES_HPP_INCLUDED

#   include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{


class BranchAfterPredicates
{
public:
    void run(program::ProgramIR_sptr sala_ir);
};


}

#endif

