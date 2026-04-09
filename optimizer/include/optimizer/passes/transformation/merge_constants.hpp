#ifndef OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED
#define OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED

#include <optimizer/programIR/program_ir.hpp>

namespace optimizer::passes
{
class MergeConstants
{
    void run(program::ProgramIR_sptr sala_ir);
};
} // namespace optimizer::passes
#endif
