#ifndef OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED
#define OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED

#include <optimizer/programIR/program_ir.hpp>

namespace optimizer::passes
{

class MergeConstants
{
  public:
    MergeConstants();
    ~MergeConstants();

    MergeConstants(const MergeConstants&)                    = delete;
    MergeConstants(MergeConstants&&)                         = delete;
    MergeConstants&         operator=(const MergeConstants&) = delete;
    MergeConstants&         operator=(MergeConstants&&)      = delete;
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace optimizer::passes
#endif
