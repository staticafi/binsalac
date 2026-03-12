
#ifndef OPTIMIZER_LOCAL_SERIALIZE_POINTS_TO_HPP_INCLUDED
#define OPTIMIZER_LOCAL_SERIALIZE_POINTS_TO_HPP_INCLUDED
#include <memory>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{

class DumpPointsTo
{
  public:
    DumpPointsTo();
    ~DumpPointsTo();
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace optimizer::passes
#endif
