#ifndef OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#define OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#include <memory>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class RemoveIndirections
{
  public:
    RemoveIndirections();
    ~RemoveIndirections();

    RemoveIndirections(const RemoveIndirections&)            = delete;
    RemoveIndirections(RemoveIndirections&&)                 = delete;
    RemoveIndirections& operator=(const RemoveIndirections&) = delete;
    RemoveIndirections& operator=(RemoveIndirections&&)      = delete;

    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace optimizer::passes
#endif
