#ifndef OPTIMIZER_PIPELINE_I_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_I_HPP_INCLUDED

#include <optimizer/pipeline/spec.hpp>

namespace optimizer::pipeline
{

class PipelineI
{
  public:
    virtual ~PipelineI()    = default;
    virtual void run()      = 0;
    virtual void run_step() = 0;

    [[nodiscard]] virtual const program::ProgramRepr& program() const = 0;
};
} // namespace optimizer::pipeline
#endif
