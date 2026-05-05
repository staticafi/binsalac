#ifndef OPTIMIZER_HPP_INCLUDED
#define OPTIMIZER_HPP_INCLUDED

#include <optimizer/config.hpp>
#include <optimizer/pipeline/pipeline_manager.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <sala/program.hpp>

namespace optimizer
{
class Optimizer
{
  public:
    explicit Optimizer(OptimizerConfig config = {});

    std::shared_ptr<sala::Program> run(std::shared_ptr<sala::Program> program);

  private:
    OptimizerConfig           config_;
    pipeline::PipelineManager pipeline_mgr_{};
};

} // namespace optimizer
#endif
