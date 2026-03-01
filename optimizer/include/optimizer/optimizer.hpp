#ifndef OPTIMIZER_HPP_INCLUDED
#define OPTIMIZER_HPP_INCLUDED

#include <optimizer/pipeline/pipeline_manager.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <sala/program.hpp>

namespace optimizer
{
struct Optimizer
{
    std::shared_ptr<sala::Program> run(std::shared_ptr<sala::Program> program);

  private:
    optimizer::pipeline::PipelineManager pipeline_mgr_{};
};

} // namespace optimizer
#endif
