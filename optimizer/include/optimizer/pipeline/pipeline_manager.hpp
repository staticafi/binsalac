#ifndef OPTIMIZER_PIPELINE_MANAGER_INCLUDED_HPP
#define OPTIMIZER_PIPELINE_MANAGER_INCLUDED_HPP

#include <optimizer/pipeline/pipeline_i.hpp>
#include <utility/assumptions.hpp>

namespace optimizer::pipeline
{

class PipelineManager
{
  public:
    void set_pipeline(std::unique_ptr<PipelineI> strategy) { piepline_ = std::move(strategy); }

    void run() const
    {
        ASSUMPTION(piepline_ != nullptr);
        piepline_->run();
    }

    void run_step() const
    {
        ASSUMPTION(piepline_ != nullptr);
        piepline_->run_step();
    };

    [[nodiscard]] const program::ProgramRepr& program() const { return piepline_->program(); }

  private:
    std::unique_ptr<PipelineI> piepline_;
};
} // namespace optimizer::pipeline
#endif
