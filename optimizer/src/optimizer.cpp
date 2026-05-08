#include <optimizer/optimizer.hpp>

#include <optimizer/translation/ir_to_sala.hpp>
#include <optimizer/translation/sala_to_ir.hpp>

#include <optimizer/pipeline/pipeline_factory.hpp>
#include <optimizer/pipeline/pipelines.hpp>

#include <utility/assumptions.hpp>
#include <utility/development.hpp>
#include <utility/invariants.hpp>
#include <utility/timeprof.hpp>

namespace optimizer
{
Optimizer::Optimizer(OptimizerConfig config) : config_(config)
{
}

std::shared_ptr<sala::Program> Optimizer::run(std::shared_ptr<sala::Program> program)
{
    if (config_.run_opt == false || !config_.pipeline.has_value())
    {
        return program;
    }

    pipeline_mgr_.set_pipeline(
            pipeline::create_pipeline(config_.pipeline.value(), std::move(program)));

    pipeline_mgr_.run();
    ASSUMPTION(std::holds_alternative<std::shared_ptr<sala::Program>>(pipeline_mgr_.program()));
    return std::get<std::shared_ptr<sala::Program>>(pipeline_mgr_.program());
}
} // namespace optimizer
