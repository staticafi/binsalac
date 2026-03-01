#include <optimizer/optimizer.hpp>

#include <optimizer/translation/ir_to_sala.hpp>
#include <optimizer/translation/sala_to_ir.hpp>

#include <optimizer/pipeline/pipelines.hpp>

#include <utility/assumptions.hpp>
#include <utility/development.hpp>
#include <utility/invariants.hpp>
#include <utility/timeprof.hpp>

namespace optimizer
{
std::shared_ptr<sala::Program> Optimizer::run(std::shared_ptr<sala::Program> program)
{
    // TODO: enable user setting of pipeline
    pipeline_mgr_.set_pipeline(std::make_unique<pipeline::TestingPipeline>(std::move(program)));
    pipeline_mgr_.run();
    return std::get<std::shared_ptr<sala::Program>>(pipeline_mgr_.program());
}

} // namespace optimizer
