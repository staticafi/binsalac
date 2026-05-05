#ifndef OPTIMIZER_PIPELINE_PIPELINE_FACTORY_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PIPELINE_FACTORY_HPP_INCLUDED
#include <optimizer/config.hpp>
#include <optimizer/pipeline/pipelines.hpp>
#include <utility/development.hpp>

namespace optimizer::pipeline
{

inline std::unique_ptr<PipelineI> create_pipeline(optimizer::PipelineKind        kind,
                                                  std::shared_ptr<sala::Program> program)
{
    switch (kind)
    {
    case optimizer::PipelineKind::exp:
        return std::make_unique<ExperimentalPipeline>(std::move(program));
    case optimizer::PipelineKind::exp_debug:
        return std::make_unique<ExperimentalDebugPipeline>(std::move(program));
    default:
        NOT_SUPPORTED();
    }
}

} // namespace optimizer::pipeline
#endif
