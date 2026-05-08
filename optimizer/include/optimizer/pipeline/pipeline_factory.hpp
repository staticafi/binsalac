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
    case optimizer::PipelineKind::default_p:
        return std::make_unique<DefaultPipeline>(std::move(program));
    case optimizer::PipelineKind::debug:
        return std::make_unique<DebugPipeline>(std::move(program));
    default:
        NOT_SUPPORTED();
    }
}

} // namespace optimizer::pipeline
#endif
