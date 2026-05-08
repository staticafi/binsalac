#ifndef OPTIMIZER_CONFIG_HPP_INCLUDED
#define OPTIMIZER_CONFIG_HPP_INCLUDED
#include <optional>

#include <stdexcept>
#include <string>

namespace optimizer
{

enum class PipelineKind
{
    default_p, // experimental
    debug,
};

struct OptimizerConfig
{
    bool                        run_opt;
    std::optional<PipelineKind> pipeline{};
};

inline PipelineKind pipeline_kind_from_string(std::string const& value)
{
    if (value == "default")
    {
        return PipelineKind::default_p;
    }

    if (value == "debug")
    {
        return PipelineKind::debug;
    }

    throw std::runtime_error("Unknown optimizer pipeline: " + value);
}

} // namespace optimizer
#endif
