#ifndef OPTIMIZER_CONFIG_HPP_INCLUDED
#define OPTIMIZER_CONFIG_HPP_INCLUDED
#include <optional>

#include <stdexcept>
#include <string>

namespace optimizer
{

enum class PipelineKind
{
    exp, // experimental
    exp_debug,
};

struct OptimizerConfig
{
    std::optional<PipelineKind> pipeline{};
};

inline PipelineKind pipeline_kind_from_string(std::string const& value)
{
    if (value == "exp")
    {
        return PipelineKind::exp;
    }

    if (value == "exp_debug")
    {
        return PipelineKind::exp_debug;
    }

    throw std::runtime_error("Unknown optimizer pipeline: " + value);
}

} // namespace optimizer
#endif
