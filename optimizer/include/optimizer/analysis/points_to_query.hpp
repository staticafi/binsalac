#ifndef OPTIMIZER_POINTS_TO_QUERY_HPP_INCLUDED
#define OPTIMIZER_POINTS_TO_QUERY_HPP_INCLUDED

#include <optimizer/utils/points_to/defines.hpp>
#include <span>

namespace optimizer::analysis
{
using objectId = utils::points_to::objectId;
using Target   = utils::points_to::Target;

struct PointsToResult
{
    std::optional<Target>   must;
    std::span<const Target> may;
};

// Optimized for quering within a function context
class PointsToQueryFunction
{
  public:
    explicit PointsToQueryFunction(const program::FunctionIR& function);

    PointsToResult before(const program::InstructionIR& instruction, const program::VariableIR& x);
    PointsToResult before(const program::InstructionIR& instruction, const program::ConstantIR& x);

    PointsToResult after(const program::InstructionIR& instruction, const program::ConstantIR& x);
    PointsToResult after(const program::InstructionIR& instruction, const program::VariableIR& x);

    std::optional<program::OperandIR_wptr> get_object(objectId id);

  private:
};
} // namespace optimizer::analysis
#endif
