#ifndef OPTIMIZER_POINTS_TO_QUERY_HPP_INCLUDED
#define OPTIMIZER_POINTS_TO_QUERY_HPP_INCLUDED
#include <optimizer/metadata/points_to.hpp>
#include <optimizer/utils/points_to/defines.hpp>

#include <functional>
#include <span>

namespace optimizer::analysis
{
using objectId = utils::points_to::objectId;
using Target   = utils::points_to::Target;

struct PointsToResult
{
    std::optional<Target>    must;
    utils::SparseSet<Target> may;
};

// Quering within a function context
class PointsToQueryFunction
{
  public:
    explicit PointsToQueryFunction(program::FunctionIR_csptr function);

    PointsToResult before(const program::InstructionIR_sptr& instruction,
                          const program::VariableIR&         x);
    PointsToResult before(const program::InstructionIR_sptr& instruction,
                          const program::ConstantIR&         x);

    PointsToResult after(const program::InstructionIR_sptr& instruction,
                         const program::ConstantIR&         x);
    PointsToResult after(const program::InstructionIR_sptr& instruction,
                         const program::VariableIR&         x);

    std::optional<program::OperandIR_sptr> get_object(objectId id) const;

  private:
    struct cache_t
    {
        program::BasicBlockIR_raw  bb;
        program::InstructionIR_raw instr;
        utils::points_to::MayState may_in;
    };

    using cach_state_t = std::pair<const program::InstructionIR*, utils::points_to::MayState>;

    // using cache_t = std::pair<const program::BasicBlockIR*, cach_state_t>;

    PointsToResult handle_request(const program::InstructionIR_sptr& instruction,
                                  const program::VariableIR& x, bool before);

    PointsToResult handle_cache_hit(const program::InstructionIR_sptr& instruction,
                                    const program::VariableIR& x, bool before);

    [[nodiscard]] const metadata::points_to::ObjectPool*
    get_object_pool(const program::InstructionIR_sptr& instruction, const program::VariableIR& x);

  private:
    program::ProgramIR_csptr  program_keepalive_;
    program::FunctionIR_csptr function_;
    objectId                  last_local_id_{0};

    cache_t cache_;

    const metadata::points_to::ProgramMeta*  program_meta_{};
    const metadata::points_to::FunctionMeta* function_meta_{};

    const metadata::points_to::ObjectPool* global_objects_{};
    const metadata::points_to::ObjectPool* local_objects_{};

    std::function<void(const utils::points_to::MayTransferContextBundle& context)> transfer_may_;
};
} // namespace optimizer::analysis
#endif
