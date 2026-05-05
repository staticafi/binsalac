#ifndef OPTIMIZER_QUERY_POINTS_TO_QUERY_HPP_INCLUDED
#define OPTIMIZER_QUERY_POINTS_TO_QUERY_HPP_INCLUDED

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/utils/points_to/defines.hpp>

#include <cstddef>
#include <functional>
#include <optional>

namespace optimizer::query
{
using objectId         = utils::points_to::objectId;
using Target           = utils::points_to::Target;
using MayAnalysisState = utils::points_to::MayAnalysisState;

struct PointsToResult
{
    bool                     poisoned{false};
    std::optional<Target>    must;
    utils::SparseSet<Target> may;
};

// Querying within a function context
class PointsToQueryFunction
{
  public:
    explicit PointsToQueryFunction(program::FunctionIR_csptr function);

    PointsToResult before(const program::InstructionIR_sptr& instruction,
                          const program::VariableIR&         x);
    PointsToResult before(const program::InstructionIR_sptr& instruction,
                          const program::ConstantIR&         x);

    PointsToResult after(const program::InstructionIR_sptr& instruction,
                         const program::VariableIR&         x);
    PointsToResult after(const program::InstructionIR_sptr& instruction,
                         const program::ConstantIR&         x);

    // Whole-state queries
    [[nodiscard]] MayAnalysisState before_state(const program::InstructionIR_sptr& instruction);

    [[nodiscard]] MayAnalysisState after_state(const program::InstructionIR_sptr& instruction);

    std::optional<program::OperandIR_sptr> get_object(objectId id) const;

  private:
    struct cache_t
    {
        program::BasicBlockIR_raw  bb{nullptr};
        program::InstructionIR_raw instr{nullptr};

        std::size_t bb_index{0};
        std::size_t instr_index{0};

        MayAnalysisState state;

        bool             after_valid{false};
        MayAnalysisState after_state;
    };

    PointsToResult handle_request(const program::InstructionIR_sptr& instruction,
                                  const program::VariableIR& x, bool before);

    PointsToResult handle_cache_hit(const program::InstructionIR_sptr& instruction,
                                    const program::VariableIR& x, bool before);

    [[nodiscard]] MayAnalysisState
    handle_state_request(const program::InstructionIR_sptr& instruction, bool before);

    [[nodiscard]] MayAnalysisState
    compute_after_state_from_cache(const program::InstructionIR_sptr& instruction);

    [[nodiscard]] const metadata::points_to::ObjectPool*
    get_object_pool(const program::InstructionIR_sptr& instruction, const program::VariableIR& x);

    void populate_cache_before(const program::InstructionIR_sptr& instruction);

    void apply_transfer_to_state(const program::InstructionIR_sptr& instruction,
                                 MayAnalysisState& state, std::size_t bb_index,
                                 std::size_t instr_index);

    void build_object_cache();

    [[nodiscard]] std::size_t get_basic_block_index(program::BasicBlockIR_raw bb_raw) const;

    void reset_cached_after_state();

    [[nodiscard]] bool try_advance_cache_to(const program::InstructionIR_sptr& instruction);

    void rebuild_cache_before(const program::InstructionIR_sptr& instruction);

  private:
    program::ProgramIR_csptr                            program_keepalive_;
    program::FunctionIR_csptr                           function_;
    objectId                                            last_local_id_{0};
    utils::SparseMap<objectId, program::OperandIR_sptr> object_cache_;

    cache_t cache_;

    const metadata::points_to::ProgramMeta*  program_meta_{nullptr};
    const metadata::points_to::FunctionMeta* function_meta_{nullptr};

    const metadata::points_to::ObjectPool* global_objects_{nullptr};
    const metadata::points_to::ObjectPool* local_objects_{nullptr};

    std::function<void(const utils::points_to::MayTransferContextBundle& context)> transfer_may_;
};

} // namespace optimizer::query

#endif
