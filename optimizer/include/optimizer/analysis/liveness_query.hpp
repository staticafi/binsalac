#ifndef OPTIMIZER_ANALYSIS_LIVENESS_QUERY_HPP_INCLUDED
#define OPTIMIZER_ANALYSIS_LIVENESS_QUERY_HPP_INCLUDED

#include <optimizer/metadata/liveness.hpp>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::analysis
{
class LivenessQueryFunction
{
  public:
    explicit LivenessQueryFunction(program::FunctionIR_csptr function);

    [[nodiscard]] bool is_live_before(const program::InstructionIR_sptr& instruction,
                                      const program::VariableIR&         variable) const;
    [[nodiscard]] bool is_live_after(const program::InstructionIR_sptr& instruction,
                                     const program::VariableIR&         variable) const;

    [[nodiscard]] bool is_live_before(const program::InstructionIR_sptr& instruction,
                                      const program::ConstantIR&         constant) const;
    [[nodiscard]] bool is_live_after(const program::InstructionIR_sptr& instruction,
                                     const program::ConstantIR&         constant) const;

    [[nodiscard]] optimizer::utils::liveness::LiveSet
    live_before(const program::InstructionIR_sptr& instruction) const;

    [[nodiscard]] optimizer::utils::liveness::LiveSet
    live_after(const program::InstructionIR_sptr& instruction) const;

    [[nodiscard]] bool is_removable(const program::InstructionIR_sptr& instruction) const;

    [[nodiscard]] const optimizer::utils::liveness::LiveSet&
    basic_block_live_out(const program::BasicBlockIR_csptr& basic_block) const;

    [[nodiscard]] const optimizer::utils::SparseSet<program::InstructionIR_raw>&
    removable_instructions(const program::BasicBlockIR_csptr& basic_block) const;

  private:
    [[nodiscard]] const metadata::liveness::BasicBlockMeta&
    get_basic_block_meta(const program::BasicBlockIR_csptr& basic_block) const;

  private:
    program::ProgramIR_csptr  program_keepalive_;
    program::FunctionIR_csptr function_;
};
} // namespace optimizer::analysis

#endif
