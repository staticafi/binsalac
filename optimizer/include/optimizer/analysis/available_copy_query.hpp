#ifndef OPTIMIZER_ANALYSIS_AVAILABLE_COPY_QUERY_HPP_INCLUDED
#define OPTIMIZER_ANALYSIS_AVAILABLE_COPY_QUERY_HPP_INCLUDED

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/available_copy/import.hpp>
#include <optimizer/utils/dynamic_bitset.hpp>
#include <optimizer/utils/sparse_map.hpp>

#include <optional>
#include <vector>

namespace optimizer::analysis
{

struct AvailableCopyResult
{
    std::optional<program::VariableIR_sptr> direct_source;
    std::optional<program::VariableIR_sptr> transitive_source;
    std::vector<program::VariableIR_sptr>   chain;
};

class AvailableCopyQueryFunction
{
  public:
    explicit AvailableCopyQueryFunction(program::FunctionIR_csptr function);

    AvailableCopyResult before(const program::InstructionIR_sptr& instruction,
                               const program::VariableIR&         x);

    AvailableCopyResult after(const program::InstructionIR_sptr& instruction,
                              const program::VariableIR&         x);

    [[nodiscard]] std::optional<program::VariableIR_sptr> get_variable(std::size_t id) const;

  private:
    using State = utils::DynamicBitset;

    struct cache_t
    {
        program::BasicBlockIR_raw  bb{};
        program::InstructionIR_raw instr{};
        State                      in_state{};
    };

    AvailableCopyResult handle_request(const program::InstructionIR_sptr& instruction,
                                       const program::VariableIR& x, bool before);

    AvailableCopyResult handle_cache_hit(const program::InstructionIR_sptr& instruction,
                                         const program::VariableIR& x, bool before);

    void reconstruct_before_state(const program::InstructionIR_sptr& instruction);

    [[nodiscard]] std::optional<std::size_t> resolve_direct_source_id(std::size_t  variable_id,
                                                                      const State& state) const;

    [[nodiscard]] std::optional<std::size_t> resolve_transitive_source_id(std::size_t  variable_id,
                                                                          const State& state) const;

    [[nodiscard]] std::vector<std::size_t> build_chain_ids(std::size_t  variable_id,
                                                           const State& state) const;

    [[nodiscard]] utils::TransferContext make_transfer_context() const;

    [[nodiscard]] static std::size_t
    get_instruction_index(const program::InstructionIR_sptr& instruction);

  private:
    program::ProgramIR_csptr  program_keepalive_;
    program::FunctionIR_csptr function_;

    cache_t cache_{};

    const metadata::available_copy::FunctionMeta* function_meta_{};

    utils::SparseMap<program::VariableIR_raw, std::size_t> variable_ids_;
    utils::SparseMap<utils::CopyFactKey, std::size_t>      fact_id_of_;
    std::vector<utils::DynamicBitset>                      kill_masks_;
};

} // namespace optimizer::analysis

#endif
