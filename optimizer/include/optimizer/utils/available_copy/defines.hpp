#ifndef OPTIMIZER_UTILS_AVAILABLE_COPY_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_AVAILABLE_COPY_DEFINES_HPP_INCLUDED

#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/ir_types.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/dynamic_bitset.hpp>
#include <optimizer/utils/sparse_map.hpp>

#include <compare>
#include <cstddef>
#include <tuple>
#include <vector>

namespace optimizer::utils::available_copy
{

using State = optimizer::utils::DynamicBitset;

struct CopyFactKey
{
    std::size_t dest_id{0};
    std::size_t source_id{0};

    bool operator==(const CopyFactKey& other) const
    {
        return std::tie(dest_id, source_id) == std::tie(other.dest_id, other.source_id);
    }

    auto operator<=>(const CopyFactKey& other) const
    {
        return std::tie(dest_id, source_id) <=> std::tie(other.dest_id, other.source_id);
    }
};

struct CopyFact
{
    std::size_t             id{0};
    std::size_t             dest_id{0};
    std::size_t             source_id{0};
    program::VariableIR_raw dest{nullptr};
    program::VariableIR_raw source{nullptr};
};

struct TransferContext
{
    const optimizer::utils::SparseMap<program::VariableIR_raw, std::size_t>& variable_ids;
    const std::vector<CopyFact>&                                             facts;
    const std::vector<std::vector<std::size_t>>&                             facts_by_dest;
    const std::vector<std::vector<std::size_t>>&                             facts_by_source;
    const optimizer::utils::SparseMap<CopyFactKey, std::size_t>&             fact_id_of;
    const std::vector<State>&                                                kill_masks;
};

bool is_direct_def_opcode(sala::Instruction::Opcode opcode);
bool has_global_memory_side_effects(sala::Instruction::Opcode opcode);
bool is_written_operand(sala::Instruction::Opcode opcode, std::size_t operand_index);

void kill_all_copies_of_variable(std::size_t variable_id, const std::vector<State>& kill_masks,
                                 State& state);

void apply_transfer(const program::InstructionIR_sptr& instruction, const TransferContext& ctx,
                    State& state);

} // namespace optimizer::utils::available_copy

#endif
