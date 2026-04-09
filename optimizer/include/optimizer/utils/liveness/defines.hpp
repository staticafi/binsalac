#ifndef OPTIMIZER_UTILS_LIVENESS_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_LIVENESS_DEFINES_HPP_INCLUDED

#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/ir_types.hpp>
#include <optimizer/utils/sparse_set.hpp>

namespace optimizer::utils::liveness
{
using Operand = program::OperandIR_raw;
using LiveSet = optimizer::utils::SparseSet<Operand>;

bool is_trackable_operand(const Operand& operand) noexcept;

bool is_written_operand(sala::Instruction::Opcode opcode, std::size_t operand_index) noexcept;

bool has_observable_side_effects(const program::InstructionIR_sptr& instruction);

bool is_pure_dead_def_candidate(const program::InstructionIR_sptr& instruction);

void collect_uses_and_defs(const program::InstructionIR_sptr& instruction, LiveSet& uses,
                           LiveSet& defs);

void apply_backward_transfer(const program::InstructionIR_sptr& instruction,
                             const LiveSet& live_after, LiveSet& live_before);

bool is_instruction_removable(const program::InstructionIR_sptr& instruction,
                              const LiveSet&                     live_after);

} // namespace optimizer::utils::liveness

#endif
