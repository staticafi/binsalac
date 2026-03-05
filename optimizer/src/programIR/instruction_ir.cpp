#include <optimizer/programIR/instruction_ir.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::program
{
BasicBlockIR* InstructionIR::get_basic_block_raw() const
{
    return basic_block_raw_;
}

BasicBlockIR_sptr InstructionIR::get_basic_block() const
{
    return basic_block_.lock();
}

sala::Instruction::Opcode InstructionIR::get_opcode() const
{
    return opcode_;
}

sala::Instruction::Modifier InstructionIR::get_modifier() const
{
    return modifier_;
}

const std::optional<InstructionIRListS_iter>& InstructionIR::get_self_it() const
{
    return self_it_;
}
const InstructionIR::Metadata& InstructionIR::get_metadata() const
{
    return metadata_;
}

const OperandIRListW& InstructionIR::get_operands() const
{
    return operands_;
}

std::optional<InstructionIRListS_iter>& InstructionIR::get_self_it()
{
    return self_it_;
}

InstructionIR::Metadata& InstructionIR::get_metadata()
{
    return metadata_;
}

sala::Instruction::Opcode& InstructionIR::get_opcode()
{
    return opcode_;
}

sala::Instruction::Modifier& InstructionIR::get_modifier()
{
    return modifier_;
}

void InstructionIR::set_basic_block(BasicBlockIR_sptr basic_block)
{
    basic_block_raw_ = basic_block.get();
    basic_block_     = basic_block;
}

void InstructionIR::assign_to_basic_block(const BasicBlockIR_sptr& basic_block)
{
    basic_block->acquire_instruction(shared_from_this());
}

void InstructionIR::push_back_operand(VariableIR_sptr variable)
{
    ASSUMPTION(variable != nullptr);
    push_back_operand_impl(std::move(variable));
}

void InstructionIR::push_back_operand(FunctionIR_sptr function)
{
    ASSUMPTION(function != nullptr);
    push_back_operand_impl(std::move(function));
}

void InstructionIR::push_back_operand(ConstantIR_sptr constant)
{
    ASSUMPTION(constant != nullptr);
    push_back_operand_impl(std::move(constant));
}

void InstructionIR::push_back_operand_impl(OperandIR_wptr operand)
{
    operands_.emplace_back(std::move(operand));
}

OperandIRListW& InstructionIR::get_operands()
{
    return operands_;
}
} // namespace optimizer::program
