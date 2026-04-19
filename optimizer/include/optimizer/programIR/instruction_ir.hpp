#ifndef OPTIMIZER_INSTRUCTION_IR_HPP_INCLUDED
#define OPTIMIZER_INSTRUCTION_IR_HPP_INCLUDED
#include <optimizer/metadata/metadata.hpp>
#include <optimizer/programIR/ir_types.hpp>

#include <sala/program.hpp>

namespace optimizer::program
{

class InstructionIR : public std::enable_shared_from_this<InstructionIR>
{
  public:
    using Metadata = metadata::Metadata<InstructionMetaEntryI>;

    InstructionIR() = default;

    void assign_to_basic_block(const BasicBlockIR_sptr& basic_block);

    BasicBlockIR_raw                              get_basic_block_raw() const;
    BasicBlockIR_sptr                             get_basic_block() const;
    sala::Instruction::Opcode                     get_opcode() const;
    sala::Instruction::Modifier                   get_modifier() const;
    const std::optional<InstructionIRListS_iter>& get_self_it() const;
    const Metadata&                               get_metadata() const;
    const OperandIRVecR&                          get_operands() const;

    std::optional<InstructionIRListS_iter>& get_self_it();
    Metadata&                               get_metadata();
    sala::Instruction::Opcode&              get_opcode();
    sala::Instruction::Modifier&            get_modifier();
    void                                    set_basic_block(BasicBlockIR_sptr basic_block);
    OperandIRVecR&                          get_operands();

    void push_back_operand(VariableIR_sptr variable);
    void push_back_operand(ConstantIR_sptr constant);
    void push_back_operand(FunctionIR_sptr function);

  private:
    BasicBlockIR_wptr basic_block_;
    BasicBlockIR_raw  basic_block_raw_;

    std::optional<InstructionIRListS_iter> self_it_;
    sala::Instruction::Opcode              opcode_{};
    sala::Instruction::Modifier            modifier_{};

    OperandIRVecR operands_;
    Metadata      metadata_;
};

} // namespace optimizer::program

#endif
