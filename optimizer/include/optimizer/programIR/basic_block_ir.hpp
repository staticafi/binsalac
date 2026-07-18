#ifndef OPTIMIZER_BASIC_BLOCK_IR_INCLUDED
#define OPTIMIZER_BASIC_BLOCK_IR_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

#include <sala/program.hpp>

#include <memory>

namespace optimizer::program
{
class BasicBlockIR : public std::enable_shared_from_this<BasicBlockIR>
{
    using Metadata = metadata::Metadata<BasicBlockMetaEntryI>;

  public:
    BasicBlockIR() = default;
    void assign_to_function(const FunctionIR_sptr& function);

    FunctionIR_raw                               get_function_raw() const;
    FunctionIR_sptr                              get_function() const;
    const std::optional<BasicBlockIRListS_iter>& get_self_it() const;
    const Metadata&                              get_metadata() const;
    const InstructionIRListS&                    get_instructions() const;
    const BasicBlockIRListW&                     get_successors() const;
    const BasicBlockIRListW&                     get_predecessors() const;

    std::optional<BasicBlockIRListS_iter>& get_self_it();
    Metadata&                              get_metadata();

    void set_function(FunctionIR_sptr function);
    void acquire_instruction(InstructionIR_sptr instruction);
    InstructionIRListS_iter release_instruction(const InstructionIR_sptr& instruction);

    void add_successor(BasicBlockIR_sptr successor);
    void add_predecessor(BasicBlockIR_sptr predecessor);

    void remove_predecessor(const BasicBlockIR_sptr& predecessor);
    void remove_successor(const BasicBlockIR_sptr& successor);

  private:
    FunctionIR_wptr                       function_;
    FunctionIR_raw                        function_raw_;
    std::optional<BasicBlockIRListS_iter> self_it_;

    BasicBlockIRListW predecessors_;
    BasicBlockIRListW successors_;

    InstructionIRListS instructions_;

    Metadata metadata_;
};

} // namespace optimizer::program
#endif
