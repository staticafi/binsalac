#ifndef OPTIMIZER_FUNCTION_IR_HPP_INCLUDED
#define OPTIMIZER_FUNCTION_IR_HPP_INCLUDED
#include <optimizer/metadata/metadata.hpp>
#include <optimizer/programIR/ir_types.hpp>

#include <sala/program.hpp>

namespace optimizer::program
{

class FunctionIR : public std::enable_shared_from_this<FunctionIR>
{
  public:
    using Metadata = metadata::Metadata<FunctionMetaEntryI>;

    FunctionIR() = default;
    void assign_to_program(const ProgramIR_sptr& program);

    ProgramIR_sptr                             get_program() const;
    const std::optional<FunctionIRListS_iter>& get_self_it() const;
    const Metadata&                            get_metadata() const;
    const BasicBlockIRListS&                   get_basic_blocks() const;
    const BasicBlockIR_sptr&                   get_entry_basic_block() const;
    const VariableIRListS&                     get_local_variables() const;
    const VariableIRListS&                     get_parameters() const;
    std::size_t                                get_initial_stack_bytes() const;
    bool                                       get_external_flag() const;
    bool                                       get_initializer_flag() const;
    bool                                       get_entry_flag() const;

    std::optional<FunctionIRListS_iter>& get_self_it();
    Metadata&                            get_metadata();
    BasicBlockIR_sptr&                   get_entry_basic_block();
    std::size_t&                         get_initial_stack_bytes();
    bool&                                get_external_flag();
    bool&                                get_initializer_flag();
    bool&                                get_entry_flag();

    void acquire_basic_block(BasicBlockIR_sptr basic_block);
    void acquire_local_variable(VariableIR_sptr variable);
    void acquire_parameter(VariableIR_sptr parameter);

    void release_basic_block(const BasicBlockIR_sptr& basic_block);
    void release_local_variable(const VariableIR_sptr& variable);
    void release_parameter(const VariableIR_sptr& parameter);

    void set_program(ProgramIR_sptr program);

  private:
    ProgramIR_wptr                      program_;
    std::optional<FunctionIRListS_iter> self_it_;

    BasicBlockIR_sptr entry_basic_block_;

    BasicBlockIRListS basic_blocks_;
    VariableIRListS   local_variables_;
    VariableIRListS   parameters_;

    Metadata    metadata_;
    bool        external_flag_{false};
    bool        initializer_flag_{false};
    bool        entry_flag_{false};
    std::size_t initial_stack_bytes_{};
};
} // namespace optimizer::program

#endif
