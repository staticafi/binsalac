#ifndef OPTIMIZER_PROGRAM_IR_HPP_INCLUDED
#define OPTIMIZER_PROGRAM_IR_HPP_INCLUDED
#include <optimizer/metadata/metadata.hpp>

#include <optimizer/programIR/ir_types.hpp>

#include <sala/program.hpp>

namespace optimizer::program
{

class ProgramIR : public std::enable_shared_from_this<ProgramIR>
{
  public:
    using Metadata = metadata::Metadata<ProgramMetaEntryI>;

    void acquire_constant(ConstantIR_sptr constant);
    void acquire_function(FunctionIR_sptr function);
    void acquire_static(VariableIR_sptr static_var);

    void release_function(const FunctionIR_sptr& function);
    void release_constant(const ConstantIR_sptr& constant);
    void release_static(const VariableIR_sptr& static_var);

    const VariableIRListS& get_static_vars() const;
    const FunctionIRListS& get_functions() const;
    const ConstantIRListS& get_constants() const;
    const FunctionIR_sptr& get_static_initializer_func() const;
    const FunctionIR_sptr& get_entry_func() const;
    const Metadata&        get_metadata() const;
    std::uint16_t          get_num_cpu_bits() const;

    VariableIRListS& get_static_vars();
    FunctionIR_sptr& get_static_initializer_func();
    ConstantIRListS& get_constants();
    FunctionIR_sptr& get_entry_func();
    Metadata&        get_metadata();
    std::uint16_t&   get_num_cpu_bits();

  private:
    FunctionIRListS functions_;
    FunctionIR_sptr entry_func_;
    FunctionIR_sptr static_initializer_func_;

    ConstantIRListS constants_;
    VariableIRListS static_vars_;
    std::uint16_t   num_cpu_bits_;

    Metadata metadata_;
};
} // namespace optimizer::program

#endif
