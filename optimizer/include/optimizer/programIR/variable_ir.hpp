#ifndef OPTIMIZER_VARIABLE_IR_INCLUDED
#define OPTIMIZER_VARIABLE_IR_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

#include <variant>
namespace optimizer::program
{

class VariableIR : public std::enable_shared_from_this<VariableIR>
{
    using Metadata = metadata::Metadata<VariableMetaEntryI>;

  public:
    enum class Context : uint8_t
    {
        UNDEFINED,
        LOCAL,
        STATIC,
        PARAMETER
    };

    VariableIR() = default;
    void assign_as_local(const FunctionIR_sptr& function);
    void assign_as_parameter(const FunctionIR_sptr& function);
    void assign_as_static(const ProgramIR_sptr& program);

    void set_program(ProgramIR_sptr program);
    void set_function(FunctionIR_sptr program);

    std::variant<std::monostate, FunctionIR_sptr, ProgramIR_sptr> get_owner() const;

    const std::optional<VariableIRListS_iter>& get_self_it() const;
    const Metadata&                            get_metadata() const;
    ProgramIR_sptr                             get_program() const;
    FunctionIR_sptr                            get_function() const;
    Context                                    get_context() const;
    std::size_t                                get_num_bytes() const;
    bool                                       get_external_flag() const;

    Context&                             get_context();
    Metadata&                            get_metadata();
    std::optional<VariableIRListS_iter>& get_self_it();
    std::size_t&                         get_num_bytes();
    bool&                                get_external_flag();

  private:
    FunctionIR_wptr function_;
    ProgramIR_wptr  program_;

    std::optional<VariableIRListS_iter> self_it_;

    std::size_t num_bytes_{0U};
    Metadata    metadata_;
    bool        external_flag_{false};

    Context context_{Context::UNDEFINED};
};

void release_variable(const VariableIR_sptr& variable);
} // namespace optimizer::program

#endif
