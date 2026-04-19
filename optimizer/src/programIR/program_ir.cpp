#include <optimizer/programIR/program_ir.hpp>

#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::program
{

void ProgramIR::acquire_function(FunctionIR_sptr function)
{
    ASSUMPTION(function != nullptr);

    if (const auto owner_before = function->get_program(); owner_before != nullptr)
    {
        owner_before->release_function(function);
    }

    function->set_program(shared_from_this());
    functions_.push_back(std::move(function));
    functions_.back()->get_self_it() = --functions_.end();
}
void ProgramIR::acquire_static(VariableIR_sptr variable)
{
    ASSUMPTION(variable != nullptr);
    release_variable(variable);

    variable->set_program(shared_from_this());
    variable->get_context() = VariableIR::Context::STATIC;
    static_vars_.push_back(std::move(variable));
    static_vars_.back()->get_self_it() = --static_vars_.end();
}

void ProgramIR::acquire_constant(ConstantIR_sptr constant)
{
    ASSUMPTION(constant != nullptr);
    if (const auto owner_before = constant->get_program(); owner_before != nullptr)
    {
        owner_before->release_constant(constant);
    }

    constant->set_program(shared_from_this());
    constants_.push_back(std::move(constant));
    constants_.back()->get_self_it() = --constants_.end();
}

void ProgramIR::release_function(const FunctionIR_sptr& function)
{
    ASSUMPTION(function->get_program().get() == this);
    ASSUMPTION(function->get_self_it().has_value());

    function->set_program(nullptr);
    functions_.erase(function->get_self_it().value());
    function->get_self_it() = std::nullopt;
}

void ProgramIR::release_constant(const ConstantIR_sptr& constat)
{
    ASSUMPTION(constat->get_program().get() == this);
    ASSUMPTION(constat->get_self_it().has_value());

    constat->set_program(nullptr);
    constants_.erase(constat->get_self_it().value());
    constat->get_self_it() = std::nullopt;
}

void ProgramIR::release_static(const VariableIR_sptr& static_var)
{
    ASSUMPTION(static_var->get_program().get() == this);
    ASSUMPTION(static_var->get_self_it().has_value());

    static_var->set_program(nullptr);
    static_var->get_context() = VariableIR::Context::UNDEFINED;
    static_vars_.erase(static_var->get_self_it().value());
    static_var->get_self_it() = std::nullopt;
}

const FunctionIRListS& ProgramIR::get_functions() const
{
    return functions_;
}

const ConstantIRListS& ProgramIR::get_constants() const
{
    return constants_;
}

const FunctionIR_sptr& ProgramIR::get_static_initializer_func() const
{
    return static_initializer_func_;
}

const FunctionIR_sptr& ProgramIR::get_entry_func() const
{
    return entry_func_;
}

const VariableIRListS& ProgramIR::get_static_vars() const
{
    return static_vars_;
}

const ProgramIR::Metadata& ProgramIR::get_metadata() const
{
    return metadata_;
}

VariableIRListS& ProgramIR::get_static_vars()
{
    return static_vars_;
}

FunctionIR_sptr& ProgramIR::get_static_initializer_func()
{
    return static_initializer_func_;
}

FunctionIR_sptr& ProgramIR::get_entry_func()
{
    return entry_func_;
}

ProgramIR::Metadata& ProgramIR::get_metadata()
{
    return metadata_;
}

ConstantIRListS& ProgramIR::get_constants()
{
    return constants_;
}

std::uint16_t ProgramIR::get_num_cpu_bits() const
{
    return num_cpu_bits_;
}

std::uint16_t& ProgramIR::get_num_cpu_bits()
{
    return num_cpu_bits_;
}
} // namespace optimizer::program
