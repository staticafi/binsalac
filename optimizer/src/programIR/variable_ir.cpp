#include <optimizer/programIR/variable_ir.hpp>

#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::program
{
void release_variable(const VariableIR_sptr& variable)
{

    ASSUMPTION(variable != nullptr);

    const auto var_context  = variable->get_context();
    const auto owner_before = variable->get_owner();
    switch (var_context)
    {
    case (VariableIR::Context::STATIC):
    {
        ASSUMPTION(std::holds_alternative<ProgramIR_sptr>(owner_before));
        const auto true_owner_before = std::get<ProgramIR_sptr>(owner_before);
        ASSUMPTION(true_owner_before != nullptr);
        true_owner_before->release_static(variable);
        break;
    }
    case (VariableIR::Context::PARAMETER):
    {
        ASSUMPTION(std::holds_alternative<FunctionIR_sptr>(owner_before));
        const auto true_owner_before = std::get<FunctionIR_sptr>(owner_before);
        ASSUMPTION(true_owner_before != nullptr);
        true_owner_before->release_parameter(variable);
        break;
    }
    case (VariableIR::Context::LOCAL):
    {
        ASSUMPTION(std::holds_alternative<FunctionIR_sptr>(owner_before));
        const auto true_owner_before = std::get<FunctionIR_sptr>(owner_before);
        ASSUMPTION(true_owner_before != nullptr);
        true_owner_before->release_local_variable(variable);
        break;
    }
    case (VariableIR::Context::UNDEFINED):
        ASSUMPTION(std::holds_alternative<std::monostate>(owner_before));
        break;
    }
}

void VariableIR::assign_as_local(const FunctionIR_sptr& function)
{
    function->acquire_local_variable(shared_from_this());
}

void VariableIR::assign_as_parameter(const FunctionIR_sptr& function)
{
    function->acquire_parameter(shared_from_this());
}

void VariableIR::assign_as_static(const ProgramIR_sptr& program)
{
    program->acquire_static(shared_from_this());
}

const std::optional<VariableIRListS_iter>& VariableIR::get_self_it() const
{
    return self_it_;
}

const VariableIR::Metadata& VariableIR::get_metadata() const
{
    return metadata_;
}

ProgramIR_sptr VariableIR::get_program() const
{
    return program_.lock();
};

FunctionIR_sptr VariableIR::get_function() const
{
    return function_.lock();
}

VariableIR::Context VariableIR::get_context() const
{
    return context_;
}

std::size_t VariableIR::get_num_bytes() const
{
    return num_bytes_;
}

bool VariableIR::get_external_flag() const
{
    return external_flag_;
}

VariableIR::Context& VariableIR::get_context()
{
    return context_;
}

VariableIR::Metadata& VariableIR::get_metadata()
{
    return metadata_;
}

std::optional<VariableIRListS_iter>& VariableIR::get_self_it()
{
    return self_it_;
}

std::size_t& VariableIR::get_num_bytes()
{
    return num_bytes_;
}

bool& VariableIR::get_external_flag()
{
    return external_flag_;
}

void VariableIR::set_program(ProgramIR_sptr program)
{
    // NOTE: only one owner at a time
    ASSUMPTION(function_.lock() == nullptr);

    program_ = std::move(program);
}

void VariableIR::set_function(FunctionIR_sptr function)
{
    // NOTE: only one owner at a time
    ASSUMPTION(program_.lock() == nullptr);

    function_ = std::move(function);
}

std::variant<std::monostate, FunctionIR_sptr, ProgramIR_sptr> VariableIR::get_owner() const
{
    if (auto locked_owner = function_.lock(); locked_owner != nullptr)
    {
        return locked_owner;
    }

    if (auto locked_owner = program_.lock(); locked_owner != nullptr)
    {
        return locked_owner;
    }

    return std::monostate();
};

} // namespace optimizer::program
