#include <optimizer/programIR/function_ir.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::program
{

void FunctionIR::assign_to_program(const ProgramIR_sptr& program)
{
    ASSUMPTION(program != nullptr);

    program->acquire_function(shared_from_this());
}

ProgramIR_sptr FunctionIR::get_program() const
{
    return program_.lock();
}

const std::optional<FunctionIRListS_iter>& FunctionIR::get_self_it() const
{
    return self_it_;
}
const FunctionIR::Metadata& FunctionIR::get_metadata() const
{
    return metadata_;
}

const BasicBlockIRListS& FunctionIR::get_basic_blocks() const
{
    return basic_blocks_;
}

const BasicBlockIR_sptr& FunctionIR::get_entry_basic_block() const
{
    return entry_basic_block_;
}

const VariableIRListS& FunctionIR::get_local_variables() const
{
    return local_variables_;
}

const VariableIRListS& FunctionIR::get_parameters() const
{
    return parameters_;
}

std::size_t FunctionIR::get_initial_stack_bytes() const
{
    return initial_stack_bytes_;
}

bool FunctionIR::get_external_flag() const
{
    return external_flag_;
}

bool FunctionIR::get_initializer_flag() const
{
    return initializer_flag_;
}

bool FunctionIR::get_entry_flag() const
{
    return entry_flag_;
}

std::optional<FunctionIRListS_iter>& FunctionIR::get_self_it()
{
    return self_it_;
}

FunctionIR::Metadata& FunctionIR::get_metadata()
{
    return metadata_;
}

BasicBlockIR_sptr& FunctionIR::get_entry_basic_block()
{
    return entry_basic_block_;
}

bool& FunctionIR::get_external_flag()
{
    return external_flag_;
}

bool& FunctionIR::get_initializer_flag()
{
    return initializer_flag_;
}

bool& FunctionIR::get_entry_flag()
{
    return entry_flag_;
}

std::size_t& FunctionIR::get_initial_stack_bytes()
{
    return initial_stack_bytes_;
}

void FunctionIR::acquire_basic_block(BasicBlockIR_sptr basic_block)
{
    ASSUMPTION(basic_block != nullptr);

    if (const auto owner_before = basic_block->get_function(); owner_before != nullptr)
    {
        owner_before->release_basic_block(basic_block);
    }

    basic_block->set_function(shared_from_this());
    basic_blocks_.push_back(std::move(basic_block));
    basic_blocks_.back()->get_self_it() = --basic_blocks_.end();
}

void FunctionIR::acquire_local_variable(VariableIR_sptr variable)
{
    ASSUMPTION(variable != nullptr);
    release_variable(variable);

    variable->set_function(shared_from_this());
    variable->get_context() = VariableIR::Context::LOCAL;
    local_variables_.push_back(std::move(variable));
    local_variables_.back()->get_self_it() = --local_variables_.end();
}

void FunctionIR::acquire_parameter(VariableIR_sptr parameter)
{
    ASSUMPTION(parameter != nullptr);
    release_variable(parameter);

    parameter->set_function(shared_from_this());
    parameter->get_context() = VariableIR::Context::PARAMETER;
    parameters_.push_back(std::move(parameter));
    parameters_.back()->get_self_it() = --local_variables_.end();
}

void FunctionIR::release_basic_block(const BasicBlockIR_sptr& basic_block)
{
    ASSUMPTION(basic_block->get_function().get() == this);
    ASSUMPTION(basic_block->get_self_it().has_value());

    basic_block->set_function(nullptr);
    basic_blocks_.erase(basic_block->get_self_it().value());
    basic_block->get_self_it() = std::nullopt;
}

void FunctionIR::release_local_variable(const VariableIR_sptr& variable)
{
    ASSUMPTION(variable->get_function().get() == this);
    ASSUMPTION(variable->get_self_it().has_value());

    variable->set_function(nullptr);
    local_variables_.erase(variable->get_self_it().value());
    variable->get_self_it() = std::nullopt;
}

void FunctionIR::release_parameter(const VariableIR_sptr& parameter)
{
    ASSUMPTION(parameter->get_function().get() == this);
    ASSUMPTION(parameter->get_self_it().has_value());

    parameter->set_function(nullptr);
    parameters_.erase(parameter->get_self_it().value());
    parameter->get_self_it() = std::nullopt;
}

void FunctionIR::set_program(ProgramIR_sptr program)
{
    program_ = std::move(program);
}
} // namespace optimizer::program
