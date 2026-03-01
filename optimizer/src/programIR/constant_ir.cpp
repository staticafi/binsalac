#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>

namespace optimizer::program
{
void ConstantIR::assign_to_program(const ProgramIR_sptr& program)
{
    program->acquire_constant(shared_from_this());
}

void ConstantIR::set_program(ProgramIR_sptr program)
{
    program_ = std::move(program);
}

ProgramIR_sptr ConstantIR::get_program() const
{
    return program_.lock();
}

const std::optional<ConstantIRListS_iter>& ConstantIR::get_self_it() const
{
    return self_it_;
}

const std::vector<std::uint8_t>& ConstantIR::get_bytes() const
{
    return bytes_;
}

std::optional<ConstantIRListS_iter>& ConstantIR::get_self_it()
{
    return self_it_;
}

std::vector<std::uint8_t>& ConstantIR::get_bytes()
{
    return bytes_;
}

const ConstantIR::Metadata& ConstantIR::get_metadata() const
{
    return metadata_;
}

ConstantIR::Metadata& ConstantIR::get_metadata()
{
    return metadata_;
}
} // namespace optimizer::program
