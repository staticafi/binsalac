#ifndef OPTIMIZER_CONSTANT_IR_HPP_INCLUDED
#define OPTIMIZER_CONSTANT_IR_HPP_INCLUDED
#include <optimizer/programIR/ir_types.hpp>

#include <vector>

namespace optimizer::program
{

class ConstantIR : public std::enable_shared_from_this<ConstantIR>
{
  public:
    using Metadata = metadata::Metadata<ConstantMetaEntryI>;

    ConstantIR() = default;
    void assign_to_program(const ProgramIR_sptr& program);

    void set_program(ProgramIR_sptr program);

    const Metadata& get_metadata() const;
    Metadata&       get_metadata();

    ProgramIR_sptr                             get_program() const;
    const std::optional<ConstantIRListS_iter>& get_self_it() const;
    const std::vector<std::uint8_t>&           get_bytes() const;

    std::optional<ConstantIRListS_iter>& get_self_it();
    std::vector<std::uint8_t>&           get_bytes();

  private:
    ProgramIR_wptr                      program_;
    std::optional<ConstantIRListS_iter> self_it_;
    Metadata                            metadata_;

    std::vector<std::uint8_t> bytes_;
};
} // namespace optimizer::program

#endif
