#ifndef OPTIMZER_LOWER_SALA_TO_IR_PASS_HPP_INCLUDED
#define OPTIMZER_LOWER_SALA_TO_IR_PASS_HPP_INCLUDED
#include <optimizer/translation/sala_to_ir.hpp>

namespace optimizer::passes
{

class LowerSalaToIR
{
  public:
    program::ProgramIR_sptr run(std::shared_ptr<sala::Program> sala)
    {
        return translation_strategy_.translate(std::move(sala));
    }

  private:
    translation::SalaToIR translation_strategy_;
};
} // namespace optimizer::passes
#endif
