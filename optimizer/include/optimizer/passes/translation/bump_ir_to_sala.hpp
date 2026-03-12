#ifndef OPTIMZER_BUMP_IR_TO_SALA_PASS_HPP_INCLUDED
#define OPTIMZER_BUMP_IR_TO_SALA_PASS_HPP_INCLUDED
#include <optimizer/translation/ir_to_sala.hpp>

namespace optimizer::passes
{

class BumpIrToSala
{
  public:
    std::shared_ptr<sala::Program> run(program::ProgramIR_sptr sala_ir)
    {
        return translation_strategy_.translate(std::move(sala_ir));
    }

  private:
    translation::IRToSala translation_strategy_;
};
} // namespace optimizer::passes
#endif
