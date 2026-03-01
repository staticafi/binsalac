#ifndef OPTIMZER_BUMP_IR_TO_SALA_PASS_HPP_INCLUDED
#define OPTIMZER_BUMP_IR_TO_SALA_PASS_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>
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

template <>
struct PassTraits<BumpIrToSala>
{
    using kind      = passes::TranslationPass;
    using needs     = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>>;
    using provides  = utils::TypeList<>;
    using preserves = utils::TypeList<>;

    static constexpr passes::Repr from_repr = passes::Repr::IR;
    static constexpr passes::Repr to_repr   = passes::Repr::Sala;
    static constexpr const char*  name      = "BumpIrToSalaTranslation";
};

} // namespace optimizer::passes
#endif
