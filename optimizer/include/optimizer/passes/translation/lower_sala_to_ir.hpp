#ifndef OPTIMZER_LOWER_SALA_TO_IR_PASS_HPP_INCLUDED
#define OPTIMZER_LOWER_SALA_TO_IR_PASS_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>
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

template <>
struct PassTraits<LowerSalaToIR>
{
    using kind      = passes::TranslationPass;
    using needs     = utils::TypeList<>;
    using provides  = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>>;
    using preserves = utils::TypeList<>;

    static constexpr passes::Repr from_repr = passes::Repr::Sala;
    static constexpr passes::Repr to_repr   = passes::Repr::IR;
    static constexpr const char*  name      = "LowerSalaToIR";
};

} // namespace optimizer::passes
#endif
