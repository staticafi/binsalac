
#ifndef OPTIMIZER_LOCAL_SERIALIZE_POINTS_TO_HPP_INCLUDED
#define OPTIMIZER_LOCAL_SERIALIZE_POINTS_TO_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>

namespace optimizer::passes
{

class SerializePointsTo
{
  public:
    SerializePointsTo();
    ~SerializePointsTo();
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

template <>
struct PassTraits<SerializePointsTo>
{
    using kind      = passes::DebugPass;
    using needs     = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>,
                                      KeyTag<metadata::MetaKey::POINTS_TO>>;
    using provides  = utils::TypeList<>;
    using preserves = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>,
                                      KeyTag<metadata::MetaKey::POINTS_TO>>;

    static constexpr Repr        required_repr = Repr::IR;
    static constexpr const char* name          = "[DEBUG] SerializePointsTo";
};
} // namespace optimizer::passes
#endif
