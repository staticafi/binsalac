#ifndef OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#define OPTIMIZER_REMOVE_INDIRECTIONS_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>

namespace optimizer::passes
{
class RemoveIndirections
{
  public:
    RemoveIndirections()  = default;
    ~RemoveIndirections() = default;

    RemoveIndirections(const RemoveIndirections&)            = delete;
    RemoveIndirections(RemoveIndirections&&)                 = delete;
    RemoveIndirections& operator=(const RemoveIndirections&) = delete;
    RemoveIndirections& operator=(RemoveIndirections&&)      = delete;

    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_{};
};

template <>
struct PassTraits<RemoveIndirections>
{
    using kind      = passes::TransformPass;
    using needs     = utils::TypeList<KeyTag<metadata::MetaKey::POINTS_TO>>;
    using provides  = utils::TypeList<>;
    using preserves = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>>;

    static constexpr Repr        required_repr = Repr::IR;
    static constexpr const char* name          = "RemoveIndirectionsTransform";
};
} // namespace optimizer::passes
#endif
