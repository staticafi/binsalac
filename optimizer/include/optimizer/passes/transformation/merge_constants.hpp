#ifndef OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED
#define OPTIMIZER_MERGE_CONSTANTS_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>
#include <optimizer/programIR/constant_ir.hpp>

namespace optimizer::passes
{

class MergeConstants
{
  public:
    MergeConstants()  = default;
    ~MergeConstants() = default;

    MergeConstants(const MergeConstants&)                    = delete;
    MergeConstants(MergeConstants&&)                         = delete;
    MergeConstants&         operator=(const MergeConstants&) = delete;
    MergeConstants&         operator=(MergeConstants&&)      = delete;
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

template <>
struct PassTraits<MergeConstants>
{
    using kind      = passes::TransformPass;
    using needs     = utils::TypeList<>;
    using provides  = utils::TypeList<>;
    using preserves = utils::TypeList<KeyTag<metadata::MetaKey::TRANSLATION_SALA_TO_IR>>;

    static constexpr Repr        required_repr = Repr::IR;
    static constexpr const char* name          = "MergeConstantsTransform";
};
} // namespace optimizer::passes
#endif
