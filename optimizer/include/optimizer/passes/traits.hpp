#ifndef OPTIMIZER_PASSES_TRAITS_INCLUDED
#define OPTIMIZER_PASSES_TRAITS_INCLUDED
#include <optimizer/metadata/meta_keys.hpp>
#include <optimizer/programIR/program_repr.hpp>
#include <optimizer/utils/type_list.hpp>

namespace optimizer::passes
{

enum class Repr : uint8_t
{
    Sala,
    IR
};

struct UnspecifiedPassKind // sentinel
{
};

struct AnalysisPass
{
};

struct TransformPass
{
};

struct TranslationPass
{
};

struct DebugPass
{
};

template <metadata::MetaKey K>
struct KeyTag
{
    static constexpr metadata::MetaKey value = K;
};

template <typename Pass>
struct PassTraits
{
    using kind      = UnspecifiedPassKind; // analysis|transform|translation|debug
    using needs     = utils::TypeList<>;   // analyses required BEFORE this pass
    using provides  = utils::TypeList<>;   // analyses produced AFTER this pass
    using preserves = utils::TypeList<>;   // analyses kept across transform/translation
    // Used for analysis/transformation/debug passes
    static constexpr Repr required_repr = Repr::IR;

    // Used for translation passes
    static constexpr Repr from_repr = Repr::IR;
    static constexpr Repr to_repr   = Repr::IR;

    static constexpr const char* name = "Undefined";
};

template <typename P>
concept HasPassTraits = !std::is_same_v<typename PassTraits<P>::kind, UnspecifiedPassKind>;

} // namespace optimizer::passes
#endif
