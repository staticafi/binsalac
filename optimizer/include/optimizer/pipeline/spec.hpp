#ifndef OPTIMIZER_PIPELINE_SPEC_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_SPEC_HPP_INCLUDED

#include <optimizer/pipeline/pass_contracts.hpp>

namespace optimizer::pipeline
{

template <const char* PassName, class MissingProducts, class AvailableProducts>
struct pipeline_dependency_error;

template <const char* PassName, Repr Expected, Repr Actual>
struct pipeline_repr_error;

template <const char* PassName, Repr Expected, Repr Actual>
struct pipeline_translation_source_error;

template <class Spec, class Available>
inline constexpr bool dependency_check_v =
        utils::is_subset_v<typename Spec::dependencies, Available>;

template <class Spec, class Available>
using missing_dependencies_t = utils::set_difference_t<typename Spec::dependencies, Available>;

template <class Spec, class Available>
using dependency_guard_t = std::conditional_t<
        dependency_check_v<Spec, Available>, void,
        pipeline_dependency_error<Spec::name, missing_dependencies_t<Spec, Available>, Available>>;

template <bool Ok, const char* PassName, class MissingProducts, class AvailableProducts>
struct dependency_enforcer;

template <const char* PassName, class MissingProducts, class AvailableProducts>
struct dependency_enforcer<true, PassName, MissingProducts, AvailableProducts>
{
    static constexpr bool value = true;
};

template <const char* PassName, class MissingProducts, class AvailableProducts>
struct dependency_enforcer<false, PassName, MissingProducts, AvailableProducts>
{
    using error = pipeline_dependency_error<PassName, MissingProducts, AvailableProducts>;
    static constexpr bool value = false;
};

template <Repr R, typename Available>
struct State
{
    static constexpr Repr repr = R;
    using available            = Available;
};

template <typename S, PipelinePassDef P, typename Kind>
struct StepImpl;

template <Repr R, typename Available, PipelinePassDef P>
struct StepImpl<State<R, Available>, P, AnalysisPass>
{
    using Spec = typename P::spec;

    using _repr_guard = std::conditional_t<(Spec::required_repr == R), void,
                                           pipeline_repr_error<Spec::name, Spec::required_repr, R>>;

    using _dep_guard = dependency_guard_t<Spec, Available>;

    static_assert(Spec::required_repr == R, "Analysis scheduled for incompatible representation");
    static_assert(dependency_enforcer<dependency_check_v<Spec, Available>, Spec::name,
                                      missing_dependencies_t<Spec, Available>, Available>::value,
                  "Analysis requires metadata products not available at this point");

    using next_available = utils::set_union_unique_t<Available, typename Spec::generates>;
    using type           = State<R, next_available>;
};

template <Repr R, typename Available, PipelinePassDef P>
struct StepImpl<State<R, Available>, P, TransformPass>
{
    using Spec = typename P::spec;

    using _repr_guard = std::conditional_t<(Spec::required_repr == R), void,
                                           pipeline_repr_error<Spec::name, Spec::required_repr, R>>;

    using _dep_guard = dependency_guard_t<Spec, Available>;

    static_assert(Spec::required_repr == R, "Transform scheduled for incompatible representation");
    static_assert(dependency_enforcer<dependency_check_v<Spec, Available>, Spec::name,
                                      missing_dependencies_t<Spec, Available>, Available>::value,
                  "Transform requires metadata products not available at this point");

    using kept           = utils::set_intersection_t<Available, typename Spec::preserves>;
    using next_available = utils::set_union_unique_t<kept, typename Spec::generates>;
    using type           = State<R, next_available>;
};

template <Repr R, typename Available, PipelinePassDef P>
struct StepImpl<State<R, Available>, P, DebugPass>
{
    using Spec = typename P::spec;

    using _repr_guard = std::conditional_t<(Spec::required_repr == R), void,
                                           pipeline_repr_error<Spec::name, Spec::required_repr, R>>;

    using _dep_guard = dependency_guard_t<Spec, Available>;

    static_assert(Spec::required_repr == R, "Debug pass scheduled for incompatible representation");
    static_assert(dependency_enforcer<dependency_check_v<Spec, Available>, Spec::name,
                                      missing_dependencies_t<Spec, Available>, Available>::value,
                  "Debug requires metadata products not available at this point");

    using next_available = utils::set_union_unique_t<Available, typename Spec::generates>;
    using type           = State<R, next_available>;
};

template <Repr R, typename Available, PipelinePassDef P>
struct StepImpl<State<R, Available>, P, TranslationPass>
{
    using Spec = typename P::spec;

    using _repr_guard =
            std::conditional_t<(Spec::from_repr == R), void,
                               pipeline_translation_source_error<Spec::name, Spec::from_repr, R>>;

    using _dep_guard = dependency_guard_t<Spec, Available>;

    static_assert(Spec::from_repr == R,
                  "Translation scheduled for incompatible source representation");
    static_assert(dependency_enforcer<dependency_check_v<Spec, Available>, Spec::name,
                                      missing_dependencies_t<Spec, Available>, Available>::value,
                  "Translations requires metadata products not available at this point");

    using kept           = utils::set_intersection_t<Available, typename Spec::preserves>;
    using next_available = utils::set_union_unique_t<kept, typename Spec::generates>;
    using type           = State<Spec::to_repr, next_available>;
};

template <typename S, PipelinePassDef P>
struct Step : StepImpl<S, P, typename P::spec::kind>
{
};

template <typename S, typename... Passes>
struct Fold
{
    using type = S;
};

template <typename S, PipelinePassDef P, typename... Passes>
struct Fold<S, P, Passes...> : Fold<typename Step<S, P>::type, Passes...>
{
};

template <Repr Start, PipelinePassDef... Passes>
struct PipelineSpec
{
    using start = State<Start, ProductList<>>;
    using final = typename Fold<start, Passes...>::type;

    static constexpr Repr end_repr = final::repr;
};

} // namespace optimizer::pipeline

#endif
