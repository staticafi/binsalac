#ifndef OPTIMIZER_PIPELINE_SPEC_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_SPEC_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>
#include <optimizer/utils/type_list.hpp>

namespace optimizer::pipeline
{

template <passes::Repr R, typename Pass>
struct State
{
    static constexpr passes::Repr repr = R;
    using avail                        = Pass;
};

template <typename S, typename P, typename Kind>
struct step_impl;

// Analysis: repr check + needs, is subset of avail, then add provides
template <passes::Repr R, typename Av, typename P>
struct step_impl<State<R, Av>, P, passes::AnalysisPass>
{
    static_assert(passes::HasPassTraits<P>, "PassTraits::kind not specified");
    constexpr static passes::Repr req = passes::PassTraits<P>::required_repr;
    static_assert(req == R, "Analysis scheduled for incompatible representation");
    using need = typename passes::PassTraits<P>::needs;
    static_assert(utils::is_subset_v<need, Av>, "Pass requires analyses not available");
    using out_avail = utils::set_union_unique_t<Av, typename passes::PassTraits<P>::provides>;
    using type      = State<R, out_avail>;
};

// Transform: repr check + needs, keep preserves, add provides
template <passes::Repr R, typename Av, typename P>
struct step_impl<State<R, Av>, P, passes::TransformPass>
{
    static_assert(passes::HasPassTraits<P>, "PassTraits::kind not specified");
    constexpr static passes::Repr req = passes::PassTraits<P>::required_repr;
    static_assert(req == R, "Transform scheduled for incompatible representation");
    using need = typename passes::PassTraits<P>::needs;
    static_assert(utils::is_subset_v<need, Av>, "Transform requires analyses not available");
    using kept      = utils::set_intersection_t<Av, typename passes::PassTraits<P>::preserves>;
    using out_avail = utils::set_union_unique_t<kept, typename passes::PassTraits<P>::provides>;
    using type      = State<R, out_avail>;
};

// Debug: repr check + needs, is subset of avail, then add provides
template <passes::Repr R, typename Av, typename P>
struct step_impl<State<R, Av>, P, passes::DebugPass>
{
    static_assert(passes::HasPassTraits<P>, "PassTraits::kind not specified");
    constexpr static passes::Repr req = passes::PassTraits<P>::required_repr;
    static_assert(req == R, "Analysis scheduled for incompatible representation");
    using need = typename passes::PassTraits<P>::needs;
    static_assert(utils::is_subset_v<need, Av>, "Pass requires analyses not available");
    using out_avail = utils::set_union_unique_t<Av, typename passes::PassTraits<P>::provides>;
    using type      = State<R, out_avail>;
};

// Translation: must match from_repr, switch to to_repr, keep preserves, add provides
template <passes::Repr R, typename Av, typename P>
struct step_impl<State<R, Av>, P, passes::TranslationPass>
{
    static_assert(passes::HasPassTraits<P>, "PassTraits::kind not specified");
    static_assert(R == passes::PassTraits<P>::from_repr,
                  "Translation scheduled for wrong source representation");
    using kept      = utils::set_intersection_t<Av, typename passes::PassTraits<P>::preserves>;
    using out_avail = utils::set_union_unique_t<kept, typename passes::PassTraits<P>::provides>;
    using type      = State<passes::PassTraits<P>::to_repr, out_avail>;
};

template <typename S, typename P>
struct step : step_impl<S, P, typename passes::PassTraits<P>::kind>
{
};

template <typename S, typename... Ps>
struct fold
{
    using type = S;
};

template <typename S, typename P, typename... Ps>
struct fold<S, P, Ps...> : fold<typename step<S, P>::type, Ps...>
{
};

template <passes::Repr Start, typename... Passes>
struct PipelineSpec
{
    using start                            = State<Start, utils::TypeList<>>;
    using final                            = typename fold<start, Passes...>::type;
    static constexpr passes::Repr end_repr = final::repr;
};

} // namespace optimizer::pipeline
#endif
