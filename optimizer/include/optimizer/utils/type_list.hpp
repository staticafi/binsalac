#ifndef OPTIMIZER_UTILS_TYPELIST_HPP_INCLUDED
#define OPTIMIZER_UTILS_TYPELIST_HPP_INCLUDED

#include <type_traits>

namespace optimizer::utils
{

template <typename... Ts>
struct TypeList
{
};

// contains<T, List>
template <typename T, typename List>
struct contains;

template <typename T, typename... Ts>
struct contains<T, TypeList<Ts...>> : std::bool_constant<(false || ... || std::is_same_v<T, Ts>)>
{
};
template <typename T, typename L>
inline constexpr bool contains_v = contains<T, L>::value;

// add_if_absent<X, L> -> if X in L, return L; else L + X
template <typename X, typename L>
struct add_if_absent;

template <typename X, typename... Ls>
struct add_if_absent<X, TypeList<Ls...>>
{
    using type =
            std::conditional_t<contains_v<X, TypeList<Ls...>>, TypeList<Ls...>, TypeList<Ls..., X>>;
};

// set_union_unique<TypeList<As...>, TypeList<Bs...>>
template <typename A, typename B>
struct set_union_unique;

template <typename... As, typename... Bs>
struct set_union_unique<TypeList<As...>, TypeList<Bs...>>
{
    template <typename Res, typename... Xs>
    struct fold;

    template <typename Res>
    struct fold<Res>
    {
        using type = Res;
    };

    template <typename Res, typename X, typename... Rest>
    struct fold<Res, X, Rest...>
    {
        using next = typename add_if_absent<X, Res>::type;
        using type = typename fold<next, Rest...>::type;
    };

    using type = typename fold<TypeList<As...>, Bs...>::type;
};
template <typename A, typename B>
using set_union_unique_t = typename set_union_unique<A, B>::type;

template <typename BsList, typename Accum, typename... Xs>
struct set_intersection_filter;

template <typename BsList, typename Accum>
struct set_intersection_filter<BsList, Accum>
{
    using type = Accum;
};

template <typename BsList, typename Accum, typename X, typename... Xs>
struct set_intersection_filter<BsList, Accum, X, Xs...>
{
  private:
    using next_accum =
            std::conditional_t<contains<X, BsList>::value,
                               typename set_union_unique<TypeList<X>, Accum>::type, Accum>;

  public:
    using type = typename set_intersection_filter<BsList, next_accum, Xs...>::type;
};

template <typename A, typename B>
struct set_intersection;

template <typename... As, typename... Bs>
struct set_intersection<TypeList<As...>, TypeList<Bs...>>
{
    using type = typename set_intersection_filter<TypeList<Bs...>, TypeList<>, As...>::type;
};

template <typename A, typename B>
using set_intersection_t = typename set_intersection<A, B>::type;

// set_difference<A, B> = elements in A that are not in B
template <typename A, typename B>
struct set_difference;

template <typename... As, typename... Bs>
struct set_difference<TypeList<As...>, TypeList<Bs...>>
{
    template <typename Accum, typename... Xs>
    struct fold;

    template <typename Accum>
    struct fold<Accum>
    {
        using type = Accum;
    };

    template <typename Accum, typename X, typename... Rest>
    struct fold<Accum, X, Rest...>
    {
        using next = std::conditional_t<contains_v<X, TypeList<Bs...>>, Accum,
                                        typename set_union_unique<Accum, TypeList<X>>::type>;
        using type = typename fold<next, Rest...>::type;
    };

    using type = typename fold<TypeList<>, As...>::type;
};

template <typename A, typename B>
using set_difference_t = typename set_difference<A, B>::type;

// is_subset<Need, Have>
template <typename Need, typename Have>
struct is_subset;

template <typename... Ns, typename... Hs>
struct is_subset<TypeList<Ns...>, TypeList<Hs...>>
    : std::conjunction<contains<Ns, TypeList<Hs...>>...>
{
};

template <typename A, typename B>
using is_subset_t = typename is_subset<A, B>::type;

template <typename A, typename B>
inline constexpr bool is_subset_v = is_subset<A, B>::value;

} // namespace optimizer::utils
#endif
