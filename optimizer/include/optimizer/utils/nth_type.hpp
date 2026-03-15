#ifndef OPTIMIZER_UTILS_NTH_TYPE_HPP_INCLUDED
#define OPTIMIZER_UTILS_NTH_TYPE_HPP_INCLUDED

#include <cstddef>

namespace optimizer::utils
{
// pack size
template <class... Ts>
inline constexpr std::size_t pack_size_v = sizeof...(Ts);

// nth type from a parameter pack
template <std::size_t I, class... Ts>
struct nth_type;
template <class T0, class... Ts>
struct nth_type<0, T0, Ts...>
{
    using type = T0;
};
template <std::size_t I, class T0, class... Ts>
struct nth_type<I, T0, Ts...>
{
    using type = typename nth_type<I - 1, Ts...>::type;
};
template <std::size_t I, class... Ts>
using nth_type_t = typename nth_type<I, Ts...>::type;

} // namespace optimizer::utils

#endif
