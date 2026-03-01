#ifndef OPTIMIZER_UTILS_REPR_OF_HPP_INCLUDED
#define OPTIMIZER_UTILS_REPR_OF_HPP_INCLUDED
#include <optimizer/passes/traits.hpp>
#include <variant>

namespace optimizer::utils
{

template <class T>
struct repr_of;

template <>
struct repr_of<std::shared_ptr<sala::Program>>
{
    static constexpr passes::Repr value = passes::Repr::Sala;
};
template <>
struct repr_of<program::ProgramIR_sptr>
{
    static constexpr passes::Repr value = passes::Repr::IR;
};

// Map repr tag -> variant alternative type
template <passes::Repr R>
struct repr_alt;
template <>
struct repr_alt<passes::Repr::Sala>
{
    using type = std::shared_ptr<sala::Program>;
};
template <>
struct repr_alt<passes::Repr::IR>
{
    using type = program::ProgramIR_sptr;
};
template <passes::Repr R>
using repr_alt_t = typename repr_alt<R>::type;

// Get current repr tag from a ProgramRepr
inline passes::Repr current_repr(const program::ProgramRepr& pr)
{
    return std::visit([](auto const& alt) { return repr_of<std::decay_t<decltype(alt)>>::value; },
                      pr);
}

#endif
} // namespace optimizer::pipeline
