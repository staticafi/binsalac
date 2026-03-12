#ifndef OPTIMIZER_UTILS_REPR_OF_HPP_INCLUDED
#define OPTIMIZER_UTILS_REPR_OF_HPP_INCLUDED

#include <optimizer/pipeline/pass_contracts.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/program_repr.hpp>

#include <memory>
#include <variant>

namespace optimizer::utils
{

template <typename T>
struct repr_of;

template <>
struct repr_of<std::shared_ptr<sala::Program>>
{
    static constexpr pipeline::Repr value = pipeline::Repr::Sala;
};

template <>
struct repr_of<program::ProgramIR>
{
    static constexpr pipeline::Repr value = pipeline::Repr::IR;
};

template <pipeline::Repr R>
struct repr_alt;

template <>
struct repr_alt<pipeline::Repr::Sala>
{
    using type = std::shared_ptr<sala::Program>;
};

template <>
struct repr_alt<pipeline::Repr::IR>
{
    using type = program::ProgramIR_sptr;
};

template <pipeline::Repr R>
using repr_alt_t = typename repr_alt<R>::type;

inline pipeline::Repr current_repr(const program::ProgramRepr& pr)
{
    if (std::holds_alternative<std::shared_ptr<sala::Program>>(pr))
    {
        return pipeline::Repr::Sala;
    }
    return pipeline::Repr::IR;
}

} // namespace optimizer::utils

#endif
