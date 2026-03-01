#ifndef OPTIMIZER_PIPELINE_INVOKERS_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_INVOKERS_HPP_INCLUDED

#include <optimizer/passes/traits.hpp>

#include <optimizer/utils/repr_of.hpp>

namespace optimizer::pipeline
{

template <passes::Repr R, class Pass>
struct run_on
{
    static void apply(program::ProgramRepr& pr)
    {
        using Alt      = utils::repr_alt_t<R>;
        const Alt& alt = std::get<Alt>(pr);
        Pass       pass{};
        pass.run(alt);
    }
};

template <passes::Repr From, passes::Repr To, class Pass>
struct translate_on
{
    static void apply(program::ProgramRepr& pr)
    {
        using Src      = utils::repr_alt_t<From>;
        using Dst      = utils::repr_alt_t<To>;
        const Src& src = std::get<Src>(pr);
        Pass       pass{};
        auto       dst = pass.run(src);
        static_assert(std::is_convertible_v<decltype(dst), Dst>,
                      "Translation must return destination Alt (convertible to Dst).");
        pr = Dst(std::move(dst));
    }
};
} // namespace optimizer::pipeline
#endif
