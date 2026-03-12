#ifndef OPTIMIZER_PIPELINE_INVOKERS_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_INVOKERS_HPP_INCLUDED

#include <optimizer/pipeline/pass_contracts.hpp>
#include <optimizer/utils/repr_of.hpp>

#include <type_traits>
#include <utility>

namespace optimizer::pipeline
{

template <Repr R, class PassImpl>
struct run_on
{
    static void apply(program::ProgramRepr& pr)
    {
        using Alt = utils::repr_alt_t<R>;

        Alt&     alt = std::get<Alt>(pr);
        PassImpl pass{};

        if constexpr (std::is_void_v<decltype(pass.run(alt))>)
        {
            pass.run(alt);
        }
        else
        {
            auto out = pass.run(alt);
            static_assert(
                    std::is_convertible_v<decltype(out), Alt>,
                    "Non-translation pass must return void or the same representation alternative");
            alt = std::move(out);
        }
    }
};

template <Repr From, Repr To, class PassImpl>
struct translate_on
{
    static void apply(program::ProgramRepr& pr)
    {
        using Src = utils::repr_alt_t<From>;
        using Dst = utils::repr_alt_t<To>;

        Src&     src = std::get<Src>(pr);
        PassImpl pass{};

        auto out = pass.run(src);
        static_assert(std::is_convertible_v<decltype(out), Dst>,
                      "Translation pass must return the destination representation alternative");

        pr = Dst(std::move(out));
    }
};

} // namespace optimizer::pipeline

#endif
