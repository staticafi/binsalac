#ifndef OTPIMZER_TYPED_PIPELINE_HPP_INCLUDED
#define OTPIMZER_TYPED_PIPELINE_HPP_INCLUDED
#include <optimizer/pipeline/pipeline_i.hpp>

#include <optimizer/pipeline/invokers.hpp>
#include <optimizer/pipeline/spec.hpp>
#include <optimizer/utils/common.hpp>
#include <utility/assumptions.hpp>

#include <utility>

namespace optimizer::pipeline
{
template <passes::Repr Start, class... Passes>
class TypedPipeline final : public PipelineI
{
  public:
    using Spec = PipelineSpec<Start, Passes...>;

    explicit TypedPipeline(program::ProgramRepr program) : program_(std::move(program)) {}

    void run() override
    {
        while (next_idx_ < total_steps())
        {
            run_step();
        }
    }

    void run_step() override
    {
        if (next_idx_ >= total_steps())
        {
            return;
        }
        dispatch_one(next_idx_++);
    }

    [[nodiscard]] const program::ProgramRepr& program() const override { return program_; }

    program::ProgramRepr& program() { return program_; }

  private:
    using _force_pipeline_spec_instantiation_t = typename Spec::final;

    static constexpr std::size_t total_steps() { return utils::pack_size_v<Passes...>; }
    void                         dispatch_one(std::size_t i) { dispatch_at<0>(i); }

    template <std::size_t I>
    void dispatch_at(std::size_t i)
    {
        if constexpr (I < total_steps())
        {
            if (i == I)
            {
                using P = utils::nth_type_t<I, Passes...>;
                run_one_pass<P>();
                return;
            }
            dispatch_at<I + 1>(i);
        }
    }

    template <typename P>
        requires passes::HasPassTraits<P>
    void run_one_pass()
    {
        using Traits = passes::PassTraits<P>;

        if constexpr (std::is_same_v<typename Traits::kind, passes::TranslationPass>)
        {
            const auto cur = utils::current_repr(program_);
            ASSUMPTION(cur == Traits::from_repr);
            translate_on<Traits::from_repr, Traits::to_repr, P>::apply(program_);
        }
        else
        {
            run_on<Traits::required_repr, P>::apply(program_);
        }
    }

  private:
    program::ProgramRepr program_; // Single source of truth
    std::size_t          next_idx_{0};
};
} // namespace optimizer::pipeline
#endif
