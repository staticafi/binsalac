#ifndef OPTIMIZER_TYPED_PIPELINE_REWORKED_HPP_INCLUDED
#define OPTIMIZER_TYPED_PIPELINE_REWORKED_HPP_INCLUDED

#include <optimizer/pipeline/invokers.hpp>
#include <optimizer/pipeline/pipeline_i.hpp>
#include <optimizer/pipeline/spec.hpp>
#include <optimizer/utils/common.hpp>
#include <utility/assumptions.hpp>

namespace optimizer::pipeline
{

template <Repr Start, PipelinePassDef... Passes>
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
    [[nodiscard]] program::ProgramRepr&       program() { return program_; }

  private:
    using _force_pipeline_spec_instantiation_t = typename Spec::final;

    static constexpr std::size_t total_steps() { return utils::pack_size_v<Passes...>; }

    void dispatch_one(std::size_t index) { dispatch_at<0>(index); }

    template <std::size_t I>
    void dispatch_at(std::size_t index)
    {
        if constexpr (I < total_steps())
        {
            if (index == I)
            {
                using Def = utils::nth_type_t<I, Passes...>;
                run_one<typename Def::impl, typename Def::spec>();
                return;
            }
            dispatch_at<I + 1>(index);
        }
    }

    template <typename Impl, typename SpecT>
    void run_one()
    {
        if constexpr (std::is_same_v<typename SpecT::kind, TranslationPass>)
        {
            const auto cur = utils::current_repr(program_);
            ASSUMPTION(cur == SpecT::from_repr);
            translate_on<SpecT::from_repr, SpecT::to_repr, Impl>::apply(program_);
        }
        else
        {
            const auto cur = utils::current_repr(program_);
            ASSUMPTION(cur == SpecT::required_repr);
            run_on<SpecT::required_repr, Impl>::apply(program_);
        }
    }

  private:
    program::ProgramRepr program_;
    std::size_t          next_idx_{0};
};

} // namespace optimizer::pipeline

#endif
