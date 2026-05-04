#include <optimizer/passes/analysis/liveness_analysis.hpp>

#include <optimizer/metadata/liveness.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/utils/liveness/import.hpp>
#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/view/flattened_cfg_view.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <memory>
#include <queue>
#include <vector>

namespace optimizer::passes
{
namespace
{
struct FunctionContext
{
  public:
    explicit FunctionContext(program::FunctionIR_sptr function) : cfg_{std::move(function)} {}

    void run()
    {
        init_data();
        solve();
        materialize();
    }

  private:
    void init_data()
    {
        init_block_local_sets();
        init_states();
    }

    void init_block_local_sets()
    {
        block_use_.assign(cfg_.size(), {});
        block_def_.assign(cfg_.size(), {});

        for (std::size_t b = 0; b < cfg_.size(); ++b)
        {
            auto& block_use = block_use_[b];
            auto& block_def = block_def_[b];

            utils::LiveSet uses;
            utils::LiveSet defs;

            for (const auto& instruction : cfg_.block(b)->get_instructions())
            {
                utils::collect_uses_and_defs(instruction, uses, defs);

                for (const auto& use : uses)
                {
                    if (!block_def.contains(use))
                    {
                        block_use.insert(use);
                    }
                }

                for (const auto& def : defs)
                {
                    block_def.insert(def);
                }
            }
        }
    }

    void init_states()
    {
        live_in_.assign(cfg_.size(), {});
        live_out_.assign(cfg_.size(), {});
    }

    void solve()
    {
        if (cfg_.empty())
        {
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<bool>       in_worklist(cfg_.size(), false);

        for (std::size_t b = 0; b < cfg_.size(); ++b)
        {
            worklist.push(b);
            in_worklist[b] = true;
        }

        while (!worklist.empty())
        {
            const auto b = worklist.front();
            worklist.pop();
            in_worklist[b] = false;

            utils::LiveSet new_out;
            for (const auto succ : cfg_.successors(b))
            {
                for (const auto& operand : live_in_[succ])
                {
                    new_out.insert(operand);
                }
            }

            utils::LiveSet new_in = new_out;
            for (const auto& def : block_def_[b])
            {
                new_in.erase(def);
            }
            for (const auto& use : block_use_[b])
            {
                new_in.insert(use);
            }

            if (!(new_out == live_out_[b]) || !(new_in == live_in_[b]))
            {
                live_out_[b] = std::move(new_out);
                live_in_[b]  = std::move(new_in);

                for (const auto pred : cfg_.predecessors(b))
                {
                    if (!in_worklist[pred])
                    {
                        worklist.push(pred);
                        in_worklist[pred] = true;
                    }
                }
            }
        }
    }

    void materialize()
    {
        for (std::size_t b = 0; b < cfg_.size(); ++b)
        {
            auto bb_meta      = std::make_unique<metadata::liveness::BasicBlockMeta>();
            bb_meta->live_out = live_out_[b];
            bb_meta->removable_instructions.clear();

            utils::LiveSet live = bb_meta->live_out;

            const auto& instructions = cfg_.block(b)->get_instructions();
            for (auto it = instructions.rbegin(); it != instructions.rend(); ++it)
            {
                ASSUMPTION(*it != nullptr);

                if (utils::is_instruction_removable(*it, live))
                {
                    bb_meta->removable_instructions.insert(it->get());
                }

                utils::LiveSet live_before;
                utils::apply_backward_transfer(*it, live, live_before);
                live = std::move(live_before);
            }

            cfg_.block(b)->get_metadata().set(std::move(bb_meta));
        }
    }

  private:
    utils::view::FlattenedCFGView cfg_;

    std::vector<utils::LiveSet> block_use_;
    std::vector<utils::LiveSet> block_def_;
    std::vector<utils::LiveSet> live_in_;
    std::vector<utils::LiveSet> live_out_;
};

class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        run();
    }

    program::ProgramIR_sptr result() const { return sala_ir_; }

  private:
    void run()
    {
        for (const auto& function : sala_ir_->get_functions())
        {
            ASSUMPTION(function != nullptr);

            FunctionContext(function).run();
        }
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void LivenessAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    TMPROF_BLOCK()
    const auto trigger = Impl(std::move(sala_ir));
}

} // namespace optimizer::passes
