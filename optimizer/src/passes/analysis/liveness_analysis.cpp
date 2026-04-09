#include <optimizer/passes/analysis/liveness_analysis.hpp>

#include <optimizer/metadata/liveness.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/utils/liveness/import.hpp>
#include <optimizer/utils/sparse_map.hpp>

#include <utility/assumptions.hpp>

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
    explicit FunctionContext(program::FunctionIR_sptr function) : function_{std::move(function)}
    {
        ASSUMPTION(function_ != nullptr);
    }

    void run()
    {
        init_data();
        solve();
        materialize();
    }

  private:
    void init_data()
    {
        build_flattened_cfg();
        init_block_local_sets();
        init_states();
    }

    void build_flattened_cfg()
    {
        NB_ = function_->get_basic_blocks().size();

        blocks_.clear();
        blocks_.reserve(NB_);

        for (const auto& bb : function_->get_basic_blocks())
        {
            blocks_.push_back(bb);
        }

        index_of_.clear();
        index_of_.reserve(NB_);

        for (std::size_t i = 0; i < NB_; ++i)
        {
            index_of_.emplace(blocks_[i].get(), i);
        }

        preds_.assign(NB_, {});
        succs_.assign(NB_, {});

        for (std::size_t i = 0; i < NB_; ++i)
        {
            for (const auto& wp : blocks_[i]->get_predecessors())
            {
                auto sp = wp.lock();
                ASSUMPTION(sp != nullptr);

                const auto it = index_of_.find(sp.get());
                ASSUMPTION(it != index_of_.end());

                preds_[i].push_back(it->second);
            }

            for (const auto& ws : blocks_[i]->get_successors())
            {
                auto ss = ws.lock();
                ASSUMPTION(ss != nullptr);

                const auto it = index_of_.find(ss.get());
                ASSUMPTION(it != index_of_.end());

                succs_[i].push_back(it->second);
            }
        }
    }

    void init_block_local_sets()
    {
        block_use_.assign(NB_, {});
        block_def_.assign(NB_, {});

        for (std::size_t b = 0; b < NB_; ++b)
        {
            auto& block_use = block_use_[b];
            auto& block_def = block_def_[b];

            utils::LiveSet uses;
            utils::LiveSet defs;

            for (const auto& instruction : blocks_[b]->get_instructions())
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
        live_in_.assign(NB_, {});
        live_out_.assign(NB_, {});
    }

    void solve()
    {
        if (NB_ == 0)
        {
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<bool>       in_worklist(NB_, false);

        for (std::size_t b = 0; b < NB_; ++b)
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
            for (const auto succ : succs_[b])
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

                for (const auto pred : preds_[b])
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
        for (std::size_t b = 0; b < NB_; ++b)
        {
            auto bb_meta      = std::make_unique<metadata::liveness::BasicBlockMeta>();
            bb_meta->live_out = live_out_[b];
            bb_meta->removable_instructions.clear();

            utils::LiveSet live = bb_meta->live_out;

            const auto& instructions = blocks_[b]->get_instructions();
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

            blocks_[b]->get_metadata().set(std::move(bb_meta));
        }
    }

  private:
    program::FunctionIR_sptr function_;

    std::size_t NB_{0};

    std::vector<program::BasicBlockIR_sptr>                             blocks_;
    std::vector<std::vector<std::size_t>>                               preds_;
    std::vector<std::vector<std::size_t>>                               succs_;
    optimizer::utils::SparseMap<program::BasicBlockIR_raw, std::size_t> index_of_;

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
    std::cout << "LiA: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir));
    std::cout << "LiA: done" << std::endl;
}

} // namespace optimizer::passes
