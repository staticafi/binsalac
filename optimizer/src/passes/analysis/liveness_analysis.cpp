#include <optimizer/passes/analysis/liveness_analysis.hpp>

#include <optimizer/metadata/liveness.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/utils/liveness/import.hpp>
#include <optimizer/utils/view/flattened_cfg_view.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <cstddef>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace optimizer::passes
{
namespace
{
// Free functions

void add_block_uses_before_defs(const utils::LiveSet& uses, utils::LiveSet& block_use,
                                const utils::LiveSet& block_def)
{
    for (const auto& use : uses)
    {
        if (!block_def.contains(use))
        {
            block_use.insert(use);
        }
    }
}

void add_block_defs(const utils::LiveSet& defs, utils::LiveSet& block_def)
{
    for (const auto& def : defs)
    {
        block_def.insert(def);
    }
}

std::size_t pop_worklist(std::queue<std::size_t>& worklist, std::vector<bool>& in_worklist)
{
    const auto block = worklist.front();
    worklist.pop();

    in_worklist[block] = false;

    return block;
}

void update_removable_instruction_set(const program::InstructionIR_sptr&  instruction,
                                      utils::LiveSet&                     live,
                                      metadata::liveness::BasicBlockMeta& bb_meta)
{
    ASSUMPTION(instruction != nullptr);

    if (utils::is_instruction_removable(instruction, live))
    {
        bb_meta.removable_instructions.insert(instruction.get());
    }

    utils::LiveSet live_before;
    utils::apply_backward_transfer(instruction, live, live_before);
    live = std::move(live_before);
}

// Class definitions

class FunctionContext
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

        for (std::size_t block = 0; block < cfg_.size(); ++block)
        {
            init_block_local_sets(block);
        }
    }

    void init_block_local_sets(std::size_t block)
    {
        auto& block_use = block_use_[block];
        auto& block_def = block_def_[block];

        utils::LiveSet uses;
        utils::LiveSet defs;

        for (const auto& instruction : cfg_.block(block)->get_instructions())
        {
            utils::collect_uses_and_defs(instruction, uses, defs);
            add_block_uses_before_defs(uses, block_use, block_def);
            add_block_defs(defs, block_def);
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

        enqueue_all_blocks(worklist, in_worklist);

        while (!worklist.empty())
        {
            process_next_block(worklist, in_worklist);
        }
    }

    void enqueue_all_blocks(std::queue<std::size_t>& worklist, std::vector<bool>& in_worklist) const
    {
        for (std::size_t block = 0; block < cfg_.size(); ++block)
        {
            worklist.push(block);
            in_worklist[block] = true;
        }
    }

    void process_next_block(std::queue<std::size_t>& worklist, std::vector<bool>& in_worklist)
    {
        const auto block = pop_worklist(worklist, in_worklist);

        auto new_out = compute_live_out(block);
        auto new_in  = compute_live_in(block, new_out);

        update_state_if_changed(block, std::move(new_in), std::move(new_out), worklist,
                                in_worklist);
    }

    utils::LiveSet compute_live_out(std::size_t block) const
    {
        utils::LiveSet live_out;

        for (const auto succ : cfg_.successors(block))
        {
            add_successor_live_in(succ, live_out);
        }

        return live_out;
    }

    void add_successor_live_in(std::size_t succ, utils::LiveSet& live_out) const
    {
        for (const auto& operand : live_in_[succ])
        {
            live_out.insert(operand);
        }
    }

    utils::LiveSet compute_live_in(std::size_t block, const utils::LiveSet& live_out) const
    {
        auto live_in = live_out;

        remove_block_defs(block, live_in);
        add_block_uses(block, live_in);

        return live_in;
    }

    void remove_block_defs(std::size_t block, utils::LiveSet& live_in) const
    {
        for (const auto& def : block_def_[block])
        {
            live_in.erase(def);
        }
    }

    void add_block_uses(std::size_t block, utils::LiveSet& live_in) const
    {
        for (const auto& use : block_use_[block])
        {
            live_in.insert(use);
        }
    }

    void update_state_if_changed(std::size_t block, utils::LiveSet new_in, utils::LiveSet new_out,
                                 std::queue<std::size_t>& worklist, std::vector<bool>& in_worklist)
    {
        if (new_out == live_out_[block] && new_in == live_in_[block])
        {
            return;
        }

        live_out_[block] = std::move(new_out);
        live_in_[block]  = std::move(new_in);

        enqueue_predecessors(block, worklist, in_worklist);
    }

    void enqueue_predecessors(std::size_t block, std::queue<std::size_t>& worklist,
                              std::vector<bool>& in_worklist) const
    {
        for (const auto pred : cfg_.predecessors(block))
        {
            if (in_worklist[pred])
            {
                continue;
            }

            worklist.push(pred);
            in_worklist[pred] = true;
        }
    }

    void materialize()
    {
        for (std::size_t block = 0; block < cfg_.size(); ++block)
        {
            materialize_block(block);
        }
    }

    void materialize_block(std::size_t block)
    {
        auto bb_meta      = std::make_unique<metadata::liveness::BasicBlockMeta>();
        bb_meta->live_out = live_out_[block];
        bb_meta->removable_instructions.clear();

        compute_removable_instructions(block, *bb_meta);

        cfg_.block(block)->get_metadata().set(std::move(bb_meta));
    }

    void compute_removable_instructions(std::size_t                         block,
                                        metadata::liveness::BasicBlockMeta& bb_meta) const
    {
        utils::LiveSet live = bb_meta.live_out;

        const auto& instructions = cfg_.block(block)->get_instructions();

        for (auto it = instructions.rbegin(); it != instructions.rend(); ++it)
        {
            update_removable_instruction_set(*it, live, bb_meta);
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
