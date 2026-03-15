#include <optimizer/passes/analysis/available_copy_analysis.hpp>

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/available_copy/import.hpp>
#include <optimizer/utils/dynamic_bitset.hpp>
#include <optimizer/utils/sparse_map.hpp>

#include <utility/assumptions.hpp>

#include <cstddef>
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
    explicit FunctionContext(program::FunctionIR_sptr function) : function_{std::move(function)} {}

    void run()
    {
        init_data();
        solve();
        materialize();
    }

  private:
    [[nodiscard]] utils::TransferContext make_transfer_context() const
    {
        return utils::TransferContext{
                .variable_ids    = variable_ids_,
                .facts           = facts_,
                .facts_by_dest   = facts_by_dest_,
                .facts_by_source = facts_by_source_,
                .fact_id_of      = fact_id_of_,
                .kill_masks      = kill_masks_,
        };
    }

    void init_data()
    {
        collect_variables();
        build_flattened_cfg();
        collect_facts();
        init_states();
    }

    void collect_variables()
    {
        variable_ids_.clear();
        NV_ = 0;

        auto add_variable = [&](const program::VariableIR_sptr& variable)
        {
            ASSUMPTION(variable != nullptr);

            const auto [it, inserted] = variable_ids_.emplace(variable.get(), NV_++);
            if (!inserted)
            {
                return;
            }

            auto var_meta = std::make_unique<metadata::available_copy::VariableMeta>();
            var_meta->id  = it->second;
            variable->get_metadata().set(std::move(var_meta));
        };

        auto program = function_->get_program();
        ASSUMPTION(program != nullptr);

        for (const auto& static_var : program->get_static_vars())
        {
            add_variable(static_var);
        }
        for (const auto& parameter : function_->get_parameters())
        {
            add_variable(parameter);
        }
        for (const auto& local : function_->get_local_variables())
        {
            add_variable(local);
        }
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
        }

        entry_ = 0;
        if (const auto entry_bb = function_->get_entry_basic_block(); entry_bb != nullptr)
        {
            const auto it = index_of_.find(entry_bb.get());
            ASSUMPTION(it != index_of_.end());
            entry_ = it->second;
        }
    }

    void collect_facts()
    {
        facts_.clear();
        fact_id_of_.clear();
        facts_by_dest_.assign(NV_, {});
        facts_by_source_.assign(NV_, {});
        kill_masks_.assign(NV_, utils::DynamicBitset{});

        for (const auto& block : blocks_)
        {
            for (const auto& instruction : block->get_instructions())
            {
                if (instruction->get_opcode() != sala::Instruction::Opcode::COPY)
                {
                    continue;
                }

                if (instruction->get_operands().size() < 2U)
                {
                    continue;
                }

                const auto* dest =
                        std::get_if<program::VariableIR_raw>(&instruction->get_operands()[0]);
                const auto* src =
                        std::get_if<program::VariableIR_raw>(&instruction->get_operands()[1]);

                if (dest == nullptr || src == nullptr || *dest == nullptr || *src == nullptr)
                {
                    continue;
                }

                if ((*dest)->get_num_bytes() != (*src)->get_num_bytes())
                {
                    continue;
                }

                const auto dest_id = variable_ids_.at(*dest);
                const auto src_id  = variable_ids_.at(*src);
                if (dest_id == src_id)
                {
                    continue;
                }

                const utils::CopyFactKey key{.dest_id = dest_id, .source_id = src_id};
                if (fact_id_of_.find(key) != fact_id_of_.end())
                {
                    continue;
                }

                const auto fact_id = facts_.size();
                fact_id_of_.emplace(key, fact_id);
                facts_.push_back(utils::CopyFact{
                        .id        = fact_id,
                        .dest_id   = dest_id,
                        .source_id = src_id,
                        .dest      = *dest,
                        .source    = *src,
                });
            }
        }

        for (auto& kill_mask : kill_masks_)
        {
            kill_mask.resize(facts_.size());
        }

        for (const auto& fact : facts_)
        {
            facts_by_dest_[fact.dest_id].push_back(fact.id);
            facts_by_source_[fact.source_id].push_back(fact.id);
            kill_masks_[fact.dest_id].set(fact.id);
            kill_masks_[fact.source_id].set(fact.id);
        }
    }

    void init_states()
    {
        out_states_.assign(NB_, utils::DynamicBitset{facts_.size()});
        in_state_scratch_.resize(facts_.size());
        has_out_.assign(NB_, false);
    }

    void build_in_state(const std::size_t b)
    {
        in_state_scratch_.reset();

        bool has_ready_pred = false;
        if (b == entry_)
        {
            has_ready_pred = true;
        }
        else
        {
            for (const auto pred : preds_[b])
            {
                if (!has_out_[pred])
                {
                    continue;
                }

                if (!has_ready_pred)
                {
                    in_state_scratch_ = out_states_[pred];
                    has_ready_pred    = true;
                }
                else
                {
                    in_state_scratch_.and_with(out_states_[pred]);
                }
            }
        }

        if (!has_ready_pred)
        {
            in_state_scratch_.reset();
        }
    }

    void apply_transfer(const program::InstructionIR_sptr& instruction,
                        utils::DynamicBitset&              state) const
    {
        utils::apply_transfer(instruction, make_transfer_context(), state);
    }

    void solve()
    {
        if (NB_ == 0)
        {
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<char>       in_worklist(NB_, 0);

        worklist.push(entry_);
        in_worklist[entry_] = 1;

        while (!worklist.empty())
        {
            const auto b = worklist.front();
            worklist.pop();
            in_worklist[b] = 0;

            build_in_state(b);

            bool has_ready_input = (b == entry_);
            if (!has_ready_input)
            {
                for (const auto pred : preds_[b])
                {
                    if (has_out_[pred])
                    {
                        has_ready_input = true;
                        break;
                    }
                }
            }

            if (!has_ready_input)
            {
                continue;
            }

            auto state = in_state_scratch_;
            for (const auto& instruction : blocks_[b]->get_instructions())
            {
                apply_transfer(instruction, state);
            }

            if (!has_out_[b] || out_states_[b] != state)
            {
                out_states_[b] = state;
                has_out_[b]    = true;

                for (const auto& weak_succ : blocks_[b]->get_successors())
                {
                    const auto succ = weak_succ.lock();
                    ASSUMPTION(succ != nullptr);

                    const auto it = index_of_.find(succ.get());
                    ASSUMPTION(it != index_of_.end());

                    if (!in_worklist[it->second])
                    {
                        worklist.push(it->second);
                        in_worklist[it->second] = 1;
                    }
                }
            }
        }
    }

    void materialize()
    {
        auto function_meta             = std::make_unique<metadata::available_copy::FunctionMeta>();
        function_meta->facts           = facts_;
        function_meta->facts_by_dest   = facts_by_dest_;
        function_meta->facts_by_source = facts_by_source_;
        function_->get_metadata().set(std::move(function_meta));

        for (std::size_t b = 0; b < NB_; ++b)
        {
            utils::DynamicBitset in_state{facts_.size()};
            bool                 has_ready_pred = false;

            if (b == entry_)
            {
                in_state.reset();
                has_ready_pred = true;
            }
            else
            {
                for (const auto pred : preds_[b])
                {
                    if (!has_out_[pred])
                    {
                        continue;
                    }

                    if (!has_ready_pred)
                    {
                        in_state       = out_states_[pred];
                        has_ready_pred = true;
                    }
                    else
                    {
                        in_state.and_with(out_states_[pred]);
                    }
                }
            }

            auto block_meta     = std::make_unique<metadata::available_copy::BasicBlockMeta>();
            block_meta->in_bits = in_state.words();
            blocks_[b]->get_metadata().set(std::move(block_meta));
        }
    }

  private:
    program::FunctionIR_sptr function_;

    std::size_t NB_{0};
    std::size_t NV_{0};
    std::size_t entry_{0};

    std::vector<program::BasicBlockIR_sptr>                  blocks_;
    std::vector<std::vector<std::size_t>>                    preds_;
    utils::SparseMap<program::BasicBlockIR_raw, std::size_t> index_of_;

    utils::SparseMap<program::VariableIR_raw, std::size_t> variable_ids_;

    std::vector<utils::CopyFact>                      facts_;
    utils::SparseMap<utils::CopyFactKey, std::size_t> fact_id_of_;
    std::vector<std::vector<std::size_t>>             facts_by_dest_;
    std::vector<std::vector<std::size_t>>             facts_by_source_;
    std::vector<utils::DynamicBitset>                 kill_masks_;

    std::vector<utils::DynamicBitset> out_states_;
    utils::DynamicBitset              in_state_scratch_;
    std::vector<bool>                 has_out_;
};

} // namespace

struct AvailableCopyAnalysis::Impl
{
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)} { run(); }

  private:
    void run()
    {
        for (const auto& function : sala_ir_->get_functions())
        {
            if (function == nullptr)
            {
                continue;
            }

            FunctionContext(function).run();
        }
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

AvailableCopyAnalysis::AvailableCopyAnalysis()  = default;
AvailableCopyAnalysis::~AvailableCopyAnalysis() = default;

program::ProgramIR_sptr AvailableCopyAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    pImpl_ = std::make_unique<Impl>(sala_ir);
    return sala_ir;
}

} // namespace optimizer::passes
