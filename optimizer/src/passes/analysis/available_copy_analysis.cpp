#include <iostream>
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
    using ReachabilityMatrix = std::vector<std::vector<bool>>;

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

        const auto program = function_->get_program();
        ASSUMPTION(program != nullptr);

        collect_program_static_variables(*program);
        collect_function_parameters();
        collect_function_locals();
    }

    void collect_program_static_variables(const program::ProgramIR& program)
    {
        for (const auto& static_var : program.get_static_vars())
        {
            add_variable(static_var);
        }
    }

    void collect_function_parameters()
    {
        for (const auto& parameter : function_->get_parameters())
        {
            add_variable(parameter);
        }
    }

    void collect_function_locals()
    {
        for (const auto& local : function_->get_local_variables())
        {
            add_variable(local);
        }
    }

    void add_variable(const program::VariableIR_sptr& variable)
    {
        ASSUMPTION(variable != nullptr);

        const auto it = variable_ids_.emplace_hint(variable_ids_.end(), variable.get(), NV_++);

        auto var_meta = std::make_unique<metadata::available_copy::VariableMeta>();
        var_meta->id  = it->second;
        variable->get_metadata().set(std::move(var_meta));
    }

    void build_flattened_cfg()
    {
        NB_ = function_->get_basic_blocks().size();

        collect_blocks();
        build_block_index();
        build_predecessor_lists();
        detect_entry_block();
    }

    void collect_blocks()
    {
        blocks_.clear();
        blocks_.reserve(NB_);

        for (const auto& bb : function_->get_basic_blocks())
        {
            blocks_.push_back(bb);
        }
    }

    void build_block_index()
    {
        index_of_.clear();
        index_of_.reserve(NB_);

        for (std::size_t i = 0; i < NB_; ++i)
        {
            index_of_.emplace(blocks_[i].get(), i);
        }
    }

    void build_predecessor_lists()
    {
        preds_.assign(NB_, {});

        for (std::size_t i = 0; i < NB_; ++i)
        {
            for (const auto& wp : blocks_[i]->get_predecessors())
            {
                const auto sp = wp.lock();
                ASSUMPTION(sp != nullptr);

                const auto it = index_of_.find(sp.get());
                ASSUMPTION(it != index_of_.end());

                preds_[i].push_back(it->second);
            }
        }
    }

    void detect_entry_block()
    {
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
        init_fact_containers();
        build_variables_by_id();

        auto reach = build_explicit_copy_reachability();
        compute_transitive_closure(reach);
        materialize_facts_from_reachability(reach);
        build_fact_indexes_and_kill_masks();
    }

    void init_fact_containers()
    {
        facts_.clear();
        fact_id_of_.clear();
        facts_by_dest_.assign(NV_, {});
        facts_by_source_.assign(NV_, {});
        kill_masks_.assign(NV_, utils::DynamicBitset{});

        variables_by_id_.reserve(NV_);
    }

    void build_variables_by_id()
    {
        const auto program = function_->get_program();
        ASSUMPTION(program != nullptr);

        register_program_static_variables_by_id(*program);
        register_function_parameters_by_id();
        register_function_locals_by_id();
    }

    void register_program_static_variables_by_id(const program::ProgramIR& program)
    {
        for (const auto& static_var : program.get_static_vars())
        {
            register_variable_by_id(static_var);
        }
    }

    void register_function_parameters_by_id()
    {
        for (const auto& parameter : function_->get_parameters())
        {
            register_variable_by_id(parameter);
        }
    }

    void register_function_locals_by_id()
    {
        for (const auto& local : function_->get_local_variables())
        {
            register_variable_by_id(local);
        }
    }

    void register_variable_by_id(const program::VariableIR_sptr& variable)
    {
        ASSUMPTION(variable != nullptr);

        const auto it = variable_ids_.find(variable.get());
        ASSUMPTION(it != variable_ids_.end());
        ASSUMPTION(it->second < NV_);

        variables_by_id_[it->second] = variable;
    }

    [[nodiscard]] ReachabilityMatrix build_explicit_copy_reachability() const
    {
        ReachabilityMatrix reach(NV_, std::vector<bool>(NV_, false));

        for (const auto& bb : function_->get_basic_blocks())
        {
            for (const auto& instruction : bb->get_instructions())
            {
                add_explicit_copy_edge(instruction, reach);
            }
        }

        return reach;
    }

    void add_explicit_copy_edge(const program::InstructionIR_sptr& instruction,
                                ReachabilityMatrix&                reach) const
    {
        if (instruction == nullptr)
        {
            return;
        }

        if (instruction->get_opcode() != sala::Instruction::Opcode::COPY)
        {
            return;
        }

        const auto& operands = instruction->get_operands();
        if (operands.size() < 2U)
        {
            return;
        }

        const auto* dest = std::get_if<program::VariableIR_raw>(&operands[0]);
        const auto* src  = std::get_if<program::VariableIR_raw>(&operands[1]);

        if (dest == nullptr || src == nullptr || *dest == nullptr || *src == nullptr)
        {
            return;
        }

        const auto dest_it = variable_ids_.find(*dest);
        const auto src_it  = variable_ids_.find(*src);

        if (dest_it == variable_ids_.end() || src_it == variable_ids_.end())
        {
            return;
        }

        const auto dest_id = dest_it->second;
        const auto src_id  = src_it->second;

        if (dest_id == src_id)
        {
            return;
        }

        if ((*dest)->get_num_bytes() != (*src)->get_num_bytes())
        {
            return;
        }

        reach[dest_id][src_id] = true;
    }

    void compute_transitive_closure(ReachabilityMatrix& reach) const
    {
        // If X <- Y and Y <- Z are possible facts, then X <- Z must also belong
        // to the universe so bridge_through_redefined_variable can activate it later.
        for (std::size_t mid = 0; mid < NV_; ++mid)
        {
            for (std::size_t dst = 0; dst < NV_; ++dst)
            {
                if (!reach[dst][mid])
                {
                    continue;
                }

                for (std::size_t src = 0; src < NV_; ++src)
                {
                    if (!reach[mid][src])
                    {
                        continue;
                    }

                    if (dst == src)
                    {
                        continue;
                    }

                    ASSUMPTION(variables_by_id_.at(dst) != nullptr);
                    ASSUMPTION(variables_by_id_.at(src) != nullptr);

                    if (variables_by_id_.at(dst)->get_num_bytes() !=
                        variables_by_id_.at(src)->get_num_bytes())
                    {
                        continue;
                    }

                    reach[dst][src] = true;
                }
            }
        }
    }

    void materialize_facts_from_reachability(const ReachabilityMatrix& reach)
    {
        for (std::size_t dest_id = 0; dest_id < NV_; ++dest_id)
        {
            materialize_facts_for_dest(dest_id, reach);
        }
    }

    void materialize_facts_for_dest(const std::size_t dest_id, const ReachabilityMatrix& reach)
    {
        for (std::size_t src_id = 0; src_id < NV_; ++src_id)
        {
            if (!reach[dest_id][src_id])
            {
                continue;
            }

            ASSUMPTION(dest_id < variables_by_id_.size());
            ASSUMPTION(src_id < variables_by_id_.size());
            ASSUMPTION(variables_by_id_[dest_id] != nullptr);
            ASSUMPTION(variables_by_id_[src_id] != nullptr);

            const auto fact_id = facts_.size();

            facts_.push_back(utils::CopyFact{
                    .id        = fact_id,
                    .dest_id   = dest_id,
                    .source_id = src_id,
                    .dest      = variables_by_id_[dest_id].get(),
                    .source    = variables_by_id_[src_id].get(),
            });

            fact_id_of_.emplace(
                    utils::CopyFactKey{
                            .dest_id   = dest_id,
                            .source_id = src_id,
                    },
                    fact_id);
        }
    }

    void build_fact_indexes_and_kill_masks()
    {
        resize_kill_masks();
        populate_fact_indexes();
        populate_kill_masks();
    }

    void resize_kill_masks()
    {
        for (auto& kill_mask : kill_masks_)
        {
            kill_mask.resize(facts_.size());
        }
    }

    void populate_fact_indexes()
    {
        for (const auto& fact : facts_)
        {
            facts_by_dest_[fact.dest_id].push_back(fact.id);
            facts_by_source_[fact.source_id].push_back(fact.id);
        }
    }

    void populate_kill_masks()
    {
        for (const auto& fact : facts_)
        {
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

    void build_in_state(const std::size_t b) { compute_joined_in_state(b, in_state_scratch_); }

    void compute_joined_in_state(const std::size_t b, utils::DynamicBitset& state) const
    {
        state.reset();

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
                    state          = out_states_[pred];
                    has_ready_pred = true;
                }
                else
                {
                    state.and_with(out_states_[pred]);
                }
            }
        }

        if (!has_ready_pred)
        {
            state.reset();
        }
    }

    [[nodiscard]] bool has_ready_input_for_block(const std::size_t b) const
    {
        if (b == entry_)
        {
            return true;
        }

        for (const auto pred : preds_[b])
        {
            if (has_out_[pred])
            {
                return true;
            }
        }

        return false;
    }

    void apply_transfer(const program::InstructionIR_sptr& instruction,
                        utils::DynamicBitset&              state) const
    {
        utils::apply_transfer(instruction, make_transfer_context(), state);
    }

    void apply_block_transfer(const std::size_t b, utils::DynamicBitset& state) const
    {
        for (const auto& instruction : blocks_[b]->get_instructions())
        {
            apply_transfer(instruction, state);
        }
    }

    void enqueue_successors(const std::size_t b, std::queue<std::size_t>& worklist,
                            std::vector<char>& in_worklist) const
    {
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

            if (!has_ready_input_for_block(b))
            {
                continue;
            }

            auto state = in_state_scratch_;
            apply_block_transfer(b, state);

            if (!has_out_[b] || out_states_[b] != state)
            {
                out_states_[b] = state;
                has_out_[b]    = true;
                enqueue_successors(b, worklist, in_worklist);
            }
        }
    }

    void materialize()
    {
        materialize_basic_block_metadata();
        materialize_function_metadata();
    }

    void materialize_basic_block_metadata()
    {
        for (std::size_t b = 0; b < NB_; ++b)
        {
            auto in_state = build_materialized_in_state(b);

            auto block_meta     = std::make_unique<metadata::available_copy::BasicBlockMeta>();
            block_meta->in_bits = in_state.words();
            blocks_[b]->get_metadata().set(std::move(block_meta));
        }
    }

    [[nodiscard]] utils::DynamicBitset build_materialized_in_state(const std::size_t b) const
    {
        utils::DynamicBitset in_state{facts_.size()};
        compute_joined_in_state(b, in_state);
        return in_state;
    }

    void materialize_function_metadata()
    {
        auto function_meta             = std::make_unique<metadata::available_copy::FunctionMeta>();
        function_meta->facts           = std::move(facts_);
        function_meta->facts_by_dest   = std::move(facts_by_dest_);
        function_meta->facts_by_source = std::move(facts_by_source_);
        function_meta->variables_by_id = std::move(variables_by_id_);
        function_->get_metadata().set(std::move(function_meta));
    }

  private:
    program::FunctionIR_sptr function_;

    std::size_t NB_{0};
    std::size_t NV_{0};
    std::size_t entry_{0};

    std::vector<program::BasicBlockIR_sptr>                  blocks_;
    std::vector<std::vector<std::size_t>>                    preds_;
    utils::SparseMap<program::BasicBlockIR_raw, std::size_t> index_of_;

    utils::SparseMap<std::size_t, program::VariableIR_sptr> variables_by_id_;
    utils::SparseMap<program::VariableIR_raw, std::size_t>  variable_ids_;

    std::vector<utils::CopyFact>                      facts_;
    utils::SparseMap<utils::CopyFactKey, std::size_t> fact_id_of_;
    std::vector<std::vector<std::size_t>>             facts_by_dest_;
    std::vector<std::vector<std::size_t>>             facts_by_source_;
    std::vector<utils::DynamicBitset>                 kill_masks_;

    std::vector<utils::DynamicBitset> out_states_;
    utils::DynamicBitset              in_state_scratch_;
    std::vector<bool>                 has_out_;
};

class Impl
{
  public:
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
} // namespace

void AvailableCopyAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    std::cout << "ACA: started" << std::endl;
    const auto trigger = Impl(sala_ir);
    std::cout << "ACA: done" << std::endl;
}

} // namespace optimizer::passes
