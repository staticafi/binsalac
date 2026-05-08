#include <optimizer/passes/analysis/available_copy_analysis.hpp>

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/query/translation_query.hpp>
#include <optimizer/utils/available_copy/import.hpp>
#include <optimizer/utils/dynamic_bitset.hpp>
#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/view/flattened_cfg_view.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <cstddef>
#include <memory>
#include <queue>
#include <utility>
#include <variant>
#include <vector>

namespace optimizer::passes
{
namespace
{
std::string me()
{
    std::ostringstream oss;
    oss << pipeline::names::available_copy;
    oss << ": ";
    return oss.str();
}

bool is_valid_copy_instruction(const program::InstructionIR_sptr& instruction)
{
    if (instruction == nullptr)
    {
        return false;
    }

    if (instruction->get_opcode() != sala::Instruction::Opcode::COPY)
    {
        return false;
    }

    return instruction->get_operands().size() >= 2U;
}

bool is_valid_copy_operand_pair(const program::VariableIR_raw* dest,
                                const program::VariableIR_raw* src)
{
    return dest != nullptr && src != nullptr && *dest != nullptr && *src != nullptr;
}

bool can_form_copy_fact(program::VariableIR_raw dest, program::VariableIR_raw src,
                        std::size_t dest_id, std::size_t src_id)
{
    if (dest_id == src_id)
    {
        return false;
    }

    return dest->get_num_bytes() == src->get_num_bytes();
}

class FunctionContext
{
  public:
    explicit FunctionContext(program::FunctionIR_sptr function)
        : function_{std::move(function)}, cfg_{function_}
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
    std::string info() const
    {
        auto program = function_->get_program();
        ASSUMPTION(program != nullptr);

        query::TranslationQuery translation{program};

        auto name = translation.function_name(function_);

        if (name.empty())
        {
            name = "<unnamed>";
        }

        std::ostringstream oss;
        oss << "[" << name << "] ";
        return oss.str();
    }

    using ReachabilityMatrix = std::vector<std::vector<bool>>;

    template <typename Fn>
    void for_each_variable(Fn&& fn) const
    {
        const auto program = function_->get_program();
        ASSUMPTION(program != nullptr);

        for (const auto& static_var : program->get_static_vars())
        {
            fn(static_var);
        }

        for (const auto& parameter : function_->get_parameters())
        {
            fn(parameter);
        }

        for (const auto& local : function_->get_local_variables())
        {
            fn(local);
        }
    }

    utils::TransferContext make_transfer_context() const
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
        LOG(LSL_DEBUG, me() << info() << "Initializing data");

        collect_variables();
        collect_facts();
        init_states();

        LOG(LSL_DEBUG, me() << info() << "Initialized data: variables=" << NV_
                            << ", facts=" << facts_.size() << ", blocks=" << cfg_.size());
    }

    void collect_variables()
    {
        LOG(LSL_DEBUG, me() << info() << "Collecting variables");

        variable_ids_.clear();
        NV_ = 0;

        for_each_variable([this](const program::VariableIR_sptr& variable)
                          { add_variable(variable); });

        LOG(LSL_DEBUG, me() << info() << "Collected variables: count=" << NV_);
    }

    void add_variable(const program::VariableIR_sptr& variable)
    {
        ASSUMPTION(variable != nullptr);

        const auto it = variable_ids_.emplace_hint(variable_ids_.end(), variable.get(), NV_++);

        auto var_meta = std::make_unique<metadata::available_copy::VariableMeta>();
        var_meta->id  = it->second;
        variable->get_metadata().set(std::move(var_meta));
    }

    void collect_facts()
    {
        LOG(LSL_DEBUG, me() << info() << "Collecting copy facts");

        init_fact_containers();
        build_variables_by_id();

        auto reach = build_explicit_copy_reachability();
        compute_transitive_closure(reach);
        materialize_facts_from_reachability(reach);
        build_fact_indexes_and_kill_masks();

        LOG(LSL_DEBUG, me() << info() << "Collected copy facts: count=" << facts_.size());
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
        for_each_variable([this](const program::VariableIR_sptr& variable)
                          { register_variable_by_id(variable); });
    }

    void register_variable_by_id(const program::VariableIR_sptr& variable)
    {
        ASSUMPTION(variable != nullptr);

        const auto it = variable_ids_.find(variable.get());
        ASSUMPTION(it != variable_ids_.end());
        ASSUMPTION(it->second < NV_);

        variables_by_id_[it->second] = variable;
    }

    ReachabilityMatrix build_explicit_copy_reachability() const
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
        if (!is_valid_copy_instruction(instruction))
        {
            return;
        }

        const auto& operands = instruction->get_operands();

        const auto* dest = std::get_if<program::VariableIR_raw>(&operands[0]);
        const auto* src  = std::get_if<program::VariableIR_raw>(&operands[1]);

        if (!is_valid_copy_operand_pair(dest, src))
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

        if (!can_form_copy_fact(*dest, *src, dest_id, src_id))
        {
            return;
        }

        reach[dest_id][src_id] = true;
    }

    void compute_transitive_closure(ReachabilityMatrix& reach) const
    {
        // If X <- Y and Y <- Z are possible facts, then X <- Z must also belong
        // to the universe so it can be activated later.
        for (std::size_t mid = 0; mid < NV_; ++mid)
        {
            for (std::size_t dst = 0; dst < NV_; ++dst)
            {
                if (!reach[dst][mid])
                {
                    continue;
                }

                add_transitive_reachability_through_mid(dst, mid, reach);
            }
        }
    }

    void add_transitive_reachability_through_mid(std::size_t dst, std::size_t mid,
                                                 ReachabilityMatrix& reach) const
    {
        for (std::size_t src = 0; src < NV_; ++src)
        {
            if (!reach[mid][src])
            {
                continue;
            }

            if (!can_materialize_transitive_fact(dst, src))
            {
                continue;
            }

            reach[dst][src] = true;
        }
    }

    bool can_materialize_transitive_fact(std::size_t dst, std::size_t src) const
    {
        if (dst == src)
        {
            return false;
        }

        ASSUMPTION(variables_by_id_.at(dst) != nullptr);
        ASSUMPTION(variables_by_id_.at(src) != nullptr);

        return variables_by_id_.at(dst)->get_num_bytes() ==
               variables_by_id_.at(src)->get_num_bytes();
    }

    void materialize_facts_from_reachability(const ReachabilityMatrix& reach)
    {
        for (std::size_t dest_id = 0; dest_id < NV_; ++dest_id)
        {
            for (std::size_t src_id = 0; src_id < NV_; ++src_id)
            {
                if (reach[dest_id][src_id])
                {
                    add_copy_fact(dest_id, src_id);
                }
            }
        }
    }

    void add_copy_fact(std::size_t dest_id, std::size_t src_id)
    {
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
        out_states_.assign(cfg_.size(), utils::DynamicBitset{facts_.size()});
        in_state_scratch_.resize(facts_.size());
        has_out_.assign(cfg_.size(), false);
    }

    void solve()
    {
        LOG(LSL_DEBUG, me() << info() << "Solving data-flow");

        if (cfg_.empty())
        {
            LOG(LSL_DEBUG, me() << info() << "Skipping solve: empty CFG");
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<char>       in_worklist(cfg_.size(), 0);

        enqueue_entry(worklist, in_worklist);

        std::size_t iterations = 0;

        while (!worklist.empty())
        {
            ++iterations;
            process_next_block(worklist, in_worklist);
        }

        LOG(LSL_DEBUG, me() << info() << "Solved data-flow: iterations=" << iterations);
    }

    void enqueue_entry(std::queue<std::size_t>& worklist, std::vector<char>& in_worklist) const
    {
        worklist.push(cfg_.entry());
        in_worklist[cfg_.entry()] = 1;
    }

    void process_next_block(std::queue<std::size_t>& worklist, std::vector<char>& in_worklist)
    {
        const auto block = pop_worklist(worklist, in_worklist);

        build_in_state(block);

        if (!has_ready_input_for_block(block))
        {
            return;
        }

        auto new_out = in_state_scratch_;
        apply_block_transfer(block, new_out);

        update_out_state_if_changed(block, new_out, worklist, in_worklist);
    }

    std::size_t pop_worklist(std::queue<std::size_t>& worklist,
                             std::vector<char>&       in_worklist) const
    {
        const auto block = worklist.front();
        worklist.pop();

        in_worklist[block] = 0;

        return block;
    }

    void build_in_state(std::size_t block) { compute_joined_in_state(block, in_state_scratch_); }

    void compute_joined_in_state(std::size_t block, utils::DynamicBitset& state) const
    {
        state.reset();

        if (block == cfg_.entry())
        {
            return;
        }

        bool has_ready_pred = false;

        for (const auto pred : cfg_.predecessors(block))
        {
            if (!has_out_[pred])
            {
                continue;
            }

            merge_predecessor_out_state(pred, state, has_ready_pred);
        }

        if (!has_ready_pred)
        {
            state.reset();
        }
    }

    void merge_predecessor_out_state(std::size_t pred, utils::DynamicBitset& state,
                                     bool& has_ready_pred) const
    {
        if (!has_ready_pred)
        {
            state          = out_states_[pred];
            has_ready_pred = true;
            return;
        }

        state.and_with(out_states_[pred]);
    }

    bool has_ready_input_for_block(std::size_t block) const
    {
        if (block == cfg_.entry())
        {
            return true;
        }

        for (const auto pred : cfg_.predecessors(block))
        {
            if (has_out_[pred])
            {
                return true;
            }
        }

        return false;
    }

    void apply_block_transfer(std::size_t block, utils::DynamicBitset& state) const
    {
        const auto context = make_transfer_context();

        for (const auto& instruction : cfg_.block(block)->get_instructions())
        {
            utils::apply_transfer(instruction, context, state);
        }
    }

    void update_out_state_if_changed(std::size_t block, const utils::DynamicBitset& new_out,
                                     std::queue<std::size_t>& worklist,
                                     std::vector<char>&       in_worklist)
    {
        if (has_out_[block] && out_states_[block] == new_out)
        {
            return;
        }

        out_states_[block] = new_out;
        has_out_[block]    = true;

        enqueue_successors(block, worklist, in_worklist);
    }

    void enqueue_successors(std::size_t block, std::queue<std::size_t>& worklist,
                            std::vector<char>& in_worklist) const
    {
        for (const auto succ : cfg_.successors(block))
        {
            if (in_worklist[succ])
            {
                continue;
            }

            worklist.push(succ);
            in_worklist[succ] = 1;
        }
    }

    void materialize()
    {
        materialize_basic_block_metadata();
        materialize_function_metadata();
    }

    void materialize_basic_block_metadata()
    {
        for (std::size_t block = 0; block < cfg_.size(); ++block)
        {
            materialize_basic_block_metadata(block);
        }
    }

    void materialize_basic_block_metadata(std::size_t block)
    {
        auto in_state = build_materialized_in_state(block);

        auto block_meta     = std::make_unique<metadata::available_copy::BasicBlockMeta>();
        block_meta->in_bits = in_state.words();

        cfg_.block(block)->get_metadata().set(std::move(block_meta));
    }

    utils::DynamicBitset build_materialized_in_state(std::size_t block) const
    {
        utils::DynamicBitset in_state{facts_.size()};
        compute_joined_in_state(block, in_state);
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
    program::FunctionIR_sptr      function_;
    utils::view::FlattenedCFGView cfg_;

    std::size_t NV_{0};

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

void AvailableCopyAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    LOG(LSL_INFO, me() << "Running");
    {
        TMPROF_BLOCK()
        const auto trigger = Impl(std::move(sala_ir));
    }
    LOG(LSL_INFO, me() << "Done");
}

} // namespace optimizer::passes
