#include <optimizer/passes/analysis/local_points_to_analysis.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/points_to/import.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

namespace optimizer::passes
{
namespace grouped_objects = utils::grouped_objects;
using utils::MayAnalysisState;
using utils::MayState;
using utils::Object;
using utils::objectId;

namespace
{
struct FunctionContext
{
  public:
    FunctionContext(std::size_t function_id, program::FunctionIR_sptr function, objectId id_start)
        : function_id_{function_id}, function_{std::move(function)}, id_start_{id_start}
    {
        ASSUMPTION(function_ != nullptr);
        ASSUMPTION(!function_->get_initializer_flag());
        init_data();
    }

    void run(MayAnalysisState global_may_seed)
    {
        global_may_seed_ = std::move(global_may_seed);
        solve();
    }

    void materialize()
    {
        for (std::size_t block = 0; block < block_count_; ++block)
        {
            MayAnalysisState may_in{};
            build_in_state(block, may_in);
            attach_block_points_to_metadata(block, std::move(may_in));
        }

        attach_function_points_to_metadata();
    }

    [[nodiscard]] std::optional<MayAnalysisState> get_after_global_states() const
    {
        std::optional<MayAnalysisState> result;

        for (const auto exit_block : exit_blocks_)
        {
            if (!is_solved_reachable_block(exit_block))
            {
                continue;
            }

            auto exported_exit = export_exit_block_to_global_state(exit_block);
            merge_optional_global_state(result, std::move(exported_exit));
        }

        return result;
    }

  private:
    void init_data()
    {
        load_global_data();
        init_states();
        init_local_objects();
        build_flattened_cfg();
        compute_reachable_blocks();
    }

    void load_global_data()
    {
        auto& program_metadata =
                function_->get_program()->get_metadata().get<metadata::points_to::ProgramMeta>();

        global_objects_ = &program_metadata.global_objects;
    }

    void init_states()
    {
        block_count_ = function_->get_basic_blocks().size();
        out_may_.reset(block_count_);
        has_out_.assign(block_count_, false);
        in_may_scratch_.clear();
    }

    void init_local_objects()
    {
        objectId id = id_start_;

        for (const auto& parameter : function_->get_parameters())
        {
            register_local_object(parameter, id, utils::RegionTag::Parameter);
            assumed_ptr_params_.insert(assumed_ptr_params_.end(), id);
            ++id;
        }

        for (const auto& local : function_->get_local_variables())
        {
            register_local_object(local, id, utils::RegionTag::Local);
            ++id;
        }

        last_local_id_ = id - 1;
    }

    void register_local_object(const program::VariableIR_sptr& variable, objectId id,
                               utils::RegionTag region)
    {
        local_objects_.emplace_hint(local_objects_.end(), id, Object{id, region});

        auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
        points_to_meta->id  = id;
        variable->get_metadata().set(std::move(points_to_meta));
    }

    void build_flattened_cfg()
    {
        collect_blocks();
        build_block_indexing();
        build_predecessor_lists_and_exit_blocks();
        detect_entry_block();
    }

    void collect_blocks()
    {
        blocks_.clear();
        blocks_.reserve(block_count_);

        for (const auto& block : function_->get_basic_blocks())
        {
            blocks_.push_back(block);
        }
    }

    void build_block_indexing()
    {
        index_of_.clear();
        index_of_.reserve(block_count_);

        for (std::size_t index = 0; index < block_count_; ++index)
        {
            index_of_[blocks_[index]] = index;
        }
    }

    void build_predecessor_lists_and_exit_blocks()
    {
        preds_.assign(block_count_, {});
        exit_blocks_.clear();

        for (std::size_t block = 0; block < block_count_; ++block)
        {
            if (blocks_[block]->get_successors().empty())
            {
                exit_blocks_.insert(block);
            }

            for (const auto& weak_pred : blocks_[block]->get_predecessors())
            {
                auto pred = weak_pred.lock();
                ASSUMPTION(pred != nullptr);
                preds_[block].push_back(block_index(pred));
            }
        }
    }

    void detect_entry_block()
    {
        entry_ = 0;

        if (auto entry_block = function_->get_entry_basic_block())
        {
            entry_ = block_index(entry_block);
        }
    }

    [[nodiscard]] std::size_t block_index(const program::BasicBlockIR_sptr& block) const
    {
        const auto it = index_of_.find(block);
        ASSUMPTION(it != index_of_.end());
        return it->second;
    }

    void compute_reachable_blocks()
    {
        reachable_.assign(block_count_, false);
        if (block_count_ == 0)
        {
            return;
        }

        std::queue<std::size_t> worklist;
        reachable_[entry_] = true;
        worklist.push(entry_);

        while (!worklist.empty())
        {
            const auto block = worklist.front();
            worklist.pop();

            enqueue_new_reachable_successors(block, worklist);
        }
    }

    void enqueue_new_reachable_successors(std::size_t block, std::queue<std::size_t>& worklist)
    {
        for (const auto& weak_succ : blocks_[block]->get_successors())
        {
            if (auto succ = weak_succ.lock())
            {
                const auto succ_index = block_index(succ);
                if (!reachable_[succ_index])
                {
                    reachable_[succ_index] = true;
                    worklist.push(succ_index);
                }
            }
        }
    }

    [[nodiscard]] bool is_solved_reachable_block(std::size_t block) const
    {
        return reachable_[block] && has_out_[block];
    }

    void solve()
    {
        if (block_count_ == 0)
        {
            return;
        }

        reset_solver_state();

        std::queue<std::size_t> worklist;
        std::vector<bool>       in_worklist(block_count_, false);
        enqueue_entry_if_reachable(worklist, in_worklist);

        while (!worklist.empty())
        {
            const auto block = pop_worklist(worklist, in_worklist);
            solve_block(block, worklist, in_worklist);
        }
    }

    void reset_solver_state() { std::fill(has_out_.begin(), has_out_.end(), false); }

    void enqueue_entry_if_reachable(std::queue<std::size_t>& worklist,
                                    std::vector<bool>&       in_worklist) const
    {
        if (!reachable_[entry_])
        {
            return;
        }

        worklist.push(entry_);
        in_worklist[entry_] = true;
    }

    std::size_t pop_worklist(std::queue<std::size_t>& worklist,
                             std::vector<bool>&       in_worklist) const
    {
        const auto block = worklist.front();
        worklist.pop();
        in_worklist[block] = false;
        return block;
    }

    void solve_block(std::size_t block, std::queue<std::size_t>& worklist,
                     std::vector<bool>& in_worklist)
    {
        auto& may = in_may_scratch_;
        build_in_state(block, may);
        apply_block_transfers(block, may);

        if (block_out_changed(block, may))
        {
            commit_block_out_state(block, may);
            enqueue_successors_if_reachable(block, worklist, in_worklist);
        }
    }

    [[nodiscard]] bool block_out_changed(std::size_t block, const MayAnalysisState& may) const
    {
        return !has_out_[block] || !out_may_.equals(block, may);
    }

    void build_in_state(std::size_t block, MayAnalysisState& may_in) const
    {
        may_in.clear();

        bool initialized = false;
        initialize_entry_seed_if_needed(block, may_in, initialized);
        merge_available_predecessors(block, may_in, initialized);

        if (!initialized)
        {
            may_in.clear();
        }
    }

    void initialize_entry_seed_if_needed(std::size_t block, MayAnalysisState& may_in,
                                         bool& initialized) const
    {
        if (block != entry_)
        {
            return;
        }

        build_entry_seed_state(may_in);
        initialized = true;
    }

    void build_entry_seed_state(MayAnalysisState& may_in) const
    {
        may_in = global_may_seed_;
        if (may_in.poisoned)
        {
            return;
        }

        for (const auto param : assumed_ptr_params_)
        {
            may_in.may.insert_or_assign(
                    param, utils::make_singleton(grouped_objects::OUT_OF_LOCAL_SCOPE, false));
        }
    }

    void merge_available_predecessors(std::size_t block, MayAnalysisState& may_in,
                                      bool& initialized) const
    {
        for (const auto pred : preds_[block])
        {
            if (!has_out_[pred])
            {
                continue;
            }

            merge_state(may_in, out_may_.get(pred), initialized);
        }
    }

    static void merge_state(MayAnalysisState& target, const MayAnalysisState& source,
                            bool& initialized)
    {
        if (!initialized)
        {
            target      = source;
            initialized = true;
            return;
        }

        utils::state_join_or_strict(target, source);
    }

    void apply_block_transfers(std::size_t block, MayAnalysisState& may)
    {
        std::size_t instruction_id = 0;

        for (const auto& instruction : blocks_[block]->get_instructions())
        {
            apply_instruction_transfer(block, instruction_id, instruction, may);
            ++instruction_id;

            if (may.poisoned)
            {
                return;
            }
        }
    }

    void apply_instruction_transfer(std::size_t block, std::size_t instruction_id,
                                    const program::InstructionIR_sptr& instruction,
                                    MayAnalysisState&                  may)
    {
        const auto relevant_ops_size = fill_operands_id(instruction);
        ASSUMPTION(operands_id_scratch_.size() >= relevant_ops_size);

        utils::MayTransferContextBundle context{
                .pp             = {.function = function_id_, .bb = block, .instr = instruction_id},
                .opcode         = instruction->get_opcode(),
                .state          = may,
                .global_objects = *global_objects_,
                .local_objects  = local_objects_,
                .operands_id    = operands_id_scratch_,
                .operands_count = relevant_ops_size,
                .last_local_id  = last_local_id_,
        };

        utils::apply_transfer_may(context);
    }

    void commit_block_out_state(std::size_t block, MayAnalysisState& may)
    {
        out_may_.set(block, may);
        has_out_[block] = true;
        may.clear();
    }

    void enqueue_successors_if_reachable(std::size_t block, std::queue<std::size_t>& worklist,
                                         std::vector<bool>& in_worklist) const
    {
        for (const auto& weak_succ : blocks_[block]->get_successors())
        {
            if (auto succ = weak_succ.lock())
            {
                const auto succ_index = block_index(succ);
                if (!reachable_[succ_index] || in_worklist[succ_index])
                {
                    continue;
                }

                worklist.push(succ_index);
                in_worklist[succ_index] = true;
            }
        }
    }

    void attach_block_points_to_metadata(std::size_t block, MayAnalysisState may_in)
    {
        auto bb_points_to_meta       = std::make_unique<metadata::points_to::BasicBlockMeta>();
        bb_points_to_meta->may_in_id = out_may_.store()->intern(std::move(may_in));
        blocks_[block]->get_metadata().set(std::move(bb_points_to_meta));
    }

    void attach_function_points_to_metadata()
    {
        auto function_meta                = std::make_unique<metadata::points_to::FunctionMeta>();
        function_meta->local_objects      = std::move(local_objects_);
        function_meta->bb_may_state_store = out_may_.store();
        function_->get_metadata().set(std::move(function_meta));
    }

    [[nodiscard]] MayAnalysisState export_exit_block_to_global_state(std::size_t exit_block) const
    {
        MayAnalysisState exported{};
        const auto&      out_may = out_may_.get(exit_block);

        if (out_may.poisoned)
        {
            utils::poison_may_state(exported);
            return exported;
        }

        for (const auto& [object_id, _] : *global_objects_)
        {
            export_global_object_may(object_id, out_may, exported);
            if (exported.poisoned)
            {
                return exported;
            }
        }

        return exported;
    }

    void export_global_object_may(objectId object_id, const MayAnalysisState& local_out_may,
                                  MayAnalysisState& exported_may) const
    {
        if (local_out_may.poisoned)
        {
            utils::poison_may_state(exported_may);
            return;
        }

        const auto object_iter = local_out_may.may.find(object_id);
        if (object_iter == local_out_may.may.end())
        {
            return;
        }

        auto projected = project_value_to_global_scope(object_iter->second);
        if (!projected.empty())
        {
            exported_may.may.insert_or_assign(object_id, std::move(projected));
        }
    }

    [[nodiscard]] utils::MayValue project_value_to_global_scope(const utils::MayValue& value) const
    {
        if (value.is_top)
        {
            return utils::MayValue::top();
        }

        utils::MayValue projected{};
        projected.loss_flags = value.loss_flags;

        for (const auto& target : value)
        {
            if (is_globally_visible_target(target.id))
            {
                projected.insert(target);
            }
            else
            {
                projected.insert(utils::Target{grouped_objects::OUT_OF_GLOBAL_SCOPE, false});
            }
        }

        return projected;
    }

    [[nodiscard]] bool is_globally_visible_target(objectId id) const
    {
        return global_objects_->contains(id) || id == grouped_objects::HEAP;
    }

    static void merge_optional_global_state(std::optional<MayAnalysisState>& target,
                                            MayAnalysisState                 source)
    {
        if (!target.has_value())
        {
            target = std::move(source);
            return;
        }

        utils::state_join_or_strict(target.value(), source);
    }

    std::size_t fill_operands_id(const program::InstructionIR_sptr& instruction)
    {
        const auto relevant_ops_count = utils::get_relevant_operands_count(*instruction);
        operands_id_scratch_.clear();
        operands_id_scratch_.reserve(relevant_ops_count);

        for (std::size_t index = 0; index < relevant_ops_count; ++index)
        {
            operands_id_scratch_.push_back(get_operand_id(instruction, index));
        }

        return relevant_ops_count;
    }

    objectId get_operand_id(const program::InstructionIR_sptr& instruction,
                            std::size_t                        position) const
    {
        const auto operand_raw = instruction->get_operands().at(position);
        if (std::holds_alternative<program::FunctionIR_raw>(operand_raw))
        {
            return grouped_objects::FUNCTION;
        }

        if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand_raw))
        {
            return (*variable_raw)->get_metadata().get<metadata::points_to::VariableMeta>().id;
        }

        if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand_raw))
        {
            return (*constant_raw)->get_metadata().get<metadata::points_to::ConstantMeta>().id;
        }

        ASSUMPTION(false);
    }

    [[nodiscard]] bool accessed_undefined() const { return accessed_undefined_; }

  private:
    bool                     accessed_undefined_{false};
    objectId                 last_local_id_{0};
    objectId                 id_start_{0};
    std::size_t              function_id_{0};
    program::FunctionIR_sptr function_;

    MayAnalysisState global_may_seed_;

    utils::ObjectPool* global_objects_{};
    utils::ObjectPool  local_objects_;

    utils::SparseSet<objectId>    assumed_ptr_params_;
    utils::SparseSet<std::size_t> exit_blocks_;

    utils::BasicBlockMayStateSlots out_may_;

    MayAnalysisState      in_may_scratch_;
    std::vector<objectId> operands_id_scratch_;

    std::size_t block_count_{0};
    std::size_t entry_{0};

    std::vector<bool>                                         reachable_;
    std::vector<bool>                                         has_out_;
    std::vector<program::BasicBlockIR_sptr>                   blocks_;
    std::vector<std::vector<std::size_t>>                     preds_;
    utils::SparseMap<program::BasicBlockIR_sptr, std::size_t> index_of_;
};

struct Impl
{
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        init_function_contexts();
        run_function_contexts();
        materialize();
    }

  private:
    void init_function_contexts()
    {
        contexts_.reserve(sala_ir_->get_functions().size());

        const auto& program_meta = sala_ir_->get_metadata().get<metadata::points_to::ProgramMeta>();
        const auto  local_id_start = static_cast<objectId>(program_meta.global_objects.size());

        std::size_t function_id = 0;
        for (const auto& function : sala_ir_->get_functions())
        {
            if (function->get_initializer_flag())
            {
                continue;
            }

            contexts_.emplace_back(
                    std::make_unique<FunctionContext>(function_id++, function, local_id_start));
        }
    }

    void run_function_contexts()
    {
        if (contexts_.empty())
        {
            return;
        }

        auto& program_meta = sala_ir_->get_metadata().get<metadata::points_to::ProgramMeta>();

        bool fixpoint_reached = false;
        do
        {
            auto next_global_state = compute_next_global_state(program_meta.may_out);
            fixpoint_reached       = update_global_state_if_needed(program_meta.may_out,
                                                                   std::move(next_global_state));
        } while (!fixpoint_reached);
    }

    [[nodiscard]] std::optional<MayAnalysisState>
    compute_next_global_state(const MayAnalysisState& current_global_state)
    {
        run_all_contexts(current_global_state);
        return merge_all_context_exports();
    }

    void run_all_contexts(const MayAnalysisState& current_global_state)
    {
        for (auto& context : contexts_)
        {
            context->run(current_global_state);
        }
    }

    [[nodiscard]] std::optional<MayAnalysisState> merge_all_context_exports()
    {
        std::optional<MayAnalysisState> next_global_state;

        for (auto& context : contexts_)
        {
            auto exported = context->get_after_global_states();
            if (!exported.has_value())
            {
                continue;
            }

            merge_optional_global_state(next_global_state, std::move(exported.value()),
                                        grouped_objects::CALL_ORDER_DISCREPANCY);
        }

        return next_global_state;
    }

    static void merge_optional_global_state(std::optional<MayAnalysisState>& target,
                                            MayAnalysisState source, objectId marker)
    {
        if (!target.has_value())
        {
            target = std::move(source);
            return;
        }

        utils::state_join_or_strict(target.value(), source, marker);
    }

    static bool update_global_state_if_needed(MayAnalysisState&               current_global_state,
                                              std::optional<MayAnalysisState> next_global_state)
    {
        if (!next_global_state.has_value())
        {
            return true;
        }

        const bool fixpoint_reached = next_global_state.value() == current_global_state;
        current_global_state        = std::move(next_global_state.value());
        return fixpoint_reached;
    }

    void materialize()
    {
        while (!contexts_.empty())
        {
            contexts_.back()->materialize();
            contexts_.pop_back();
        }
    }

  private:
    program::ProgramIR_sptr                       sala_ir_;
    std::vector<std::unique_ptr<FunctionContext>> contexts_;
};
} // namespace

void LocalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    const auto trigger = Impl(std::move(sala_ir));
}
} // namespace optimizer::passes
