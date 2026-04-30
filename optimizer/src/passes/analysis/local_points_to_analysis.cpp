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
#include <queue>

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
    FunctionContext(std::size_t function_id, program::FunctionIR_sptr function, int id_start)
        : _id{function_id}, function_{std::move(function)}, id_start_{id_start}
    {
        if (function_->get_initializer_flag())
        {
            throw std::logic_error("Static initializer is not part of the local analysis");
        }

        init_data();
    }

    void run(MayAnalysisState global_may_seed)
    {
        global_may_seed_ = std::move(global_may_seed);
        solve();
    }

    void materialize()
    {
        for (std::size_t b = 0; b < NB_; ++b)
        {
            auto may_in = reconstruct_materialized_in_state(b);
            attach_block_points_to_metadata(b, std::move(may_in));
        }

        attach_function_points_to_metadata();
    }

    std::optional<MayAnalysisState> get_after_global_states()
    {
        std::optional<MayAnalysisState> result_may;

        MayAnalysisState bb_out_global_may_scratch;
        bb_out_global_may_scratch.may.reserve(global_may_seed_.may.size());

        for (const auto exit_bb : exit_basic_blocks)
        {
            if (!reachable_[exit_bb] || !has_out_[exit_bb])
            {
                continue;
            }

            export_exit_block_to_global_state(exit_bb, bb_out_global_may_scratch);

            if (!result_may.has_value())
            {
                result_may = bb_out_global_may_scratch;
            }
            else
            {
                utils::points_to::state_join_or_strict(result_may.value(),
                                                       bb_out_global_may_scratch);
            }
        }

        return result_may;
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

    void compute_reachable_blocks()
    {
        reachable_.assign(NB_, false);

        if (NB_ == 0)
        {
            return;
        }

        std::queue<std::size_t> wl;
        wl.push(entry_);
        reachable_[entry_] = true;

        while (!wl.empty())
        {
            const auto b = wl.front();
            wl.pop();

            for (const auto& weak_succ : blocks_[b]->get_successors())
            {
                if (auto succ = weak_succ.lock())
                {
                    const auto it = index_of_.find(succ);
                    ASSUMPTION(it != index_of_.end());

                    const auto succ_idx = it->second;
                    if (!reachable_[succ_idx])
                    {
                        reachable_[succ_idx] = true;
                        wl.push(succ_idx);
                    }
                }
            }
        }
    }

    void export_global_object_may(const objectId object_id, const MayAnalysisState& local_out_may,
                                  MayAnalysisState& exported_may) const
    {
        if (local_out_may.poisoned)
        {
            utils::poison_may_state(exported_may);
            return;
        }

        const auto object_end_bb_iter = local_out_may.may.find(object_id);
        if (object_end_bb_iter == local_out_may.may.end())
        {
            return;
        }

        if (object_end_bb_iter->second.is_top)
        {
            exported_may.may.insert_or_assign(object_id, utils::MayValue::top());
            return;
        }

        std::optional<MayState::iterator> exported_iter;

        const auto add_to_result = [&](const utils::Target& elem)
        {
            if (exported_iter.has_value())
            {
                exported_iter.value()->second.insert(elem);
            }
            else
            {
                const auto [iter, _] = exported_may.may.insert(
                        std::make_pair(object_id, utils::MayValue::singleton(elem)));
                exported_iter = iter;
            }
        };

        for (const auto target : object_end_bb_iter->second)
        {
            if (global_objects_->contains(target.id) || target.id == grouped_objects::HEAP)
            {
                add_to_result(target);
            }
            else
            {
                add_to_result({grouped_objects::OUT_OF_GLOBAL_SCOPE, false});
            }
        }
    }

    void export_exit_block_to_global_state(const std::size_t end_bb,
                                           MayAnalysisState& exported_may) const
    {
        exported_may.clear();

        const auto& out_may = out_may_.get(end_bb);
        if (out_may.poisoned)
        {
            utils::poison_may_state(exported_may);
            return;
        }

        for (const auto& [object_id, _] : *global_objects_)
        {
            export_global_object_may(object_id, out_may, exported_may);
            if (exported_may.poisoned)
            {
                return;
            }
        }
    }

    void attach_block_points_to_metadata(const std::size_t b, MayAnalysisState may_in)
    {
        auto bb_points_to_meta       = std::make_unique<metadata::points_to::BasicBlockMeta>();
        bb_points_to_meta->may_in_id = out_may_.store()->intern(std::move(may_in));
        blocks_[b]->get_metadata().set(std::move(bb_points_to_meta));
    }

    void attach_function_points_to_metadata()
    {
        auto function_meta                = std::make_unique<metadata::points_to::FunctionMeta>();
        function_meta->local_objects      = std::move(local_objects_);
        function_meta->bb_may_state_store = out_may_.store();
        function_->get_metadata().set(std::move(function_meta));
    }

    MayAnalysisState reconstruct_materialized_in_state(const std::size_t b) const
    {
        MayAnalysisState may_in{};

        bool in_initialized = false;

        if (b == entry_)
        {
            build_entry_seed_states(may_in);
            in_initialized = true;
        }

        for (const auto p : preds_[b])
        {
            if (!has_out_[p])
            {
                continue;
            }

            if (!in_initialized)
            {
                may_in         = out_may_.get(p);
                in_initialized = true;
            }
            else
            {
                utils::state_join_or_strict(may_in, out_may_.get(p));
            }
        }

        if (!in_initialized)
        {
            may_in.clear();
        }

        return may_in;
    }

    void init_local_objects()
    {
        objectId id = id_start_;

        auto assign_object_ids_variables =
                [&](const program::VariableIRListS& variables, const utils::RegionTag region)
        {
            for (const auto& variable : variables)
            {
                local_objects_.emplace_hint(local_objects_.end(), id, Object{id, region});
                auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
                points_to_meta->id  = id++;
                variable->get_metadata().set(std::move(points_to_meta));
            }
        };

        for (const auto& parameter : function_->get_parameters())
        {
            local_objects_.emplace_hint(local_objects_.end(), id,
                                        Object{id, utils::RegionTag::Parameter});
            auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
            points_to_meta->id  = id;
            assumed_ptr_params_.insert(assumed_ptr_params_.end(), id);
            ++id;
            parameter->get_metadata().set(std::move(points_to_meta));
        }

        assign_object_ids_variables(function_->get_local_variables(), utils::RegionTag::Local);
        last_local_id_ = id - 1;
    }

    void load_global_data()
    {
        auto& program_metadata =
                function_->get_program()->get_metadata().get<metadata::points_to::ProgramMeta>();

        global_objects_ = &program_metadata.global_objects;
    }

    void init_states()
    {
        NB_ = function_->get_basic_blocks().size();
        out_may_.reset(NB_);

        has_out_.assign(NB_, false);

        in_may_scratch_.clear();
    }

    void build_flattened_cfg()
    {
        NB_ = function_->get_basic_blocks().size();
        build_block_indexing();
        build_predecessor_lists_and_exit_blocks();
        detect_entry_block();
    }

    void build_block_indexing()
    {
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
            index_of_[blocks_[i]] = i;
        }
    }

    void build_predecessor_lists_and_exit_blocks()
    {
        preds_.assign(NB_, {});
        exit_basic_blocks.clear();

        for (std::size_t i = 0; i < NB_; ++i)
        {
            if (blocks_[i]->get_successors().empty())
            {
                exit_basic_blocks.insert(i);
            }

            for (const auto& wp : blocks_[i]->get_predecessors())
            {
                auto sp = wp.lock();
                ASSUMPTION(sp != nullptr);

                const auto it = index_of_.find(sp);
                ASSUMPTION(it != index_of_.end());

                preds_[i].push_back(it->second);
            }
        }
    }

    void detect_entry_block()
    {
        entry_ = 0;

        if (auto eb = function_->get_entry_basic_block())
        {
            const auto it = index_of_.find(eb);
            ASSUMPTION(it != index_of_.end());
            entry_ = it->second;
        }
    }

    void build_entry_seed_states(MayAnalysisState& may_in) const
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

    void build_solver_in_state(const std::size_t b, MayAnalysisState& may_in)
    {
        may_in.clear();

        bool in_initialized = false;

        if (b == entry_)
        {
            build_entry_seed_states(may_in);
            in_initialized = true;
        }

        for (const auto p : preds_[b])
        {
            if (!has_out_[p])
            {
                continue;
            }

            if (!in_initialized)
            {
                may_in         = out_may_.get(p);
                in_initialized = true;
            }
            else
            {
                utils::points_to::state_join_or_strict(may_in, out_may_.get(p));
            }
        }

        if (!in_initialized)
        {
            may_in.clear();
        }
    }

    void apply_block_transfers(const std::size_t b, MayAnalysisState& may_in)
    {
        std::size_t instruction_id = 0;

        for (const auto& instruction : blocks_[b]->get_instructions())
        {
            const auto relevant_ops_size = fill_operands_id(instruction);
            ASSUMPTION(operands_id_scratch_.size() >= relevant_ops_size);

            utils::MayTransferContextBundle may_transfer_context{
                    .pp             = {.function = _id, .bb = b, .instr = instruction_id},
                    .opcode         = instruction->get_opcode(),
                    .state          = may_in,
                    .global_objects = *global_objects_,
                    .local_objects  = local_objects_,
                    .operands_id    = operands_id_scratch_,
                    .operands_count = relevant_ops_size,
                    .last_local_id  = last_local_id_,
            };

            utils::apply_transfer_may(may_transfer_context);
            ++instruction_id;

            if (may_in.poisoned)
            {
                return;
            }
        }
    }

    bool block_out_changed(const std::size_t b, const MayAnalysisState& may_in) const
    {
        return !out_may_.equals(b, may_in);
    }

    void commit_block_out_state(const std::size_t b, MayAnalysisState& may_in)
    {
        out_may_.set(b, may_in);
        may_in.clear();
    }

    void enqueue_successors_if_reachable(const std::size_t b, std::queue<std::size_t>& worklist,
                                         std::vector<bool>& in_worklist) const
    {
        for (const auto& weak_succ : blocks_[b]->get_successors())
        {
            if (auto succ = weak_succ.lock())
            {
                const auto it = index_of_.find(succ);
                ASSUMPTION(it != index_of_.end());

                const auto succ_idx = it->second;
                if (!reachable_[succ_idx])
                {
                    continue;
                }

                if (!in_worklist[succ_idx])
                {
                    worklist.push(succ_idx);
                    in_worklist[succ_idx] = true;
                }
            }
        }
    }

    void solve()
    {
        if (NB_ == 0)
        {
            return;
        }

        std::fill(has_out_.begin(), has_out_.end(), false);

        std::queue<std::size_t> worklist;
        std::vector<bool>       in_worklist(NB_, false);

        if (reachable_[entry_])
        {
            worklist.push(entry_);
            in_worklist[entry_] = true;
        }

        while (!worklist.empty())
        {
            const auto b = worklist.front();
            worklist.pop();
            in_worklist[b] = false;

            auto& may_in = in_may_scratch_;

            build_solver_in_state(b, may_in);
            apply_block_transfers(b, may_in);

            const bool changed = !has_out_[b] || block_out_changed(b, may_in);
            if (changed)
            {
                commit_block_out_state(b, may_in);
                has_out_[b] = true;
                enqueue_successors_if_reachable(b, worklist, in_worklist);
            }
        }
    }

    inline std::size_t fill_operands_id(const program::InstructionIR_sptr& instruction)
    {
        const auto relevant_ops_count = utils::get_relevant_operands_count(*instruction);
        operands_id_scratch_.clear();
        if (relevant_ops_count > operands_id_scratch_.size())
        {
            operands_id_scratch_.reserve(relevant_ops_count);
        }

        for (std::size_t i = 0; i < relevant_ops_count; ++i)
        {
            operands_id_scratch_.push_back(get_operand_id(instruction, i));
        }

        return relevant_ops_count;
    }

    inline objectId get_operand_id(const program::InstructionIR_sptr& instruction,
                                   const std::size_t                  position)
    {
        const auto operand_raw = instruction->get_operands().at(position);
        if (std::holds_alternative<program::FunctionIR_raw>(operand_raw))
        {
            return grouped_objects::FUNCTION;
        }
        else if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand_raw))
        {
            return (*variable_raw)->get_metadata().get<metadata::points_to::VariableMeta>().id;
        }
        else if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand_raw))
        {
            return (*constant_raw)->get_metadata().get<metadata::points_to::ConstantMeta>().id;
        }
        else
        {
            ASSUMPTION(false);
        }
    }

    bool accessed_undefined() const { return accessed_undefined_; }

  private:
    bool                     accessed_undefined_{false};
    objectId                 last_local_id_{0};
    int                      id_start_;
    std::size_t              _id;
    program::FunctionIR_sptr function_;

    MayAnalysisState global_may_seed_;

    utils::ObjectPool* global_objects_{};
    utils::ObjectPool  local_objects_;

    utils::SparseSet<objectId>    assumed_ptr_params_;
    utils::SparseSet<std::size_t> exit_basic_blocks;

    utils::BasicBlockMayStateSlots out_may_;

    MayAnalysisState      in_may_scratch_;
    std::vector<objectId> operands_id_scratch_;

    std::size_t NB_{0};
    std::size_t entry_{0};

    std::vector<bool>                                         reachable_;
    std::vector<bool>                                         has_out_;
    std::vector<program::BasicBlockIR_sptr>                   blocks_;
    std::vector<std::vector<std::size_t>>                     preds_;
    utils::SparseMap<program::BasicBlockIR_sptr, std::size_t> index_of_;
};

struct Impl
{
    Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        init_function_contexts();
        run_function_contexts();
        materialize();
    }

  private:
    void init_function_contexts()
    {
        contexts_.reserve(sala_ir_->get_functions().size());
        const auto& points_to_program_meta =
                sala_ir_->get_metadata().get<metadata::points_to::ProgramMeta>();

        const auto local_id_start =
                static_cast<objectId>(points_to_program_meta.global_objects.size());

        std::size_t func_id = 0;
        for (const auto& function : sala_ir_->get_functions())
        {
            if (function->get_initializer_flag())
            {
                continue;
            }

            contexts_.emplace_back(
                    std::make_unique<FunctionContext>(func_id++, function, local_id_start));
        }
    }

    void run_function_contexts()
    {
        if (contexts_.empty())
        {
            return;
        }

        auto& program_metadata = sala_ir_->get_metadata().get<metadata::points_to::ProgramMeta>();
        bool  fixpoint_reached = false;

        do
        {
            std::optional<MayAnalysisState> fixpoint_may;

            for (auto& context : contexts_)
            {
                context->run(program_metadata.may_out);
            }

            for (auto& context : contexts_)
            {
                auto possible_out_state = context->get_after_global_states();
                if (!possible_out_state.has_value())
                {
                    continue;
                }

                auto& global_may_out = possible_out_state;

                if (!fixpoint_may.has_value())
                {
                    fixpoint_may = std::move(global_may_out);
                }
                else
                {
                    utils::points_to::state_join_or_strict(fixpoint_may.value(),
                                                           global_may_out.value(),
                                                           grouped_objects::CALL_ORDER_DISCREPANCY);
                }
            }

            if (!fixpoint_may.has_value())
            {
                fixpoint_reached = true;
            }
            else
            {
                fixpoint_reached         = fixpoint_may.value() == program_metadata.may_out;
                program_metadata.may_out = std::move(fixpoint_may.value());
            }
        } while (!fixpoint_reached);
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
    program::ProgramIR_sptr sala_ir_;

    std::vector<std::unique_ptr<FunctionContext>> contexts_;
};
} // namespace

void LocalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    const auto trigger = Impl(std::move(sala_ir));
}
} // namespace optimizer::passes
