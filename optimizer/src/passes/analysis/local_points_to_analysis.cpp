#include <optimizer/passes/analysis/local_points_to_analysis.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/points_to_import.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <algorithm>
#include <execution>
#include <map>
#include <queue>

namespace optimizer::passes
{

namespace grouped_objects = utils::grouped_objects;
using utils::MayState;
using utils::MustState;
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
        // TODO: think about using pmr may must scratch states to avoid allocation
    }

    void run()
    {
        if (function_->get_initializer_flag())
        {
            // We assume this was filled by the GlobalPointsToAnalysis
            return;
        }
        init_data();
        solve();
    }

    void resolve(const MustState& global_must_seed, const MayState& global_may_seed)
    {
        global_may_seed_  = global_may_seed;
        global_must_seed_ = global_must_seed;
        solve();
        materialize();
    }

    void materialize()
    {
        for (std::size_t b = 0; b < NB_; ++b)
        {
            // IN from final OUTs of preds
            MayState  may_in{};
            MustState must_in{};
            bool      first_pred_found = false;
            if (b == entry_)
            {
                must_in = global_must_seed_;
                may_in  = global_may_seed_;
                // FIXME: do this somehow better
                for (const auto param : assumed_ptr_params_)
                {
                    may_in.insert(std::make_pair(
                            param, utils::MaySet{{grouped_objects::OUT_OF_LOCAL_SCOPE, false}}));
                }

                for (const auto elem : utils::points_to::grouped_objects::ABSTRACT_NODES)
                {
                    may_in[elem] = utils::MaySet{{.id = elem, .offset_flag = false}};
                }
                first_pred_found = true;
            }

            for (auto p : preds_[b])
            {
                if (!first_pred_found)
                {
                    may_in           = out_may_[p];
                    first_pred_found = true;
                }
                else
                {
                    utils::state_join_or_strict(may_in, out_may_[p]);
                }
            }

            if (!preds_[b].empty())
            {
                must_in = out_must_[preds_[b][0]];
                for (std::size_t j = 1; j < preds_[b].size(); ++j)
                {
                    utils::state_join_and(must_in, out_must_[preds_[b][j]]);
                }
            }

            auto bb_points_to_meta     = std::make_unique<metadata::points_to::BasicBlockMeta>();
            bb_points_to_meta->may_in  = std::move(may_in);
            bb_points_to_meta->must_in = std::move(must_in);
            blocks_[b]->get_metadata().set(std::move(bb_points_to_meta));
        }

        auto function_meta           = std::make_unique<metadata::points_to::FunctionMeta>();
        function_meta->local_objects = std::move(local_objects_);
        function_->get_metadata().set(std::move(function_meta));
    }

    std::pair<MustState, MayState> get_after_global_states()
    {
        MustState result_must;
        MustState bb_out_global_must_scratch;
        result_must.reserve(global_must_seed_.size());
        bb_out_global_must_scratch.reserve(global_must_seed_.size());

        MayState result_may;
        MayState bb_out_global_may_scratch;
        result_may.reserve(global_may_seed_.size());
        bb_out_global_may_scratch.reserve(global_may_seed_.size());

        for (const auto end_bb : func_end_bbs)
        {
            // std::cout << end_bb << std::endl;
            bb_out_global_may_scratch.clear();
            bb_out_global_must_scratch.clear();

            const auto& out_must = out_must_[end_bb];
            const auto& out_may  = out_may_[end_bb];

            for (const auto& [object_id, object_info] : *global_objects_)
            {
                // MAY points to
                {
                    const auto                        object_end_bb_iter = out_may.find(object_id);
                    std::optional<MayState::iterator> bb_out_may_iter;
                    const auto                        add_to_result = [&](const utils::Target& elem)
                    {
                        if (bb_out_may_iter.has_value())
                        {
                            bb_out_may_iter.value()->second.emplace(elem);
                        }
                        else
                        {
                            const auto [iter, _] = bb_out_global_may_scratch.insert(
                                    std::make_pair(object_id, utils::points_to::MaySet{elem}));
                            bb_out_may_iter = iter;
                        }
                    };
                    if (object_end_bb_iter != out_may.end())
                    {
                        for (const auto target : object_end_bb_iter->second)
                        {
                            if (local_objects_.contains(target.id))
                            {
                                add_to_result({grouped_objects::OUT_OF_GLOBAL_SCOPE, false});
                            }
                            else if (target.id == grouped_objects::OUT_OF_LOCAL_SCOPE)
                            {
                                add_to_result({grouped_objects::UNKOWN_GLOBAL, false});
                            }
                            else
                            {
                                add_to_result(target);
                            }
                        }
                    }
                }
                // MUST points to
                {
                    if (const auto object_end_bb_iter = out_must.find(object_id);
                        object_end_bb_iter != out_must.end())
                    {
                        bb_out_global_must_scratch[object_id] = object_end_bb_iter->second;
                    }
                }
            }

            if (result_must.empty())
            {
                result_must = bb_out_global_must_scratch;
            }
            else
            {
                utils::points_to::state_join_and(result_must, bb_out_global_must_scratch);
            }

            if (result_may.empty())
            {
                result_may = bb_out_global_may_scratch;
            }
            else
            {
                utils::points_to::state_join_or_strict(result_may, bb_out_global_may_scratch);
            }
        }

        return std::make_pair(std::move(result_must), std::move(result_may));
    }

  private:
    void init_data()
    {
        load_global_data();
        init_states();
        init_local_objects();
        build_flattened_cfg();
    }

    void init_local_objects()
    {
        objectId id = id_start_;

        auto assign_object_ids_variables =
                [&](const program::VariableIRListS& variables, const utils::RegionTag region)
        {
            for (const auto& variable : variables)
            {
                local_objects_.emplace(id, Object{id, region});
                auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
                points_to_meta->id  = id++;
                variable->get_metadata().set(std::move(points_to_meta));
            }
        };
        for (const auto& parameter : function_->get_parameters())
        {
            local_objects_.emplace(id, Object{id, utils::RegionTag::Parameter});
            auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
            points_to_meta->id  = id;
            if (parameter->get_num_bytes() == function_->get_program()->get_num_cpu_bits() / 8)
            {
                assumed_ptr_params_.insert(id);
            }
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

        global_objects_   = &program_metadata.global_objects;
        global_must_seed_ = program_metadata.must_out;
        global_may_seed_  = program_metadata.may_out;
    }

    void init_states()
    {
        NB_ = function_->get_basic_blocks().size();
        out_may_.assign(NB_, {});
        out_must_.assign(NB_, {});

        in_may_scratch_.clear();
        in_must_scratch_.clear();
    }

    void build_flattened_cfg()
    {
        // blocks_ + index_of_
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
            index_of_[blocks_[i]] = i;
        }

        // preds_
        preds_.assign(NB_, {});
        for (std::size_t i = 0; i < NB_; ++i)
        {
            if (blocks_[i]->get_successors().empty())
            {
                func_end_bbs.insert(i);
            }

            for (const auto& wp : blocks_[i]->get_predecessors())
            {
                auto sp = wp.lock();
                ASSUMPTION(sp != nullptr);
                auto it = index_of_.find(sp);
                ASSUMPTION(it != index_of_.end());
                preds_[i].push_back(it->second);
            }
        }

        // entry_
        entry_ = 0;
        if (auto eb = function_->get_entry_basic_block())
        {
            auto it = index_of_.find(eb);
            ASSUMPTION(it != index_of_.end());
            entry_ = it->second;
        }
    }

    void solve()
    {
        if (NB_ == 0)
        {
            return;
        }
        first_visit_.assign(NB_, true);

        std::queue<std::size_t> wl;
        std::vector<char>       in_wl(NB_, 0);
        wl.push(entry_);
        in_wl[entry_] = 1;

        while (!wl.empty())
        {
            const std::size_t b = wl.front();
            wl.pop();
            in_wl[b] = 0;

            auto& may_in     = in_may_scratch_;
            auto& must_in    = in_must_scratch_;
            auto  first_pred = false;
            if (b == entry_ && first_visit_[b])
            {
                may_in = global_may_seed_;
                for (const auto param : assumed_ptr_params_)
                {
                    may_in.insert(std::make_pair(
                            param, utils::MaySet{{.id = grouped_objects::OUT_OF_LOCAL_SCOPE,
                                                  .offset_flag = false}}));
                }

                for (const auto elem : utils::points_to::grouped_objects::ABSTRACT_NODES)
                {
                    may_in[elem] = utils::MaySet{{.id = elem, .offset_flag = false}};
                }

                // may[grouped_objects::OUT_OF_LOCAL_SCOPE] = utils::MaySet{
                //         {.id = grouped_objects::OUT_OF_LOCAL_SCOPE, .offset_flag = false},
                // };
                must_in    = global_must_seed_;
                first_pred = true;
            }

            for (const auto p : preds_[b])
            {
                if (first_visit_[p])
                {
                    continue;
                }

                if (!first_pred)
                {
                    must_in    = out_must_[p];
                    may_in     = out_may_[p];
                    first_pred = true;
                }
                else
                {
                    utils::state_join_or_strict(may_in, out_may_[p]);
                    utils::state_join_and(must_in, out_must_[p]);
                }
            }
            // clear states only if necessary (not set)
            if (!first_pred)
            {
                ASSUMPTION(b == entry_);
                may_in.clear();
                must_in.clear();
            }

            // apply transfer functions modifying may/must-IN to OUT inplace
            std::size_t instruciton_id = 0;
            for (const auto& instruction : blocks_[b]->get_instructions())
            {
                const auto relevant_ops_size = fill_operands_id(instruction);

                // One state needs to be copied -> the modification is done in-place
                // we choose the must state since it contains less data
                utils::MayTransferContextBundle may_transfer_context{
                        .pp             = {.function = _id, .bb = b, .instr = instruciton_id},
                        .opcode         = instruction->get_opcode(),
                        .may_in         = may_in,
                        .must_in        = MustState{must_in},
                        .global_objects = *global_objects_,
                        .local_objects  = local_objects_,
                        .operands_id    = operands_id_scratch_,
                        .operands_count = relevant_ops_size,
                        .last_local_id  = last_local_id_,
                };

                utils::MustTransferContextBundle must_transfer_context{
                        .opcode         = instruction->get_opcode(),
                        .must_in        = must_in,
                        .may_in         = may_in,
                        .global_objects = *global_objects_,
                        .local_objects  = local_objects_,
                        .operands_id    = operands_id_scratch_,
                        .operands_count = relevant_ops_size,
                        .last_local_id  = last_local_id_,
                };

                // WARNING: order of calls is important since we modify the states in-place
                utils::apply_transfer_must(must_transfer_context);
                utils::apply_transfer_may(may_transfer_context);
                ++instruciton_id;
            }

            first_visit_[b] = false;
            if (out_may_[b] != may_in || out_must_[b] != must_in)
            {
                std::swap(out_may_[b], may_in);
                std::swap(out_must_[b], must_in);

                for (const auto& weak_succ : blocks_[b]->get_successors())
                {
                    if (auto succ = weak_succ.lock())
                    {
                        auto it = index_of_.find(succ);
                        ASSUMPTION(it != index_of_.end());
                        bool has_ready_pred =
                                std::any_of(preds_[it->second].begin(), preds_[it->second].end(),
                                            [&](auto p) { return !first_visit_[p]; });

                        if (has_ready_pred && !in_wl[it->second])
                        {
                            wl.push(it->second);
                            in_wl[it->second] = 1;
                        }
                    }
                }
            }
        }
    }

    inline std::size_t fill_operands_id(const program::InstructionIR_sptr& instruction)
    {
        const auto relevant_ops_count = utils::get_relevant_operands_count(*instruction);
        if (relevant_ops_count >= operands_id_scratch_.size())
        {
            operands_id_scratch_.resize(relevant_ops_count, grouped_objects::UNDEFINED);
        }
        for (std::size_t i = 0; i < relevant_ops_count; ++i)
        {
            operands_id_scratch_[i] = get_operand_id(instruction, i);
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
    bool                     accessed_undefined_;
    objectId                 last_local_id_{0};
    int                      id_start_;
    std::size_t              _id;
    program::FunctionIR_sptr function_;

    MayState  global_may_seed_;
    MustState global_must_seed_;

    utils::ObjectPool* global_objects_{};
    utils::ObjectPool  local_objects_;

    std::unordered_set<objectId>    assumed_ptr_params_;
    std::unordered_set<std::size_t> func_end_bbs;

    std::vector<MayState>  out_may_;
    std::vector<MustState> out_must_;

    MayState              in_may_scratch_;
    MustState             in_must_scratch_;
    std::vector<objectId> operands_id_scratch_;

    std::size_t NB_{0};
    std::size_t entry_{0};

    std::vector<bool>                                           first_visit_;
    std::vector<program::BasicBlockIR_sptr>                     blocks_;
    std::vector<std::vector<std::size_t>>                       preds_;
    std::unordered_map<program::BasicBlockIR_sptr, std::size_t> index_of_;
};
} // namespace

struct LocalPointsToAnalysis::Impl
{
    Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        run_function_contexts();
    }

  private:
    void run_function_contexts()
    {
        const auto exec_policy = std::execution::seq;
        contexts_.reserve(sala_ir_->get_functions().size());
        auto& pm             = sala_ir_->get_metadata();
        auto& points_to_meta = pm.get<metadata::points_to::ProgramMeta>();
        auto  local_id_start = static_cast<objectId>(points_to_meta.global_objects.size());

        std::size_t func_id = 0;
        for (const auto& function : sala_ir_->get_functions())
        {
            if (function->get_initializer_flag())
            {
                continue;
            }
            contexts_.emplace_back(func_id++, function, local_id_start);
        }
        // first clean run
        std::for_each(exec_policy, contexts_.begin(), contexts_.end(),
                      [](auto& ctx) { ctx.run(); });

        MustState global_must_fixpoint;
        MayState  global_may_fixpoint;
        global_must_fixpoint.reserve(points_to_meta.must_out.size());
        global_may_fixpoint.reserve(points_to_meta.may_out.size());

        const auto fill_fixpoint_info = [&]()
        {
            global_must_fixpoint = points_to_meta.must_out;
            global_may_fixpoint  = points_to_meta.may_out;
            for (auto& ctx : contexts_)
            {
                const auto& [context_global_must, context_global_may] =
                        ctx.get_after_global_states();
                utils::points_to::state_join_or_strict(global_may_fixpoint, context_global_may,
                                                       grouped_objects::CALL_ORDER_DISCREPENCY);
                utils::points_to::state_join_and(global_must_fixpoint, context_global_must);
            }
        };

        fill_fixpoint_info();
        // resolve global points to information until a fixpoint occurs
        while (points_to_meta.must_out != global_must_fixpoint ||
               points_to_meta.may_out != global_may_fixpoint)
        {
            std::for_each(exec_policy, contexts_.begin(), contexts_.end(), [&](auto& ctx)
                          { ctx.resolve(global_must_fixpoint, global_may_fixpoint); });
            fill_fixpoint_info();
            points_to_meta.must_out = global_must_fixpoint;
            points_to_meta.may_out  = global_may_fixpoint;
        }

        std::for_each(contexts_.begin(), contexts_.end(), [&](auto& ctx) { ctx.materialize(); });
    }

  private:
    program::ProgramIR_sptr sala_ir_;

    std::vector<FunctionContext> contexts_;
};

LocalPointsToAnalysis::~LocalPointsToAnalysis() = default;
LocalPointsToAnalysis::LocalPointsToAnalysis()  = default;

program::ProgramIR_sptr LocalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    pImpl_ = std::make_unique<Impl>(sala_ir);
    return sala_ir;
}
} // namespace optimizer::passes
