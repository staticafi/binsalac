#include <optimizer/passes/analysis/global_points_to_analysis.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/points_to/import.hpp>

#include <queue>
#include <utility/assumptions.hpp>

namespace optimizer::passes
{
using utils::MayAnalysisState;
using utils::MayState;
using utils::Object;
using utils::objectId;

class Impl
{
  public:
    Impl(program::ProgramIR_sptr sala_ir)
        : sala_ir_(std::move(sala_ir)), static_init_{sala_ir_->get_static_initializer_func()}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        ASSUMPTION(static_init_ != nullptr);
        run();
    }

  private:
    void run()
    {
        init_object_data();
        build_flattened_cfg();
        init_states();
        solve_static_intializer();
        materialize();
    }

    void init_object_data()
    {
        objectId id = 0;

        auto assign_object_ids_variables = [&](const program::VariableIRListS& variables,
                                               auto&                           target_mapping,
                                               utils::RegionTag                region) mutable
        {
            for (const auto& variable : variables)
            {
                target_mapping.emplace(id, Object{id, region});
                operand_to_id_.emplace(variable.get(), id);

                auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
                points_to_meta->id  = id++;
                variable->get_metadata().set(std::move(points_to_meta));
            }
        };

        for (const auto& constant : sala_ir_->get_constants())
        {
            global_objects_.emplace(id, Object{id, utils::RegionTag::Constant});
            operand_to_id_.emplace(constant.get(), id);

            auto points_to_meta = std::make_unique<metadata::points_to::ConstantMeta>();
            points_to_meta->id  = id++;
            constant->get_metadata().set(std::move(points_to_meta));
        }

        assign_object_ids_variables(sala_ir_->get_static_vars(), global_objects_,
                                    utils::RegionTag::Static);
        assign_object_ids_variables(static_init_->get_parameters(), local_objects_,
                                    utils::RegionTag::Parameter);
        assign_object_ids_variables(static_init_->get_local_variables(), local_objects_,
                                    utils::RegionTag::Local);

        ASSUMPTION(id > 0);
        last_local_id_ = id - 1;
        NB_            = static_init_->get_basic_blocks().size();
    }

    void build_flattened_cfg()
    {
        blocks_.clear();
        blocks_.reserve(NB_);
        for (const auto& bb : static_init_->get_basic_blocks())
        {
            blocks_.push_back(bb);
        }

        index_of_.clear();
        index_of_.reserve(NB_);
        for (std::size_t i = 0; i < NB_; ++i)
        {
            index_of_[blocks_[i]] = i;
        }

        preds_.assign(NB_, {});
        for (std::size_t i = 0; i < NB_; ++i)
        {
            for (const auto& wp : blocks_[i]->get_predecessors())
            {
                auto sp = wp.lock();
                ASSUMPTION(sp != nullptr);

                const auto it = index_of_.find(sp);
                ASSUMPTION(it != index_of_.end());
                preds_[i].push_back(it->second);
            }
        }

        entry_        = 0;
        const auto eb = static_init_->get_entry_basic_block();
        ASSUMPTION(eb != nullptr);

        const auto it = index_of_.find(eb);
        ASSUMPTION(it != index_of_.end());
        entry_ = it->second;
    }

    void init_states()
    {
        out_may_.reset(NB_);
        in_may_scratch_.clear();
        first_visit_.assign(NB_, true);
    }

    void solve_static_intializer()
    {
        if (NB_ == 0)
        {
            return;
        }

        std::queue<std::size_t> wl;
        std::vector<char>       in_wl(NB_, 0);

        wl.push(entry_);
        in_wl[entry_] = 1;

        while (!wl.empty())
        {
            const std::size_t b = wl.front();
            wl.pop();
            in_wl[b] = 0;

            auto& may        = in_may_scratch_;
            bool  first_pred = false;

            may.clear();

            for (const auto p : preds_[b])
            {
                if (first_visit_[p])
                {
                    continue;
                }

                if (!first_pred)
                {
                    may        = out_may_.get(p);
                    first_pred = true;
                }
                else
                {
                    utils::state_join_or_strict(may, out_may_.get(p));
                }
            }

            if (!first_pred)
            {
                may.clear();
            }

            std::size_t instruction_id = 0;
            for (const auto& instruction : blocks_[b]->get_instructions())
            {
                const auto relevant_ops_count = fill_operands_id(instruction);
                ASSUMPTION(operands_id_scratch_.size() >= relevant_ops_count);

                utils::MayTransferContextBundle may_transfer_context{
                        .pp             = {.function = 0, .bb = b, .instr = instruction_id},
                        .opcode         = instruction->get_opcode(),
                        .state          = may,
                        .global_objects = global_objects_,
                        .local_objects  = local_objects_,
                        .operands_id    = operands_id_scratch_,
                        .operands_count = relevant_ops_count,
                        .last_local_id  = last_local_id_,
                };

                utils::apply_transfer_may(may_transfer_context);
                ++instruction_id;

                if (may.poisoned)
                {
                    break;
                }
            }

            const bool changed = first_visit_[b] || !out_may_.equals(b, may);

            first_visit_[b] = false;
            if (changed)
            {
                out_may_.set(b, may);
                may.clear();

                for (const auto& ws : blocks_[b]->get_successors())
                {
                    if (auto s = ws.lock())
                    {
                        const auto it = index_of_.find(s);
                        ASSUMPTION(it != index_of_.end());

                        if (!in_wl[it->second])
                        {
                            wl.push(it->second);
                            in_wl[it->second] = 1;
                        }
                    }
                }
            }
        }
    }

    void materialize()
    {
        if (NB_ == 0)
        {
            return;
        }

        auto program_points_to_meta = std::make_unique<metadata::points_to::ProgramMeta>();

        bool first_end_found = false;

        for (std::size_t b = 0; b < NB_; ++b)
        {
            MayAnalysisState in_may{};

            for (const auto p : preds_[b])
            {
                if (first_visit_[p])
                {
                    continue;
                }

                if (!in_may.poisoned && !out_may_.get(p).poisoned && in_may.may.empty())
                {
                    in_may = out_may_.get(p);
                }
                else
                {
                    utils::state_join_or_strict(in_may, out_may_.get(p));
                }
            }

            auto bb_points_to_meta       = std::make_unique<metadata::points_to::BasicBlockMeta>();
            bb_points_to_meta->may_in_id = out_may_.store()->intern(in_may);
            blocks_[b]->get_metadata().set(std::move(bb_points_to_meta));

            std::size_t instruction_id = 0;
            for (const auto& instruction : blocks_[b]->get_instructions())
            {
                const auto relevant_ops_size = fill_operands_id(instruction);

                utils::MayTransferContextBundle may_transfer_context{
                        .pp             = {.function = 0, .bb = b, .instr = instruction_id},
                        .opcode         = instruction->get_opcode(),
                        .state          = in_may,
                        .global_objects = global_objects_,
                        .local_objects  = local_objects_,
                        .operands_id    = operands_id_scratch_,
                        .operands_count = relevant_ops_size,
                        .last_local_id  = last_local_id_,
                };

                utils::apply_transfer_may(may_transfer_context);
                ++instruction_id;

                if (in_may.poisoned)
                {
                    break;
                }
            }

            if (blocks_[b]->get_successors().empty())
            {
                if (first_end_found)
                {
                    utils::state_join_or_strict(program_points_to_meta->may_out, out_may_.get(b));
                }
                else
                {
                    program_points_to_meta->may_out = out_may_.get(b);
                    first_end_found                 = true;
                }
            }
        }

        auto function_meta                = std::make_unique<metadata::points_to::FunctionMeta>();
        function_meta->local_objects      = std::move(local_objects_);
        function_meta->bb_may_state_store = out_may_.store();
        static_init_->get_metadata().set(std::move(function_meta));

        program_points_to_meta->global_objects = std::move(global_objects_);
        program_points_to_meta->transfer_may_  = utils::apply_transfer_may;
        sala_ir_->get_metadata().set(std::move(program_points_to_meta));
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
        const auto& operand = instruction->get_operands().at(position);

        if (std::holds_alternative<program::FunctionIR_raw>(operand))
        {
            return utils::grouped_objects::FUNCTION;
        }

        const auto operand_id_iter = operand_to_id_.find(operand);
        ASSUMPTION(operand_id_iter != operand_to_id_.end());
        return operand_id_iter->second;
    }

  private:
    program::ProgramIR_sptr  sala_ir_;
    program::FunctionIR_sptr static_init_;

    MayAnalysisState in_may_scratch_;

    utils::ObjectPool global_objects_;
    utils::ObjectPool local_objects_;

    std::unordered_map<program::OperandIR_raw, objectId> operand_to_id_{};

    std::vector<objectId> operands_id_scratch_;
    objectId              last_local_id_{0};

    utils::BasicBlockMayStateSlots out_may_;

    std::size_t N_{0}, NB_{0};
    std::size_t entry_{0};

    std::vector<bool>                                           first_visit_;
    std::vector<program::BasicBlockIR_sptr>                     blocks_;
    std::vector<std::vector<std::size_t>>                       preds_;
    std::unordered_map<program::BasicBlockIR_sptr, std::size_t> index_of_;
};

void GlobalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    std::cout << "GPA: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir));
    std::cout << "GPA: done" << std::endl;
}
} // namespace optimizer::passes
