#include <optimizer/passes/analysis/global_points_to_analysis.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/points_to/import.hpp>

#include <utility/assumptions.hpp>

#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
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
class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir)
        : sala_ir_{std::move(sala_ir)}, static_init_{sala_ir_->get_static_initializer_func()}
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
        solve_static_initializer();
        materialize();
    }

    void init_object_data()
    {
        objectId id = 0;

        for (const auto& constant : sala_ir_->get_constants())
        {
            register_constant_object(constant, id++);
        }

        for (const auto& variable : sala_ir_->get_static_vars())
        {
            register_variable_object(variable, global_objects_, id++, utils::RegionTag::Static);
        }

        for (const auto& parameter : static_init_->get_parameters())
        {
            register_variable_object(parameter, local_objects_, id++, utils::RegionTag::Parameter);
        }

        for (const auto& local : static_init_->get_local_variables())
        {
            register_variable_object(local, local_objects_, id++, utils::RegionTag::Local);
        }

        last_local_id_ = id - 1;
        block_count_   = static_init_->get_basic_blocks().size();
    }

    void register_constant_object(const program::ConstantIR_sptr& constant, objectId id)
    {
        global_objects_.emplace(id, Object{id, utils::RegionTag::Constant});
        operand_to_id_.emplace(constant.get(), id);

        auto points_to_meta = std::make_unique<metadata::points_to::ConstantMeta>();
        points_to_meta->id  = id;
        constant->get_metadata().set(std::move(points_to_meta));
    }

    void register_variable_object(const program::VariableIR_sptr& variable,
                                  utils::ObjectPool& target_mapping, objectId id,
                                  utils::RegionTag region)
    {
        target_mapping.emplace(id, Object{id, region});
        operand_to_id_.emplace(variable.get(), id);

        auto points_to_meta = std::make_unique<metadata::points_to::VariableMeta>();
        points_to_meta->id  = id;
        variable->get_metadata().set(std::move(points_to_meta));
    }

    void build_flattened_cfg()
    {
        collect_blocks();
        build_block_indexing();
        build_predecessor_lists();
        detect_entry_block();
    }

    void collect_blocks()
    {
        blocks_.clear();
        blocks_.reserve(block_count_);

        for (const auto& block : static_init_->get_basic_blocks())
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

    void build_predecessor_lists()
    {
        preds_.assign(block_count_, {});

        for (std::size_t block = 0; block < block_count_; ++block)
        {
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
        const auto entry_block = static_init_->get_entry_basic_block();
        ASSUMPTION(entry_block != nullptr);
        entry_ = block_index(entry_block);
    }

    [[nodiscard]] std::size_t block_index(const program::BasicBlockIR_sptr& block) const
    {
        const auto it = index_of_.find(block);
        ASSUMPTION(it != index_of_.end());
        return it->second;
    }

    void init_states()
    {
        out_may_.reset(block_count_);
        in_may_scratch_.clear();
        first_visit_.assign(block_count_, true);
    }

    void solve_static_initializer()
    {
        if (block_count_ == 0)
        {
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<char>       in_worklist(block_count_, 0);

        worklist.push(entry_);
        in_worklist[entry_] = 1;

        while (!worklist.empty())
        {
            const auto block = pop_worklist(worklist, in_worklist);
            solve_block(block, worklist, in_worklist);
        }
    }

    std::size_t pop_worklist(std::queue<std::size_t>& worklist,
                             std::vector<char>&       in_worklist) const
    {
        const auto block = worklist.front();
        worklist.pop();
        in_worklist[block] = 0;
        return block;
    }

    void solve_block(std::size_t block, std::queue<std::size_t>& worklist,
                     std::vector<char>& in_worklist)
    {
        auto& may = in_may_scratch_;
        build_in_state(block, may);
        apply_block_transfers(block, may);

        if (block_out_changed(block, may))
        {
            mark_block_visited(block);
            commit_block_out_state(block, may);
            enqueue_successors(block, worklist, in_worklist);
        }
        else
        {
            mark_block_visited(block);
        }
    }

    void mark_block_visited(std::size_t block) { first_visit_[block] = false; }

    [[nodiscard]] bool block_out_changed(std::size_t block, const MayAnalysisState& may) const
    {
        return first_visit_[block] || !out_may_.equals(block, may);
    }

    void build_in_state(std::size_t block, MayAnalysisState& may) const
    {
        may.clear();

        bool initialized = false;
        for (const auto pred : preds_[block])
        {
            if (first_visit_[pred])
            {
                continue;
            }

            merge_state(may, out_may_.get(pred), initialized);
        }

        if (!initialized)
        {
            may.clear();
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
        const auto relevant_ops_count = fill_operands_id(instruction);
        ASSUMPTION(operands_id_scratch_.size() >= relevant_ops_count);

        utils::MayTransferContextBundle context{
                .pp             = {.function = 0, .bb = block, .instr = instruction_id},
                .opcode         = instruction->get_opcode(),
                .state          = may,
                .global_objects = global_objects_,
                .local_objects  = local_objects_,
                .operands_id    = operands_id_scratch_,
                .operands_count = relevant_ops_count,
                .last_local_id  = last_local_id_,
        };

        utils::apply_transfer_may(context);
    }

    void commit_block_out_state(std::size_t block, MayAnalysisState& may)
    {
        out_may_.set(block, may);
        may.clear();
    }

    void enqueue_successors(std::size_t block, std::queue<std::size_t>& worklist,
                            std::vector<char>& in_worklist) const
    {
        for (const auto& weak_succ : blocks_[block]->get_successors())
        {
            if (auto succ = weak_succ.lock())
            {
                const auto succ_index = block_index(succ);
                if (in_worklist[succ_index])
                {
                    continue;
                }

                worklist.push(succ_index);
                in_worklist[succ_index] = 1;
            }
        }
    }

    void materialize()
    {
        if (block_count_ == 0)
        {
            return;
        }

        auto program_points_to_meta = std::make_unique<metadata::points_to::ProgramMeta>();

        materialize_basic_blocks_and_exit_summary(*program_points_to_meta);
        attach_static_initializer_metadata();
        attach_program_metadata(std::move(program_points_to_meta));
    }

    void materialize_basic_blocks_and_exit_summary(
            metadata::points_to::ProgramMeta& program_points_to_meta)
    {
        bool first_exit_found = false;

        for (std::size_t block = 0; block < block_count_; ++block)
        {
            MayAnalysisState in_may{};
            build_in_state(block, in_may);
            attach_block_points_to_metadata(block, in_may);

            if (is_exit_block(block))
            {
                merge_exit_block_into_program_summary(block, program_points_to_meta.may_out,
                                                      first_exit_found);
            }
        }
    }

    void attach_block_points_to_metadata(std::size_t block, const MayAnalysisState& in_may)
    {
        auto bb_points_to_meta       = std::make_unique<metadata::points_to::BasicBlockMeta>();
        bb_points_to_meta->may_in_id = out_may_.store()->intern(in_may);
        blocks_[block]->get_metadata().set(std::move(bb_points_to_meta));
    }

    [[nodiscard]] bool is_exit_block(std::size_t block) const
    {
        return blocks_[block]->get_successors().empty();
    }

    void merge_exit_block_into_program_summary(std::size_t block, MayAnalysisState& program_may_out,
                                               bool& first_exit_found) const
    {
        auto exported_exit = export_static_initializer_state(out_may_.get(block));

        if (!first_exit_found)
        {
            program_may_out  = std::move(exported_exit);
            first_exit_found = true;
            return;
        }

        utils::state_join_or_strict(program_may_out, exported_exit);
    }

    [[nodiscard]] MayAnalysisState
    export_static_initializer_state(const MayAnalysisState& source) const
    {
        MayAnalysisState exported{};

        if (source.poisoned)
        {
            utils::poison_may_state(exported);
            return exported;
        }

        for (const auto& [object_id, _] : global_objects_)
        {
            export_global_object_may(object_id, source, exported);
            if (exported.poisoned)
            {
                return exported;
            }
        }

        return exported;
    }

    void export_global_object_may(objectId object_id, const MayAnalysisState& source,
                                  MayAnalysisState& exported) const
    {
        if (source.poisoned)
        {
            utils::poison_may_state(exported);
            return;
        }

        const auto source_iter = source.may.find(object_id);
        if (source_iter == source.may.end())
        {
            return;
        }

        auto projected = project_value_to_global_scope(source_iter->second);
        if (!projected.empty())
        {
            exported.may.insert_or_assign(object_id, std::move(projected));
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
        return global_objects_.contains(id) || id == grouped_objects::HEAP;
    }

    void attach_static_initializer_metadata()
    {
        auto function_meta                = std::make_unique<metadata::points_to::FunctionMeta>();
        function_meta->local_objects      = std::move(local_objects_);
        function_meta->bb_may_state_store = out_may_.store();
        static_init_->get_metadata().set(std::move(function_meta));
    }

    void attach_program_metadata(std::unique_ptr<metadata::points_to::ProgramMeta> program_meta)
    {
        program_meta->global_objects = std::move(global_objects_);
        program_meta->transfer_may_  = utils::apply_transfer_may;
        sala_ir_->get_metadata().set(std::move(program_meta));
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
        const auto& operand = instruction->get_operands().at(position);

        if (std::holds_alternative<program::FunctionIR_raw>(operand))
        {
            return grouped_objects::FUNCTION;
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

    std::size_t block_count_{0};
    std::size_t entry_{0};

    std::vector<bool>                                           first_visit_;
    std::vector<program::BasicBlockIR_sptr>                     blocks_;
    std::vector<std::vector<std::size_t>>                       preds_;
    std::unordered_map<program::BasicBlockIR_sptr, std::size_t> index_of_;
};
} // namespace

void GlobalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    std::cout << "GPA: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir));
    std::cout << "GPA: done" << std::endl;
}
} // namespace optimizer::passes
