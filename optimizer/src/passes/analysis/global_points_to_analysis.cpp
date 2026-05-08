#include <optimizer/passes/analysis/global_points_to_analysis.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/query/translation_query.hpp>
#include <optimizer/utils/points_to/import.hpp>
#include <optimizer/utils/view/flattened_cfg_view.hpp>

#include <utility/assumptions.hpp>
#include <utility/log.hpp>
#include <utility/timeprof.hpp>

#include <memory>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace optimizer::passes
{
namespace abstract_nodes = utils::abstract_nodes;
using utils::MayAnalysisState;
using utils::MayState;
using utils::Object;
using utils::objectId;

namespace
{

std::string me()
{
    std::ostringstream oss;
    oss << pipeline::names::global_points_to;
    oss << ": ";
    return oss.str();
}

class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir)
        : sala_ir_{std::move(sala_ir)}, static_init_{sala_ir_->get_static_initializer_func()},
          cfg_{static_init_}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        ASSUMPTION(static_init_ != nullptr);
        run();
    }

  private:
    std::string info() const
    {
        query::TranslationQuery translation{sala_ir_};

        auto name = translation.function_name(static_init_);
        if (name.empty())
        {
            name = "<static-initializer>";
        }

        std::ostringstream oss;
        oss << "[" << name << "] ";
        return oss.str();
    }

    void run()
    {
        LOG(LSL_DEBUG, me() << info() << "Running static initializer context");

        init_object_data();
        init_states();
        solve_static_initializer();
        materialize();

        LOG(LSL_DEBUG, me() << info() << "Done static initializer context");
    }

    void init_object_data()
    {
        LOG(LSL_DEBUG, me() << info() << "Initializing object data");

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

        LOG(LSL_DEBUG, me() << info() << "Initialized object data: global_objects="
                            << global_objects_.size() << ", local_objects=" << local_objects_.size()
                            << ", last_local_id=" << last_local_id_);
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

    void init_states()
    {
        LOG(LSL_DEBUG, me() << info() << "Initializing states");

        out_may_.reset(cfg_.size());
        in_may_scratch_.clear();
        first_visit_.assign(cfg_.size(), true);

        LOG(LSL_DEBUG, me() << info() << "Initialized states: blocks=" << cfg_.size());
    }

    void solve_static_initializer()
    {
        LOG(LSL_DEBUG, me() << info() << "Solving static initializer");

        if (cfg_.size() == 0)
        {
            LOG(LSL_DEBUG, me() << info() << "Skipping solve: empty CFG");
            return;
        }

        std::queue<std::size_t> worklist;
        std::vector<char>       in_worklist(cfg_.size(), 0);

        worklist.push(cfg_.entry());
        in_worklist[cfg_.entry()] = 1;

        std::size_t iterations = 0;

        while (!worklist.empty())
        {
            ++iterations;
            const auto block = pop_worklist(worklist, in_worklist);
            solve_block(block, worklist, in_worklist);
        }

        LOG(LSL_DEBUG, me() << info() << "Solved static initializer: iterations=" << iterations);
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

    bool block_out_changed(std::size_t block, const MayAnalysisState& may) const
    {
        return first_visit_[block] || !out_may_.equals(block, may);
    }

    void build_in_state(std::size_t block, MayAnalysisState& may) const
    {
        may.clear();

        bool initialized = false;
        for (const auto pred : cfg_.predecessors(block))
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

    void apply_block_transfers(std::size_t block, MayAnalysisState& may)
    {
        std::size_t instruction_id = 0;

        for (const auto& instruction : cfg_.block(block)->get_instructions())
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
        LOG(LSL_DEBUG, me() << info() << "Materializing metadata");

        if (cfg_.empty())
        {
            LOG(LSL_DEBUG, me() << info() << "Skipping materialize: empty CFG");
            return;
        }

        auto program_points_to_meta = std::make_unique<metadata::points_to::ProgramMeta>();

        materialize_basic_blocks_and_exit_summary(*program_points_to_meta);
        attach_static_initializer_metadata();
        attach_program_metadata(std::move(program_points_to_meta));

        LOG(LSL_DEBUG, me() << info() << "Materialized metadata");
    }

    void materialize_basic_blocks_and_exit_summary(
            metadata::points_to::ProgramMeta& program_points_to_meta)
    {
        bool first_exit_found = false;

        for (std::size_t block = 0; block < cfg_.size(); ++block)
        {
            MayAnalysisState in_may{};
            build_in_state(block, in_may);
            attach_block_points_to_metadata(block, in_may);

            if (cfg_.is_exit_block(block))
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
        cfg_.block(block)->get_metadata().set(std::move(bb_points_to_meta));
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

        utils::state_join_cfg(program_may_out, exported_exit);
    }

    MayAnalysisState export_static_initializer_state(const MayAnalysisState& source) const
    {
        MayAnalysisState exported{};

        if (source.poisoned)
        {
            utils::poison_may_state(exported);
            return exported;
        }

        for (const auto& [object_id, _] : global_objects_)
        {
            export_global_object_state_cell(object_id, source, exported);
            if (exported.poisoned)
            {
                return exported;
            }
        }

        export_global_object_state_cell(abstract_nodes::HEAP, source, exported);
        return exported;
    }

    void export_global_object_state_cell(objectId object_id, const MayAnalysisState& source,
                                         MayAnalysisState& exported) const
    {
        if (source.poisoned)
        {
            utils::poison_may_state(exported);
            return;
        }

        const auto source_iter = source.may.find(object_id);
        if (source_iter != source.may.end())
        {
            auto projected = project_value_to_global_scope(source_iter->second);
            if (!projected.empty())
            {
                exported.may.insert_or_assign(object_id, std::move(projected));
            }
        }

        const auto source_must_iter = source.must.find(object_id);
        if (source_must_iter != source.must.end())
        {
            if (auto projected_must = project_must_to_global_scope(source_must_iter->second);
                projected_must.has_value())
            {
                exported.must.insert_or_assign(object_id, projected_must.value());
            }
        }
    }

    std::optional<utils::Target> project_must_to_global_scope(const utils::Target& target) const
    {
        if (!target.offset_flag && global_objects_.contains(target.id))
        {
            return target;
        }

        return std::nullopt;
    }

    utils::MayValue project_value_to_global_scope(const utils::MayValue& value) const
    {
        if (value.is_top)
        {
            return utils::MayValue::top();
        }

        utils::MayValue projected{};

        for (const auto& target : value)
        {
            if (is_globally_visible_target(global_objects_, target.id))
            {
                projected.insert(target);
            }
            else
            {
                projected.insert(utils::Target{abstract_nodes::OUT_OF_GLOBAL_SCOPE, false});
            }
        }

        return projected;
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
            return abstract_nodes::FUNCTION;
        }

        const auto operand_id_iter = operand_to_id_.find(operand);
        ASSUMPTION(operand_id_iter != operand_to_id_.end());
        return operand_id_iter->second;
    }

  private:
    program::ProgramIR_sptr       sala_ir_;
    program::FunctionIR_sptr      static_init_;
    utils::view::FlattenedCFGView cfg_;

    MayAnalysisState in_may_scratch_;

    utils::ObjectPool global_objects_;
    utils::ObjectPool local_objects_;

    std::unordered_map<program::OperandIR_raw, objectId> operand_to_id_{};

    std::vector<objectId> operands_id_scratch_;
    objectId              last_local_id_{0};

    utils::BasicBlockMayStateSlots out_may_;

    std::vector<bool> first_visit_;
};

} // namespace

void GlobalPointsToAnalysis::run(program::ProgramIR_sptr sala_ir)
{
    LOG(LSL_INFO, me() << "Running");
    {
        TMPROF_BLOCK()
        const auto trigger = Impl(std::move(sala_ir));
    }
    LOG(LSL_INFO, me() << "Done");
}

} // namespace optimizer::passes
