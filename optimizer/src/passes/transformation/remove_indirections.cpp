#include <optimizer/passes/transformation/remove_indirections.hpp>

#include <optimizer/analysis/points_to_query.hpp>
#include <optimizer/metadata/points_to.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>

#include <utility/development.hpp>

#include <execution>

namespace optimizer::passes
{
using MetaKey = metadata::MetaKey;
namespace
{
struct TransformationContext
{
    program::InstructionIR&     instruction;
    program::OperandIRVecR_iter indirection_iter;
    program::OperandIR_raw      target;
};

class FunctionContext
{
  public:
    explicit FunctionContext(program::FunctionIR_sptr function_ir)
        : function_{std::move(function_ir)}, function_query_(function_ir)
    {
    }

    void run()
    {
        if (function_->get_initializer_flag())
        {
            // TODO: consider how
            return;
        }
        for (const auto& bb_ptr : function_->get_basic_blocks())
        {
            resolve_basic_block(*bb_ptr);
        }
    };

  private:
    void resolve_basic_block(const program::BasicBlockIR& basic_block)
    {
        const auto& points_to_meta =
                basic_block.get_metadata().get<metadata::points_to::BasicBlockMeta>();

        for (const auto& instr_ptr : basic_block.get_instructions())
        {
            switch (instr_ptr->get_opcode())
            {
            case sala::Instruction::Opcode::LOAD:
                queue_replace_load(instr_ptr);
                break;
            case sala::Instruction::Opcode::STORE:
                queue_replace_store(instr_ptr);
                break;
            default:
                break;
            }

            // we have to perform the transformations at the end of the processed basic_block
            process_bb_queue();
        }
    }

    void process_bb_queue()
    {
        for (const auto& transformation : to_transform_bb_queue)
        {
            transformation.instruction.get_opcode() = sala::Instruction::Opcode::COPY;
            *transformation.indirection_iter        = transformation.target;
        }
    }

    void queue_replace_load(const program::InstructionIR_sptr& instruction)
    {
        queue_replace(instruction, std::next(instruction->get_operands().begin()));
    }

    void queue_replace_store(const program::InstructionIR_sptr& instruction)
    {
        queue_replace(instruction, instruction->get_operands().begin());
    }

    void queue_replace(const program::InstructionIR_sptr& instruction,
                       const program::OperandIRVecR_iter  indirect_operand_iter)
    {
        const auto& indirect_operand = *indirect_operand_iter;
        if (!std::holds_alternative<program::VariableIR_raw>(indirect_operand))
        {
            return;
        }

        const auto& indirect_variable = *(std::get<program::VariableIR_raw>(indirect_operand));
        if (indirect_variable.get_context() != program::VariableIR::Context::LOCAL)
        {
            // TODO: think about replacing also not local variables
            // - we can have *ptr_to_static = x, then we can perform static = x ?
            return;
        }

        const auto points_to_result = function_query_.before(instruction, indirect_variable);
        if (!points_to_result.must.has_value())
        {
            return;
        }

        if (points_to_result.must->offset_flag)
        {
            return;
        }

        const auto target_operand = function_query_.get_object(points_to_result.must->id);
        if (!target_operand.has_value())
        {
            return;
        }

        // FIXME: fix this
        //  to_transform_bb_queue.emplace_back(instruction, indirect_operand_iter,
        //                                     target_operand.value());
    }

  private:
    program::FunctionIR_sptr        function_;
    analysis::PointsToQueryFunction function_query_;

    std::vector<TransformationContext> to_transform_bb_queue;
};
} // namespace

class RemoveIndirections::Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr salal_ir) : sala_ir_{std::move(salal_ir)}
    {
        run_function_contexts();
    };

  private:
    void run_function_contexts()
    {

        const auto exec_policy = std::execution::seq;
        contexts_.reserve(sala_ir_->get_functions().size());

        for (const auto& function : sala_ir_->get_functions())
        {
            if (function->get_initializer_flag())
            {
                // TODO: currently ignored look into correctness
                continue;
            }
            contexts_.emplace_back(function);
        }

        std::for_each(exec_policy, contexts_.begin(), contexts_.end(),
                      [](auto& ctx) { ctx.run(); });
    };

  private:
    program::ProgramIR_sptr sala_ir_;

    std::vector<FunctionContext> contexts_;
};

RemoveIndirections::~RemoveIndirections() = default;
RemoveIndirections::RemoveIndirections()  = default;

program::ProgramIR_sptr RemoveIndirections::run(program::ProgramIR_sptr sala_ir)
{
    ASSUMPTION(sala_ir->get_metadata().has<metadata::points_to::ProgramMeta>());
    pImpl_ = std::make_unique<Impl>(sala_ir);
    return sala_ir;
}
} // namespace optimizer::passes
