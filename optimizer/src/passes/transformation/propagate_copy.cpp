#include <optimizer/passes/transformation/propagate_copy.hpp>

#include <optimizer/passes/analysis/available_copy_analysis.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/query/available_copy_query.hpp>
#include <optimizer/utils/available_copy/import.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <vector>

namespace optimizer::passes
{
namespace
{
struct PendingTransform
{
    program::InstructionIR_sptr instruction;
    std::size_t                 operand_index;
    program::VariableIR_sptr    replacement;
};

bool is_variable_use_operand(const program::InstructionIR_sptr& instruction,
                             const std::size_t                  operand_index)
{
    ASSUMPTION(instruction != nullptr);

    if (utils::available_copy::is_written_operand(instruction->get_opcode(), operand_index))
    {
        return false;
    }

    const auto& operands = instruction->get_operands();
    if (operand_index >= operands.size())
    {
        return false;
    }

    const auto* variable = std::get_if<program::VariableIR_raw>(&operands[operand_index]);
    return variable != nullptr && *variable != nullptr;
}

std::optional<PendingTransform> compute_replacement(const program::InstructionIR_sptr& instruction,
                                                    const std::size_t operand_index,
                                                    query::AvailableCopyQueryFunction& query)
{
    ASSUMPTION(instruction != nullptr);

    if (!is_variable_use_operand(instruction, operand_index))
    {
        return std::nullopt;
    }

    const auto& operands = instruction->get_operands();
    const auto* variable = std::get_if<program::VariableIR_raw>(&operands[operand_index]);
    ASSUMPTION(variable != nullptr);
    ASSUMPTION(*variable != nullptr);

    const auto result = query.before(instruction, **variable);
    if (!result.transitive_source.has_value())
    {
        return std::nullopt;
    }

    const auto& replacement = result.transitive_source.value();
    if (replacement == nullptr || replacement.get() == *variable)
    {
        return std::nullopt;
    }

    ASSUMPTION(replacement->get_num_bytes() == (*variable)->get_num_bytes());

    return PendingTransform{
            .instruction   = instruction,
            .operand_index = operand_index,
            .replacement   = replacement,
    };
}

void apply_replacements(std::vector<PendingTransform>& replacements)
{
    while (!replacements.empty())
    {
        const auto& item = replacements.back();
        ASSUMPTION(item.instruction != nullptr);
        ASSUMPTION(item.replacement != nullptr);
        ASSUMPTION(item.operand_index < item.instruction->get_operands().size());

        item.instruction->get_operands()[item.operand_index] = item.replacement.get();
        replacements.pop_back();
    }
}

void propagate_in_function(const program::FunctionIR_sptr& function)
{
    ASSUMPTION(function != nullptr);

    if (function->get_external_flag())
    {
        return;
    }

    query::AvailableCopyQueryFunction query(function);
    std::vector<PendingTransform>     replacements;

    for (const auto& basic_block : function->get_basic_blocks())
    {
        ASSUMPTION(basic_block != nullptr);

        for (const auto& instruction : basic_block->get_instructions())
        {
            ASSUMPTION(instruction != nullptr);

            const auto operand_count = instruction->get_operands().size();
            for (std::size_t i = 0; i < operand_count; ++i)
            {
                if (auto replacement = compute_replacement(instruction, i, query);
                    replacement.has_value())
                {
                    replacements.push_back(std::move(replacement.value()));
                }
            }
        }
    }

    apply_replacements(replacements);
}

class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        run();
    }

    program::ProgramIR_sptr result() const { return sala_ir_; }

  private:
    void run()
    {
        for (const auto& function : sala_ir_->get_functions())
        {
            ASSUMPTION(function != nullptr);
            ASSUMPTION(function->get_metadata().has<metadata::available_copy::FunctionMeta>());
            if (function == nullptr)
            {
                continue;
            }

            propagate_in_function(function);
        }
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};
} // namespace

void PropagateCopy::run(program::ProgramIR_sptr sala_ir)
{
    TMPROF_BLOCK();
    const auto trigger = Impl(std::move(sala_ir));
}
} // namespace optimizer::passes
