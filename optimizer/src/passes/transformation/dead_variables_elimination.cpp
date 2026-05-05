#include <optimizer/passes/transformation/dead_variables_elimination.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <optimizer/utils/sparse_set.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <vector>

namespace optimizer::passes
{
namespace
{
using PendingRemoval = std::vector<program::VariableIR_sptr>;

inline void collect_removable_variables_in_function(const program::FunctionIR_sptr& function,
                                                    PendingRemoval&                 removals)
{
    utils::SparseSet<program::VariableIR_raw> used;
    used.reserve(function->get_local_variables().size());

    for (const auto& basic_block : function->get_basic_blocks())
    {
        for (const auto& instruction : basic_block->get_instructions())
        {
            for (const auto operand : instruction->get_operands())
            {
                if (auto variable_ir_raw = std::get_if<program::VariableIR_raw>(&operand);
                    variable_ir_raw &&
                    (*variable_ir_raw)->get_context() == program::VariableIR::Context::LOCAL)
                {
                    used.emplace(*variable_ir_raw);
                }
            }
        }
    }

    for (const auto& local_variable : function->get_local_variables())
    {
        if (!used.contains(local_variable.get()))
        {
            removals.push_back(local_variable);
        }
    }
}

inline void remove_variables(PendingRemoval& removals)
{
    while (!removals.empty())
    {
        program::release_variable(removals.back());
        removals.pop_back();
    }
}

class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        run();
    }

  private:
    program::ProgramIR_sptr run()
    {
        PendingRemoval removals;

        for (const auto& function : sala_ir_->get_functions())
        {
            if (function == nullptr)
            {
                continue;
            }

            collect_removable_variables_in_function(function, removals);
        }

        remove_variables(removals);
        return sala_ir_;
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};
} // namespace

void DeadVariablesElimination::run(program::ProgramIR_sptr sala_ir)
{
    TMPROF_BLOCK();
    const auto trigger = Impl(std::move(sala_ir));
}

} // namespace optimizer::passes
