#include <optimizer/passes/transformation/dead_variables_elimination.hpp>

#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/query/translation_query.hpp>

#include <optimizer/utils/sparse_set.hpp>

#include <utility/assumptions.hpp>
#include <utility/log.hpp>
#include <utility/timeprof.hpp>

#include <sstream>
#include <vector>

namespace optimizer::passes
{
namespace
{

using PendingRemoval = std::vector<program::VariableIR_sptr>;

std::string me()
{
    std::ostringstream oss;
    oss << pipeline::names::dead_variables_elimination;
    oss << ": ";
    return oss.str();
}

std::string info(const program::FunctionIR_sptr& function)
{
    ASSUMPTION(function != nullptr);

    const auto program = function->get_program();
    ASSUMPTION(program != nullptr);

    query::TranslationQuery translation{program};

    auto name = translation.function_name(function);
    if (name.empty())
    {
        name = "<unnamed>";
    }

    std::ostringstream oss;
    oss << "[" << name << "] ";
    return oss.str();
}

inline void collect_removable_variables_in_function(const program::FunctionIR_sptr& function,
                                                    PendingRemoval&                 removals)
{
    LOG(LSL_DEBUG, me() << info(function) << "Collecting removable variables");

    const auto previous_count = removals.size();

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

    LOG(LSL_DEBUG, me() << info(function) << "Collected removable variables: count="
                        << (removals.size() - previous_count));
}

inline void remove_variables(PendingRemoval& removals)
{
    LOG(LSL_DEBUG, me() << "Removing variables: count=" << removals.size());

    while (!removals.empty())
    {
        program::release_variable(removals.back());
        removals.pop_back();
    }

    LOG(LSL_DEBUG, me() << "Removed variables");
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
        LOG(LSL_DEBUG, me() << "Running implementation");

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

        LOG(LSL_DEBUG, me() << "Done implementation");

        return sala_ir_;
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void DeadVariablesElimination::run(program::ProgramIR_sptr sala_ir)
{
    LOG(LSL_INFO, me() << "Running");
    {
        TMPROF_BLOCK();
        const auto trigger = Impl(std::move(sala_ir));
    }
    LOG(LSL_INFO, me() << "Done");
}

} // namespace optimizer::passes
