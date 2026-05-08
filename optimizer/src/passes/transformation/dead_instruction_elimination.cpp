#include <optimizer/passes/transformation/dead_instruction_elimination.hpp>

#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/query/liveness_query.hpp>
#include <optimizer/query/translation_query.hpp>

#include <utility/assumptions.hpp>
#include <utility/log.hpp>
#include <utility/timeprof.hpp>

#include <sstream>
#include <utility>
#include <vector>

namespace optimizer::passes
{
namespace
{

using PendingRemoval = std::vector<program::InstructionIR_sptr>;

std::string me()
{
    std::ostringstream oss;
    oss << pipeline::names::dead_instruction_elimination;
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

void collect_removable_instructions_in_basic_block(const program::BasicBlockIR_sptr&   basic_block,
                                                   const query::LivenessQueryFunction& query,
                                                   PendingRemoval&                     removals)
{
    ASSUMPTION(basic_block != nullptr);

    const auto& removable_instructions = query.removable_instructions(basic_block);

    if (removable_instructions.empty())
    {
        return;
    }

    for (const auto& instruction : basic_block->get_instructions())
    {
        ASSUMPTION(instruction != nullptr);

        if (removable_instructions.contains(instruction.get()))
        {
            removals.push_back(instruction);
        }
    }
}

void collect_removable_instructions_in_function(const program::FunctionIR_sptr& function,
                                                PendingRemoval&                 removals)
{
    ASSUMPTION(function != nullptr);

    LOG(LSL_DEBUG, me() << info(function) << "Collecting removable instructions");

    if (function->get_external_flag())
    {
        LOG(LSL_DEBUG, me() << info(function) << "Skipping external function");
        return;
    }

    const auto previous_count = removals.size();

    query::LivenessQueryFunction query(function);

    for (const auto& basic_block : function->get_basic_blocks())
    {
        collect_removable_instructions_in_basic_block(basic_block, query, removals);
    }

    LOG(LSL_DEBUG, me() << info(function) << "Collected removable instructions: count="
                        << (removals.size() - previous_count));
}

void collect_removable_instructions(const program::ProgramIR_sptr& sala_ir,
                                    PendingRemoval&                removals)
{
    ASSUMPTION(sala_ir != nullptr);

    LOG(LSL_DEBUG, me() << "Collecting removable instructions");

    for (const auto& function : sala_ir->get_functions())
    {
        if (function == nullptr)
        {
            continue;
        }

        collect_removable_instructions_in_function(function, removals);
    }

    LOG(LSL_DEBUG, me() << "Collected removable instructions: total=" << removals.size());
}

void remove_instruction(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto basic_block = instruction->get_basic_block();
    ASSUMPTION(basic_block != nullptr);

    basic_block->release_instruction(instruction);
}

void remove_instructions(const PendingRemoval& removals)
{
    LOG(LSL_DEBUG, me() << "Removing instructions: count=" << removals.size());

    for (const auto& instruction : removals)
    {
        remove_instruction(instruction);
    }

    LOG(LSL_DEBUG, me() << "Removed instructions");
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
    void run()
    {
        LOG(LSL_DEBUG, me() << "Running implementation");

        PendingRemoval removals;

        collect_removable_instructions(sala_ir_, removals);
        remove_instructions(removals);

        LOG(LSL_DEBUG, me() << "Done implementation");
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void DeadInstructionElimination::run(program::ProgramIR_sptr sala_ir)
{
    LOG(LSL_INFO, me() << "Running");
    {
        TMPROF_BLOCK();
        const auto trigger = Impl(std::move(sala_ir));
    }
    LOG(LSL_INFO, me() << "Done");
}

} // namespace optimizer::passes
