#include <optimizer/passes/transformation/dead_instruction_elimination.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/query/liveness_query.hpp>

#include <utility/assumptions.hpp>
#include <utility/timeprof.hpp>

#include <utility>
#include <vector>

namespace optimizer::passes
{
namespace
{

using PendingRemoval = std::vector<program::InstructionIR_sptr>;

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

    if (function->get_external_flag())
    {
        return;
    }

    query::LivenessQueryFunction query(function);

    for (const auto& basic_block : function->get_basic_blocks())
    {
        collect_removable_instructions_in_basic_block(basic_block, query, removals);
    }
}

void collect_removable_instructions(const program::ProgramIR_sptr& sala_ir,
                                    PendingRemoval&                removals)
{
    ASSUMPTION(sala_ir != nullptr);

    for (const auto& function : sala_ir->get_functions())
    {
        if (function == nullptr)
        {
            continue;
        }

        collect_removable_instructions_in_function(function, removals);
    }
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
    for (const auto& instruction : removals)
    {
        remove_instruction(instruction);
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
    void run()
    {
        PendingRemoval removals;

        collect_removable_instructions(sala_ir_, removals);
        remove_instructions(removals);
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void DeadInstructionElimination::run(program::ProgramIR_sptr sala_ir)
{
    TMPROF_BLOCK();
    const auto trigger = Impl(std::move(sala_ir));
}

} // namespace optimizer::passes
