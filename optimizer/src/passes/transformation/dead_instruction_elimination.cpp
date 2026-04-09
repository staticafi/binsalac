#include <optimizer/passes/transformation/dead_instruction_elimination.hpp>

#include <optimizer/analysis/liveness_query.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>

#include <utility/assumptions.hpp>

#include <vector>

namespace optimizer::passes
{
namespace
{
using PendingRemoval = std::vector<program::InstructionIR_sptr>;

inline void collect_removable_instructions_in_function(const program::FunctionIR_sptr& function,
                                                       PendingRemoval&                 removals)
{
    ASSUMPTION(function != nullptr);

    if (function->get_external_flag())
    {
        return;
    }

    analysis::LivenessQueryFunction query(function);

    for (const auto& basic_block : function->get_basic_blocks())
    {
        ASSUMPTION(basic_block != nullptr);

        for (const auto& instruction : basic_block->get_instructions())
        {
            ASSUMPTION(instruction != nullptr);

            if (query.is_removable(instruction))
            {
                removals.push_back(instruction);
            }
        }
    }
}

inline void remove_instructions(const PendingRemoval& removals)
{
    for (const auto& instruction : removals)
    {
        ASSUMPTION(instruction != nullptr);

        const auto basic_block = instruction->get_basic_block();
        ASSUMPTION(basic_block != nullptr);

        basic_block->release_instruction(instruction);
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

            collect_removable_instructions_in_function(function, removals);
        }

        remove_instructions(removals);
        return sala_ir_;
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};
} // namespace

void DeadInstructionElimination::run(program::ProgramIR_sptr sala_ir)
{
    std::cout << "DIE: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir));
    std::cout << "DIE: done" << std::endl;
}

} // namespace optimizer::passes
