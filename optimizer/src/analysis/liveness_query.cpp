#include <optimizer/analysis/liveness_query.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/liveness/import.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::analysis
{

namespace
{

bool live_set_contains(const utils::LiveSet& live, const program::VariableIR& variable)
{
    const auto self_iter = variable.get_self_it();
    ASSUMPTION(self_iter.has_value());
    return live.contains(self_iter.value()->get());
}

bool live_set_contains(const utils::LiveSet& live, const program::ConstantIR& constant)
{
    const auto self_iter = constant.get_self_it();
    ASSUMPTION(self_iter.has_value());
    return live.contains(self_iter.value()->get());
}

} // namespace

LivenessQueryFunction::LivenessQueryFunction(program::FunctionIR_csptr function)
    : program_keepalive_{}, function_{std::move(function)}
{
    ASSUMPTION(function_ != nullptr);

    program_keepalive_ = function_->get_program();
    ASSUMPTION(program_keepalive_ != nullptr);
}

const metadata::liveness::BasicBlockMeta&
LivenessQueryFunction::get_basic_block_meta(const program::BasicBlockIR_csptr& basic_block) const
{
    ASSUMPTION(basic_block != nullptr);
    ASSUMPTION(basic_block->get_function_raw() == function_.get());
    ASSUMPTION(basic_block->get_metadata().has<metadata::liveness::BasicBlockMeta>());

    return basic_block->get_metadata().get<metadata::liveness::BasicBlockMeta>();
}

utils::LiveSet
LivenessQueryFunction::live_after(const program::InstructionIR_sptr& instruction) const
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    const auto& bb_meta = get_basic_block_meta(instruction->get_basic_block());

    utils::LiveSet live = bb_meta.live_out;

    const auto& instructions = bb_raw->get_instructions();
    for (auto it = instructions.rbegin(); it != instructions.rend(); ++it)
    {
        ASSUMPTION(*it != nullptr);

        if (it->get() == instruction.get())
        {
            return live;
        }

        utils::LiveSet live_before;
        utils::apply_backward_transfer(*it, live, live_before);
        live = std::move(live_before);
    }

    ASSUMPTION(false);
}

utils::LiveSet
LivenessQueryFunction::live_before(const program::InstructionIR_sptr& instruction) const
{
    ASSUMPTION(instruction != nullptr);

    auto after = live_after(instruction);

    utils::LiveSet before;
    utils::apply_backward_transfer(instruction, after, before);
    return before;
}

bool LivenessQueryFunction::is_live_before(const program::InstructionIR_sptr& instruction,
                                           const program::VariableIR&         variable) const
{
    const auto live = live_before(instruction);
    return live_set_contains(live, variable);
}

bool LivenessQueryFunction::is_live_after(const program::InstructionIR_sptr& instruction,
                                          const program::VariableIR&         variable) const
{
    const auto live = live_after(instruction);
    return live_set_contains(live, variable);
}

bool LivenessQueryFunction::is_live_before(const program::InstructionIR_sptr& instruction,
                                           const program::ConstantIR&         constant) const
{
    const auto live = live_before(instruction);
    return live_set_contains(live, constant);
}

bool LivenessQueryFunction::is_live_after(const program::InstructionIR_sptr& instruction,
                                          const program::ConstantIR&         constant) const
{
    const auto live = live_after(instruction);
    return live_set_contains(live, constant);
}

bool LivenessQueryFunction::is_removable(const program::InstructionIR_sptr& instruction) const
{
    ASSUMPTION(instruction != nullptr);

    const auto bb = instruction->get_basic_block();
    ASSUMPTION(bb != nullptr);

    const auto& bb_meta = get_basic_block_meta(bb);
    return bb_meta.removable_instructions.contains(instruction.get());
}

const utils::LiveSet&
LivenessQueryFunction::basic_block_live_out(const program::BasicBlockIR_csptr& basic_block) const
{
    return get_basic_block_meta(basic_block).live_out;
}

const optimizer::utils::SparseSet<program::InstructionIR_raw>&
LivenessQueryFunction::removable_instructions(const program::BasicBlockIR_csptr& basic_block) const
{
    return get_basic_block_meta(basic_block).removable_instructions;
}

} // namespace optimizer::analysis
