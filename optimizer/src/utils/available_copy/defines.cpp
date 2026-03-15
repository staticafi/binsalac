#include <optimizer/utils/available_copy/defines.hpp>

#include <utility/assumptions.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace optimizer::utils::available_copy
{
namespace
{

std::optional<std::size_t> get_variable_id(
        const program::OperandIR_raw&                                            operand,
        const optimizer::utils::SparseMap<program::VariableIR_raw, std::size_t>& variable_ids)
{
    const auto* variable = std::get_if<program::VariableIR_raw>(&operand);
    if (variable == nullptr || *variable == nullptr)
    {
        return std::nullopt;
    }

    const auto it = variable_ids.find(*variable);
    if (it == variable_ids.end())
    {
        return std::nullopt;
    }

    return it->second;
}

void bridge_through_redefined_variable(const std::size_t variable_id, const TransferContext& ctx,
                                       State& state)
{
    if (variable_id >= ctx.facts_by_dest.size() || variable_id >= ctx.facts_by_source.size())
    {
        return;
    }

    std::vector<std::size_t> incoming;
    std::vector<std::size_t> outgoing;

    // incoming: X -> variable_id
    for (const auto fact_id : ctx.facts_by_source[variable_id])
    {
        if (state.test(fact_id))
        {
            incoming.push_back(fact_id);
        }
    }

    // outgoing: variable_id -> Y
    for (const auto fact_id : ctx.facts_by_dest[variable_id])
    {
        if (state.test(fact_id))
        {
            outgoing.push_back(fact_id);
        }
    }

    for (const auto in_fact_id : incoming)
    {
        const auto& in_fact = ctx.facts[in_fact_id];
        const auto  x_id    = in_fact.dest_id;

        for (const auto out_fact_id : outgoing)
        {
            const auto& out_fact = ctx.facts[out_fact_id];
            const auto  y_id     = out_fact.source_id;

            if (x_id == y_id)
            {
                continue;
            }

            const CopyFactKey bridged_key{.dest_id = x_id, .source_id = y_id};
            const auto        it = ctx.fact_id_of.find(bridged_key);
            if (it == ctx.fact_id_of.end())
            {
                continue;
            }

            state.set(it->second);
        }
    }
}

void kill_written_operands(const program::InstructionIR_sptr& instruction,
                           const TransferContext& ctx, State& state)
{
    ASSUMPTION(instruction != nullptr);

    const auto  opcode   = instruction->get_opcode();
    const auto& operands = instruction->get_operands();

    for (std::size_t i = 0; i < operands.size(); ++i)
    {
        if (!is_written_operand(opcode, i))
        {
            continue;
        }

        const auto variable_id = get_variable_id(operands[i], ctx.variable_ids);
        if (!variable_id.has_value())
        {
            continue;
        }

        bridge_through_redefined_variable(variable_id.value(), ctx, state);
        kill_all_copies_of_variable(variable_id.value(), ctx.kill_masks, state);
    }
}

void generate_copy_fact(const program::InstructionIR_sptr& instruction, const TransferContext& ctx,
                        State& state)
{
    ASSUMPTION(instruction != nullptr);

    if (instruction->get_opcode() != sala::Instruction::Opcode::COPY)
    {
        return;
    }

    const auto& operands = instruction->get_operands();
    if (operands.size() < 2U)
    {
        return;
    }

    const auto* dest = std::get_if<program::VariableIR_raw>(&operands[0]);
    const auto* src  = std::get_if<program::VariableIR_raw>(&operands[1]);

    if (dest == nullptr || src == nullptr || *dest == nullptr || *src == nullptr)
    {
        return;
    }

    const auto dest_it = ctx.variable_ids.find(*dest);
    const auto src_it  = ctx.variable_ids.find(*src);

    if (dest_it == ctx.variable_ids.end() || src_it == ctx.variable_ids.end())
    {
        return;
    }

    const auto dest_id = dest_it->second;
    const auto src_id  = src_it->second;

    if (dest_id == src_id)
    {
        return;
    }

    if ((*dest)->get_num_bytes() != (*src)->get_num_bytes())
    {
        return;
    }

    const auto fact_it = ctx.fact_id_of.find(CopyFactKey{.dest_id = dest_id, .source_id = src_id});
    if (fact_it == ctx.fact_id_of.end())
    {
        return;
    }

    state.set(fact_it->second);
}

} // namespace

bool is_direct_def_opcode(const sala::Instruction::Opcode opcode)
{
    using Op = sala::Instruction::Opcode;

    switch (opcode)
    {
    case Op::ADDRESS:
    case Op::LOAD:
    case Op::COPY:
    case Op::MOVEPTR:
    case Op::ALLOCA:
    case Op::STACKSAVE:
    case Op::MALLOC:
    case Op::ADD:
    case Op::SUB:
    case Op::MUL:
    case Op::DIV:
    case Op::REM:
    case Op::AND:
    case Op::OR:
    case Op::XOR:
    case Op::SHL:
    case Op::SHR:
    case Op::NEG:
    case Op::EXTEND:
    case Op::TRUNCATE:
    case Op::F2I:
    case Op::I2F:
    case Op::P2I:
    case Op::I2P:
    case Op::LESS:
    case Op::LESS_EQUAL:
    case Op::GREATER:
    case Op::GREATER_EQUAL:
    case Op::EQUAL:
    case Op::UNEQUAL:
    case Op::ISNAN:
    case Op::VA_COPY:
        return true;

    default:
        return false;
    }
}

bool has_global_memory_side_effects(const sala::Instruction::Opcode opcode)
{
    using Op = sala::Instruction::Opcode;

    switch (opcode)
    {
    case Op::STORE:
    case Op::MEMCPY:
    case Op::MEMMOVE:
    case Op::MEMSET:
    case Op::CALL:
    case Op::STACKRESTORE:
    case Op::VA_START:
    case Op::VA_END:
    case Op::VA_ARG:
        return true;

    default:
        return false;
    }
}

bool is_written_operand(const sala::Instruction::Opcode opcode, const std::size_t operand_index)
{
    if (operand_index == 0U && is_direct_def_opcode(opcode))
    {
        return true;
    }

    return opcode == sala::Instruction::Opcode::VA_ARG &&
           (operand_index == 0U || operand_index == 1U);
}

void kill_all_copies_of_variable(const std::size_t         variable_id,
                                 const std::vector<State>& kill_masks, State& state)
{
    if (variable_id >= kill_masks.size())
    {
        return;
    }

    state.subtract(kill_masks[variable_id]);
}

void apply_transfer(const program::InstructionIR_sptr& instruction, const TransferContext& ctx,
                    State& state)
{
    ASSUMPTION(instruction != nullptr);

    const auto opcode = instruction->get_opcode();

    if (has_global_memory_side_effects(opcode))
    {
        state.reset();
        return;
    }

    kill_written_operands(instruction, ctx, state);
    generate_copy_fact(instruction, ctx, state);
}

} // namespace optimizer::utils::available_copy
