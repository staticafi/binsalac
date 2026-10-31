#include <optimizer/utils/liveness/defines.hpp>

#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <utility/assumptions.hpp>

namespace optimizer::utils::liveness
{
namespace
{
template <typename T>
inline bool holds_trackable(const Operand& operand) noexcept
{
    if (const auto* ptr = std::get_if<T>(&operand))
    {
        return *ptr != nullptr;
    }
    return false;
}

inline void insert_if_trackable(LiveSet& set, const Operand& operand)
{
    if (is_trackable_operand(operand))
    {
        set.insert(operand);
    }
}

inline bool is_static_variable_operand(const Operand& operand) noexcept
{
    const auto* variable_raw = std::get_if<program::VariableIR_raw>(&operand);
    if (variable_raw == nullptr || *variable_raw == nullptr)
    {
        return false;
    }

    return (*variable_raw)->get_context() == program::VariableIR::Context::STATIC;
}

inline bool is_constant_operand(const Operand& operand) noexcept
{
    return std::holds_alternative<program::ConstantIR_raw>(operand);
}

inline bool writes_static_storage(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto  opcode = instruction->get_opcode();
    const auto& ops    = instruction->get_operands();

    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        if (!is_written_operand(opcode, i))
        {
            continue;
        }

        if (is_static_variable_operand(ops[i]))
        {
            return true;
        }
    }

    return false;
}

inline bool writes_constant_storage(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto  opcode = instruction->get_opcode();
    const auto& ops    = instruction->get_operands();

    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        if (!is_written_operand(opcode, i))
        {
            continue;
        }

        if (is_constant_operand(ops[i]))
        {
            return true;
        }
    }

    return false;
}

bool reads_indirect_memory(const sala::Instruction::Opcode opcode) noexcept
{
    using Op = sala::Instruction::Opcode;
    switch (opcode)
    {
    case Op::LOAD:
    case Op::CALL:
    case Op::MEMCPY:
    case Op::MEMMOVE:
    case Op::VA_START:
    case Op::VA_END:
    case Op::VA_ARG:
    case Op::VA_COPY:
        return true;

    default:
        return false;
    }
}

} // namespace

bool is_trackable_operand(const Operand& operand) noexcept
{
    return holds_trackable<program::VariableIR_raw>(operand) ||
           holds_trackable<program::ConstantIR_raw>(operand);
}

bool is_written_operand(const sala::Instruction::Opcode opcode,
                        const std::size_t               operand_index) noexcept
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
        return operand_index == 0U;

    case Op::VA_ARG:
        return operand_index == 0U || operand_index == 1U;

    default:
        return false;
    }
}

bool has_observable_side_effects(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    using Op          = sala::Instruction::Opcode;
    const auto opcode = instruction->get_opcode();

    switch (opcode)
    {
    case Op::STORE:
    case Op::MEMCPY:
    case Op::MEMMOVE:
    case Op::MEMSET:
    case Op::CALL:
    case Op::FREE:
    case Op::RET:
    case Op::JUMP:
    case Op::BRANCH:
    case Op::HALT:
    case Op::STACKRESTORE:
    case Op::VA_START:
    case Op::VA_END:
    case Op::VA_ARG:
    case Op::__INVALID__:
        return true;

    default:
        break;
    }

    return writes_constant_storage(instruction) || writes_static_storage(instruction);
}

bool is_pure_dead_def_candidate(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    if (has_observable_side_effects(instruction))
    {
        return false;
    }

    using Op = sala::Instruction::Opcode;
    switch (instruction->get_opcode())
    {
    case Op::NOP:
    case Op::ADDRESS:
    case Op::COPY:
    case Op::LOAD:
    case Op::MOVEPTR:
    case Op::ALLOCA:
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
        return true;

    default:
        return false;
    }
}

LiveSet memory_read_set(const program::FunctionIR& function)
{
    const auto program = function.get_program();
    ASSUMPTION(program != nullptr);

    LiveSet memory;
    memory.reserve(function.get_parameters().size()
                 + function.get_local_variables().size()
                 + program->get_static_vars().size()
                 + program->get_constants().size());

    for (const auto& parameter : function.get_parameters())
    {
        memory.insert(parameter.get());
    }
    for (const auto& local : function.get_local_variables())
    {
        memory.insert(local.get());
    }
    for (const auto& variable : program->get_static_vars())
    {
        memory.insert(variable.get());
    }
    for (const auto& constant : program->get_constants())
    {
        memory.insert(constant.get());
    }

    return memory;
}

void collect_uses_and_defs(const program::InstructionIR_sptr& instruction,
                           const LiveSet& memory, LiveSet& uses, LiveSet& defs)
{
    ASSUMPTION(instruction != nullptr);

    uses.clear();
    defs.clear();

    const auto  opcode = instruction->get_opcode();
    const auto& ops    = instruction->get_operands();

    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        if (is_written_operand(opcode, i))
        {
            insert_if_trackable(defs, ops[i]);
        }
        else
        {
            insert_if_trackable(uses, ops[i]);
        }
    }

    if (reads_indirect_memory(opcode))
    {
        uses.union_with(memory);
    }
}

void apply_backward_transfer(const program::InstructionIR_sptr& instruction,
                             const LiveSet& memory, const LiveSet& live_after,
                             LiveSet& live_before)
{
    ASSUMPTION(instruction != nullptr);

    live_before = live_after;

    LiveSet uses;
    LiveSet defs;
    collect_uses_and_defs(instruction, memory, uses, defs);

    for (const auto& def : defs)
    {
        live_before.erase(def);
    }

    for (const auto& use : uses)
    {
        live_before.insert(use);
    }
}
bool is_instruction_removable(const program::InstructionIR_sptr& instruction,
                              const LiveSet&                     live_after)
{
    ASSUMPTION(instruction != nullptr);

    if (instruction->get_opcode() == sala::Instruction::Opcode::NOP)
    {
        return true;
    }

    if (!is_pure_dead_def_candidate(instruction))
    {
        return false;
    }

    bool has_def = false;
    const auto opcode = instruction->get_opcode();
    const auto& ops   = instruction->get_operands();

    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        if (!is_written_operand(opcode, i) || !is_trackable_operand(ops[i]))
        {
            continue;
        }

        has_def = true;
        if (live_after.contains(ops[i]))
        {
            return false;
        }
    }

    ASSUMPTION(has_def);
    return true;
}

} // namespace optimizer::utils::liveness
