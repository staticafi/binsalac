#include <optimizer/utils/common.hpp>

#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>

#include <utility/development.hpp>

namespace optimizer::utils
{
std::string instruction_opcode_to_string(sala::Instruction::Opcode const opcode)
{
    switch (opcode)
    {
    case sala::Instruction::Opcode::__INVALID__:
        return "__INVALID__";
    case sala::Instruction::Opcode::NOP:
        return "NOP";
    case sala::Instruction::Opcode::HALT:
        return "HALT";
    case sala::Instruction::Opcode::ADDRESS:
        return "ADDRESS";
    case sala::Instruction::Opcode::LOAD:
        return "LOAD";
    case sala::Instruction::Opcode::STORE:
        return "STORE";
    case sala::Instruction::Opcode::COPY:
        return "COPY";
    case sala::Instruction::Opcode::MEMCPY:
        return "MEMCPY";
    case sala::Instruction::Opcode::MEMMOVE:
        return "MEMMOVE";
    case sala::Instruction::Opcode::MEMSET:
        return "MEMSET";
    case sala::Instruction::Opcode::MOVEPTR:
        return "MOVEPTR";
    case sala::Instruction::Opcode::ALLOCA:
        return "ALLOCA";
    case sala::Instruction::Opcode::STACKSAVE:
        return "STACKSAVE";
    case sala::Instruction::Opcode::STACKRESTORE:
        return "STACKRESTORE";
    case sala::Instruction::Opcode::MALLOC:
        return "MALLOC";
    case sala::Instruction::Opcode::FREE:
        return "FREE";
    case sala::Instruction::Opcode::ADD:
        return "ADD";
    case sala::Instruction::Opcode::SUB:
        return "SUB";
    case sala::Instruction::Opcode::MUL:
        return "MUL";
    case sala::Instruction::Opcode::DIV:
        return "DIV";
    case sala::Instruction::Opcode::REM:
        return "REM";
    case sala::Instruction::Opcode::AND:
        return "AND";
    case sala::Instruction::Opcode::OR:
        return "OR";
    case sala::Instruction::Opcode::XOR:
        return "XOR";
    case sala::Instruction::Opcode::SHL:
        return "SHL";
    case sala::Instruction::Opcode::SHR:
        return "SHR";
    case sala::Instruction::Opcode::NEG:
        return "NEG";
    case sala::Instruction::Opcode::EXTEND:
        return "EXTEND";
    case sala::Instruction::Opcode::TRUNCATE:
        return "TRUNCATE";
    case sala::Instruction::Opcode::F2I:
        return "F2I";
    case sala::Instruction::Opcode::I2F:
        return "I2F";
    case sala::Instruction::Opcode::P2I:
        return "P2I";
    case sala::Instruction::Opcode::I2P:
        return "I2P";
    case sala::Instruction::Opcode::LESS:
        return "LESS";
    case sala::Instruction::Opcode::LESS_EQUAL:
        return "LESS_EQUAL";
    case sala::Instruction::Opcode::GREATER:
        return "GREATER";
    case sala::Instruction::Opcode::GREATER_EQUAL:
        return "GREATER_EQUAL";
    case sala::Instruction::Opcode::EQUAL:
        return "EQUAL";
    case sala::Instruction::Opcode::UNEQUAL:
        return "UNEQUAL";
    case sala::Instruction::Opcode::ISNAN:
        return "ISNAN";
    case sala::Instruction::Opcode::JUMP:
        return "JUMP";
    case sala::Instruction::Opcode::BRANCH:
        return "BRANCH";
    case sala::Instruction::Opcode::CALL:
        return "CALL";
    case sala::Instruction::Opcode::RET:
        return "RET";
    case sala::Instruction::Opcode::VA_START:
        return "VA_START";
    case sala::Instruction::Opcode::VA_END:
        return "VA_END";
    case sala::Instruction::Opcode::VA_ARG:
        return "VA_ARG";
    case sala::Instruction::Opcode::VA_COPY:
        return "VA_COPY";
    default:
        NOT_SUPPORTED();
    }
}

std::string get_offset(int offset)
{
    return std::string(offset, ' ');
}

std::string_view get_function_name(const program::FunctionIR& function)
{
    const auto& metadata = function.get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::FunctionMeta>());
    const auto& translation_meta = metadata.get<metadata::translation::FunctionMeta>();

    return translation_meta.name;
}

std::string sanitize_filename(std::string name)
{
    for (char& ch : name)
    {
        switch (ch)
        {
        case '/':
        case '\\':
        case ':':
        case '*':
        case '?':
        case '"':
        case '<':
        case '>':
        case '|':
            ch = '_';
            break;
        default:
            break;
        }
    }

    if (name.empty())
    {
        name = "program";
    }

    return name;
}

std::string get_program_name(const program::ProgramIR& program)
{
    const auto& metadata = program.get_metadata();
    ASSUMPTION(metadata.has<metadata::translation::ProgramMeta>());

    const auto& translation_meta = metadata.get<metadata::translation::ProgramMeta>();
    return sanitize_filename(translation_meta.name);
}
} // namespace optimizer::utils
