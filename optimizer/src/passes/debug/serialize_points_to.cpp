#include <iostream>
#include <optimizer/passes/debug/serialize_points_to.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/points_to_import.hpp>

#include <sala/streaming.hpp>
#include <sstream>
#include <utility/assumptions.hpp>
#include <utility/development.hpp>

namespace optimizer::passes
{
static constexpr std::string_view ERROR = "ERROR";

constexpr int OFFSET_MULT = 2;

static std::string instruction_opcode_to_string(sala::Instruction::Opcode const opcode)
{
    switch (opcode)
    {
    case sala::Instruction::Opcode::__INVALID__:
        return "__INVALID__";
        break;
    case sala::Instruction::Opcode::NOP:
        return "NOP";
        break;
    case sala::Instruction::Opcode::HALT:
        return "HALT";
        break;
    case sala::Instruction::Opcode::ADDRESS:
        return "ADDRESS";
        break;
    case sala::Instruction::Opcode::LOAD:
        return "LOAD";
        break;
    case sala::Instruction::Opcode::STORE:
        return "STORE";
        break;
    case sala::Instruction::Opcode::COPY:
        return "COPY";
        break;
    case sala::Instruction::Opcode::MEMCPY:
        return "MEMCPY";
        break;
    case sala::Instruction::Opcode::MEMMOVE:
        return "MEMMOVE";
        break;
    case sala::Instruction::Opcode::MEMSET:
        return "MEMSET";
        break;
    case sala::Instruction::Opcode::MOVEPTR:
        return "MOVEPTR";
        break;
    case sala::Instruction::Opcode::ALLOCA:
        return "ALLOCA";
        break;
    case sala::Instruction::Opcode::STACKSAVE:
        return "STACKSAVE";
        break;
    case sala::Instruction::Opcode::STACKRESTORE:
        return "STACKRESTORE";
        break;
    case sala::Instruction::Opcode::MALLOC:
        return "MALLOC";
        break;
    case sala::Instruction::Opcode::FREE:
        return "FREE";
        break;
    case sala::Instruction::Opcode::ADD:
        return "ADD";
        break;
    case sala::Instruction::Opcode::SUB:
        return "SUB";
        break;
    case sala::Instruction::Opcode::MUL:
        return "MUL";
        break;
    case sala::Instruction::Opcode::DIV:
        return "DIV";
        break;
    case sala::Instruction::Opcode::REM:
        return "REM";
        break;
    case sala::Instruction::Opcode::AND:
        return "AND";
        break;
    case sala::Instruction::Opcode::OR:
        return "OR";
        break;
    case sala::Instruction::Opcode::XOR:
        return "XOR";
        break;
    case sala::Instruction::Opcode::SHL:
        return "SHL";
        break;
    case sala::Instruction::Opcode::SHR:
        return "SHR";
        break;
    case sala::Instruction::Opcode::NEG:
        return "NEG";
        break;
    case sala::Instruction::Opcode::EXTEND:
        return "EXTEND";
        break;
    case sala::Instruction::Opcode::TRUNCATE:
        return "TRUNCATE";
        break;
    case sala::Instruction::Opcode::F2I:
        return "F2I";
        break;
    case sala::Instruction::Opcode::I2F:
        return "I2F";
        break;
    case sala::Instruction::Opcode::P2I:
        return "P2I";
        break;
    case sala::Instruction::Opcode::I2P:
        return "I2P";
        break;
    case sala::Instruction::Opcode::LESS:
        return "LESS";
        break;
    case sala::Instruction::Opcode::LESS_EQUAL:
        return "LESS_EQUAL";
        break;
    case sala::Instruction::Opcode::GREATER:
        return "GREATER";
        break;
    case sala::Instruction::Opcode::GREATER_EQUAL:
        return "GREATER_EQUAL";
        break;
    case sala::Instruction::Opcode::EQUAL:
        return "EQUAL";
        break;
    case sala::Instruction::Opcode::UNEQUAL:
        return "UNEQUAL";
        break;
    case sala::Instruction::Opcode::ISNAN:
        return "ISNAN";
        break;
    case sala::Instruction::Opcode::JUMP:
        return "JUMP";
        break;
    case sala::Instruction::Opcode::BRANCH:
        return "BRANCH";
        break;
    case sala::Instruction::Opcode::CALL:
        return "CALL";
        break;
    case sala::Instruction::Opcode::RET:
        return "RET";
        break;
    case sala::Instruction::Opcode::VA_START:
        return "VA_START";
        break;
    case sala::Instruction::Opcode::VA_END:
        return "VA_END";
        break;
    case sala::Instruction::Opcode::VA_ARG:
        return "VA_ARG";
        break;
    case sala::Instruction::Opcode::VA_COPY:
        return "VA_COPY";
        break;
    default:
        NOT_SUPPORTED();
        break;
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

struct SerializePointsTo::Impl
{
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir)
    {
        builder_.flush();
        builder_ << "__CONSTANTS__\n [\n";
        for (const auto& constant : sala_ir->get_constants())
        {
            serialize_constant(*constant, 2);
            builder_ << "\n";
        }
        builder_ << " ]\n";

        builder_ << "__STATIC__\n [\n";
        for (const auto& static_var : sala_ir->get_static_vars())
        {
            serialize_variable(*static_var, 2);
            builder_ << "\n";
        }
        builder_ << " ]\n";

        for (const auto& function : sala_ir->get_functions())
        {
            serialize_function_def(*function, OFFSET_MULT);
        }

        std::cout << builder_.str() << std::endl;
        return sala_ir;
    }

    void serialize_function_def(const program::FunctionIR& function, int offset)
    {
        if (function.get_initializer_flag())
        {
            builder_ << "__init__ ";
        }
        if (function.get_entry_flag())
        {
            builder_ << "__entry__ ";
        }
        if (function.get_external_flag())
        {
            builder_ << "__extern__ ";
        }
        builder_ << get_function_name(function) << ": \n";
        builder_ << get_offset(offset) << "__params__: \n";
        builder_ << get_offset(offset) << "(\n";
        offset += OFFSET_MULT;
        for (const auto& param : function.get_parameters())
        {
            seralize_param(*param, offset);
            builder_ << "\n";
        }
        offset -= OFFSET_MULT;
        builder_ << get_offset(offset) << ")\n";

        builder_ << get_offset(offset) << "__locals__: \n";
        builder_ << get_offset(offset) << "[\n";
        offset += OFFSET_MULT;
        for (const auto& local : function.get_local_variables())
        {
            seralize_local(*local, offset);
            builder_ << "\n";
        }
        offset -= OFFSET_MULT;
        builder_ << get_offset(offset) << "]\n";

        fill_bb_map(function);
        builder_ << get_offset(offset) << "__basic_blocks__: \n";
        builder_ << get_offset(offset) << "<\n";
        offset += OFFSET_MULT;
        for (const auto& basic_block : function.get_basic_blocks())
        {
            serialize_basic_block(basic_block, offset);
        }
        offset -= OFFSET_MULT;
        builder_ << get_offset(offset) << ">\n\n";
    }
    void seralize_param(const program::VariableIR& param, int offset)
    {
        serialize_variable(param, offset);
    }

    void fill_bb_map(const program::FunctionIR& function)
    {
        bb_id_map_.clear();
        int id = 0;
        for (const auto& bb : function.get_basic_blocks())
        {
            bb_id_map_[bb] = id++;
        }
    }

    void serialize_basic_block(const program::BasicBlockIR_sptr& basic_block, int offset)
    {
        if (basic_block == basic_block->get_function()->get_entry_basic_block())
        {
            builder_ << get_offset(offset) << "__entry__" << "\n";
        }

        builder_ << get_offset(offset) << (bb_id_map_[basic_block]) << ":\n";

        // Points_to
        if (!basic_block->get_metadata().has<metadata::points_to::BasicBlockMeta>())
        {
            builder_ << get_offset(offset) << " >>ERROR METADATA NOT PRESENT<< \n";
        }
        else
        {
            const auto& points_to_meta =
                    basic_block->get_metadata().get<metadata::points_to::BasicBlockMeta>();

            builder_ << get_offset(offset) << " >>MAY IN: ";
            for (const auto& kvp : points_to_meta.may_in)
            {
                builder_ << kvp.first << " = { ";
                for (auto elem : kvp.second)
                {
                    builder_ << elem << " ";
                }
                builder_ << "}; ";
            }
            builder_ << "<<\n";

            builder_ << get_offset(offset) << " >>MUST IN: ";
            for (const auto& kvp : points_to_meta.must_in)
            {
                builder_ << kvp.first << " -> " << kvp.second << "; ";
            }
            builder_ << "<<\n";
        }

        builder_ << get_offset(offset) << "{\n";
        serialize_instructions(basic_block->get_instructions(), offset + OFFSET_MULT);
        builder_ << get_offset(offset) << "}\n";
        builder_ << get_offset(offset) << "|_succ__:";
        for (const auto& succ : basic_block->get_successors())
        {
            builder_ << get_offset(1) << bb_id_map_[succ.lock()];
        }
        builder_ << "\n";
        builder_ << get_offset(offset) << "|_pred__:";
        for (const auto& pred : basic_block->get_predecessors())
        {
            builder_ << get_offset(1) << bb_id_map_[pred.lock()];
        }
        builder_ << "\n\n";
    }

    void seralize_local(const program::VariableIR& local, int offset)
    {
        serialize_variable(local, offset);
    }

    void serialize_instructions(const program::InstructionIRListS& instructions, int offset)
    {
        for (const auto& instruction : instructions)
        {
            const auto& metadata = instruction->get_metadata();
            ASSUMPTION(metadata.has<metadata::translation::InstructionMeta>());
            // const auto& points_to_meta = metadata.get<metadata::points_to::InstructionMeta>();
            //
            // builder_ << get_offset(offset) << " >>MAY IN: ";
            // for (const auto& kvp : points_to_meta.may_in)
            // {
            //     builder_ << kvp.first << " = { ";
            //     for (auto elem : kvp.second)
            //     {
            //         builder_ << elem << " ";
            //     }
            //     builder_ << "}; ";
            // }
            // builder_ << "<<\n";
            //
            // builder_ << get_offset(offset) << " >>MUST IN: ";
            // for (const auto& kvp : points_to_meta.must_in)
            // {
            //     builder_ << kvp.first << " -> " << kvp.second << "; ";
            // }
            // builder_ << "<<\n";

            builder_ << get_offset(offset)
                     << instruction_opcode_to_string(instruction->get_opcode());
            constexpr int operand_sep = 1;
            for (const auto& operand : instruction->get_operands())
            {
                serialize_operand(operand, operand_sep);
            }
            builder_ << "\n";
        }
    }
    void serialize_operand(const program::OperandIR_raw& operand, int operand_sep)
    {
        if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand))
        {
            serialize_variable(**variable_raw, operand_sep);
        }
        else if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand))
        {
            serialize_constant(**constant_raw, operand_sep);
        }
        else if (const auto function_raw = std::get_if<program::FunctionIR_raw>(&operand))
        {
            serialize_function_pointer(**function_raw, operand_sep);
        }
    }
    void serialize_function_pointer(const program::FunctionIR& function, int offset)
    {
        builder_ << get_offset(offset) << utils::grouped_objects::FUNCTION << "="
                 << get_function_name(function);
    }
    void serialize_constant(const program::ConstantIR& constant, int offset)
    {
        const auto& metadata       = constant.get_metadata();
        const auto& points_to_meta = metadata.try_get<metadata::points_to::ConstantMeta>();
        if (points_to_meta.has_value())
        {
            builder_ << get_offset(offset) << points_to_meta.value()->id;
        }
        else
        {
            builder_ << get_offset(offset) << ERROR;
        }
    }

    void serialize_variable(const program::VariableIR& variable, int offset)
    {
        const auto& metadata       = variable.get_metadata();
        const auto& points_to_meta = metadata.try_get<metadata::points_to::VariableMeta>();
        if (points_to_meta.has_value())
        {
            builder_ << get_offset(offset) << points_to_meta.value()->id;
        }
        else
        {
            builder_ << get_offset(offset) << ERROR;
        }
    }

  private:
    std::ostringstream builder_;

    std::unordered_map<program::BasicBlockIR_sptr, int> bb_id_map_;
};

SerializePointsTo::SerializePointsTo()
{
    pImpl_ = std::make_unique<SerializePointsTo::Impl>();
}

SerializePointsTo::~SerializePointsTo() = default;

program::ProgramIR_sptr SerializePointsTo::run(program::ProgramIR_sptr sala_ir)
{
    INVARIANT(pImpl_ != nullptr);
    return pImpl_->run(std::move(sala_ir));
};

} // namespace optimizer::passes
