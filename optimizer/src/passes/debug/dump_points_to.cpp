#include <optimizer/passes/debug/dump_points_to.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/common.hpp>
#include <optimizer/utils/points_to/import.hpp>

#include <sala/streaming.hpp>
#include <utility/assumptions.hpp>
#include <utility/development.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <unordered_map>

namespace optimizer::passes
{
namespace
{
static constexpr std::string_view ERROR = "ERROR";

constexpr int OFFSET_MULT = 2;
} // namespace

struct DumpPointsTo::Impl
{
    program::ProgramIR_sptr run(program::ProgramIR_sptr sala_ir)
    {
        const auto output_path =
                std::filesystem::path(utils::get_program_name(*sala_ir) + ".points_to_debug");

        std::ofstream out(output_path, std::ios::out | std::ios::trunc);
        ASSUMPTION(out.is_open());

        out << "__CONSTANTS__\n [\n";
        for (const auto& constant : sala_ir->get_constants())
        {
            serialize_constant(out, *constant, 2);
            out << "\n";
        }
        out << " ]\n";

        out << "__STATIC__\n [\n";
        for (const auto& static_var : sala_ir->get_static_vars())
        {
            serialize_variable(out, *static_var, 2);
            out << "\n";
        }
        out << " ]\n";

        for (const auto& function : sala_ir->get_functions())
        {
            serialize_function_def(out, *function, OFFSET_MULT);
        }

        return sala_ir;
    }

    void serialize_function_def(std::ostream& out, const program::FunctionIR& function, int offset)
    {
        const auto* function_points_to_meta =
                function.get_metadata().get_raw<metadata::points_to::FunctionMeta>();
        ASSUMPTION(function_points_to_meta != nullptr);
        ASSUMPTION(function_points_to_meta->bb_may_state_store != nullptr);

        if (function.get_initializer_flag())
        {
            out << "__init__ ";
        }
        if (function.get_entry_flag())
        {
            out << "__entry__ ";
        }
        if (function.get_external_flag())
        {
            out << "__extern__ ";
        }
        out << utils::get_function_name(function) << ": \n";
        out << utils::get_offset(offset) << "__params__: \n";
        out << utils::get_offset(offset) << "(\n";
        offset += OFFSET_MULT;
        for (const auto& param : function.get_parameters())
        {
            serialize_param(out, *param, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ")\n";

        out << utils::get_offset(offset) << "__locals__: \n";
        out << utils::get_offset(offset) << "[\n";
        offset += OFFSET_MULT;
        for (const auto& local : function.get_local_variables())
        {
            serialize_local(out, *local, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << "]\n";

        fill_bb_map(function);
        out << utils::get_offset(offset) << "__basic_blocks__: \n";
        out << utils::get_offset(offset) << "<\n";
        offset += OFFSET_MULT;
        for (const auto& basic_block : function.get_basic_blocks())
        {
            serialize_basic_block(out, basic_block, offset, *function_points_to_meta);
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ">\n\n";
    }

    void serialize_param(std::ostream& out, const program::VariableIR& param, int offset)
    {
        serialize_variable(out, param, offset);
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

    void serialize_basic_block(std::ostream& out, const program::BasicBlockIR_sptr& basic_block,
                               int                                      offset,
                               const metadata::points_to::FunctionMeta& function_points_to_meta)
    {
        if (basic_block == basic_block->get_function()->get_entry_basic_block())
        {
            out << utils::get_offset(offset) << "__entry__"
                << "\n";
        }

        out << utils::get_offset(offset) << (bb_id_map_[basic_block]) << ":\n";

        if (!basic_block->get_metadata().has<metadata::points_to::BasicBlockMeta>())
        {
            out << utils::get_offset(offset) << " >>ERROR METADATA NOT PRESENT<< \n";
        }
        else
        {
            const auto& points_to_meta =
                    basic_block->get_metadata().get<metadata::points_to::BasicBlockMeta>();

            const auto& may_in =
                    function_points_to_meta.bb_may_state_store->get(points_to_meta.may_in_id);
            out << utils::get_offset(offset) << " >>MAY IN: ";
            for (const auto& kvp : may_in)
            {
                out << kvp.first << " = " << kvp.second;
                out << "; ";
            }
            out << "<<\n";

            out << utils::get_offset(offset) << " >>MUST IN: ";
            for (const auto& kvp : may_in)
            {
                const auto must_fact = kvp.second.must_fact();
                if (must_fact.has_value())
                {
                    out << kvp.first << " -> " << must_fact.value() << "; ";
                }
            }
            out << "<<\n";
        }

        out << utils::get_offset(offset) << "{\n";
        serialize_instructions(out, basic_block->get_instructions(), offset + OFFSET_MULT);
        out << utils::get_offset(offset) << "}\n";
        out << utils::get_offset(offset) << "|_succ__:";
        for (const auto& succ : basic_block->get_successors())
        {
            out << utils::get_offset(1) << bb_id_map_[succ.lock()];
        }
        out << "\n";
        out << utils::get_offset(offset) << "|_pred__:";
        for (const auto& pred : basic_block->get_predecessors())
        {
            out << utils::get_offset(1) << bb_id_map_[pred.lock()];
        }
        out << "\n\n";
    }

    void serialize_local(std::ostream& out, const program::VariableIR& local, int offset)
    {
        serialize_variable(out, local, offset);
    }

    void serialize_instructions(std::ostream& out, const program::InstructionIRListS& instructions,
                                int offset)
    {
        for (const auto& instruction : instructions)
        {
            const auto& metadata = instruction->get_metadata();
            ASSUMPTION(metadata.has<metadata::translation::InstructionMeta>());
            out << utils::get_offset(offset)
                << utils::instruction_opcode_to_string(instruction->get_opcode());
            constexpr int operand_sep = 1;
            for (const auto& operand : instruction->get_operands())
            {
                serialize_operand(out, operand, operand_sep);
            }
            out << "\n";
        }
    }

    void serialize_operand(std::ostream& out, const program::OperandIR_raw& operand,
                           int operand_sep)
    {
        if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand))
        {
            serialize_variable(out, **variable_raw, operand_sep);
        }
        else if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand))
        {
            serialize_constant(out, **constant_raw, operand_sep);
        }
        else if (const auto function_raw = std::get_if<program::FunctionIR_raw>(&operand))
        {
            serialize_function_pointer(out, **function_raw, operand_sep);
        }
    }

    void serialize_function_pointer(std::ostream& out, const program::FunctionIR& function,
                                    int offset)
    {
        out << utils::get_offset(offset) << utils::grouped_objects::FUNCTION << "="
            << utils::get_function_name(function);
    }

    void serialize_constant(std::ostream& out, const program::ConstantIR& constant, int offset)
    {
        const auto& metadata       = constant.get_metadata();
        const auto& points_to_meta = metadata.try_get<metadata::points_to::ConstantMeta>();
        if (points_to_meta.has_value())
        {
            out << utils::get_offset(offset) << points_to_meta.value()->id;
        }
        else
        {
            out << utils::get_offset(offset) << ERROR;
        }
    }

    void serialize_variable(std::ostream& out, const program::VariableIR& variable, int offset)
    {
        const auto& metadata       = variable.get_metadata();
        const auto& points_to_meta = metadata.try_get<metadata::points_to::VariableMeta>();
        if (points_to_meta.has_value())
        {
            out << utils::get_offset(offset) << points_to_meta.value()->id;
        }
        else
        {
            out << utils::get_offset(offset) << ERROR;
        }
    }

  private:
    std::unordered_map<program::BasicBlockIR_sptr, int> bb_id_map_;
};

DumpPointsTo::DumpPointsTo()
{
    pImpl_ = std::make_unique<DumpPointsTo::Impl>();
}

DumpPointsTo::~DumpPointsTo() = default;

program::ProgramIR_sptr DumpPointsTo::run(program::ProgramIR_sptr sala_ir)
{
    INVARIANT(pImpl_ != nullptr);
    return pImpl_->run(std::move(sala_ir));
};

} // namespace optimizer::passes
