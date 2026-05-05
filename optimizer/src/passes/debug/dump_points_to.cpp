#include <optimizer/passes/debug/dump_points_to.hpp>

#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/query/points_to_query.hpp>
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

void dump_may_state(std::ostream& out, const utils::MayAnalysisState& state, std::string_view label,
                    int offset)
{
    out << utils::get_offset(offset) << " >>" << label << " MAY: ";

    if (state.poisoned)
    {
        out << "POISONED";
    }
    else
    {
        for (const auto& kvp : state.may)
        {
            out << kvp.first << " = " << kvp.second << "; ";
        }
    }

    out << "<<\n";
}

void dump_must_state(std::ostream& out, const utils::MayAnalysisState& state,
                     std::string_view label, int offset)
{
    out << utils::get_offset(offset) << " >>" << label << " MUST: ";

    if (state.poisoned)
    {
        out << "POISONED";
    }
    else
    {
        for (const auto& kvp : state.must)
        {
            const auto must_fact = kvp.second;
            out << kvp.first << " -> " << must_fact << "; ";
        }
    }

    out << "<<\n";
}

struct Impl
{
    Impl(program::ProgramIR_sptr sala_ir, std::filesystem::path output_path)
        : sala_ir_{std::move(sala_ir)}, output_path_{std::move(output_path)}
    {
        run();
    }

    program::ProgramIR_sptr run()
    {
        const auto default_filename = utils::get_program_name(*sala_ir_) + ".points_to_debug";

        std::filesystem::path resolved_output_path;

        if (output_path_.empty())
        {
            resolved_output_path = default_filename;
        }
        else if (!output_path_.has_extension())
        {
            // treat as directory
            resolved_output_path = output_path_ / default_filename;
        }
        else
        {
            // treat as full file path
            resolved_output_path = output_path_;
        }

        std::ofstream out(resolved_output_path, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            throw std::runtime_error("Failed to open output file: " +
                                     resolved_output_path.string());
        }

        out << "__CONSTANTS__\n [\n";
        for (const auto& constant : sala_ir_->get_constants())
        {
            serialize_constant(out, *constant, 2);
            out << "\n";
        }
        out << " ]\n";

        out << "__STATIC__\n [\n";
        for (const auto& static_var : sala_ir_->get_static_vars())
        {
            serialize_variable(out, *static_var, 2);
            out << "\n";
        }
        out << " ]\n";

        for (const auto& function : sala_ir_->get_functions())
        {
            serialize_function_def(out, function, OFFSET_MULT);
        }

        return sala_ir_;
    }

    void serialize_function_def(std::ostream& out, const program::FunctionIR_sptr& function,
                                int offset)
    {
        ASSUMPTION(function != nullptr);

        const auto* function_points_to_meta =
                function->get_metadata().get_raw<metadata::points_to::FunctionMeta>();
        ASSUMPTION(function_points_to_meta != nullptr);
        ASSUMPTION(function_points_to_meta->bb_may_state_store != nullptr);

        query::PointsToQueryFunction query(function);

        if (function->get_initializer_flag())
        {
            out << "__init__ ";
        }
        if (function->get_entry_flag())
        {
            out << "__entry__ ";
        }
        if (function->get_external_flag())
        {
            out << "__extern__ ";
        }

        out << utils::get_function_name(*function) << ":\n";

        out << utils::get_offset(offset) << "__params__:\n";
        out << utils::get_offset(offset) << "(\n";
        offset += OFFSET_MULT;
        for (const auto& param : function->get_parameters())
        {
            serialize_param(out, *param, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ")\n";

        out << utils::get_offset(offset) << "__locals__:\n";
        out << utils::get_offset(offset) << "[\n";
        offset += OFFSET_MULT;
        for (const auto& local : function->get_local_variables())
        {
            serialize_local(out, *local, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << "]\n";

        fill_bb_map(*function);
        out << utils::get_offset(offset) << "__basic_blocks__:\n";
        out << utils::get_offset(offset) << "<\n";
        offset += OFFSET_MULT;
        for (const auto& basic_block : function->get_basic_blocks())
        {
            serialize_basic_block(out, basic_block, offset, *function_points_to_meta, query);
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
                               const metadata::points_to::FunctionMeta& function_points_to_meta,
                               query::PointsToQueryFunction&            query)
    {
        if (basic_block == basic_block->get_function()->get_entry_basic_block())
        {
            out << utils::get_offset(offset) << "__entry__\n";
        }

        out << utils::get_offset(offset) << bb_id_map_[basic_block] << ":\n";

        if (!basic_block->get_metadata().has<metadata::points_to::BasicBlockMeta>())
        {
            out << utils::get_offset(offset) << " >>ERROR METADATA NOT PRESENT<< \n";
        }
        else
        {
            const auto& points_to_meta =
                    basic_block->get_metadata().get<metadata::points_to::BasicBlockMeta>();

            const auto& state =
                    function_points_to_meta.bb_may_state_store->get(points_to_meta.may_in_id);

            dump_may_state(out, state, "BB-IN", offset);
            dump_must_state(out, state, "BB-IN", offset);
        }

        out << utils::get_offset(offset) << "{\n";
        serialize_instructions(out, basic_block->get_instructions(), offset + OFFSET_MULT, query);
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
                                int offset, query::PointsToQueryFunction& query)
    {
        for (const auto& instruction : instructions)
        {
            const auto& metadata = instruction->get_metadata();
            ASSUMPTION(metadata.has<metadata::translation::InstructionMeta>());

            const auto before_state = query.before_state(instruction);
            const auto after_state  = query.after_state(instruction);

            dump_may_state(out, before_state, "BEFORE", offset);
            dump_must_state(out, before_state, "BEFORE", offset);

            out << utils::get_offset(offset)
                << utils::instruction_opcode_to_string(instruction->get_opcode());

            constexpr int operand_sep = 1;
            for (const auto& operand : instruction->get_operands())
            {
                serialize_operand(out, operand, operand_sep);
            }
            out << "\n";

            dump_may_state(out, after_state, "AFTER", offset);
            dump_must_state(out, after_state, "AFTER", offset);
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
    std::filesystem::path                               output_path_;
    program::ProgramIR_sptr                             sala_ir_;
};

} // namespace

void DumpPointsTo::set_output_path(std::filesystem::path output_path)
{
    output_path_ = std::move(output_path);
}

void DumpPointsTo::run(program::ProgramIR_sptr sala_ir)
{
    ASSUMPTION(sala_ir != nullptr);
    const auto trigger = Impl(std::move(sala_ir), output_path_);
}
} // namespace optimizer::passes
