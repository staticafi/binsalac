#include <optimizer/passes/debug/dump_available_copy.hpp>

#include <optimizer/analysis/available_copy_query.hpp>
#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/metadata/metadata.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/available_copy/import.hpp>
#include <optimizer/utils/common.hpp>

#include <utility/assumptions.hpp>
#include <utility/development.hpp>
#include <utility/invariants.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <unordered_map>

namespace optimizer::passes
{
namespace
{
static constexpr std::string_view ERROR       = "ERROR";
constexpr int                     OFFSET_MULT = 2;

struct Impl
{
    Impl(program::ProgramIR_sptr sala_ir, std::filesystem::path output_path)
        : sala_ir_{std::move(sala_ir)}, output_path_{std::move(output_path)}
    {
        run();
    }

    program::ProgramIR_sptr run()
    {
        const auto default_filename = utils::get_program_name(*sala_ir_) + ".available_copy_debug";

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
            std::runtime_error("Failed to open output file: " + resolved_output_path.string());
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
            serialize_variable(out, *param, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ")\n";

        out << utils::get_offset(offset) << "__locals__:\n";
        out << utils::get_offset(offset) << "[\n";
        offset += OFFSET_MULT;
        for (const auto& local : function->get_local_variables())
        {
            serialize_variable(out, *local, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << "]\n";

        serialize_function_metadata(out, *function, offset);

        fill_bb_map(*function);

        analysis::AvailableCopyQueryFunction query(function);

        out << utils::get_offset(offset) << "__basic_blocks__:\n";
        out << utils::get_offset(offset) << "<\n";
        offset += OFFSET_MULT;
        for (const auto& basic_block : function->get_basic_blocks())
        {
            serialize_basic_block(out, basic_block, query, offset);
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ">\n\n";
    }

    void serialize_function_metadata(std::ostream& out, const program::FunctionIR& function,
                                     int offset)
    {
        out << utils::get_offset(offset) << "__available_copy_function_meta__:\n";

        if (!function.get_metadata().has<metadata::available_copy::FunctionMeta>())
        {
            out << utils::get_offset(offset) << " >>ERROR METADATA NOT PRESENT<<\n";
            return;
        }

        const auto& meta = function.get_metadata().get<metadata::available_copy::FunctionMeta>();

        out << utils::get_offset(offset) << " facts: [";
        for (std::size_t i = 0; i < meta.facts.size(); ++i)
        {
            if (i != 0)
            {
                out << "; ";
            }

            const auto& fact = meta.facts[i];
            out << "#" << fact.id << " (" << fact.dest_id << " <- " << fact.source_id << ")";

            if (fact.dest != nullptr)
            {
                serialize_variable(out, *fact.dest, 0);
            }
            else
            {
                out << ERROR;
            }

            out << " = ";

            if (fact.source != nullptr)
            {
                serialize_variable(out, *fact.source, 0);
            }
            else
            {
                out << ERROR;
            }
        }
        out << "]\n";

        out << utils::get_offset(offset) << " facts_by_dest: [";
        for (std::size_t dest = 0; dest < meta.facts_by_dest.size(); ++dest)
        {
            if (dest != 0)
            {
                out << "; ";
            }

            out << dest << " -> {";
            bool first = true;
            for (const auto fact_id : meta.facts_by_dest[dest])
            {
                if (!first)
                {
                    out << ", ";
                }
                first = false;
                out << fact_id;
            }
            out << "}";
        }
        out << "]\n";
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
                               analysis::AvailableCopyQueryFunction& query, int offset)
    {
        if (basic_block == basic_block->get_function()->get_entry_basic_block())
        {
            out << utils::get_offset(offset) << "__entry__\n";
        }

        out << utils::get_offset(offset) << bb_id_map_[basic_block] << ":\n";

        if (!basic_block->get_metadata().has<metadata::available_copy::BasicBlockMeta>())
        {
            out << utils::get_offset(offset) << " >>ERROR METADATA NOT PRESENT<<\n";
        }
        else
        {
            const auto& meta =
                    basic_block->get_metadata().get<metadata::available_copy::BasicBlockMeta>();

            out << utils::get_offset(offset) << " >>IN_BITS: ";
            for (const auto word : meta.in_bits)
            {
                out << word << " ";
            }
            out << "<<\n";
        }

        out << utils::get_offset(offset) << "{\n";
        serialize_instructions(out, basic_block->get_instructions(), query, offset + OFFSET_MULT);
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

    void serialize_instructions(std::ostream& out, const program::InstructionIRListS& instructions,
                                analysis::AvailableCopyQueryFunction& query, int offset)
    {
        for (const auto& instruction : instructions)
        {
            out << utils::get_offset(offset)
                << utils::instruction_opcode_to_string(instruction->get_opcode());

            constexpr int operand_sep = 1;
            for (const auto& operand : instruction->get_operands())
            {
                serialize_operand(out, operand, operand_sep);
            }

            dump_instruction_query_info(out, instruction, query);

            out << "\n";
        }
    }

    void dump_instruction_query_info(std::ostream&                         out,
                                     const program::InstructionIR_sptr&    instruction,
                                     analysis::AvailableCopyQueryFunction& query)
    {
        const auto  opcode   = instruction->get_opcode();
        const auto& operands = instruction->get_operands();

        bool printed_any = false;

        for (std::size_t i = 0; i < operands.size(); ++i)
        {
            if (utils::is_written_operand(opcode, i))
            {
                continue;
            }

            const auto* variable_raw = std::get_if<program::VariableIR_raw>(&operands[i]);
            if (variable_raw == nullptr || *variable_raw == nullptr)
            {
                continue;
            }

            const auto result = query.before(instruction, **variable_raw);

            const bool has_direct     = result.direct_source.has_value();
            const bool has_transitive = result.transitive_source.has_value();
            const bool has_chain      = result.chain.size() > 1;

            if (!has_direct && !has_transitive && !has_chain)
            {
                continue;
            }

            if (!printed_any)
            {
                out << "    >>QUERY: ";
                printed_any = true;
            }

            out << "[op=" << i;

            if (has_direct)
            {
                out << ", direct=";
                serialize_variable(out, *result.direct_source.value(), 0);
            }

            if (has_transitive)
            {
                out << ", transitive=";
                serialize_variable(out, *result.transitive_source.value(), 0);
            }

            if (has_chain)
            {
                out << ", chain={";
                for (std::size_t ci = 0; ci < result.chain.size(); ++ci)
                {
                    if (ci != 0)
                    {
                        out << " -> ";
                    }
                    serialize_variable(out, *result.chain[ci], 0);
                }
                out << "}";
            }

            out << "] ";
        }

        if (printed_any)
        {
            out << "<<";
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
        out << utils::get_offset(offset) << utils::get_function_name(function);
    }

    void serialize_constant(std::ostream& out, const program::ConstantIR&, int offset)
    {
        out << utils::get_offset(offset) << "CONST";
    }

    void serialize_variable(std::ostream& out, const program::VariableIR& variable, int offset)
    {
        const auto& metadata = variable.get_metadata();

        if (const auto avail_meta = metadata.try_get<metadata::available_copy::VariableMeta>();
            avail_meta.has_value())
        {
            out << utils::get_offset(offset) << avail_meta.value()->id;
            return;
        }

        out << utils::get_offset(offset) << ERROR;
    }

  private:
    std::unordered_map<program::BasicBlockIR_sptr, int> bb_id_map_;
    program::ProgramIR_sptr                             sala_ir_;

    std::filesystem::path output_path_;
};
} // namespace

void DumpAvailCopy::set_output_path(std::filesystem::path output_path)
{
    output_path_ = std::move(output_path);
}

void DumpAvailCopy::run(program::ProgramIR_sptr sala_ir)
{
    ASSUMPTION(sala_ir != nullptr);
    // std::cout << "DLi: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir), output_path_);
    // std::cout << "DLi: done" << std::endl;
}

} // namespace optimizer::passes
