#include <optimizer/passes/debug/dump_liveness.hpp>

#include <optimizer/analysis/liveness_query.hpp>
#include <optimizer/metadata/liveness.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/common.hpp>
#include <optimizer/utils/liveness/import.hpp>
#include <optimizer/utils/sparse_set.hpp>

#include <utility/assumptions.hpp>

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
    Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)} { run(); }

    program::ProgramIR_sptr run()
    {
        ASSUMPTION(sala_ir_ != nullptr);

        build_operand_display_ids();

        const auto output_path =
                std::filesystem::path(utils::get_program_name(*sala_ir_) + ".liveness_debug");

        std::ofstream out(output_path, std::ios::out | std::ios::trunc);
        ASSUMPTION(out.is_open());

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

    void build_operand_display_ids()
    {
        next_display_id_ = 0;
        variable_ids_.clear();
        constant_ids_.clear();

        for (const auto& constant : sala_ir_->get_constants())
        {
            ASSUMPTION(constant != nullptr);
            constant_ids_.emplace(constant.get(), next_display_id_++);
        }

        for (const auto& variable : sala_ir_->get_static_vars())
        {
            ASSUMPTION(variable != nullptr);
            variable_ids_.emplace(variable.get(), next_display_id_++);
        }

        for (const auto& function : sala_ir_->get_functions())
        {
            ASSUMPTION(function != nullptr);

            for (const auto& parameter : function->get_parameters())
            {
                ASSUMPTION(parameter != nullptr);
                variable_ids_.emplace(parameter.get(), next_display_id_++);
            }

            for (const auto& local : function->get_local_variables())
            {
                ASSUMPTION(local != nullptr);
                variable_ids_.emplace(local.get(), next_display_id_++);
            }
        }
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
        out << utils::get_function_name(*function) << ": \n";

        out << utils::get_offset(offset) << "__params__: \n";
        out << utils::get_offset(offset) << "(\n";
        offset += OFFSET_MULT;
        for (const auto& param : function->get_parameters())
        {
            serialize_variable(out, *param, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ")\n";

        out << utils::get_offset(offset) << "__locals__: \n";
        out << utils::get_offset(offset) << "[\n";
        offset += OFFSET_MULT;
        for (const auto& local : function->get_local_variables())
        {
            serialize_variable(out, *local, offset);
            out << "\n";
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << "]\n";

        fill_bb_map(*function);

        analysis::LivenessQueryFunction query(function);

        out << utils::get_offset(offset) << "__basic_blocks__: \n";
        out << utils::get_offset(offset) << "<\n";
        offset += OFFSET_MULT;
        for (const auto& basic_block : function->get_basic_blocks())
        {
            serialize_basic_block(out, basic_block, query, offset);
        }
        offset -= OFFSET_MULT;
        out << utils::get_offset(offset) << ">\n\n";
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
                               analysis::LivenessQueryFunction& query, int offset)
    {
        ASSUMPTION(basic_block != nullptr);

        if (basic_block == basic_block->get_function()->get_entry_basic_block())
        {
            out << utils::get_offset(offset) << "__entry__\n";
        }

        out << utils::get_offset(offset) << bb_id_map_[basic_block] << ":\n";

        if (!basic_block->get_metadata().has<metadata::liveness::BasicBlockMeta>())
        {
            out << utils::get_offset(offset) << " >>ERROR METADATA NOT PRESENT<<\n";
        }
        else
        {
            const auto& meta =
                    basic_block->get_metadata().get<metadata::liveness::BasicBlockMeta>();

            out << utils::get_offset(offset) << " >>LIVE_OUT: ";
            serialize_live_set(out, meta.live_out);
            out << "<<\n";

            out << utils::get_offset(offset) << " >>REMOVABLE: {";
            bool        first = true;
            std::size_t idx   = 0;
            for (const auto& instruction : basic_block->get_instructions())
            {
                if (meta.removable_instructions.contains(instruction.get()))
                {
                    if (!first)
                    {
                        out << ", ";
                    }
                    first = false;
                    out << idx;
                }
                ++idx;
            }
            out << "}<<\n";
        }

        out << utils::get_offset(offset) << "{\n";
        serialize_instructions(out, basic_block->get_instructions(), query, offset + OFFSET_MULT);
        out << utils::get_offset(offset) << "}\n";

        out << utils::get_offset(offset) << "|_succ__:";
        for (const auto& succ : basic_block->get_successors())
        {
            auto sp = succ.lock();
            ASSUMPTION(sp != nullptr);
            out << utils::get_offset(1) << bb_id_map_[sp];
        }
        out << "\n";

        out << utils::get_offset(offset) << "|_pred__:";
        for (const auto& pred : basic_block->get_predecessors())
        {
            auto pp = pred.lock();
            ASSUMPTION(pp != nullptr);
            out << utils::get_offset(1) << bb_id_map_[pp];
        }
        out << "\n\n";
    }

    void serialize_instructions(std::ostream& out, const program::InstructionIRListS& instructions,
                                analysis::LivenessQueryFunction& query, int offset)
    {
        for (const auto& instruction : instructions)
        {
            ASSUMPTION(instruction != nullptr);

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

    void dump_instruction_query_info(std::ostream&                      out,
                                     const program::InstructionIR_sptr& instruction,
                                     analysis::LivenessQueryFunction&   query)
    {
        const auto live_before = query.live_before(instruction);
        const auto live_after  = query.live_after(instruction);
        const bool removable   = query.is_removable(instruction);

        out << "    >>QUERY: [removable=" << (removable ? "yes" : "no") << ", live_before=";
        serialize_live_set(out, live_before);
        out << ", live_after=";
        serialize_live_set(out, live_after);

        const auto  opcode   = instruction->get_opcode();
        const auto& operands = instruction->get_operands();

        bool printed_operand_info = false;
        for (std::size_t i = 0; i < operands.size(); ++i)
        {
            if (!utils::is_trackable_operand(operands[i]))
            {
                continue;
            }

            if (!printed_operand_info)
            {
                out << ", operands={";
                printed_operand_info = true;
            }
            else
            {
                out << "; ";
            }

            out << "op=" << i << ", kind=" << (utils::is_written_operand(opcode, i) ? "def" : "use")
                << ", value=";
            serialize_operand(out, operands[i], 0);

            if (const auto* var = std::get_if<program::VariableIR_raw>(&operands[i]);
                var != nullptr && *var != nullptr)
            {
                out << ", before=" << (query.is_live_before(instruction, **var) ? "live" : "dead");
                out << ", after=" << (query.is_live_after(instruction, **var) ? "live" : "dead");
            }
            else if (const auto* cnst = std::get_if<program::ConstantIR_raw>(&operands[i]);
                     cnst != nullptr && *cnst != nullptr)
            {
                out << ", before=" << (query.is_live_before(instruction, **cnst) ? "live" : "dead");
                out << ", after=" << (query.is_live_after(instruction, **cnst) ? "live" : "dead");
            }
        }

        if (printed_operand_info)
        {
            out << "}";
        }

        out << "] <<";
    }

    void serialize_live_set(std::ostream& out, const utils::LiveSet& live_set)
    {
        out << "{";
        bool first = true;
        for (const auto& operand : live_set)
        {
            if (!first)
            {
                out << ", ";
            }
            first = false;
            serialize_operand(out, operand, 0);
        }
        out << "} ";
    }

    void serialize_operand(std::ostream& out, const program::OperandIR_raw& operand, int offset)
    {
        if (const auto variable_raw = std::get_if<program::VariableIR_raw>(&operand))
        {
            ASSUMPTION(*variable_raw != nullptr);
            serialize_variable(out, **variable_raw, offset);
        }
        else if (const auto constant_raw = std::get_if<program::ConstantIR_raw>(&operand))
        {
            ASSUMPTION(*constant_raw != nullptr);
            serialize_constant(out, **constant_raw, offset);
        }
        else if (const auto function_raw = std::get_if<program::FunctionIR_raw>(&operand))
        {
            ASSUMPTION(*function_raw != nullptr);
            serialize_function_pointer(out, **function_raw, offset);
        }
    }

    void serialize_function_pointer(std::ostream& out, const program::FunctionIR& function,
                                    int offset)
    {
        out << utils::get_offset(offset) << utils::get_function_name(function);
    }

    void serialize_constant(std::ostream& out, const program::ConstantIR& constant, int offset)
    {
        const auto it = constant_ids_.find(const_cast<program::ConstantIR*>(&constant));
        if (it != constant_ids_.end())
        {
            out << utils::get_offset(offset) << it->second;
        }
        else
        {
            out << utils::get_offset(offset) << ERROR;
        }
    }

    void serialize_variable(std::ostream& out, const program::VariableIR& variable, int offset)
    {
        const auto it = variable_ids_.find(const_cast<program::VariableIR*>(&variable));
        if (it != variable_ids_.end())
        {
            out << utils::get_offset(offset) << it->second;
        }
        else
        {
            out << utils::get_offset(offset) << ERROR;
        }
    }

  private:
    std::unordered_map<program::BasicBlockIR_sptr, int> bb_id_map_;

    std::unordered_map<program::VariableIR_raw, int> variable_ids_;
    std::unordered_map<program::ConstantIR_raw, int> constant_ids_;
    int                                              next_display_id_{0};

    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void DumpLiveness::run(program::ProgramIR_sptr sala_ir)
{
    ASSUMPTION(sala_ir != nullptr);
    std::cout << "DLi: started" << std::endl;
    const auto trigger = Impl(std::move(sala_ir));
    std::cout << "DLi: done" << std::endl;
}

} // namespace optimizer::passes
