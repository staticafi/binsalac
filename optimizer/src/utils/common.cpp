#include <optimizer/utils/common.hpp>

#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>

#include <utility/development.hpp>

namespace optimizer::utils
{
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
