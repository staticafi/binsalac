#include <llvm2sala/program_info.hpp>

std::string  get_program_name()
{
    return "llvm2sala";
}

std::string  get_program_version()
{
    return "0.1";
}

std::string  get_program_description()
{
    return "Translates a LLVM file (.ll) to a Sala file (.json).\n";
}
