#include <llvm2llvm/program_info.hpp>

std::string  get_program_name()
{
    return "llvm2llvm";
}

std::string  get_program_version()
{
    return "0.1";
}

std::string  get_program_description()
{
    return "Translates a LLVM file (.ll) to a simplified LLVM file (.ll).\n";
}
