#include <llvm2llvm/program_info.hpp>
#include <llvm2llvm/program_options.hpp>
#include <llvm2llvm/llvm_simplifier.hpp>
#include <utility/timeprof.hpp>
#include <iostream>
#include <filesystem>

void run(int argc, char* argv[])
{
    TMPROF_BLOCK();

    if (get_program_options()->has("help"))
    {
        std::cout << get_program_options() << std::endl;
        return;
    }
    if (get_program_options()->has("version"))
    {
        std::cout << get_program_options()->value("version") << std::endl;
        return;
    }
    if (!get_program_options()->has("input"))
    {
        std::cout << "The input file was not specified." << std::endl;
        return;
    }

    simplify_llvm_file(
        get_program_options()->value("input"),
        get_program_options()->has("output") ?
            get_program_options()->value("output") :
            std::filesystem::path{ get_program_options()->value("input") }.replace_extension(".sim.ll").string()
            );
}
