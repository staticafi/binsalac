#include <llvm2sala/program_info.hpp>
#include <llvm2sala/program_options.hpp>
#include <llvm2sala/compiler.hpp>
#include <sala/program.hpp>
#include <sala/streaming.hpp>
#include <utility/timeprof.hpp>
#include <iostream>
#include <fstream>

int run(int argc, char* argv[])
{
    TMPROF_BLOCK();

    if (get_program_options()->has("help"))
    {
        std::cout << get_program_options() << std::endl;
        return 1;
    }
    if (get_program_options()->has("version"))
    {
        std::cout << get_program_options()->value("version") << std::endl;
        return 2;
    }
    if (!get_program_options()->has("input"))
    {
        std::cout << "The input file was not specified." << std::endl;
        return 3;
    }

    sala::Program P;
    compile_llvm_file(get_program_options()->value("input"), P);

    std::string const entry_function_name{ get_program_options()->has("entry") ? get_program_options()->value("entry") : "main" };
    bool entry_was_set{ false };
    for (sala::Function const& function : P.functions())
        if (function.name() == entry_function_name)
        {
            P.set_entry_function(function.index());
            entry_was_set = true;
        }
    if (!entry_was_set)
    {
        std::cout << "Could not find the entry function '" << entry_function_name << "'." << std::endl;
        return 4;
    }

    std::filesystem::path output_json_file{
        get_program_options()->has("output") ?
            std::filesystem::path{ get_program_options()->value("output") } :
            std::filesystem::path{ get_program_options()->value("input") }.replace_extension(".json")
        };

    std::ofstream ostr(output_json_file.c_str(), std::ios_base::binary);
    ostr << P;
    if (get_program_options()->has("jsonc"))
    {
        ostr.close();
        ostr.open(output_json_file.replace_extension("jsonc").c_str(), std::ios_base::binary);
        ostr << sala::enable_json_comments << P << sala::disable_json_comments;
    }

    return 0;
}
