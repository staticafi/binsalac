#include <llvm2sala/program_info.hpp>
#include <llvm2sala/program_options.hpp>
#include <llvm2sala/compiler.hpp>
#include <sala/program.hpp>
#include <sala/streaming.hpp>
#include <utility/timeprof.hpp>
#include <iostream>
#include <fstream>

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

    sala::Program P;
    compile_llvm_file(get_program_options()->value("input"), P);

    std::filesystem::path output_json_file{
        get_program_options()->has("output") ?
            std::filesystem::path{ get_program_options()->value("output") } :
            std::filesystem::path{ get_program_options()->value("input") }.replace_extension(".json")
        };

    std::ofstream ostr(output_json_file.c_str(), std::ios_base::binary);
    ostr << P;
    if (get_program_options()->has("jsonx"))
    {
        ostr.close();
        ostr.open(output_json_file.replace_extension("jsonx").c_str(), std::ios_base::binary);
        ostr << sala::enable_json_comments << P << sala::disable_json_comments;
    }
}
