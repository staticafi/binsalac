#include <sala2sala/program_info.hpp>
#include <sala2sala/program_options.hpp>
#include <sala2sala/optimizer.hpp>
#include <sala/program.hpp>
#include <sala/streaming.hpp>
#include <utility/timeprof.hpp>
#include <iostream>
#include <fstream>
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

    sala::Program P;
    {
        std::filesystem::path input_json_file{ get_program_options()->value("input") };
        std::ifstream istr(input_json_file.c_str(), std::ios_base::binary);
        istr >> P;
    }

    Optimizer optimizer{ P };
    optimizer.run();

    std::filesystem::path output_json_file{ get_program_options()->value("output") };
    if (std::filesystem::is_directory(output_json_file))
        output_json_file.append(P.name() + ".json");
    std::ofstream ostr(output_json_file.c_str(), std::ios_base::binary);
    ostr << P;
    if (get_program_options()->has("jsonc"))
    {
        ostr.close();
        ostr.open(output_json_file.replace_extension("jsonc").c_str(), std::ios_base::binary);
        ostr << sala::enable_json_comments << P << sala::disable_json_comments;
    }
}
