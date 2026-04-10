#ifndef OPTIMIZER_TESTS_ANALYSIS_UTILS_HPP_INLCUDED
#define OPTIMIZER_TESTS_ANALYSIS_UTILS_HPP_INLCUDED

#include <optimizer/passes/translation/lower_sala_to_ir.hpp>
#include <optimizer/programIR/ir_types.hpp>

#include <sala/program.hpp>
#include <sala/streaming.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
namespace optimizer::tests
{
optimizer::program::ProgramIR_sptr load_program_via_lowering(const fs::path& input)
{
    auto P = std::make_shared<sala::Program>();
    {
        std::ifstream istr(input.c_str(), std::ios_base::binary);
        istr >> *P;
    }

    optimizer::passes::LowerSalaToIR lower;
    return lower.run(P);
}
} // namespace optimizer::tests
#endif
