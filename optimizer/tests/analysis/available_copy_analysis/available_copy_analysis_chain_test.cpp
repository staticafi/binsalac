#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "../analysis_test_harness.hpp"
#include "../analysis_test_registry.hpp"
#include "../utils.hpp"

#include <optimizer/passes/analysis/available_copy_analysis.hpp>
#include <optimizer/passes/debug/dump_available_copy.hpp>

#include <optimizer/passes/translation/lower_sala_to_ir.hpp>
#include <sala/program.hpp>
#include <sala/streaming.hpp>

namespace fs = std::filesystem;

TEST_CASE("available-copy analysis golden files match expected debug dumps",
          "[analysis][aca][golden][available copy]")
{
    const fs::path programs_dir = fs::absolute(AVAILABLE_COPY_ANALYSIS_PROGRAMS_DIR);

    const auto cases = optimizer::tests::collect_cases_from_directory(programs_dir, ".json");

    REQUIRE_FALSE(cases.empty());

    for (const auto& test_case : cases)
    {
        DYNAMIC_SECTION(test_case.name)
        {

            optimizer::tests::AnalysisTestHarness harness(
                    optimizer::tests::load_program_via_lowering);
            optimizer::passes::AvailableCopyAnalysis available_copy;
            optimizer::passes::DumpAvailCopy         dump_available_copy;
            dump_available_copy.set_output_path(programs_dir);

            harness.add_program(test_case.input_program)
                    .add_analysis("available_copy_analysis",
                                  [&](const auto& ir) { available_copy.run(ir); })
                    .add_analysis("dump_available_copy",
                                  [&](const auto& ir) { dump_available_copy.run(ir); });

            REQUIRE_NOTHROW(harness.run_and_compare(".available_copy_debug",
                                                    ".available_copy_debug.expected"));
            harness.cleanup_generated_files();
        }
    }
}
