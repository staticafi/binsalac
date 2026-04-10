#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "../analysis_test_harness.hpp"
#include "../analysis_test_registry.hpp"
#include "../utils.hpp"

#include <optimizer/passes/analysis/global_points_to_analysis.hpp>
#include <optimizer/passes/analysis/local_points_to_analysis.hpp>
#include <optimizer/passes/debug/dump_points_to.hpp>

#include <optimizer/passes/translation/lower_sala_to_ir.hpp>
#include <sala/program.hpp>
#include <sala/streaming.hpp>

namespace fs = std::filesystem;

TEST_CASE("points-to golden files match expected debug dumps", "[analysis][points_to][golden]")
{
    const fs::path programs_dir = fs::absolute(fs::path{POINTS_TO_ANALYSIS_PROGRAMS_DIR});

    const auto cases = optimizer::tests::collect_cases_from_directory(programs_dir, ".json");

    REQUIRE_FALSE(cases.empty());

    for (const auto& test_case : cases)
    {
        DYNAMIC_SECTION(test_case.name)
        {

            optimizer::tests::AnalysisTestHarness harness(
                    optimizer::tests::load_program_via_lowering);
            optimizer::passes::GlobalPointsToAnalysis global_points;
            optimizer::passes::LocalPointsToAnalysis  local_points;
            optimizer::passes::DumpPointsTo           dump_may_points;
            dump_may_points.set_output_path(programs_dir);

            harness.add_program(test_case.input_program)
                    .add_analysis("global_points_to_analysis",
                                  [&](const auto& ir) { global_points.run(ir); })
                    .add_analysis("local_points_to_analysis",
                                  [&](const auto& ir) { local_points.run(ir); })
                    .add_analysis("dump_may_points",
                                  [&](const auto& ir) { dump_may_points.run(ir); });

            REQUIRE_NOTHROW(
                    harness.run_and_compare(".points_to_debug", ".points_to_debug.expected"));
            harness.cleanup_generated_files();
        }
    }
}
