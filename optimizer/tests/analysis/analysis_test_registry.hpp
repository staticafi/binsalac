#ifndef OPTIMIZER_TESTS_ANALYSIS_TEST_REGISTRY_HPP_INCLUDED
#define OPTIMIZER_TESTS_ANALYSIS_TEST_REGISTRY_HPP_INCLUDED
#include <filesystem>
#include <string>
#include <vector>

namespace optimizer::tests
{

struct AnalysisGoldenCase
{
    std::string           name;
    std::filesystem::path input_program;
};

inline std::vector<AnalysisGoldenCase>
collect_cases_from_directory(const std::filesystem::path& dir, const std::string& input_ext)
{
    std::vector<AnalysisGoldenCase> cases;

    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const auto& path = entry.path();
        if (path.extension() != input_ext)
        {
            continue;
        }

        cases.push_back({path.stem().string(), path});
    }

    return cases;
}

} // namespace optimizer::tests
#endif
