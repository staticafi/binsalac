#ifndef OPTIMIZER_TESTS_ANALYSIS_TEST_HARNESS_HPP_INCLUDED
#define OPTIMIZER_TESTS_ANALYSIS_TEST_HARNESS_HPP_INCLUDED

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <optimizer/programIR/program_ir.hpp>

namespace optimizer::tests
{

class AnalysisTestHarness
{
  public:
    using ProgramLoader = std::function<program::ProgramIR_sptr(const std::filesystem::path&)>;
    using AnalysisStep  = std::function<void(const program::ProgramIR_sptr&)>;

    explicit AnalysisTestHarness(ProgramLoader loader) : loader_(std::move(loader)) {}

    ~AnalysisTestHarness() noexcept
    {
        if (cleanup_on_destruction_)
        {
            cleanup_generated_files();
        }
    }

    AnalysisTestHarness& add_program(const std::filesystem::path& input_program)
    {
        programs_.push_back(input_program);
        return *this;
    }

    AnalysisTestHarness& add_analysis(std::string name, AnalysisStep step)
    {
        analyses_.push_back({std::move(name), std::move(step)});
        return *this;
    }

    AnalysisTestHarness& set_cleanup_on_destruction(bool value = true)
    {
        cleanup_on_destruction_ = value;
        return *this;
    }

    void cleanup_generated_files() const noexcept
    {
        for (const auto& path : generated_files_)
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    void cleanup_generated_files(const std::string& produced_ext) const noexcept
    {
        for (const auto& program_path : programs_)
        {
            std::error_code ec;
            std::filesystem::remove(replace_extension_preserve_stem(program_path, produced_ext),
                                    ec);
        }
    }

    void run_and_compare(const std::string& produced_ext, const std::string& expected_ext)
    {
        generated_files_.clear();

        for (const auto& program_path : programs_)
        {
            const auto produced = replace_extension_preserve_stem(program_path, produced_ext);
            generated_files_.push_back(produced);

            {
                std::error_code ec;
                std::filesystem::remove(produced, ec);
            }

            const auto ir = loader_(program_path);
            if (ir == nullptr)
            {
                throw std::runtime_error("Failed to load program: " + program_path.string());
            }

            for (const auto& analysis : analyses_)
            {
                analysis.step(ir);
            }

            const auto expected = replace_extension_preserve_stem(program_path, expected_ext);

            const auto produced_text = read_file(produced);
            const auto expected_text = read_file(expected);

            if (produced_text != expected_text)
            {
                throw std::runtime_error(build_diff_message(program_path, produced, expected,
                                                            produced_text, expected_text));
            }
        }
    }

  private:
    struct NamedAnalysis
    {
        std::string  name;
        AnalysisStep step;
    };

    static std::filesystem::path replace_extension_preserve_stem(const std::filesystem::path& path,
                                                                 const std::string& new_ext)
    {
        auto result = path;
        result.replace_extension(new_ext);
        return result;
    }

    static std::string read_file(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        if (!in)
        {
            throw std::runtime_error("Failed to open file: " + path.string());
        }

        std::ostringstream oss;
        oss << in.rdbuf();
        return oss.str();
    }

    static std::vector<std::string> split_lines(const std::string& text)
    {
        std::vector<std::string> lines;
        std::stringstream        ss(text);
        std::string              line;

        while (std::getline(ss, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            lines.push_back(line);
        }

        if (!text.empty() && text.back() == '\n')
        {
            lines.emplace_back();
        }

        return lines;
    }

    static std::size_t first_diff_pos(std::string_view a, std::string_view b)
    {
        const std::size_t min_len = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < min_len; ++i)
        {
            if (a[i] != b[i])
            {
                return i;
            }
        }
        return min_len;
    }

    static std::string marker_line(std::size_t pos) { return std::string(pos, ' ') + '^'; }

    static std::string build_diff_message(const std::filesystem::path& input,
                                          const std::filesystem::path& produced,
                                          const std::filesystem::path& expected,
                                          const std::string&           produced_text,
                                          const std::string&           expected_text)
    {
        const auto produced_lines = split_lines(produced_text);
        const auto expected_lines = split_lines(expected_text);

        std::ostringstream oss;
        oss << "Analysis output mismatch for input: " << input << '\n'
            << "Produced: " << produced << '\n'
            << "Expected: " << expected << '\n';

        const std::size_t max_lines = std::max(produced_lines.size(), expected_lines.size());

        bool any_diff = false;
        for (std::size_t i = 0; i < max_lines; ++i)
        {
            const bool has_expected = i < expected_lines.size();
            const bool has_produced = i < produced_lines.size();

            const std::string_view exp =
                    has_expected ? std::string_view(expected_lines[i]) : std::string_view{};
            const std::string_view pro =
                    has_produced ? std::string_view(produced_lines[i]) : std::string_view{};

            if (!has_expected || !has_produced || exp != pro)
            {
                any_diff = true;
                oss << "\nLine " << (i + 1) << " differs:\n";
                oss << "  expected: " << (has_expected ? std::string(exp) : "<missing>") << '\n';
                oss << "  produced: " << (has_produced ? std::string(pro) : "<missing>") << '\n';

                if (has_expected && has_produced)
                {
                    const std::size_t diff_pos = first_diff_pos(exp, pro);
                    oss << "            " << marker_line(diff_pos) << '\n';
                }
            }
        }

        if (!any_diff)
        {
            oss << "\nTexts differ, but no line-level difference was detected. ";
        }

        return oss.str();
    }

  private:
    ProgramLoader                      loader_;
    std::vector<std::filesystem::path> programs_;
    std::vector<NamedAnalysis>         analyses_;
    std::vector<std::filesystem::path> generated_files_{};
    bool                               cleanup_on_destruction_{false};
};

} // namespace optimizer::tests

#endif
