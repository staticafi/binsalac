#ifndef OPTIMIZER_PASSES_DEBUG_DUMP_POINTS_TO_HPP_INCLUDED
#define OPTIMIZER_PASSES_DEBUG_DUMP_POINTS_TO_HPP_INCLUDED
#include <filesystem>
#include <optimizer/programIR/ir_types.hpp>

namespace optimizer::passes
{
class DumpPointsTo
{
  public:
    void run(program::ProgramIR_sptr sala_ir);

    void set_output_path(std::filesystem::path output_path);

  private:
    std::filesystem::path output_path_{};
};
} // namespace optimizer::passes
#endif
