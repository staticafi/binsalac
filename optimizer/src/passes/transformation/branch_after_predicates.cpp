#include <optimizer/passes/transformation/branch_after_predicates.hpp>

#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/query/translation_query.hpp>

#include <utility/assumptions.hpp>
#include <utility/log.hpp>
#include <utility/timeprof.hpp>

#include <sstream>
#include <utility>
#include <vector>

namespace optimizer::passes
{
namespace
{

using PendingRemoval = std::vector<program::InstructionIR_sptr>;

std::string me()
{
    std::ostringstream oss;
    oss << pipeline::names::branch_after_predicates;
    oss << ": ";
    return oss.str();
}

std::string info(const program::FunctionIR_sptr& function)
{
    ASSUMPTION(function != nullptr);

    const auto program = function->get_program();
    ASSUMPTION(program != nullptr);

    query::TranslationQuery translation{program};

    auto name = translation.function_name(function);
    if (name.empty())
    {
        name = "<unnamed>";
    }

    std::ostringstream oss;
    oss << "[" << name << "] ";
    return oss.str();
}


class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_{std::move(sala_ir)}
    {
        ASSUMPTION(sala_ir_ != nullptr);
        run();
    }

  private:
    void run()
    {
        LOG(LSL_DEBUG, me() << "Running implementation");

        // TODO!

        LOG(LSL_DEBUG, me() << "Done implementation");
    }

  private:
    program::ProgramIR_sptr sala_ir_;
};

} // namespace

void BranchAfterPredicates::run(program::ProgramIR_sptr sala_ir)
{
    LOG(LSL_INFO, me() << "Running");
    {
        TMPROF_BLOCK();
        const auto trigger = Impl(std::move(sala_ir));
    }
    LOG(LSL_INFO, me() << "Done");
}

} // namespace optimizer::passes
