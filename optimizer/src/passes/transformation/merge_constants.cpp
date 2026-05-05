#include <optimizer/passes/transformation/merge_constants.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/utils/sparse_map.hpp>
#include <optimizer/utils/sparse_set.hpp>

#include <utility/development.hpp>
#include <utility/invariants.hpp>
#include <utility/timeprof.hpp>

#include <algorithm>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace optimizer::passes
{
using ConstantView = std::span<const uint8_t>;

namespace
{
struct ConstantViewHash
{
    std::size_t operator()(ConstantView s) const noexcept
    {
        auto sv = std::string_view(reinterpret_cast<const char*>(s.data()), s.size());
        return std::hash<std::string_view>{}(sv);
    }
};

struct ConstantViewEq
{
    bool operator()(ConstantView a, ConstantView b) const noexcept
    {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    }
};

class Impl
{
  public:
    explicit Impl(program::ProgramIR_sptr sala_ir) : sala_ir_(std::move(sala_ir)) { run(); }

  private:
    void run()
    {
        if (!group_constants())
        {
            return;
        }

        collect_possibly_redefined_constants();
        select_canonical_constants();

        if (replacement_map_.empty())
        {
            return;
        }

        redirect_to_canonical();
        release_duplicates();
    }

  private:
    bool group_constants()
    {
        bool duplicate_found = false;

        for (const auto& constant : sala_ir_->get_constants())
        {
            auto [it, inserted] = grouped_constants_.try_emplace(
                    constant->get_bytes(), std::vector<program::ConstantIR_sptr>{});

            if (!inserted)
            {
                duplicate_found = true;
            }

            it->second.push_back(constant);
        }

        return duplicate_found;
    }

    void collect_possibly_redefined_constants()
    {
        const auto static_init = sala_ir_->get_static_initializer_func();

        for (const auto& basic_block : static_init->get_basic_blocks())
        {
            for (const auto& instruction : basic_block->get_instructions())
            {
                switch (instruction->get_opcode())
                {
                case sala::Instruction::Opcode::ADDRESS:
                {
                    /*
                     * ADDRESS is treated conservatively: if the address of a
                     * constant is taken during static initialization, the
                     * constant may be modified indirectly later.
                     */
                    for (const auto& operand : instruction->get_operands())
                    {
                        mark_possibly_redefined_constant(operand);
                    }
                    break;
                }

                case sala::Instruction::Opcode::LOAD:
                case sala::Instruction::Opcode::COPY:
                case sala::Instruction::Opcode::MALLOC:
                case sala::Instruction::Opcode::ADD:
                case sala::Instruction::Opcode::SUB:
                case sala::Instruction::Opcode::MUL:
                case sala::Instruction::Opcode::DIV:
                case sala::Instruction::Opcode::REM:
                case sala::Instruction::Opcode::AND:
                case sala::Instruction::Opcode::OR:
                case sala::Instruction::Opcode::XOR:
                case sala::Instruction::Opcode::SHL:
                case sala::Instruction::Opcode::SHR:
                case sala::Instruction::Opcode::NEG:
                case sala::Instruction::Opcode::F2I:
                case sala::Instruction::Opcode::I2F:
                case sala::Instruction::Opcode::P2I:
                case sala::Instruction::Opcode::I2P:
                case sala::Instruction::Opcode::EXTEND:
                case sala::Instruction::Opcode::TRUNCATE:
                case sala::Instruction::Opcode::LESS:
                case sala::Instruction::Opcode::LESS_EQUAL:
                case sala::Instruction::Opcode::GREATER:
                case sala::Instruction::Opcode::GREATER_EQUAL:
                case sala::Instruction::Opcode::EQUAL:
                case sala::Instruction::Opcode::UNEQUAL:
                case sala::Instruction::Opcode::MOVEPTR:
                {
                    /*
                     * For these instructions, the first operand is treated as
                     * the destination. If it is a constant, that constant is
                     * unsafe as a canonical target.
                     */
                    const auto& operand = *instruction->get_operands().begin();
                    mark_possibly_redefined_constant(operand);
                    break;
                }

                case sala::Instruction::Opcode::__INVALID__:
                case sala::Instruction::Opcode::STORE:
                case sala::Instruction::Opcode::NOP:
                case sala::Instruction::Opcode::HALT:
                case sala::Instruction::Opcode::MEMCPY:
                case sala::Instruction::Opcode::MEMMOVE:
                case sala::Instruction::Opcode::MEMSET:
                case sala::Instruction::Opcode::ALLOCA:
                case sala::Instruction::Opcode::STACKSAVE:
                case sala::Instruction::Opcode::STACKRESTORE:
                case sala::Instruction::Opcode::FREE:
                case sala::Instruction::Opcode::ISNAN:
                case sala::Instruction::Opcode::JUMP:
                case sala::Instruction::Opcode::BRANCH:
                case sala::Instruction::Opcode::CALL:
                case sala::Instruction::Opcode::RET:
                case sala::Instruction::Opcode::VA_START:
                case sala::Instruction::Opcode::VA_END:
                case sala::Instruction::Opcode::VA_ARG:
                case sala::Instruction::Opcode::VA_COPY:
                    break;

                default:
                    NOT_SUPPORTED();
                }
            }
        }
    }

    void mark_possibly_redefined_constant(const program::OperandIR_raw operand)
    {
        if (!std::holds_alternative<program::ConstantIR_raw>(operand))
        {
            return;
        }

        const auto constant = std::get<program::ConstantIR_raw>(operand);
        possibly_redefined_constants_.insert(constant);
    }

    void select_canonical_constants()
    {
        for (const auto& [_, constants] : grouped_constants_)
        {
            if (constants.size() < 2)
            {
                continue;
            }

            const auto canonical_it = std::find_if(
                    constants.begin(), constants.end(),
                    [this](const program::ConstantIR_sptr& constant)
                    { return !possibly_redefined_constants_.contains(constant.get()); });

            /*
             * If every constant with this byte sequence may be redefined during
             * static initialization, no safe representative exists.
             */
            if (canonical_it == constants.end())
            {
                continue;
            }

            const auto& canonical = *canonical_it;

            for (const auto& constant : constants)
            {
                if (constant == canonical)
                {
                    continue;
                }

                /*
                 * A possibly redefined constant must keep its own storage.
                 * It must not be redirected to another constant, and it must
                 * not be used as the canonical target for other constants.
                 */
                if (possibly_redefined_constants_.contains(constant.get()))
                {
                    continue;
                }

                const auto [_, inserted] = replacement_map_.try_emplace(constant.get(), canonical);
                INVARIANT(inserted);

                duplicates_to_release_.push_back(constant);
            }
        }
    }

    void redirect_to_canonical() const
    {
        for (const auto& function : sala_ir_->get_functions())
        {
            for (const auto& basic_block : function->get_basic_blocks())
            {
                for (const auto& instruction : basic_block->get_instructions())
                {
                    for (auto& operand : instruction->get_operands())
                    {
                        if (!std::holds_alternative<program::ConstantIR_raw>(operand))
                        {
                            continue;
                        }

                        const auto constant       = std::get<program::ConstantIR_raw>(operand);
                        const auto replacement_it = replacement_map_.find(constant);

                        if (replacement_it == replacement_map_.end())
                        {
                            continue;
                        }

                        operand = replacement_it->second.get();
                    }
                }
            }
        }
    }

    void release_duplicates()
    {
        for (const auto& duplicate : duplicates_to_release_)
        {
            sala_ir_->release_constant(duplicate);
        }
    }

  private:
    program::ProgramIR_sptr sala_ir_;

    std::unordered_map<ConstantView, std::vector<program::ConstantIR_sptr>, ConstantViewHash,
                       ConstantViewEq>
            grouped_constants_;

    utils::SparseSet<program::ConstantIR_raw> possibly_redefined_constants_;

    utils::SparseMap<program::ConstantIR_raw, program::ConstantIR_sptr> replacement_map_;

    std::vector<program::ConstantIR_sptr> duplicates_to_release_;
};
} // namespace

void MergeConstants::run(program::ProgramIR_sptr sala_ir)
{
    TMPROF_BLOCK();
    const auto trigger = Impl(std::move(sala_ir));
}
} // namespace optimizer::passes
