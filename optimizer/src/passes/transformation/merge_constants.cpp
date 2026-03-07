#include <optimizer/passes/transformation/merge_constants.hpp>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>

#include <utility/invariants.hpp>

#include <span>
#include <unordered_set>
#include <utility/development.hpp>

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
} // namespace

class MergeConstants::Impl
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

        exclude_possibly_redefined();
        redirect_to_canonical();
        return;
    }

  private:
    void exclude_possibly_redefined()
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
                    for (const auto& operand : instruction->get_operands())
                    {
                        exclude_possible_const_operand(operand);
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
                    const auto& operand = *instruction->get_operands().begin();
                    exclude_possible_const_operand(operand);
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

    void exclude_possible_const_operand(const program::OperandIR_raw operand)
    {
        if (!std::holds_alternative<program::ConstantIR_raw>(operand))
        {
            return;
        }

        const auto constant = std::get<program::ConstantIR_raw>(operand);
        duplicates_.erase(constant);
    }

    void clear_context()
    {
        canonical_constant_map_.clear();
        duplicates_.clear();
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
                        const auto constant = std::get<program::ConstantIR_raw>(operand);
                        if (duplicates_.contains(constant))
                        {
                            operand = canonical_constant_map_.at(constant->get_bytes()).get();
                            continue;
                        }
                    }
                }
            }
        }
    }

    bool group_constants()
    {
        bool duplicit_found = false;
        for (auto constant_it = sala_ir_->get_constants().begin();
             constant_it != sala_ir_->get_constants().end();)
        {
            const auto& constant = *constant_it;
            const auto [_, succes] =
                    canonical_constant_map_.try_emplace(constant->get_bytes(), constant);
            if (!succes)
            {
                const auto next_constant = std::next(constant_it);
                const auto [_, success]  = duplicates_.insert(constant.get());
                INVARIANT(success);
                sala_ir_->release_constant(constant);
                duplicit_found = true;
                constant_it    = next_constant;
            }
            else
            {
                ++constant_it;
            }
        }

        return duplicit_found;
    }

  private:
    program::ProgramIR_sptr sala_ir_;
    std::unordered_map<ConstantView, program::ConstantIR_sptr, ConstantViewHash, ConstantViewEq>
                                                canonical_constant_map_;
    std::unordered_set<program::ConstantIR_raw> duplicates_;
};

program::ProgramIR_sptr MergeConstants::run(program::ProgramIR_sptr sala_ir)
{
    pImpl_ = std::make_unique<MergeConstants::Impl>(sala_ir);
    return sala_ir;
}
} // namespace optimizer::passes
