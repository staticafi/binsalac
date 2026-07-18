#include <optimizer/passes/transformation/branch_after_predicates.hpp>

#include <optimizer/pipeline/pass_names.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/query/translation_query.hpp>

#include <utility/assumptions.hpp>
#include <utility/invariants.hpp>
#include <utility/log.hpp>
#include <utility/timeprof.hpp>

#include <sstream>
#include <utility>
#include <vector>

namespace optimizer::passes
{
namespace
{

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

void collect_internal_predicates(
        program::ProgramIR_sptr const& sala_ir,
        std::vector<program::InstructionIR_sptr>& internal_predicates
        )
{
    ASSUMPTION(sala_ir != nullptr);

    LOG(LSL_DEBUG, me() << "Collecting internal predicates");

    for (const auto& function : sala_ir->get_functions())
    {
        if (function == nullptr)
        {
            continue;
        }
        if (function->get_external_flag())
        {
            LOG(LSL_DEBUG, me() << info(function) << "Skipping external function");
            continue;
        }
        if (function->get_initializer_flag())
        {
            LOG(LSL_DEBUG, me() << info(function) << "Skipping static initializer");
            continue;
        }

        const auto previous_count = internal_predicates.size();

        for (const auto& basic_block : function->get_basic_blocks())
        {
            for (const auto& instruction : basic_block->get_instructions())
            {
                ASSUMPTION(instruction != nullptr);

                switch (instruction->get_opcode())
                {
                    case sala::Instruction::Opcode::LESS:
                    case sala::Instruction::Opcode::LESS_EQUAL:
                    case sala::Instruction::Opcode::GREATER:
                    case sala::Instruction::Opcode::GREATER_EQUAL:
                    case sala::Instruction::Opcode::EQUAL:
                    case sala::Instruction::Opcode::UNEQUAL:
                    case sala::Instruction::Opcode::ISNAN:
                        break;
                    default:
                        continue;
                }

                if (instruction->get_self_it())
                {
                    const auto& succ_instruction = *std::next(*instruction->get_self_it());
                    if (succ_instruction->get_opcode() != sala::Instruction::Opcode::BRANCH)
                        internal_predicates.push_back(instruction);
                }
            }
        }

        LOG(LSL_DEBUG, me() << info(function) << "Collected internal predicates: count="
                            << (internal_predicates.size() - previous_count));
    }


    LOG(LSL_DEBUG, me() << "Collected internal predicates: total=" << internal_predicates.size());
}

std::unique_ptr<metadata::translation::InstructionMeta> clone_instruction_metadata(program::InstructionIR_sptr const& instruction)
{
    auto  translation_meta = std::make_unique<metadata::translation::InstructionMeta>();
    translation_meta->source_back_mapping = instruction->get_metadata().get<metadata::translation::InstructionMeta>().source_back_mapping;
    return translation_meta;
}

void create_diamond_after(program::InstructionIR_sptr const& instruction)
{
    //metadata::translation::InstructionMeta const meta = instruction->get_metadata().get<metadata::translation::InstructionMeta>();
    sala::SourceBackMapping const back_mapping = instruction->get_metadata().get<metadata::translation::InstructionMeta>().source_back_mapping;
    auto basic_block = instruction->get_basic_block();
    const auto& function = instruction->get_basic_block()->get_function();
    
    program::BasicBlockIR_sptr const basic_block_end = std::make_shared<program::BasicBlockIR>();
    function->acquire_basic_block(basic_block_end);
    for (auto it = std::next(*instruction->get_self_it()); it != basic_block->get_instructions().end(); )
    {
        program::InstructionIR_sptr instr = *it;
        it = basic_block->release_instruction(instr);
        basic_block_end->acquire_instruction(instr);
    }

    program::InstructionIR_sptr new_instr = std::make_shared<program::InstructionIR>();
    new_instr->get_opcode() = sala::Instruction::Opcode::BRANCH;
    new_instr->get_modifier() = sala::Instruction::Modifier::NONE;
    new_instr->get_operands().push_back(instruction->get_operands().front());
    new_instr->get_metadata().set(clone_instruction_metadata(instruction));
    basic_block->acquire_instruction(new_instr);
    
    program::BasicBlockIR_sptr const basic_block_0 = std::make_shared<program::BasicBlockIR>();
    function->acquire_basic_block(basic_block_0);
    new_instr = std::make_shared<program::InstructionIR>();
    new_instr->get_opcode() = sala::Instruction::Opcode::JUMP;
    new_instr->get_modifier() = sala::Instruction::Modifier::NONE;
    new_instr->get_metadata().set(clone_instruction_metadata(instruction));
    basic_block_0->acquire_instruction(new_instr);

    program::BasicBlockIR_sptr const basic_block_1 = std::make_shared<program::BasicBlockIR>();
    function->acquire_basic_block(basic_block_1);
    new_instr = std::make_shared<program::InstructionIR>();
    new_instr->get_opcode() = sala::Instruction::Opcode::JUMP;
    new_instr->get_modifier() = sala::Instruction::Modifier::NONE;
    new_instr->get_metadata().set(clone_instruction_metadata(instruction));
    basic_block_1->acquire_instruction(new_instr);

    while (!basic_block->get_successors().empty())
    {
        auto const succ_basic_block = basic_block->get_successors().front().lock();
        basic_block->remove_successor(succ_basic_block);
        succ_basic_block->remove_predecessor(basic_block);
        succ_basic_block->add_predecessor(basic_block_end);
        basic_block_end->add_successor(succ_basic_block);
    }

    basic_block->add_successor(basic_block_0);
    basic_block_0->add_predecessor(basic_block);

    basic_block->add_successor(basic_block_1);
    basic_block_1->add_predecessor(basic_block);

    basic_block_0->add_successor(basic_block_end);
    basic_block_end->add_predecessor(basic_block_0);

    basic_block_1->add_successor(basic_block_end);
    basic_block_end->add_predecessor(basic_block_1);
}

bool branching_needs_fix(program::InstructionIR_sptr const& branch_instr)
{
    ASSUMPTION(branch_instr->get_basic_block()->get_successors().size() == 2ULL);
    for (auto const& succ_block_wptr: branch_instr->get_basic_block()->get_successors())
    {
        auto succ_basic_block = succ_block_wptr.lock();

        ASSUMPTION(succ_basic_block != nullptr && !succ_basic_block->get_instructions().empty());
        auto const& first_instr = succ_basic_block->get_instructions().front();

        if (first_instr->get_opcode() != sala::Instruction::Opcode::COPY)
            return true;

        ASSUMPTION(first_instr->get_operands().size() == 2ULL);
        if (first_instr->get_operands().front() != branch_instr->get_operands().front())
            return true;
        if (!std::holds_alternative<program::ConstantIR_raw>(first_instr->get_operands().back()))
            return true;
    }
    return false;
}

void insert_copy_after_branch(program::ProgramIR_sptr const& sala_ir)
{
    ASSUMPTION(sala_ir != nullptr);

    LOG(LSL_DEBUG, me() << "Inserting COPY after BRANCH.");

    std::vector<program::InstructionIR_sptr> branchings_to_fix;
    for (const auto& function : sala_ir->get_functions())
    {
        if (function == nullptr)
            continue;
        if (function->get_external_flag())
        {
            LOG(LSL_DEBUG, me() << info(function) << "Skipping external function");
            continue;
        }
        if (function->get_initializer_flag())
        {
            LOG(LSL_DEBUG, me() << info(function) << "Skipping static initializer");
            continue;
        }

        for (const auto& basic_block : function->get_basic_blocks())
            for (const auto& instruction : basic_block->get_instructions())
            {
                ASSUMPTION(instruction != nullptr);
                if (instruction->get_opcode() == sala::Instruction::Opcode::BRANCH && branching_needs_fix(instruction))
                    branchings_to_fix.push_back(instruction);
            }
    }

    if (branchings_to_fix.empty())
    {
        LOG(LSL_DEBUG, me() << "No branching needs fixing => we are done.");
        return;
    }

    LOG(LSL_DEBUG, me() << "Found " << branchings_to_fix.size() << " BRANCH instructions to be fixed.");

    program::ConstantIR_sptr constants[2];
    {
        constants[0] = std::make_shared<program::ConstantIR>();
        constants[0]->get_bytes().push_back(0U);
        sala_ir->acquire_constant(constants[0]);

        constants[1] = std::make_shared<program::ConstantIR>();
        constants[1]->get_bytes().push_back(1U);
        sala_ir->acquire_constant(constants[1]);
    }

    for (auto const& branch_instr : branchings_to_fix)
    {
        auto const basic_block{ branch_instr->get_basic_block() };

        program::BasicBlockIR_sptr const succ_basic_blocks[2] {
            basic_block->get_successors().front().lock(),
            basic_block->get_successors().back().lock()
        };

        for (std::size_t i = 0ULL; i != 2ULL; ++i)
        {
            program::InstructionIR_sptr new_instr = std::make_shared<program::InstructionIR>();
            new_instr->get_opcode() = sala::Instruction::Opcode::COPY;
            new_instr->get_modifier() = sala::Instruction::Modifier::NONE;
            new_instr->get_operands().push_back(branch_instr->get_operands().front());
            new_instr->push_back_operand(constants[i]);
            new_instr->get_metadata().set(clone_instruction_metadata(branch_instr));

            if (succ_basic_blocks[i]->get_predecessors().size() == 1ULL)
            {
                INVARIANT(succ_basic_blocks[i]->get_predecessors().front().lock() == basic_block);
                succ_basic_blocks[i]->acquire_instruction_front(new_instr);
                continue;
            }

            program::BasicBlockIR_sptr const new_basic_block = std::make_shared<program::BasicBlockIR>();
            basic_block->get_function()->acquire_basic_block(new_basic_block);

            new_basic_block->acquire_instruction_front(new_instr);

            new_instr = std::make_shared<program::InstructionIR>();
            new_instr->get_opcode() = sala::Instruction::Opcode::JUMP;
            new_instr->get_modifier() = sala::Instruction::Modifier::NONE;
            new_instr->get_metadata().set(clone_instruction_metadata(branch_instr));
            new_basic_block->acquire_instruction(new_instr);

            basic_block->remove_successor(succ_basic_blocks[i]);
            succ_basic_blocks[i]->remove_predecessor(basic_block);

            if (i == 0UL)
                basic_block->add_successor_front(new_basic_block);
            else
                basic_block->add_successor(new_basic_block);
            new_basic_block->add_predecessor(basic_block);

            new_basic_block->add_successor(succ_basic_blocks[i]);
            succ_basic_blocks[i]->add_predecessor(new_basic_block);
        }
    }

    LOG(LSL_DEBUG, me() << "Finished insertion of COPY after BRANCH: total fixes=" << 0);
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

        {
            std::vector<program::InstructionIR_sptr> internal_predicates;
            collect_internal_predicates(sala_ir_, internal_predicates);
            if (!internal_predicates.empty())
            {
                LOG(LSL_DEBUG, me() << "Building diamonds after collected instructions");
                for (auto const& instruction : internal_predicates)
                    create_diamond_after(instruction);
            }
        }

        insert_copy_after_branch(sala_ir_);

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
