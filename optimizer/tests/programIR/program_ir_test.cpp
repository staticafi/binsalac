#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <variant>
#include <vector>

#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

namespace
{
using optimizer::program::BasicBlockIR;
using optimizer::program::BasicBlockIR_sptr;
using optimizer::program::ConstantIR;
using optimizer::program::ConstantIR_sptr;
using optimizer::program::FunctionIR;
using optimizer::program::FunctionIR_sptr;
using optimizer::program::InstructionIR;
using optimizer::program::InstructionIR_sptr;
using optimizer::program::ProgramIR;
using optimizer::program::ProgramIR_sptr;
using optimizer::program::release_variable;
using optimizer::program::VariableIR;
using optimizer::program::VariableIR_sptr;

template <typename ListT>
std::size_t list_size(const ListT& list)
{
    return static_cast<std::size_t>(std::distance(list.begin(), list.end()));
}
} // namespace

TEST_CASE("ProgramIR acquires and releases functions", "[ProgramIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();

    REQUIRE(function->get_program() == nullptr);
    REQUIRE_FALSE(function->get_self_it().has_value());
    REQUIRE(list_size(program->get_functions()) == 0);

    program->acquire_function(function);

    REQUIRE(function->get_program() == program);
    REQUIRE(function->get_self_it().has_value());
    REQUIRE(list_size(program->get_functions()) == 1);
    REQUIRE((*function->get_self_it())->get() == function.get());

    program->release_function(function);

    REQUIRE(function->get_program() == nullptr);
    REQUIRE_FALSE(function->get_self_it().has_value());
    REQUIRE(list_size(program->get_functions()) == 0);
}

TEST_CASE("FunctionIR can be reassigned between programs", "[ProgramIR][FunctionIR]")
{
    auto program1 = std::make_shared<ProgramIR>();
    auto program2 = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();

    program1->acquire_function(function);
    REQUIRE(function->get_program() == program1);
    REQUIRE(list_size(program1->get_functions()) == 1);
    REQUIRE(list_size(program2->get_functions()) == 0);

    program2->acquire_function(function);

    REQUIRE(function->get_program() == program2);
    REQUIRE(function->get_self_it().has_value());
    REQUIRE(list_size(program1->get_functions()) == 0);
    REQUIRE(list_size(program2->get_functions()) == 1);
}

TEST_CASE("FunctionIR assign_to_program delegates to program acquisition", "[FunctionIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();

    function->assign_to_program(program);

    REQUIRE(function->get_program() == program);
    REQUIRE(function->get_self_it().has_value());
    REQUIRE(list_size(program->get_functions()) == 1);
}

TEST_CASE("ProgramIR acquires and releases constants", "[ProgramIR][ConstantIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto constant = std::make_shared<ConstantIR>();

    REQUIRE(constant->get_program() == nullptr);
    REQUIRE_FALSE(constant->get_self_it().has_value());

    program->acquire_constant(constant);

    REQUIRE(constant->get_program() == program);
    REQUIRE(constant->get_self_it().has_value());
    REQUIRE(list_size(program->get_constants()) == 1);

    program->release_constant(constant);

    REQUIRE(constant->get_program() == nullptr);
    REQUIRE_FALSE(constant->get_self_it().has_value());
    REQUIRE(list_size(program->get_constants()) == 0);
}

TEST_CASE("ConstantIR assign_to_program delegates to program acquisition", "[ConstantIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto constant = std::make_shared<ConstantIR>();

    constant->assign_to_program(program);

    REQUIRE(constant->get_program() == program);
    REQUIRE(constant->get_self_it().has_value());
    REQUIRE(list_size(program->get_constants()) == 1);
}

TEST_CASE("ConstantIR bytes are mutable through getter", "[ConstantIR]")
{
    ConstantIR constant;

    constant.get_bytes().push_back(0x12);
    constant.get_bytes().push_back(0x34);

    REQUIRE(constant.get_bytes().size() == 2);
    REQUIRE(constant.get_bytes()[0] == 0x12);
    REQUIRE(constant.get_bytes()[1] == 0x34);
}

TEST_CASE("ProgramIR acquires and releases static variables with context updates",
          "[ProgramIR][VariableIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto variable = std::make_shared<VariableIR>();

    REQUIRE(variable->get_program() == nullptr);
    REQUIRE(variable->get_function() == nullptr);
    REQUIRE(variable->get_context() == VariableIR::Context::UNDEFINED);
    REQUIRE(std::holds_alternative<std::monostate>(variable->get_owner()));

    program->acquire_static(variable);

    REQUIRE(variable->get_program() == program);
    REQUIRE(variable->get_function() == nullptr);
    REQUIRE(variable->get_context() == VariableIR::Context::STATIC);
    REQUIRE(variable->get_self_it().has_value());
    REQUIRE(list_size(program->get_static_vars()) == 1);
    REQUIRE(std::holds_alternative<ProgramIR_sptr>(variable->get_owner()));

    program->release_static(variable);

    REQUIRE(variable->get_program() == nullptr);
    REQUIRE(variable->get_context() == VariableIR::Context::UNDEFINED);
    REQUIRE_FALSE(variable->get_self_it().has_value());
    REQUIRE(list_size(program->get_static_vars()) == 0);
    REQUIRE(std::holds_alternative<std::monostate>(variable->get_owner()));
}

TEST_CASE("VariableIR assign_as_static delegates to program acquisition", "[VariableIR][ProgramIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto variable = std::make_shared<VariableIR>();

    variable->assign_as_static(program);

    REQUIRE(variable->get_program() == program);
    REQUIRE(variable->get_context() == VariableIR::Context::STATIC);
    REQUIRE(variable->get_self_it().has_value());
    REQUIRE(list_size(program->get_static_vars()) == 1);
}

TEST_CASE("FunctionIR acquires and releases local variables", "[FunctionIR][VariableIR]")
{
    auto function = std::make_shared<FunctionIR>();
    auto variable = std::make_shared<VariableIR>();

    function->acquire_local_variable(variable);

    REQUIRE(variable->get_function() == function);
    REQUIRE(variable->get_program() == nullptr);
    REQUIRE(variable->get_context() == VariableIR::Context::LOCAL);
    REQUIRE(variable->get_self_it().has_value());
    REQUIRE(list_size(function->get_local_variables()) == 1);
    REQUIRE(std::holds_alternative<FunctionIR_sptr>(variable->get_owner()));

    function->release_local_variable(variable);

    REQUIRE(variable->get_function() == nullptr);
    REQUIRE_FALSE(variable->get_self_it().has_value());
    REQUIRE(list_size(function->get_local_variables()) == 0);
    REQUIRE(std::holds_alternative<std::monostate>(variable->get_owner()));
}

TEST_CASE("FunctionIR acquires and releases parameters", "[FunctionIR][VariableIR]")
{
    auto function  = std::make_shared<FunctionIR>();
    auto parameter = std::make_shared<VariableIR>();

    function->acquire_parameter(parameter);

    REQUIRE(parameter->get_function() == function);
    REQUIRE(parameter->get_context() == VariableIR::Context::PARAMETER);
    REQUIRE(parameter->get_self_it().has_value());
    REQUIRE(list_size(function->get_parameters()) == 1);
    REQUIRE(std::holds_alternative<FunctionIR_sptr>(parameter->get_owner()));

    function->release_parameter(parameter);

    REQUIRE(parameter->get_function() == nullptr);
    REQUIRE_FALSE(parameter->get_self_it().has_value());
    REQUIRE(list_size(function->get_parameters()) == 0);
    REQUIRE(std::holds_alternative<std::monostate>(parameter->get_owner()));
}

TEST_CASE("VariableIR can move between static local and parameter owners",
          "[VariableIR][FunctionIR][ProgramIR]")
{
    auto program   = std::make_shared<ProgramIR>();
    auto function1 = std::make_shared<FunctionIR>();
    auto function2 = std::make_shared<FunctionIR>();
    auto variable  = std::make_shared<VariableIR>();

    program->acquire_static(variable);
    REQUIRE(variable->get_program() == program);
    REQUIRE(variable->get_context() == VariableIR::Context::STATIC);
    REQUIRE(list_size(program->get_static_vars()) == 1);

    function1->acquire_local_variable(variable);
    REQUIRE(list_size(program->get_static_vars()) == 0);
    REQUIRE(list_size(function1->get_local_variables()) == 1);
    REQUIRE(variable->get_program() == nullptr);
    REQUIRE(variable->get_function() == function1);
    REQUIRE(variable->get_context() == VariableIR::Context::LOCAL);

    function2->acquire_parameter(variable);
    REQUIRE(list_size(function1->get_local_variables()) == 0);
    REQUIRE(list_size(function2->get_parameters()) == 1);
    REQUIRE(variable->get_function() == function2);
    REQUIRE(variable->get_context() == VariableIR::Context::PARAMETER);
}

TEST_CASE("VariableIR assign helpers delegate to owners", "[VariableIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();
    auto local    = std::make_shared<VariableIR>();
    auto param    = std::make_shared<VariableIR>();

    local->assign_as_local(function);
    param->assign_as_parameter(function);

    REQUIRE(local->get_function() == function);
    REQUIRE(local->get_context() == VariableIR::Context::LOCAL);
    REQUIRE(list_size(function->get_local_variables()) == 1);

    REQUIRE(param->get_function() == function);
    REQUIRE(param->get_context() == VariableIR::Context::PARAMETER);
    REQUIRE(list_size(function->get_parameters()) == 1);

    auto staticVar = std::make_shared<VariableIR>();
    staticVar->assign_as_static(program);

    REQUIRE(staticVar->get_program() == program);
    REQUIRE(staticVar->get_context() == VariableIR::Context::STATIC);
    REQUIRE(list_size(program->get_static_vars()) == 1);
}

TEST_CASE("release_variable dispatches based on variable context", "[VariableIR]")
{
    auto program    = std::make_shared<ProgramIR>();
    auto function   = std::make_shared<FunctionIR>();
    auto static_var = std::make_shared<VariableIR>();
    auto local_var  = std::make_shared<VariableIR>();
    auto param_var  = std::make_shared<VariableIR>();

    program->acquire_static(static_var);
    function->acquire_local_variable(local_var);
    function->acquire_parameter(param_var);

    release_variable(static_var);
    release_variable(local_var);
    release_variable(param_var);

    REQUIRE(list_size(program->get_static_vars()) == 0);
    REQUIRE(list_size(function->get_local_variables()) == 0);
    REQUIRE(list_size(function->get_parameters()) == 0);

    REQUIRE(static_var->get_context() == VariableIR::Context::UNDEFINED);
    REQUIRE(static_var->get_program() == nullptr);
    REQUIRE(local_var->get_function() == nullptr);
    REQUIRE(param_var->get_function() == nullptr);
}

TEST_CASE("FunctionIR acquires and releases basic blocks", "[FunctionIR][BasicBlockIR]")
{
    auto function    = std::make_shared<FunctionIR>();
    auto basic_block = std::make_shared<BasicBlockIR>();

    function->acquire_basic_block(basic_block);

    REQUIRE(basic_block->get_function() == function);
    REQUIRE(basic_block->get_function_raw() == function.get());
    REQUIRE(basic_block->get_self_it().has_value());
    REQUIRE(list_size(function->get_basic_blocks()) == 1);
    REQUIRE((*basic_block->get_self_it())->get() == basic_block.get());

    function->release_basic_block(basic_block);

    REQUIRE(basic_block->get_function() == nullptr);
    REQUIRE(basic_block->get_function_raw() == nullptr);
    REQUIRE_FALSE(basic_block->get_self_it().has_value());
    REQUIRE(list_size(function->get_basic_blocks()) == 0);
}

TEST_CASE("BasicBlockIR assign_to_function delegates to function acquisition",
          "[BasicBlockIR][FunctionIR]")
{
    auto function    = std::make_shared<FunctionIR>();
    auto basic_block = std::make_shared<BasicBlockIR>();

    basic_block->assign_to_function(function);

    REQUIRE(basic_block->get_function() == function);
    REQUIRE(basic_block->get_self_it().has_value());
    REQUIRE(list_size(function->get_basic_blocks()) == 1);
}

TEST_CASE("BasicBlockIR can move between functions", "[BasicBlockIR][FunctionIR]")
{
    auto function1   = std::make_shared<FunctionIR>();
    auto function2   = std::make_shared<FunctionIR>();
    auto basic_block = std::make_shared<BasicBlockIR>();

    function1->acquire_basic_block(basic_block);
    REQUIRE(list_size(function1->get_basic_blocks()) == 1);

    function2->acquire_basic_block(basic_block);

    REQUIRE(list_size(function1->get_basic_blocks()) == 0);
    REQUIRE(list_size(function2->get_basic_blocks()) == 1);
    REQUIRE(basic_block->get_function() == function2);
    REQUIRE(basic_block->get_function_raw() == function2.get());
}

TEST_CASE("BasicBlockIR stores and removes predecessors and successors", "[BasicBlockIR]")
{
    auto a = std::make_shared<BasicBlockIR>();
    auto b = std::make_shared<BasicBlockIR>();
    auto c = std::make_shared<BasicBlockIR>();

    a->add_successor(b);
    a->add_successor(c);
    b->add_predecessor(a);
    c->add_predecessor(a);

    REQUIRE(list_size(a->get_successors()) == 2);
    REQUIRE(list_size(b->get_predecessors()) == 1);
    REQUIRE(list_size(c->get_predecessors()) == 1);

    a->remove_successor(b);
    c->remove_predecessor(a);

    REQUIRE(list_size(a->get_successors()) == 1);
    REQUIRE(list_size(b->get_predecessors()) == 1);
    REQUIRE(list_size(c->get_predecessors()) == 0);
}

TEST_CASE("BasicBlockIR acquires and releases instructions", "[BasicBlockIR][InstructionIR]")
{
    auto basic_block = std::make_shared<BasicBlockIR>();
    auto instruction = std::make_shared<InstructionIR>();

    basic_block->acquire_instruction(instruction);

    REQUIRE(instruction->get_basic_block() == basic_block);
    REQUIRE(instruction->get_basic_block_raw() == basic_block.get());
    REQUIRE(instruction->get_self_it().has_value());
    REQUIRE(list_size(basic_block->get_instructions()) == 1);
    REQUIRE((*instruction->get_self_it())->get() == instruction.get());

    basic_block->release_instruction(instruction);

    REQUIRE(instruction->get_basic_block() == nullptr);
    REQUIRE(instruction->get_basic_block_raw() == nullptr);
    REQUIRE_FALSE(instruction->get_self_it().has_value());
    REQUIRE(list_size(basic_block->get_instructions()) == 0);
}

TEST_CASE("InstructionIR assign_to_basic_block delegates to block acquisition",
          "[InstructionIR][BasicBlockIR]")
{
    auto basic_block = std::make_shared<BasicBlockIR>();
    auto instruction = std::make_shared<InstructionIR>();

    instruction->assign_to_basic_block(basic_block);

    REQUIRE(instruction->get_basic_block() == basic_block);
    REQUIRE(instruction->get_self_it().has_value());
    REQUIRE(list_size(basic_block->get_instructions()) == 1);
}

TEST_CASE("InstructionIR can move between basic blocks", "[InstructionIR][BasicBlockIR]")
{
    auto bb1         = std::make_shared<BasicBlockIR>();
    auto bb2         = std::make_shared<BasicBlockIR>();
    auto instruction = std::make_shared<InstructionIR>();

    bb1->acquire_instruction(instruction);
    REQUIRE(list_size(bb1->get_instructions()) == 1);

    bb2->acquire_instruction(instruction);

    REQUIRE(list_size(bb1->get_instructions()) == 0);
    REQUIRE(list_size(bb2->get_instructions()) == 1);
    REQUIRE(instruction->get_basic_block() == bb2);
    REQUIRE(instruction->get_basic_block_raw() == bb2.get());
}

TEST_CASE("InstructionIR stores operands of different IR kinds", "[InstructionIR]")
{
    auto instruction = std::make_shared<InstructionIR>();
    auto variable    = std::make_shared<VariableIR>();
    auto function    = std::make_shared<FunctionIR>();
    auto constant    = std::make_shared<ConstantIR>();

    REQUIRE(instruction->get_operands().empty());

    instruction->push_back_operand(variable);
    instruction->push_back_operand(function);
    instruction->push_back_operand(constant);

    REQUIRE(instruction->get_operands().size() == 3);
}

TEST_CASE("ProgramIR mutable scalar and pointer accessors work", "[ProgramIR]")
{
    ProgramIR program;
    auto      entry = std::make_shared<FunctionIR>();
    auto      init  = std::make_shared<FunctionIR>();

    program.get_num_cpu_bits()            = 64;
    program.get_entry_func()              = entry;
    program.get_static_initializer_func() = init;

    REQUIRE(program.get_num_cpu_bits() == 64);
    REQUIRE(program.get_entry_func() == entry);
    REQUIRE(program.get_static_initializer_func() == init);
}

TEST_CASE("FunctionIR mutable accessors work", "[FunctionIR]")
{
    FunctionIR function;
    auto       entryBlock = std::make_shared<BasicBlockIR>();

    function.get_entry_basic_block()   = entryBlock;
    function.get_external_flag()       = true;
    function.get_initializer_flag()    = true;
    function.get_entry_flag()          = true;
    function.get_initial_stack_bytes() = 32;

    REQUIRE(function.get_entry_basic_block() == entryBlock);
    REQUIRE(function.get_external_flag());
    REQUIRE(function.get_initializer_flag());
    REQUIRE(function.get_entry_flag());
    REQUIRE(function.get_initial_stack_bytes() == 32);
}

TEST_CASE("VariableIR mutable scalar accessors work", "[VariableIR]")
{
    VariableIR variable;

    variable.get_num_bytes()     = 8;
    variable.get_external_flag() = true;

    REQUIRE(variable.get_num_bytes() == 8);
    REQUIRE(variable.get_external_flag());
}

TEST_CASE("ProgramIR reassigns constants between programs", "[ProgramIR][ConstantIR]")
{
    auto program1 = std::make_shared<ProgramIR>();
    auto program2 = std::make_shared<ProgramIR>();
    auto constant = std::make_shared<ConstantIR>();

    program1->acquire_constant(constant);

    REQUIRE(constant->get_program() == program1);
    REQUIRE(constant->get_self_it().has_value());
    REQUIRE(list_size(program1->get_constants()) == 1);
    REQUIRE(list_size(program2->get_constants()) == 0);

    program2->acquire_constant(constant);

    REQUIRE(constant->get_program() == program2);
    REQUIRE(constant->get_self_it().has_value());
    REQUIRE(list_size(program1->get_constants()) == 0);
    REQUIRE(list_size(program2->get_constants()) == 1);
}

TEST_CASE("VariableIR get_owner reflects current owner kind", "[VariableIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();
    auto variable = std::make_shared<VariableIR>();

    REQUIRE(std::holds_alternative<std::monostate>(variable->get_owner()));

    program->acquire_static(variable);
    REQUIRE(std::holds_alternative<ProgramIR_sptr>(variable->get_owner()));
    REQUIRE(std::get<ProgramIR_sptr>(variable->get_owner()) == program);

    function->acquire_local_variable(variable);
    REQUIRE(std::holds_alternative<FunctionIR_sptr>(variable->get_owner()));
    REQUIRE(std::get<FunctionIR_sptr>(variable->get_owner()) == function);

    function->release_local_variable(variable);
    REQUIRE(std::holds_alternative<std::monostate>(variable->get_owner()));
}

TEST_CASE("FunctionIR reassigns variable between local and parameter lists",
          "[FunctionIR][VariableIR]")
{
    auto function = std::make_shared<FunctionIR>();
    auto variable = std::make_shared<VariableIR>();

    function->acquire_local_variable(variable);

    REQUIRE(list_size(function->get_local_variables()) == 1);
    REQUIRE(list_size(function->get_parameters()) == 0);
    REQUIRE(variable->get_context() == VariableIR::Context::LOCAL);
    REQUIRE(variable->get_function() == function);

    function->acquire_parameter(variable);

    REQUIRE(list_size(function->get_local_variables()) == 0);
    REQUIRE(list_size(function->get_parameters()) == 1);
    REQUIRE(variable->get_context() == VariableIR::Context::PARAMETER);
    REQUIRE(variable->get_function() == function);
    REQUIRE(variable->get_self_it().has_value());
}

TEST_CASE("ProgramIR acquire_static removes variable from function owner",
          "[ProgramIR][VariableIR]")
{
    auto program  = std::make_shared<ProgramIR>();
    auto function = std::make_shared<FunctionIR>();
    auto variable = std::make_shared<VariableIR>();

    function->acquire_parameter(variable);

    REQUIRE(list_size(function->get_parameters()) == 1);
    REQUIRE(variable->get_context() == VariableIR::Context::PARAMETER);

    program->acquire_static(variable);

    REQUIRE(list_size(function->get_parameters()) == 0);
    REQUIRE(list_size(program->get_static_vars()) == 1);
    REQUIRE(variable->get_program() == program);
    REQUIRE(variable->get_function() == nullptr);
    REQUIRE(variable->get_context() == VariableIR::Context::STATIC);
}

TEST_CASE("BasicBlockIR remove_successor and remove_predecessor only erase matching block",
          "[BasicBlockIR]")
{
    auto a = std::make_shared<BasicBlockIR>();
    auto b = std::make_shared<BasicBlockIR>();
    auto c = std::make_shared<BasicBlockIR>();

    a->add_successor(b);
    a->add_successor(c);
    a->add_successor(b);

    REQUIRE(list_size(a->get_successors()) == 3);

    a->remove_successor(b);

    REQUIRE(list_size(a->get_successors()) == 1);

    c->add_predecessor(a);
    c->add_predecessor(b);
    c->add_predecessor(a);

    REQUIRE(list_size(c->get_predecessors()) == 3);

    c->remove_predecessor(a);

    REQUIRE(list_size(c->get_predecessors()) == 1);
}

TEST_CASE("BasicBlockIR preserves instruction order when acquiring multiple instructions",
          "[BasicBlockIR][InstructionIR]")
{
    auto block = std::make_shared<BasicBlockIR>();
    auto i1    = std::make_shared<InstructionIR>();
    auto i2    = std::make_shared<InstructionIR>();
    auto i3    = std::make_shared<InstructionIR>();

    block->acquire_instruction(i1);
    block->acquire_instruction(i2);
    block->acquire_instruction(i3);

    auto it = block->get_instructions().begin();
    REQUIRE(it->get() == i1.get());
    ++it;
    REQUIRE(it->get() == i2.get());
    ++it;
    REQUIRE(it->get() == i3.get());

    REQUIRE((*i1->get_self_it())->get() == i1.get());
    REQUIRE((*i2->get_self_it())->get() == i2.get());
    REQUIRE((*i3->get_self_it())->get() == i3.get());
}

TEST_CASE("InstructionIR release from one block allows reacquire into same block tail",
          "[InstructionIR][BasicBlockIR]")
{
    auto block = std::make_shared<BasicBlockIR>();
    auto i1    = std::make_shared<InstructionIR>();
    auto i2    = std::make_shared<InstructionIR>();

    block->acquire_instruction(i1);
    block->acquire_instruction(i2);

    block->release_instruction(i1);

    REQUIRE(list_size(block->get_instructions()) == 1);
    REQUIRE(block->get_instructions().front().get() == i2.get());
    REQUIRE_FALSE(i1->get_self_it().has_value());
    REQUIRE(i1->get_basic_block() == nullptr);

    block->acquire_instruction(i1);

    REQUIRE(list_size(block->get_instructions()) == 2);
    auto it = block->get_instructions().begin();
    REQUIRE(it->get() == i2.get());
    ++it;
    REQUIRE(it->get() == i1.get());
}

TEST_CASE("InstructionIR mutable operands accessor allows replacing operands", "[InstructionIR]")
{
    auto instruction = std::make_shared<InstructionIR>();
    auto variable1   = std::make_shared<VariableIR>();
    auto variable2   = std::make_shared<VariableIR>();
    auto constant    = std::make_shared<ConstantIR>();

    instruction->push_back_operand(variable1);
    REQUIRE(instruction->get_operands().size() == 1);

    auto& operands = instruction->get_operands();
    operands.clear();
    operands.emplace_back(variable2.get());
    operands.emplace_back(constant.get());

    REQUIRE(instruction->get_operands().size() == 2);
}

TEST_CASE("FunctionIR preserves basic block order", "[FunctionIR][BasicBlockIR]")
{
    auto function = std::make_shared<FunctionIR>();
    auto bb1      = std::make_shared<BasicBlockIR>();
    auto bb2      = std::make_shared<BasicBlockIR>();
    auto bb3      = std::make_shared<BasicBlockIR>();

    function->acquire_basic_block(bb1);
    function->acquire_basic_block(bb2);
    function->acquire_basic_block(bb3);

    auto it = function->get_basic_blocks().begin();
    REQUIRE(it->get() == bb1.get());
    ++it;
    REQUIRE(it->get() == bb2.get());
    ++it;
    REQUIRE(it->get() == bb3.get());
}

TEST_CASE("ProgramIR preserves function order", "[ProgramIR][FunctionIR]")
{
    auto program = std::make_shared<ProgramIR>();
    auto f1      = std::make_shared<FunctionIR>();
    auto f2      = std::make_shared<FunctionIR>();
    auto f3      = std::make_shared<FunctionIR>();

    program->acquire_function(f1);
    program->acquire_function(f2);
    program->acquire_function(f3);

    auto it = program->get_functions().begin();
    REQUIRE(it->get() == f1.get());
    ++it;
    REQUIRE(it->get() == f2.get());
    ++it;
    REQUIRE(it->get() == f3.get());
}

TEST_CASE("ProgramIR preserves static variable order", "[ProgramIR][VariableIR]")
{
    auto program = std::make_shared<ProgramIR>();
    auto v1      = std::make_shared<VariableIR>();
    auto v2      = std::make_shared<VariableIR>();
    auto v3      = std::make_shared<VariableIR>();

    program->acquire_static(v1);
    program->acquire_static(v2);
    program->acquire_static(v3);

    auto it = program->get_static_vars().begin();
    REQUIRE(it->get() == v1.get());
    ++it;
    REQUIRE(it->get() == v2.get());
    ++it;
    REQUIRE(it->get() == v3.get());
}

TEST_CASE("ProgramIR preserves constant order", "[ProgramIR][ConstantIR]")
{
    auto program = std::make_shared<ProgramIR>();
    auto c1      = std::make_shared<ConstantIR>();
    auto c2      = std::make_shared<ConstantIR>();
    auto c3      = std::make_shared<ConstantIR>();

    program->acquire_constant(c1);
    program->acquire_constant(c2);
    program->acquire_constant(c3);

    auto it = program->get_constants().begin();
    REQUIRE(it->get() == c1.get());
    ++it;
    REQUIRE(it->get() == c2.get());
    ++it;
    REQUIRE(it->get() == c3.get());
}
