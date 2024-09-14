#include <llvm2sala/compiler.hpp>
#include <llvmutl/llvm_utils.hpp>
#include <utility/endian.hpp>
#include <utility/timeprof.hpp>
#include <utility/assumptions.hpp>
#include <utility/invariants.hpp>
#include <utility/development.hpp>
#include <utility/config.hpp>
#if COMPILER() == COMPILER_VC()
#    pragma warning(push)
#    pragma warning(disable : 4624 4996 4146 4800 4996 4005 4355 4244 4267)
#endif
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/DebugInfo/DIContext.h>
#if COMPILER() == COMPILER_VC()
#    pragma warning(pop)
#endif
#include <limits>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <iostream>


static inline void push_back_operand(sala::Instruction& sala_instruction, Compiler::MemoryObject const& mo)
{
    sala_instruction.push_back_operand(mo.index, mo.descriptor);
}


static void copy_bytes_of_value(std::uint8_t const* value_ptr, std::size_t const num_bytes, sala::Constant& constant)
{
    if (is_this_little_endian_machine())
        for (std::size_t i = 0U; i < num_bytes; ++i)
            constant.push_back_byte(value_ptr[i]);
    else
        for (std::size_t i = num_bytes; i > 0U; --i)
            constant.push_back_byte(value_ptr[i - 1U]);
}


static std::string to_hex_string(std::uint8_t const* const value_ptr, std::size_t const num_bytes)
{
    std::stringstream sstr;
    if (is_this_little_endian_machine())
        for (std::size_t i = 0U; i < num_bytes; ++i)
            sstr << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)value_ptr[i];
    else
        for (std::size_t i = num_bytes; i > 0U; --i)
            sstr << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)value_ptr[i];
    return sstr.str();
}


static bool is_nop_in_basic_block(sala::BasicBlock& sala_block, std::uint32_t const start_index = 0U)
{
    for (auto it = std::next(sala_block.instructions().begin(), start_index); it != sala_block.instructions().end(); ++it)
        if (it->opcode() == sala::Instruction::Opcode::NOP)
            return true;
    return false;
}


static sala::Instruction::Opcode get_interpreted_function_opcode(
    std::string const& llvm_function_name,
    sala::Instruction::Opcode const failure_opcode = sala::Instruction::Opcode::__INVALID__
    )
{
    static std::unordered_map<std::string, sala::Instruction::Opcode> const mapping{
        { "malloc", sala::Instruction::Opcode::MALLOC },
        { "free", sala::Instruction::Opcode::FREE },
    };
    auto const it = mapping.find(llvm_function_name);
    return it == mapping.end() ? failure_opcode : it->second;
}


static sala::Instruction::Opcode get_interpreted_function_opcode(
    llvm::Function const* const llvm_function,
    sala::Instruction::Opcode const failure_opcode = sala::Instruction::Opcode::__INVALID__
    )
{
    if (llvm_function->isDeclaration())
        return  get_interpreted_function_opcode(llvm_function->getName().str(), failure_opcode);
    return failure_opcode;
}


static sala::Instruction::Opcode get_interpreted_function_opcode(
    llvm::Value const* const llvm_value,
    sala::Instruction::Opcode const failure_opcode = sala::Instruction::Opcode::__INVALID__
    )
{
    if (auto llvm_function = llvm::dyn_cast<llvm::Function const>(llvm_value))
        return get_interpreted_function_opcode(llvm_function, failure_opcode);
    return failure_opcode;
}


static void remove_nops_from_basic_block(sala::BasicBlock& sala_block)
{
    std::vector<sala::Instruction> instructions;
    for (auto& instruction : sala_block.instructions())
        if (instruction.opcode() != sala::Instruction::Opcode::NOP)
            instructions.push_back(instruction);
    while (sala_block.instructions().size() > instructions.size())
        sala_block.pop_back_instruction();
    for (std::size_t i = 0ULL; i != instructions.size(); ++i)
        sala_block.assign_instruction(i, instructions.at(i));
}


void compile_llvm_file(std::filesystem::path const& llvm_file_pathname, sala::Program& P)
{
    TMPROF_BLOCK();

    llvm::SMDiagnostic D;
    llvm::LLVMContext C;
    std::unique_ptr<llvm::Module> M;
    {
 
        M = llvm::parseIRFile(llvm_file_pathname.string(), D, C);
        if (M == nullptr)
        {
            llvm::raw_os_ostream ros(std::cout);
            D.print(llvm_file_pathname.filename().string().c_str(), ros, false);
            ros.flush();
            return;
        }
    }

    P.set_system(M->getTargetTriple());
    P.set_num_cpu_bits((std::uint16_t)(8U * M->getDataLayout().getPointerSize()));

    std::string program_name{ llvm_file_pathname.filename().replace_extension("").string() };
    (void)program_name.ends_with(".sim");
    program_name = std::filesystem::path{program_name}.replace_extension("").string();

    P.set_name(program_name);

    Compiler compiler{ P, *M };
    compiler.run();
}


Compiler::Compiler(sala::Program& P, llvm::Module& M)
    : program_{ P }
    , module_{ M }
    , created_memory_objects_{}
    , moveptr_constants_{}
    , numeric_constants_{}
    , compile_constant_variable_indices_{}
    , compiled_function_{ nullptr }
    , compiled_basic_block_{ nullptr }
{}


Compiler::MemoryObject const& Compiler::register_parameter(llvm::Value* const llvm_value, sala::Variable const& sala_parameter)
{
    auto it_and_state = created_memory_objects_.insert({ llvm_value, { sala_parameter.index(), sala::Instruction::Descriptor::PARAMETER } });
    ASSUMPTION(it_and_state.second);
    return it_and_state.first->second;
}


Compiler::MemoryObject const& Compiler::register_variable(llvm::Value* const llvm_value, sala::Variable const& sala_variable)
{
    auto it_and_state = created_memory_objects_.insert({
        llvm_value,
        {
            sala_variable.index(),
            sala_variable.region() == sala::Variable::Region::STATIC ?
                sala::Instruction::Descriptor::STATIC : sala::Instruction::Descriptor::LOCAL
        } 
    });
    ASSUMPTION(it_and_state.second);
    return it_and_state.first->second;
}


Compiler::MemoryObject const& Compiler::register_constant(llvm::Value* llvm_value, sala::Constant const& sala_constant)
{
    auto it_and_state = created_memory_objects_.insert({ llvm_value, { sala_constant.index(), sala::Instruction::Descriptor::CONSTANT } });
    ASSUMPTION(it_and_state.second);
    return it_and_state.first->second;
}


Compiler::MemoryObject const& Compiler::register_function(llvm::Value* llvm_value, sala::Function const& sala_function)
{
    auto it_and_state = created_memory_objects_.insert({ llvm_value, { sala_function.index(), sala::Instruction::Descriptor::FUNCTION } });
    ASSUMPTION(it_and_state.second);
    return it_and_state.first->second;
}


Compiler::MemoryObject const& Compiler::update_variable(llvm::Value& llvm_value, llvm::Type* const llvm_type, sala::SourceBackMapping const& back_mapping)
{
    auto const& mo = memory_object(llvm_value);
    INVARIANT(mo.descriptor == sala::Instruction::Descriptor::LOCAL);
    auto& sala_variable = const_cast<sala::Variable&>(compiled_function()->local_variables().at(mo.index));
    sala_variable.set_num_bytes(llvm_sizeof(llvm_type, module()));
    sala_variable.source_back_mapping() = back_mapping;
    return mo;
}


Compiler::MemoryObject const& Compiler::memory_object(llvm::Value* const llvm_value)
{
    auto it = created_memory_objects_.find(llvm_value);
    if (it == created_memory_objects_.end())
    {
        if (auto llvm_constant = llvm::dyn_cast<llvm::Constant>(llvm_value))
        {
            std::vector<std::pair<llvm::Value*, std::uint64_t> > pointer_initializations;
            MemoryObject sala_constant_mo;
            {
                auto& sala_constant = program().push_back_constant();
                if (auto llvm_static_constant = llvm::dyn_cast<llvm::GlobalVariable>(llvm_constant))
                    compile_constant(llvm_static_constant->getInitializer(), sala_constant, pointer_initializations);
                else
                    compile_constant(llvm_constant, sala_constant, pointer_initializations);
                sala_constant_mo = { sala_constant.index(), sala::Instruction::Descriptor::CONSTANT };
            }

            if (llvm::isa<llvm::GlobalVariable>(llvm_constant))
            {
                ASSUMPTION(llvm_constant->getType()->isPointerTy());

                MemoryObject sala_constant_ptr_mo;
                {
                    auto& sala_constant_ptr = program().push_back_constant();
                    for (unsigned int i = 0U, n = module().getDataLayout().getPointerSize(); i < n; ++i)
                        sala_constant_ptr.push_back_byte(171U); // hex: AB, bin: 10101011 - indicates not initialized pointer.
                    sala_constant_ptr_mo = register_constant(llvm_value, sala_constant_ptr);
                }

                auto& sala_instruction = program().function_ref(program().static_initializer()).last_basic_block_ref().push_back_instruction();
                sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
                push_back_operand(sala_instruction, sala_constant_ptr_mo);
                push_back_operand(sala_instruction, sala_constant_mo);
            }
            else
                register_constant(llvm_value, program().constant_ref(sala_constant_mo.index));

            if (!pointer_initializations.empty())
            {
                sala::Function* old_compiled_function = compiled_function();
                sala::BasicBlock* old_compiled_basic_block = compiled_basic_block();
                set_compiled_function(&program().function_ref(program().static_initializer()));
                set_compiled_basic_block(&compiled_function()->last_basic_block_ref());

                auto& sala_function{ *compiled_function() };
                auto& sala_block{ *compiled_basic_block() };

                if (compile_constant_variable_indices_.address_variable_index == std::numeric_limits<std::uint32_t>::max())
                {
                    {
                        auto& sala_variable = sala_function.push_back_local_variable();
                        sala_variable.set_num_bytes(module().getDataLayout().getPointerSize());
                        compile_constant_variable_indices_.moveptr_variable_index = sala_variable.index();
                    }
                    {
                        auto& sala_variable = sala_function.push_back_local_variable();
                        sala_variable.set_num_bytes(module().getDataLayout().getPointerSize());
                        compile_constant_variable_indices_.address_variable_index = sala_variable.index();
                    }
                }

                MemoryObject const moveptr_variable_mo{
                    compile_constant_variable_indices_.moveptr_variable_index, sala::Instruction::Descriptor::LOCAL
                    };
                MemoryObject const address_variable_mo{
                    compile_constant_variable_indices_.address_variable_index, sala::Instruction::Descriptor::LOCAL
                    };
                for (auto const& ptr_init : pointer_initializations)
                {
                    {
                        auto& sala_instruction = sala_block.push_back_instruction();
                        if (llvm::isa<llvm::GlobalVariable>(llvm_value))
                            sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
                        else
                            sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
                        push_back_operand(sala_instruction, moveptr_variable_mo);
                        push_back_operand(sala_instruction, created_memory_objects_.find(llvm_value)->second);
                    }
                    if (ptr_init.second > 0ULL)
                    {
                        MemoryObject const constant_one_mo{ moveptr_constant_index(1ULL), sala::Instruction::Descriptor::CONSTANT };
                        auto& sala_instruction = sala_block.push_back_instruction();
                        sala_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);
                        push_back_operand(sala_instruction, moveptr_variable_mo);
                        push_back_operand(sala_instruction, moveptr_variable_mo);
                        push_back_operand(sala_instruction, { moveptr_constant_index(ptr_init.second), sala::Instruction::Descriptor::CONSTANT });
                        push_back_operand(sala_instruction, constant_one_mo);
                    }
                    if (llvm::ConstantExpr* llvm_const_expr = llvm::dyn_cast<llvm::ConstantExpr>(ptr_init.first))
                    {
                        std::vector<llvm::Instruction*> llvm_instructions;
                        std::vector<llvm::ConstantExpr*> llvm_const_expr_stack{ llvm_const_expr };
                        do
                        {
                            llvm_const_expr = llvm_const_expr_stack.back();
                            llvm_const_expr_stack.pop_back();
                            if (!has_memory_object(llvm_const_expr))
                                register_variable(llvm_const_expr, compiled_function()->local_variables().at(address_variable_mo.index));
                            llvm_instructions.push_back(llvm_constant_expr_to_instruction(llvm_const_expr));
                            if (!has_memory_object(llvm_instructions.back()))
                                register_variable(llvm_instructions.back(), compiled_function()->local_variables().at(address_variable_mo.index));
                            if (auto const llvm_const_sub_expr = llvm::dyn_cast<llvm::ConstantExpr>(llvm_const_expr->getOperand(0)))
                                llvm_const_expr_stack.push_back(llvm_const_sub_expr);
                        }
                        while (!llvm_const_expr_stack.empty());
                        std::vector<llvm::PHINode*> phi_nodes;
                        for (auto it = llvm_instructions.rbegin(); it != llvm_instructions.rend(); ++it)
                            compile_instruction(**it, sala_block.push_back_instruction(), phi_nodes);
                        ASSUMPTION(phi_nodes.empty());
                        for (auto it = llvm_instructions.rbegin(); it != llvm_instructions.rend(); ++it)
                        {
                            (*it)->replaceAllUsesWith(llvm::UndefValue::get((*it)->getType())); 
                            (*it)->dropAllReferences();
                            (*it)->deleteValue();
                        }
                    }
                    else
                    {
                        ASSUMPTION(!llvm::isa<llvm::ConstantExpr>(ptr_init.first));
                        auto const ptr_init_mo{ memory_object(ptr_init.first) };
                        auto& sala_instruction = sala_block.push_back_instruction();
                        sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
                        push_back_operand(sala_instruction, address_variable_mo);
                        push_back_operand(sala_instruction, ptr_init_mo);
                    }
                    {
                        auto& sala_instruction = sala_block.push_back_instruction();
                        sala_instruction.set_opcode(sala::Instruction::Opcode::STORE);
                        push_back_operand(sala_instruction, moveptr_variable_mo);
                        push_back_operand(sala_instruction, address_variable_mo);
                    }
                }

                set_compiled_function(old_compiled_function);
                set_compiled_basic_block(old_compiled_basic_block);
            }
    
            return created_memory_objects_.find(llvm_value)->second;
        }
        return register_variable(llvm_value, compiled_function()->push_back_local_variable());
    }
    return it->second;
}


std::uint32_t Compiler::moveptr_constant_index(std::int64_t const ptr_move)
{
    auto it = moveptr_constants_.find(ptr_move);
    if (it == moveptr_constants_.end())
    {
        auto& sala_constant = program().push_back_constant();
        if (sizeof(void*) == 4U)
        {
            std::int32_t const ptr_move_32 = (std::int32_t)ptr_move;
            copy_bytes_of_value((std::uint8_t const *)&ptr_move, sizeof(ptr_move_32), sala_constant);
        }
        else
        {
            ASSUMPTION(sizeof(void*) == sizeof(ptr_move));
            copy_bytes_of_value((std::uint8_t const *)&ptr_move, sizeof(ptr_move), sala_constant);
        }
        it = moveptr_constants_.insert({ ptr_move, sala_constant.index() }).first;
    }
    return it->second;
}


std::uint32_t Compiler::numeric_constant_index_impl(std::uint8_t const* const value_ptr, std::size_t const num_bytes)
{
    std::string value_string = to_hex_string(value_ptr, num_bytes);
    auto it_and_state = numeric_constants_.insert({ value_string, (std::uint32_t)program().constants().size() });
    if (it_and_state.second)
    {
        auto& sala_constant = program().push_back_constant();
        ASSUMPTION(sala_constant.index() == it_and_state.first->second);
        copy_bytes_of_value(value_ptr, num_bytes, sala_constant);
    }
    return it_and_state.first->second;
}


void Compiler::run()
{
    // Initialize the static initializer

    ASSUMPTION(program().functions().empty());
    program().push_back_function(sala::Program::static_initializer_name()).push_back_basic_block();

    // Register static objects

    for (auto global_it = module().global_begin(); global_it != module().global_end(); ++global_it)
    {
        auto& llvm_static = *global_it;
        if (!llvm_static.isConstant())
        {
            ASSUMPTION(llvm_static.getType()->isPointerTy());

            sala::SourceBackMapping back_mapping;
            {
                llvm::SmallVector<llvm::DIGlobalVariableExpression*> gvs;
                llvm_static.getDebugInfo(gvs);
                if (!gvs.empty())
                    back_mapping.line = gvs.front()->getVariable()->getLine();
            }

            MemoryObject sala_static_variable_ptr_mo;
            {
                auto& sala_static_variable_ptr = program().push_back_static_variable();
                sala_static_variable_ptr.set_num_bytes(llvm_sizeof(llvm_static.getType(), module()));
                sala_static_variable_ptr.source_back_mapping() = back_mapping;
                sala_static_variable_ptr_mo = register_variable(&llvm_static, sala_static_variable_ptr);
            }

            MemoryObject sala_static_variable_mo;
            {
                auto& sala_static_variable = program().push_back_static_variable();
                sala_static_variable.set_num_bytes(llvm_sizeof(llvm_static.getValueType(), module()));
                sala_static_variable.source_back_mapping() = back_mapping;
                sala_static_variable_mo = { sala_static_variable.index(), sala::Instruction::Descriptor::STATIC };
            }

            auto& sala_instruction = program().function_ref(program().static_initializer()).last_basic_block_ref().push_back_instruction();
            sala_instruction.source_back_mapping() = back_mapping;
            sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
            push_back_operand(sala_instruction, sala_static_variable_ptr_mo);
            push_back_operand(sala_instruction, sala_static_variable_mo);

            if (!llvm_static.hasInitializer())
                program().push_back_external_variable(sala_static_variable_mo.index, llvm_static.getName().str());
        }
    }
    for (auto it = module().begin(); it != module().end(); ++it)
    {
        std::string function_name{ it->getName().str() };
        if (it->isDeclaration())
        {
            bool do_register_function{ true };
            if (it->isIntrinsic())
                switch (it->getIntrinsicID())
                {
                    // Here we process llvm intrinsics which we translate as external functions.

                    case llvm::Intrinsic::fabs:
                        function_name = it->getReturnType()->isFloatTy() ? "fabsf" : "fabs";
                        break;
                    case llvm::Intrinsic::bswap:
                        function_name = "__llvm_intrinsic__bswap_" + std::to_string(8U * llvm_sizeof(it->getReturnType(), module()));
                        break;
                    case llvm::Intrinsic::ctlz:
                        function_name = "__llvm_intrinsic__ctlz_" + std::to_string(8U * llvm_sizeof(it->getReturnType(), module()));
                        break;
                    case llvm::Intrinsic::trunc:
                        function_name = "__llvm_intrinsic__trunc_" + std::to_string(8U * llvm_sizeof(it->getReturnType(), module()));
                        break;
                    case llvm::Intrinsic::rint:
                        function_name = "__llvm_intrinsic__rint_" + std::to_string(8U * llvm_sizeof(it->getReturnType(), module()));
                        break;

                    default:
                        do_register_function = false;
                        break;
                }
            else if (get_interpreted_function_opcode(&*it) != sala::Instruction::Opcode::__INVALID__)
                do_register_function = false;

            if (!do_register_function)
                continue;
        }
        auto const mo{ register_function(&*it, program().push_back_function(function_name)) };
    }

    // Compile static objects

    for (auto global_it = module().global_begin(); global_it != module().global_end(); ++global_it)
    {
        auto& llvm_static = *global_it;
        if (!llvm_static.isConstant() && llvm_static.hasInitializer())
        {
            auto variable_mo = memory_object(llvm_static);
            auto self_mo{ memory_object(llvm_static) };
            auto initializer_mo{ memory_object(llvm_static.getInitializer()) };
            auto& sala_instruction = program().function_ref(program().static_initializer()).basic_block_ref(0U).push_back_instruction();
            sala_instruction.set_opcode(sala::Instruction::Opcode::STORE);
            push_back_operand(sala_instruction, self_mo);
            push_back_operand(sala_instruction, initializer_mo);
        }
    }
    for (auto it = module().begin(); it != module().end(); ++it)
        if (has_memory_object(&*it) && it->isDeclaration())
        {
            auto& sala_function = program().function_ref(memory_object(*it).index);
            compile_function_parameters(*it, sala_function);
            sala_function.push_back_basic_block().push_back_instruction().set_opcode(sala::Instruction::Opcode::RET);
            program().push_back_external_function(sala_function.index());
        }
    for (auto it = module().begin(); it != module().end(); ++it)
        if (has_memory_object(&*it) && !it->isDeclaration())
            compile_function(*it, program().function_ref(memory_object(*it).index));

    // Remove NOPs from static initializer

    auto& sala_initializer = program().function_ref(program().static_initializer());
    for (std::size_t i = 0ULL, n = sala_initializer.basic_blocks().size(); i < n; ++i)
    {
        auto& sala_block = sala_initializer.basic_block_ref((std::uint32_t)i);
        if (is_nop_in_basic_block(sala_block))
            remove_nops_from_basic_block(sala_block);
    }

    // Finish the static initializer

    program().function_ref(sala::Program::static_initializer())
             .last_basic_block_ref()
             .push_back_instruction()
             .set_opcode(sala::Instruction::Opcode::RET)
             ;
}


void Compiler::compile_constant(
    llvm::Constant const* const llvm_constant,
    sala::Constant& sala_constant,
    std::vector<std::pair<llvm::Value*, std::uint64_t> >& pointer_initializations)
{
    if (llvm_constant->getType()->isPointerTy())
    {
        if (llvm_constant->isZeroValue())
            for (unsigned int i = 0U, n = module().getDataLayout().getPointerSize(); i < n; ++i)
                sala_constant.push_back_byte(0U);
        else
        {
            pointer_initializations.push_back({ (llvm::Value*)llvm_constant, sala_constant.num_bytes() });
            for (unsigned int i = 0U, n = module().getDataLayout().getPointerSize(); i < n; ++i)
                sala_constant.push_back_byte(171U); // hex: AB, bin: 10101011 - indicates not initialized pointer.
        }
        return;
    }

    if (auto llvm_int = llvm::dyn_cast<llvm::ConstantInt>(llvm_constant))
    {
        std::size_t const num_bits = std::max(8U, llvm_int->getValue().getBitWidth());
        ASSUMPTION((num_bits % 8U) == 0 && num_bits <= 64U);
        copy_bytes_of_value(
            (std::uint8_t const *)llvm_int->getValue().getRawData(),
            num_bits / 8U,
            sala_constant
            );
    }
    else if (auto llvm_float = llvm::dyn_cast<llvm::ConstantFP>(llvm_constant))
    {
        if (llvm_float->getType()->isFloatTy())
        {
            float data = llvm_float->getValue().convertToFloat();
            copy_bytes_of_value((std::uint8_t const *)&data, sizeof(data), sala_constant);
        }
        else if (llvm_float->getType()->isDoubleTy())
        {
            double data = llvm_float->getValue().convertToDouble();
            copy_bytes_of_value((std::uint8_t const *)&data, sizeof(data), sala_constant);
        }
        else if (llvm_float->getType()->isX86_FP80Ty() || llvm_float->getType()->isFP128Ty())
        {
            // WARNING: Here we are not correct as we represent 80-bit and 128-bit floats as 64-bit floats.
            unsigned int constexpr NUM_HEX_CHARS{ 2U * sizeof(double) };
            char hex_string[NUM_HEX_CHARS + 1U];
            llvm_float->getValue().convertToHexString(hex_string, NUM_HEX_CHARS, false, llvm::RoundingMode::TowardZero);
            hex_string[NUM_HEX_CHARS] = 0;
            double data = std::strtod(hex_string, nullptr);
            copy_bytes_of_value((std::uint8_t const *)&data, sizeof(data), sala_constant);
        }
        else UNREACHABLE();
    }
    else if (auto llvm_array = llvm::dyn_cast<llvm::ConstantDataArray>(llvm_constant))
    {
        for (std::size_t i = 0U, n = llvm_array->getType()->getArrayNumElements(); i < n; ++i)
            compile_constant(llvm_array->getAggregateElement((unsigned int)i), sala_constant, pointer_initializations);
    }
    else if (auto llvm_array = llvm::dyn_cast<llvm::ConstantArray>(llvm_constant))
    {
        for (std::size_t i = 0U, n = llvm_array->getType()->getArrayNumElements(); i < n; ++i)
            compile_constant(llvm_array->getAggregateElement((unsigned int)i), sala_constant, pointer_initializations);
    }
    else if (auto llvm_zero_initializer = llvm::dyn_cast<llvm::ConstantAggregateZero>(llvm_constant))
    {
        if (llvm_zero_initializer->getType()->isArrayTy())
            for (std::size_t i = 0U, n = llvm_zero_initializer->getType()->getArrayNumElements(); i < n; ++i)
                compile_constant(llvm_zero_initializer->getAggregateElement((unsigned int)i), sala_constant, pointer_initializations);
        else if (llvm_zero_initializer->getType()->isStructTy())
        {
            auto start_size = sala_constant.bytes().size();
            auto llvm_struct_layout = module().getDataLayout().getStructLayout(
                llvm::dyn_cast<llvm::StructType>(llvm_zero_initializer->getType())
                );
            for (std::size_t i = 0U, n = llvm_zero_initializer->getType()->getStructNumElements(); i < n; ++i)
            {
                auto num_bytes_to_move_over = llvm_struct_layout->getElementOffset((unsigned int)i);
                while (sala_constant.bytes().size() - start_size < num_bytes_to_move_over)
                    sala_constant.push_back_byte(205U); // hex: CD, bin: 11001101 - indicates not initialized memory.
                compile_constant(llvm_zero_initializer->getAggregateElement((unsigned int)i), sala_constant, pointer_initializations);
            }
            auto num_bytes_to_move_over = llvm_struct_layout->getSizeInBytes();
            while (sala_constant.bytes().size() - start_size < num_bytes_to_move_over)
                sala_constant.push_back_byte(205U); // hex: CD, bin: 11001101 - indicates not initialized memory.
        }
        else { NOT_IMPLEMENTED_YET(); }
    }
    else if (auto llvm_struct = llvm::dyn_cast<llvm::ConstantStruct>(llvm_constant))
    {
        auto start_size = sala_constant.bytes().size();
        auto llvm_struct_layout = module().getDataLayout().getStructLayout(llvm_struct->getType());
        for (std::size_t i = 0U, n = llvm_struct->getType()->getStructNumElements(); i < n; ++i)
        {
            auto num_bytes_to_move_over = llvm_struct_layout->getElementOffset((unsigned int)i);
            while (sala_constant.bytes().size() - start_size < num_bytes_to_move_over)
                sala_constant.push_back_byte(205U); // hex: CD, bin: 11001101 - indicates not initialized memory.
            compile_constant(llvm_struct->getAggregateElement((unsigned int)i), sala_constant, pointer_initializations);
        }
        auto num_bytes_to_move_over = llvm_struct_layout->getSizeInBytes();
        while (sala_constant.bytes().size() - start_size < num_bytes_to_move_over)
            sala_constant.push_back_byte(205U); // hex: CD, bin: 11001101 - indicates not initialized memory.
    }
    else if (llvm::isa<llvm::UndefValue>(llvm_constant))
        for (std::size_t i = 0ULL, n = llvm_sizeof(llvm_constant->getType(), module()); i < n; ++i)
            sala_constant.push_back_byte(205U); // hex: CD, bin: 11001101 - indicates not initialized memory.
    else if (auto llvm_ptr2int = llvm::dyn_cast<llvm::PtrToIntOperator>(llvm_constant))
    {
        // ASSUMPTION(llvm_sizeof(llvm_ptr2int->getType(), module()) == 8ULL && llvm::isa<llvm::Constant>(llvm_ptr2int->getOperand(0)));
        // compile_constant(llvm::dyn_cast<llvm::Constant>(llvm_ptr2int->getOperand(0)), sala_constant, pointer_initializations);
        NOT_IMPLEMENTED_YET();
    }
    else
    {
        //std::cout << "\ncompile_constant() - NOT_IMPLEMENTED_YET:\n" << llvm_to_str(llvm_constant) << "\n";
        NOT_IMPLEMENTED_YET();
    }
}


void Compiler::compile_function(llvm::Function& llvm_function, sala::Function& sala_function)
{
    set_compiled_function(&sala_function);
    set_uses_stacksave(llvm_function_contains_intrinsic(&llvm_function, llvm::Intrinsic::stacksave));

    if (sala_function.name() == "main")
        program().set_entry_function(sala_function.index());

    std::unordered_map<llvm::BasicBlock*, std::uint32_t> mapping;
    for (auto it = llvm_function.begin(); it != llvm_function.end(); ++it)
        mapping.insert({ &*it, sala_function.push_back_basic_block().index() });
    for (auto llvm_and_sala : mapping)
    {
        std::vector<std::uint32_t> successors;
        for (auto it = llvm::succ_begin(llvm_and_sala.first); it != llvm::succ_end(llvm_and_sala.first); ++it)
            successors.push_back(mapping.at(*it));
        ASSUMPTION(successors.size() <= 2ULL);
        auto& sala_basic_block = compiled_function()->basic_block_ref(llvm_and_sala.second);
        for (auto it = successors.rbegin(); it != successors.rend(); ++it)
            sala_basic_block.push_back_successor(*it);
    }
    if (llvm_function.getSubprogram() != nullptr)
    {
        sala_function.source_back_mapping().line = llvm_function.getSubprogram()->getLine();
        sala_function.source_back_mapping().column = 1;
    }

    compile_function_parameters(llvm_function, sala_function);

    // We first allocate all llvm registers as Sala stack variables.
    for (auto it_bb = llvm_function.begin(); it_bb != llvm_function.end(); ++it_bb)
        for (auto it_instr = it_bb->begin(); it_instr != it_bb->end(); ++it_instr)
            if (!it_instr->getType()->isVoidTy())
            {
                auto& sala_variable = sala_function.push_back_local_variable();
                sala_variable.set_num_bytes(llvm_sizeof(it_instr->getType(), module()));
                if (llvm::DILocation const* dbg_loc = it_instr->getDebugLoc())
                {
                    sala_variable.source_back_mapping().line = dbg_loc->getLine();
                    sala_variable.source_back_mapping().column = dbg_loc->getColumn();
                }
                register_variable(&*it_instr, sala_variable);
            }

    std::vector<llvm::PHINode*> phi_nodes;
    for (auto it = llvm_function.begin(); it != llvm_function.end(); ++it)
        // We do NOT iterate over 'mapping', because the order can be arbitrary!
    {
        auto mapping_it = mapping.find(&*it);
        compile_basic_block(*mapping_it->first, compiled_function()->basic_block_ref(mapping_it->second), phi_nodes);
    }

    for (llvm::PHINode* phi : phi_nodes)
    {
        auto& phi_memory_object = memory_object(phi);

        for (unsigned int i = 0U, n = phi->getNumIncomingValues(); i < n; ++i)
        {
            auto const phi_block_index = mapping.at(phi->getParent());

            auto& new_sala_block = sala_function.push_back_basic_block();
            new_sala_block.push_back_successor(phi_block_index);

            auto& incoming_sala_block = sala_function.basic_block_ref(mapping.at(phi->getIncomingBlock(i)));
            for (std::size_t i = 0ULL, n = incoming_sala_block.successors().size(); i < n; ++i)
                if (incoming_sala_block.successors().at(i) == phi_block_index)
                {
                    incoming_sala_block.successor_ref((std::uint32_t)i) = new_sala_block.index();
                    break;
                }

            {
                auto& sala_instruction = new_sala_block.push_back_instruction();
                sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
                push_back_operand(sala_instruction, phi_memory_object);
                push_back_operand(sala_instruction, memory_object(phi->getIncomingValue(i)));
            }
            {
                auto& sala_instruction = new_sala_block.push_back_instruction();
                sala_instruction.set_opcode(sala::Instruction::Opcode::JUMP);
            }
        }
    }

    set_uses_stacksave();
    set_compiled_function();
}


void Compiler::compile_function_parameters(llvm::Function& llvm_function, sala::Function& sala_function)
{
    if (!llvm_function.getReturnType()->isVoidTy())
    {
        auto& sala_return_parameter = sala_function.push_back_parameter();
        sala_return_parameter.set_num_bytes(module().getDataLayout().getPointerSize());
        sala_return_parameter.source_back_mapping() = sala_function.source_back_mapping();
    }
    for (auto it = llvm_function.arg_begin(); it != llvm_function.arg_end(); ++it)
    {
        auto& sala_parameter = sala_function.push_back_parameter();
        register_parameter(&*it, sala_parameter);
        sala_parameter.set_num_bytes(llvm_sizeof(it->getType(), module()));
        sala_parameter.source_back_mapping() = sala_function.source_back_mapping();
    }
}


void Compiler::compile_basic_block(llvm::BasicBlock& llvm_block, sala::BasicBlock& sala_block, std::vector<llvm::PHINode*>& phi_nodes)
{
    set_compiled_basic_block(&sala_block);

    for (auto it = llvm_block.begin(); it != llvm_block.end(); ++it)
    {
        llvm::IntrinsicInst* const intrinsic_instr = llvm::dyn_cast<llvm::IntrinsicInst>(&*it);
        if (intrinsic_instr != nullptr)
            switch (intrinsic_instr->getIntrinsicID())
            {
                case llvm::Intrinsic::dbg_declare:
                case llvm::Intrinsic::dbg_label:
                    continue;
                default:
                    break;
            }

        std::uint32_t sala_instruction_index;
        {
            auto& sala_instruction = sala_block.push_back_instruction();
            sala_instruction_index = sala_instruction.index();

            if (llvm::DILocation const* dbg_loc = it->getDebugLoc())
            {
                sala_instruction.source_back_mapping().line = dbg_loc->getLine();
                sala_instruction.source_back_mapping().column = dbg_loc->getColumn();
            }

            compile_instruction(*it, sala_instruction, phi_nodes);
        }
    }

    if (is_nop_in_basic_block(sala_block))
        remove_nops_from_basic_block(sala_block);

    set_compiled_basic_block();
}


void Compiler::compile_instruction(llvm::Instruction& llvm_instruction, sala::Instruction& sala_instruction, std::vector<llvm::PHINode*>& phi_nodes)
{
    switch (llvm_instruction.getOpcode())
    {
    case llvm::Instruction::Unreachable:
        compile_instruction_unreachable(*llvm::dyn_cast<llvm::UnreachableInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Alloca:
        compile_instruction_alloca(*llvm::dyn_cast<llvm::AllocaInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::BitCast:
        compile_instruction_bitcast(*llvm::dyn_cast<llvm::BitCastInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Load:
        compile_instruction_load(*llvm::dyn_cast<llvm::LoadInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Store:
        compile_instruction_store(*llvm::dyn_cast<llvm::StoreInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Add:
    case llvm::Instruction::FAdd:
        compile_instruction_add(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Sub:
    case llvm::Instruction::FSub:
        compile_instruction_sub(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Mul:
    case llvm::Instruction::FMul:
        compile_instruction_mul(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::SDiv:
    case llvm::Instruction::UDiv:
    case llvm::Instruction::FDiv:
        compile_instruction_div(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::SRem:
    case llvm::Instruction::URem:
        compile_instruction_rem(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::And:
        compile_instruction_and(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Or:
        compile_instruction_or(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Xor:
        compile_instruction_xor(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Shl:
        compile_instruction_shl(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::AShr:
        compile_instruction_ashr(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::LShr:
        compile_instruction_lshr(*llvm::dyn_cast<llvm::BinaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::FNeg:
        compile_instruction_fneg(*llvm::dyn_cast<llvm::UnaryOperator>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::SExt:
    case llvm::Instruction::ZExt:
        compile_instruction_cast(*llvm::dyn_cast<llvm::CastInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::FPExt:
        compile_instruction_cast(*llvm::dyn_cast<llvm::FPExtInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Trunc:
        compile_instruction_cast(*llvm::dyn_cast<llvm::TruncInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::FPTrunc:
        compile_instruction_cast(*llvm::dyn_cast<llvm::FPTruncInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::SIToFP:
        compile_instruction_cast(*llvm::dyn_cast<llvm::SIToFPInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::UIToFP:
        compile_instruction_cast(*llvm::dyn_cast<llvm::UIToFPInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::FPToSI:
        compile_instruction_cast(*llvm::dyn_cast<llvm::FPToSIInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::FPToUI:
        compile_instruction_cast(*llvm::dyn_cast<llvm::FPToUIInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::PtrToInt:
        compile_instruction_ptrtoint(*llvm::dyn_cast<llvm::PtrToIntInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::IntToPtr:
        compile_instruction_inttoptr(*llvm::dyn_cast<llvm::IntToPtrInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::GetElementPtr:
        compile_instruction_getelementptr(*llvm::dyn_cast<llvm::GetElementPtrInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::ICmp:
    case llvm::Instruction::FCmp:
        compile_instruction_cmp(*llvm::dyn_cast<llvm::CmpInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Br:
        compile_instruction_br(*llvm::dyn_cast<llvm::BranchInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::PHI:
        {
            sala_instruction.set_opcode(sala::Instruction::Opcode::NOP);
            phi_nodes.push_back(llvm::dyn_cast<llvm::PHINode>(&llvm_instruction));
        }
        break;
    case llvm::Instruction::ExtractValue:
        compile_instruction_extractvalue(*llvm::dyn_cast<llvm::ExtractValueInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::InsertValue:
        compile_instruction_insertvalue(*llvm::dyn_cast<llvm::InsertValueInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Call:
        compile_instruction_call(*llvm::dyn_cast<llvm::CallInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::Ret:
        compile_instruction_ret(*llvm::dyn_cast<llvm::ReturnInst>(&llvm_instruction), sala_instruction);
        break;
    case llvm::Instruction::VAArg:
        compile_instruction_vaarg(*llvm::dyn_cast<llvm::VAArgInst>(&llvm_instruction), sala_instruction);
        break;
    default:
        NOT_IMPLEMENTED_YET();
        break;
    }
}


void Compiler::compile_instruction_unreachable(llvm::UnreachableInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    sala_instruction.set_opcode(sala::Instruction::Opcode::HALT);
}


void Compiler::compile_instruction_alloca(llvm::AllocaInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if ((uses_stacksave()
             && (sala_instruction.basic_block_index() != 0U ||
                 llvm_basic_block_contains_intrinsic_before_instruction(
                        llvm_instruction.getParent(),
                        llvm::Intrinsic::stacksave,
                        &llvm_instruction)))
        || (llvm_instruction.getNumOperands() > 0U && !llvm::isa<llvm::ConstantInt>(llvm_instruction.getOperand(0U)))
        )
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::ALLOCA);
        push_back_operand(sala_instruction, memory_object(&llvm_instruction));
        if (llvm_instruction.getNumOperands() == 0U)
            push_back_operand(sala_instruction, { moveptr_constant_index(1ULL), sala::Instruction::Descriptor::CONSTANT });
        else
            push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
        push_back_operand(sala_instruction, {
            moveptr_constant_index(llvm_sizeof(llvm_instruction.getAllocatedType(), module())),
            sala::Instruction::Descriptor::CONSTANT
            });
    }
    else
    {
        auto& var = compiled_function()->push_back_local_variable();
        var.source_back_mapping() = sala_instruction.source_back_mapping();

        std::uint64_t num_elements{ 1ULL };
        if (llvm_instruction.getNumOperands() > 0U)
        {
            auto llvm_constant = llvm::dyn_cast<llvm::ConstantInt>(llvm_instruction.getOperand(0U));
            switch (llvm_constant->getValue().getBitWidth() / 8U)
            {
            case 1U:
                num_elements = (std::uint64_t)*(std::int8_t const *)llvm_constant->getValue().getRawData();
                break;
            case 2U:
                num_elements = (std::uint64_t)*(std::int16_t const *)llvm_constant->getValue().getRawData();
                break;
            case 4U:
                num_elements = (std::uint64_t)*(std::int32_t const *)llvm_constant->getValue().getRawData();
                break;
            case 8U:
                num_elements = *(std::uint64_t const *)llvm_constant->getValue().getRawData();
                break;
            default: UNREACHABLE(); break;
            }
        }

        var.set_num_bytes(num_elements * llvm_sizeof(llvm_instruction.getAllocatedType(), module()));

        sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
        push_back_operand(sala_instruction, memory_object(&llvm_instruction));
        push_back_operand(sala_instruction, { var.index(), sala::Instruction::Descriptor::LOCAL });
    }
}


void Compiler::compile_instruction_bitcast(llvm::BitCastInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm::isa<llvm::Function>(llvm_instruction.getOperand(0)))
        sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
    else
        sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
    push_back_operand(sala_instruction, memory_object(&llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}


void Compiler::compile_instruction_load(llvm::LoadInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    sala_instruction.set_opcode(sala::Instruction::Opcode::LOAD);
    push_back_operand(sala_instruction, memory_object(&llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}


void Compiler::compile_instruction_store(llvm::StoreInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm::isa<llvm::Function>(llvm_instruction.getOperand(0)))
    {
        auto const sala_instruction_back_mapping{ sala_instruction.source_back_mapping() };

        MemoryObject const var_mo { compiled_function()->push_back_local_variable().index(), sala::Instruction::Descriptor::LOCAL };
        compiled_function()->last_local_variable_ref().set_num_bytes(module().getDataLayout().getPointerSize());
        compiled_function()->last_local_variable_ref().source_back_mapping() = sala_instruction_back_mapping;

        sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
        push_back_operand(sala_instruction, var_mo);
        push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));

        auto& instr = compiled_basic_block()->push_back_instruction();
        instr.source_back_mapping() = sala_instruction_back_mapping;
        instr.set_opcode(sala::Instruction::Opcode::STORE);
        push_back_operand(instr, memory_object(llvm_instruction.getOperand(1)));
        push_back_operand(instr, var_mo);
    }
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::STORE);
        push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(1)));
        push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
    }
}


void Compiler::compile_instruction_add(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::ADD);

    if (llvm_instruction.getOpcode() == llvm::Instruction::FAdd)
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    else if (llvm_instruction.hasNoSignedWrap())
        sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
    else
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_sub(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::SUB);

    if (llvm_instruction.getOpcode() == llvm::Instruction::FSub)
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    else if (llvm_instruction.hasNoSignedWrap())
        sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
    else
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_mul(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::MUL);

    if (llvm_instruction.getOpcode() == llvm::Instruction::FMul)
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    else if (llvm_instruction.hasNoSignedWrap())
        sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
    else
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_div(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::DIV);

    if (llvm_instruction.getOpcode() == llvm::Instruction::FDiv)
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    else if (llvm_instruction.getOpcode() == llvm::Instruction::SDiv)
        sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
    else
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_rem(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::REM);

    if (llvm_instruction.getOpcode() == llvm::Instruction::SRem)
        sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
    else
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_and(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::AND);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_or(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::OR);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_xor(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::XOR);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_shl(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::SHL);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_ashr(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::SHR);
    sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_lshr(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));
    auto& op1 = memory_object(llvm_instruction.getOperand(1));

    sala_instruction.set_opcode(sala::Instruction::Opcode::SHR);
    sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
    push_back_operand(sala_instruction, op1);
}


void Compiler::compile_instruction_fneg(llvm::UnaryOperator& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::NEG);
    sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_cast(llvm::CastInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm_sizeof(llvm_instruction.getType(), module()) == llvm_sizeof(llvm_instruction.getOperand(0)->getType(), module()))
        sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::EXTEND);
        sala_instruction.set_modifier(llvm_instruction.getOpcode() == llvm::Instruction::SExt ?
                sala::Instruction::Modifier::SIGNED : sala::Instruction::Modifier::UNSIGNED
                );
    }

    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}


void Compiler::compile_instruction_cast(llvm::FPExtInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm_sizeof(llvm_instruction.getType(), module()) == llvm_sizeof(llvm_instruction.getOperand(0)->getType(), module()))
        sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::EXTEND);
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    }

    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}


void Compiler::compile_instruction_cast(llvm::TruncInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));

    if (llvm_sizeof(llvm_instruction.getType(), module()) == llvm_sizeof(llvm_instruction.getOperand(0)->getType(), module()))
    {
        ASSUMPTION(llvm_instruction.getType()->isIntegerTy(1) && llvm_instruction.getOperand(0)->getType()->isIntegerTy(8));
        sala_instruction.set_opcode(sala::Instruction::Opcode::AND);
        sala_instruction.push_back_operand(numeric_constant_index<std::uint8_t>(1U), sala::Instruction::Descriptor::CONSTANT);
    }
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::TRUNCATE);
        sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
    }
}


void Compiler::compile_instruction_cast(llvm::FPTruncInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm_sizeof(llvm_instruction.getType(), module()) == llvm_sizeof(llvm_instruction.getOperand(0)->getType(), module()))
        sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::TRUNCATE);
        sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
    }

    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}


void Compiler::compile_instruction_cast(llvm::SIToFPInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::I2F);
    sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_cast(llvm::UIToFPInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::I2F);
    sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_cast(llvm::FPToSIInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::F2I);
    sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_cast(llvm::FPToUIInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::F2I);
    sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_ptrtoint(llvm::PtrToIntInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::P2I);
    sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_inttoptr(llvm::IntToPtrInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto& self = memory_object(llvm_instruction);
    auto& op0 = memory_object(llvm_instruction.getOperand(0));

    sala_instruction.set_opcode(sala::Instruction::Opcode::I2P);
    sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);

    push_back_operand(sala_instruction, self);
    push_back_operand(sala_instruction, op0);
}


void Compiler::compile_instruction_getelementptr(llvm::GetElementPtrInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto const sala_instruction_back_mapping{ sala_instruction.source_back_mapping() };

    auto self = memory_object(llvm_instruction);
    auto ptr_operand = memory_object(llvm_instruction.getPointerOperand());

    std::vector<MemoryObject> idx_memory_objects;
    std::vector<std::int64_t> constant_indices;
    for (unsigned int i = 1; i < llvm_instruction.getNumOperands(); ++i)
    {
        auto llvm_index = llvm_instruction.getOperand(i);
        if (auto llvm_constant = llvm::dyn_cast<llvm::ConstantInt>(llvm_index))
        {
            std::int64_t index;
            switch (llvm_constant->getValue().getBitWidth() / 8U)
            {
            case 1U:
                index = (std::int64_t)*(std::int8_t const *)llvm_constant->getValue().getRawData();
                break;
            case 2U:
                index = (std::int64_t)*(std::int16_t const *)llvm_constant->getValue().getRawData();
                break;
            case 4U:
                index = (std::int64_t)*(std::int32_t const *)llvm_constant->getValue().getRawData();
                break;
            case 8U:
                index = *(std::int64_t const *)llvm_constant->getValue().getRawData();
                break;
            default: UNREACHABLE(); break;
            }
            idx_memory_objects.push_back({ moveptr_constant_index(index), sala::Instruction::Descriptor::CONSTANT });
            constant_indices.push_back(index);
        }
        else
        {
            idx_memory_objects.push_back(memory_object(llvm_index));
            constant_indices.push_back(std::numeric_limits<std::int64_t>::min());
        }
    }

    ASSUMPTION(!idx_memory_objects.empty());
    INVARIANT(idx_memory_objects.size() == constant_indices.size());

    std::vector<std::uint32_t> sala_instructions{ compiled_basic_block()->last_instruction_ref().index() };
    for (unsigned int i = 2, n = llvm_instruction.getNumOperands(); i < n; ++i)
    {
        auto& sala_instruction_new = compiled_basic_block()->push_back_instruction();
        sala_instruction_new.source_back_mapping() = sala_instruction_back_mapping;
        sala_instructions.push_back(sala_instruction_new.index());
    }

    INVARIANT(idx_memory_objects.size() == sala_instructions.size());

    llvm::Type* llvm_operand0_type = llvm_instruction.getPointerOperand()->getType();
    ASSUMPTION(llvm_operand0_type->isPointerTy());

    auto operand_ptr{ &ptr_operand };

    std::size_t num_nops{ 0ULL };
    for (std::size_t i = 0ULL; i < idx_memory_objects.size(); ++i)
    {
        auto& sala_moveptr_instruction = compiled_basic_block()->instruction_ref(sala_instructions.at(i));
        switch (llvm_operand0_type->getTypeID())
        {
        case llvm::Type::ArrayTyID:
            llvm_operand0_type = llvm_operand0_type->getArrayElementType();
            if (constant_indices.at(i) == 0ULL)
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::NOP);
                ++num_nops;
            }
            else
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);

                push_back_operand(sala_moveptr_instruction, self);
                push_back_operand(sala_moveptr_instruction, *operand_ptr);
                push_back_operand(sala_moveptr_instruction, idx_memory_objects.at(i));

                auto num_pointed_bytes = llvm_sizeof(llvm_operand0_type, module());
                MemoryObject size_memory_object{ moveptr_constant_index(num_pointed_bytes), sala::Instruction::Descriptor::CONSTANT };
                push_back_operand(sala_moveptr_instruction, size_memory_object);
            }
            break;
        case llvm::Type::StructTyID:
            ASSUMPTION(constant_indices.at(i) >= 0LL);
            if (constant_indices.at(i) == 0ULL)
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::NOP);
                ++num_nops;
            }
            else
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);

                push_back_operand(sala_moveptr_instruction, self);
                push_back_operand(sala_moveptr_instruction, *operand_ptr);
                push_back_operand(sala_moveptr_instruction, { moveptr_constant_index(1LL), sala::Instruction::Descriptor::CONSTANT });

                auto llvm_struct_layout = module().getDataLayout().getStructLayout(llvm::dyn_cast<llvm::StructType>(llvm_operand0_type));
                auto num_bytes_to_move_over = (std::int64_t)llvm_struct_layout->getElementOffset((unsigned int)constant_indices.at(i));
                push_back_operand(sala_moveptr_instruction, { moveptr_constant_index(num_bytes_to_move_over), sala::Instruction::Descriptor::CONSTANT });
            }
            llvm_operand0_type = llvm_operand0_type->getStructElementType((unsigned int)constant_indices.at(i));
            break;
        case llvm::Type::PointerTyID:
            INVARIANT(i == 0ULL);
            llvm_operand0_type = llvm_instruction.getSourceElementType();
            if (constant_indices.at(i) == 0ULL)
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::NOP);
                ++num_nops;
            }
            else
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);

                push_back_operand(sala_moveptr_instruction, self);
                push_back_operand(sala_moveptr_instruction, *operand_ptr);
                push_back_operand(sala_moveptr_instruction, idx_memory_objects.at(i));

                auto num_pointed_bytes = llvm_sizeof(llvm_operand0_type, module());
                MemoryObject size_memory_object{ moveptr_constant_index(num_pointed_bytes), sala::Instruction::Descriptor::CONSTANT };
                push_back_operand(sala_moveptr_instruction, size_memory_object);
            }
            break;
        default: ASSUMPTION(i + 1ULL ==  idx_memory_objects.size()); break;
        }

        if (sala_moveptr_instruction.opcode() != sala::Instruction::Opcode::NOP)
            operand_ptr = &self;
    }

    if (num_nops == idx_memory_objects.size() && self != ptr_operand)
    {
        auto& sala_instruction = compiled_basic_block()->instruction_ref(sala_instructions.front());
        sala_instruction.set_opcode(sala::Instruction::Opcode::COPY);
        push_back_operand(sala_instruction, self);
        push_back_operand(sala_instruction, ptr_operand);
    }
}


void Compiler::compile_instruction_cmp(llvm::CmpInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(1)));

    switch (llvm_instruction.getPredicate())
    {
        case llvm::CmpInst::Predicate::FCMP_OEQ:
            sala_instruction.set_opcode(sala::Instruction::Opcode::EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;
        case llvm::CmpInst::Predicate::FCMP_OGT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;
        case llvm::CmpInst::Predicate::FCMP_OGE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;
        case llvm::CmpInst::Predicate::FCMP_OLT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;
        case llvm::CmpInst::Predicate::FCMP_OLE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;
        case llvm::CmpInst::Predicate::FCMP_ONE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::UNEQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
            break;

        case llvm::CmpInst::Predicate::FCMP_UEQ:
            sala_instruction.set_opcode(sala::Instruction::Opcode::EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;
        case llvm::CmpInst::Predicate::FCMP_UGT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;
        case llvm::CmpInst::Predicate::FCMP_UGE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;
        case llvm::CmpInst::Predicate::FCMP_ULT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;
        case llvm::CmpInst::Predicate::FCMP_ULE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;
        case llvm::CmpInst::Predicate::FCMP_UNE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::UNEQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;

        case llvm::CmpInst::Predicate::FCMP_UNO:
            sala_instruction.set_opcode(sala::Instruction::Opcode::ISNAN);
            sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING_UNORDERED);
            break;

        case llvm::CmpInst::Predicate::ICMP_EQ:
            sala_instruction.set_opcode(sala::Instruction::Opcode::EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_NE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::UNEQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_UGT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_UGE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_ULT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_ULE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::UNSIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_SGT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER);
            sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_SGE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::GREATER_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_SLT:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS);
            sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
            break;
        case llvm::CmpInst::Predicate::ICMP_SLE:
            sala_instruction.set_opcode(sala::Instruction::Opcode::LESS_EQUAL);
            sala_instruction.set_modifier(sala::Instruction::Modifier::SIGNED);
            break;

        default:
            NOT_IMPLEMENTED_YET();
            break;
    }
}


void Compiler::compile_instruction_br(llvm::BranchInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm_instruction.isUnconditional())
        sala_instruction.set_opcode(sala::Instruction::Opcode::JUMP);
    else
    {
        sala_instruction.set_opcode(sala::Instruction::Opcode::BRANCH);
        push_back_operand(sala_instruction, memory_object(llvm_instruction.getCondition()));
    }
}


void Compiler::compile_instruction_extractvalue(llvm::ExtractValueInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto const sala_instruction_back_mapping{ sala_instruction.source_back_mapping() };

    auto self = memory_object(llvm_instruction);
    auto operand0 = memory_object(llvm_instruction.getAggregateOperand());

    MemoryObject sala_ptr_variable_mo;
    {
        auto& sala_ptr_variable = compiled_function()->push_back_local_variable();
        sala_ptr_variable.set_num_bytes(module().getDataLayout().getPointerSize());
        sala_ptr_variable.source_back_mapping() = sala_instruction.source_back_mapping();
        sala_ptr_variable_mo.index = sala_ptr_variable.index();
        sala_ptr_variable_mo.descriptor = sala::Instruction::Descriptor::LOCAL;
    }

    sala_instruction.set_opcode(sala::Instruction::Opcode::ADDRESS);
    push_back_operand(sala_instruction, sala_ptr_variable_mo);
    push_back_operand(sala_instruction, operand0);

    std::vector<MemoryObject> idx_memory_objects;
    std::vector<std::int64_t> constant_indices;
    for (auto it = llvm_instruction.idx_begin(); it != llvm_instruction.idx_end(); ++it)
    {
        idx_memory_objects.push_back({ moveptr_constant_index(*it), sala::Instruction::Descriptor::CONSTANT });
        constant_indices.push_back(*it);
    }

    ASSUMPTION(!idx_memory_objects.empty());
    INVARIANT(idx_memory_objects.size() == constant_indices.size());

    std::vector<std::uint32_t> sala_instructions;
    for (std::size_t i = 0ULL; i < idx_memory_objects.size(); ++i)
    {
        auto& sala_instruction_new = compiled_basic_block()->push_back_instruction();
        sala_instruction_new.source_back_mapping() = sala_instruction_back_mapping;
        sala_instructions.push_back(sala_instruction_new.index());
    }

    INVARIANT(idx_memory_objects.size() == sala_instructions.size());

    llvm::Type* llvm_operand0_type = llvm_instruction.getAggregateOperand()->getType();
    ASSUMPTION(llvm_operand0_type->isArrayTy() || llvm_operand0_type->isStructTy());

    std::size_t num_nops{ 0ULL };
    for (std::size_t i = 0ULL; i < idx_memory_objects.size(); ++i)
    {
        auto& sala_moveptr_instruction = compiled_basic_block()->instruction_ref(sala_instructions.at(i));
        switch (llvm_operand0_type->getTypeID())
        {
        case llvm::Type::ArrayTyID:
            llvm_operand0_type = llvm_operand0_type->getArrayElementType();
            if (constant_indices.at(i) == 0ULL)
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::NOP);
                ++num_nops;
            }
            else
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);

                push_back_operand(sala_moveptr_instruction, sala_ptr_variable_mo);
                push_back_operand(sala_moveptr_instruction, sala_ptr_variable_mo);
                push_back_operand(sala_moveptr_instruction, idx_memory_objects.at(i));

                auto num_pointed_bytes = llvm_sizeof(llvm_operand0_type, module());
                MemoryObject size_memory_object{ moveptr_constant_index(num_pointed_bytes), sala::Instruction::Descriptor::CONSTANT };
                push_back_operand(sala_moveptr_instruction, size_memory_object);
            }
            break;
        case llvm::Type::StructTyID:
            ASSUMPTION(constant_indices.at(i) >= 0LL);
            if (constant_indices.at(i) == 0ULL)
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::NOP);
                ++num_nops;
            }
            else
            {
                sala_moveptr_instruction.set_opcode(sala::Instruction::Opcode::MOVEPTR);

                push_back_operand(sala_moveptr_instruction, sala_ptr_variable_mo);
                push_back_operand(sala_moveptr_instruction, sala_ptr_variable_mo);
                push_back_operand(sala_moveptr_instruction, { moveptr_constant_index(1LL), sala::Instruction::Descriptor::CONSTANT });

                auto llvm_struct_layout = module().getDataLayout().getStructLayout(llvm::dyn_cast<llvm::StructType>(llvm_operand0_type));
                auto num_bytes_to_move_over = (std::int64_t)llvm_struct_layout->getElementOffset((unsigned int)constant_indices.at(i));
                push_back_operand(sala_moveptr_instruction, { moveptr_constant_index(num_bytes_to_move_over), sala::Instruction::Descriptor::CONSTANT });
            }
            llvm_operand0_type = llvm_operand0_type->getStructElementType((unsigned int)constant_indices.at(i));
            break;
        case llvm::Type::PointerTyID:
            UNREACHABLE();
            break;
        default: ASSUMPTION(i + 1ULL ==  idx_memory_objects.size()); break;
        }
    }

    auto& sala_instruction_new = compiled_basic_block()->push_back_instruction();
    sala_instruction_new.source_back_mapping() = sala_instruction_back_mapping;
    sala_instruction_new.set_opcode(sala::Instruction::Opcode::LOAD);
    push_back_operand(sala_instruction_new, self);
    push_back_operand(sala_instruction_new, sala_ptr_variable_mo);
}


void Compiler::compile_instruction_insertvalue(llvm::InsertValueInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    NOT_IMPLEMENTED_YET();
}


void Compiler::compile_instruction_call(llvm::CallInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    auto const sala_instruction_back_mapping{ sala_instruction.source_back_mapping() };
    auto opcode { sala::Instruction::Opcode::CALL };

    if (auto llvm_intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&llvm_instruction))
    {
        switch (llvm_intrinsic->getIntrinsicID())
        {
        case llvm::Intrinsic::memcpy:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::MEMCPY);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(1)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(2)));
            }
            return;
        case llvm::Intrinsic::memmove:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::MEMMOVE);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(1)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(2)));
            }
            return;
        case llvm::Intrinsic::memset:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::MEMSET);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(1)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(2)));
            }
            return;
        case llvm::Intrinsic::stacksave:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::STACKSAVE);
                push_back_operand(sala_instruction, memory_object(llvm_instruction));
            }
            return;
        case llvm::Intrinsic::stackrestore:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::STACKRESTORE);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
            }
            return;
        case llvm::Intrinsic::vastart:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::VA_START);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
            }
            return;
        case llvm::Intrinsic::vaend:
            {
                sala_instruction.set_opcode(sala::Instruction::Opcode::VA_END);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
            }
            return;
        case llvm::Intrinsic::fmuladd:
            {
                auto const result_varibale_mo{ memory_object(llvm_instruction) };

                sala_instruction.set_opcode(sala::Instruction::Opcode::MUL);
                sala_instruction.set_modifier(sala::Instruction::Modifier::FLOATING);
                push_back_operand(sala_instruction, result_varibale_mo);
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(0)));
                push_back_operand(sala_instruction, memory_object(llvm_intrinsic->getOperand(1)));

                compiled_basic_block()->push_back_instruction();
                compiled_basic_block()->last_instruction_ref().source_back_mapping() = sala_instruction_back_mapping;
                compiled_basic_block()->last_instruction_ref().set_opcode(sala::Instruction::Opcode::ADD);
                compiled_basic_block()->last_instruction_ref().set_modifier(sala::Instruction::Modifier::FLOATING);
                push_back_operand(compiled_basic_block()->last_instruction_ref(), result_varibale_mo);
                push_back_operand(compiled_basic_block()->last_instruction_ref(), result_varibale_mo);
                push_back_operand(compiled_basic_block()->last_instruction_ref(), memory_object(llvm_intrinsic->getOperand(2)));
            }
            return;

        // In this section we list llvm intrinsics which we translate as external functions.
        case llvm::Intrinsic::fabs:
        case llvm::Intrinsic::bswap:
        case llvm::Intrinsic::ctlz:
        case llvm::Intrinsic::trunc:
        case llvm::Intrinsic::rint:
            break;

        default:
            NOT_IMPLEMENTED_YET();
            return;
        }
    }
    else
        opcode = get_interpreted_function_opcode(llvm_instruction.getCalledOperand(), sala::Instruction::Opcode::CALL);

    std::vector<MemoryObject> arguments;
    {
        auto append_call_value = [this, &arguments, &sala_instruction_back_mapping](llvm::Value* const llvm_arg)
        {
            compiled_function()->push_back_local_variable();
            compiled_function()->last_local_variable_ref().set_num_bytes(module().getDataLayout().getPointerSize());
            compiled_function()->last_local_variable_ref().source_back_mapping() = sala_instruction_back_mapping;

            arguments.push_back({ compiled_function()->last_local_variable_ref().index(), sala::Instruction::Descriptor::LOCAL });

            compiled_basic_block()->last_instruction_ref().set_opcode(sala::Instruction::Opcode::ADDRESS);
            compiled_basic_block()->last_instruction_ref().source_back_mapping() = sala_instruction_back_mapping;
            push_back_operand(compiled_basic_block()->last_instruction_ref(), arguments.back());
            push_back_operand(compiled_basic_block()->last_instruction_ref(), memory_object(llvm_arg));

            compiled_basic_block()->push_back_instruction();
        };

        if (opcode == sala::Instruction::Opcode::CALL)
        {
            arguments.push_back(memory_object(llvm_instruction.getCalledOperand()));

            if (!llvm_instruction.getType()->isVoidTy())
                append_call_value(&llvm_instruction);
        }
        else if (!llvm_instruction.getType()->isVoidTy())
            arguments.push_back(memory_object(llvm_instruction));

        for (auto it = llvm_instruction.arg_begin(); it != llvm_instruction.arg_end(); ++it)
            if (llvm::isa<llvm::Function const>(*it))
                append_call_value(*it);
            else
                arguments.push_back(memory_object(it->get()));
    }

    compiled_basic_block()->last_instruction_ref().set_opcode(opcode);
    compiled_basic_block()->last_instruction_ref().source_back_mapping() = sala_instruction_back_mapping;
    for (MemoryObject argument : arguments)
        push_back_operand(compiled_basic_block()->last_instruction_ref(), argument);
}


void Compiler::compile_instruction_ret(llvm::ReturnInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    if (llvm_instruction.getReturnValue() != nullptr)
    {
        auto const sala_instruction_back_mapping{ sala_instruction.source_back_mapping() };
        auto const sala_store_instruction_index = sala_instruction.index();
        auto const sala_ret_instruction_index = compiled_basic_block()->push_back_instruction().index();

        auto& sala_store_instruction = compiled_basic_block()->instruction_ref(sala_store_instruction_index);
        sala_store_instruction.set_opcode(sala::Instruction::Opcode::STORE);
        sala_store_instruction.push_back_operand(0U, sala::Instruction::Descriptor::PARAMETER);
        push_back_operand(sala_store_instruction, memory_object(llvm_instruction.getReturnValue()));

        auto& sala_ret_instruction = compiled_basic_block()->instruction_ref(sala_ret_instruction_index);
        sala_ret_instruction.set_opcode(sala::Instruction::Opcode::RET);
        sala_ret_instruction.source_back_mapping() = sala_instruction_back_mapping;
    }
    else
        sala_instruction.set_opcode(sala::Instruction::Opcode::RET);
}


void Compiler::compile_instruction_vaarg(llvm::VAArgInst& llvm_instruction, sala::Instruction& sala_instruction)
{
    sala_instruction.set_opcode(sala::Instruction::Opcode::VA_ARG);
    push_back_operand(sala_instruction, memory_object(llvm_instruction));
    push_back_operand(sala_instruction, memory_object(llvm_instruction.getOperand(0)));
}
