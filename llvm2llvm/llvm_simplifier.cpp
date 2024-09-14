#include <llvm2llvm/llvm_simplifier.hpp>
#include <llvmutl/llvm_utils.hpp>
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
#include <llvm/IR/ReplaceConstant.h>
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
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>


struct InstructionAndConstantExpr
{
    llvm::Instruction* instruction;
    llvm::ConstantExpr* constant_expr;
};


static InstructionAndConstantExpr get_first_constant_expr(llvm::Instruction* const llvm_instruction)
{
    for (unsigned int i = 0; i < llvm_instruction->getNumOperands(); ++i)
        if (auto const constant_expr = llvm::dyn_cast<llvm::ConstantExpr>(llvm_instruction->getOperand(i)))
            return { llvm_instruction, constant_expr };
    return { llvm_instruction, nullptr };
}


static llvm::Instruction* eliminate_phi_node_constant_expr(llvm::PHINode* const phi_node, llvm::ConstantExpr* const expression)
{
    unsigned int i;
    for (i = 0; phi_node->getIncomingValue(i) != expression; )
    {
        ++i;
        ASSUMPTION(i < phi_node->getNumIncomingValues());
    }
    auto const basic_block = phi_node->getIncomingBlock(i);

    auto const new_instruction = llvm_constant_expr_to_instruction(expression, basic_block->getTerminator());
    for ( ; i < phi_node->getNumIncomingValues(); ++i)
        if (phi_node->getIncomingBlock(i) == basic_block)
            phi_node->setIncomingValue(i, new_instruction);

    return new_instruction;
}


static void eliminate_const_expressions(llvm::Module* const llvm_module)
{
    for (auto it_function = llvm_module->begin(); it_function != llvm_module->end(); ++it_function)
    {
        std::vector<InstructionAndConstantExpr> constant_expressions;
        for (auto it_block = it_function->begin(); it_block != it_function->end(); ++it_block)
            for (auto it_instruction = it_block->begin(); it_instruction != it_block->end(); ++it_instruction)
            {
                auto const constant_expr = get_first_constant_expr(&*it_instruction);
                if (constant_expr.constant_expr != nullptr)
                    constant_expressions.push_back(constant_expr);
            }
        while (!constant_expressions.empty())
        {
            auto expression = constant_expressions.back();
            constant_expressions.pop_back();

            llvm::Instruction* new_llvm_instruction;
            if (auto const phi_node = llvm::dyn_cast<llvm::PHINode>(expression.instruction))
                new_llvm_instruction = eliminate_phi_node_constant_expr(phi_node, expression.constant_expr);
            else
            {
                new_llvm_instruction = llvm_constant_expr_to_instruction(expression.constant_expr, expression.instruction);
                expression.instruction->replaceUsesOfWith(expression.constant_expr, new_llvm_instruction);
            }

            expression = get_first_constant_expr(expression.instruction);
            if (expression.constant_expr != nullptr)
                constant_expressions.push_back(expression);

            expression = get_first_constant_expr(new_llvm_instruction);
            if (expression.constant_expr != nullptr)
                constant_expressions.push_back(expression);
        }
    }
}


static void eliminate_switch_instructions(llvm::Module* const llvm_module)
{
    auto manager = std::make_unique<llvm::legacy::FunctionPassManager>(llvm_module);
    manager->add(llvm::createLowerSwitchPass());
    for (auto it_function = llvm_module->begin(); it_function != llvm_module->end(); ++it_function)
        if (!it_function->isDeclaration())
            manager->run(*it_function);
}


static void eliminate_select_instruction(llvm::SelectInst* const llvm_select_instr)
{
    llvm::BasicBlock* const original_block = llvm_select_instr->getParent();
    llvm::BasicBlock* const end_block = original_block->splitBasicBlock(llvm_select_instr);

    llvm::BasicBlock* const true_block = llvm::BasicBlock::Create(
        original_block->getParent()->getContext(),
        "",
        original_block->getParent(),
        end_block
        );
    llvm::BasicBlock* const false_block = llvm::BasicBlock::Create(
        original_block->getParent()->getContext(),
        "",
        original_block->getParent(),
        end_block
        );

    if (original_block->hasName())
    {
        end_block->setName(original_block->getName() + ".SelectEnd");
        true_block->setName(original_block->getName() + ".SelectTrue");
        false_block->setName(original_block->getName() + ".SelectEnd");
    }

    original_block->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(
        true_block,
        false_block,
        llvm_select_instr->getCondition(),
        original_block
        )->setDebugLoc(llvm_select_instr->getDebugLoc());

    llvm::BranchInst::Create(end_block, true_block)->setDebugLoc(llvm_select_instr->getDebugLoc());
    llvm::BranchInst::Create(end_block, false_block)->setDebugLoc(llvm_select_instr->getDebugLoc());

    llvm::PHINode* const phi = llvm::PHINode::Create(
        llvm_select_instr->getOperand(1)->getType(),
        llvm_select_instr->getNumOperands(),
        "",
        llvm_select_instr
        );
    phi->addIncoming(llvm_select_instr->getOperand(1), true_block);
    phi->addIncoming(llvm_select_instr->getOperand(2), false_block);
    if (llvm_select_instr->hasName())
        phi->setName(llvm_select_instr->getName() + ".PHINode");
    phi->setDebugLoc(llvm_select_instr->getDebugLoc());

    llvm_select_instr->replaceAllUsesWith(phi);
    llvm_select_instr->eraseFromParent();
}


static void eliminate_select_instructions(llvm::Module* const llvm_module)
{
    for (auto it_function = llvm_module->begin(); it_function != llvm_module->end(); ++it_function)
    {
        std::vector<llvm::SelectInst*> selects;
        for (auto it_block = it_function->begin(); it_block != it_function->end(); ++it_block)
            for (auto it_instruction = it_block->begin(); it_instruction != it_block->end(); ++it_instruction)
                if (auto const select_instr = llvm::dyn_cast<llvm::SelectInst>(&*it_instruction))
                    selects.push_back(select_instr);
        for (auto select_instr : selects)
            eliminate_select_instruction(select_instr);
    }
}


void simplify_llvm_file(std::filesystem::path const& src_llvm_file_pathname, std::filesystem::path const& dst_llvm_file_pathname)
{
    TMPROF_BLOCK();

    llvm::SMDiagnostic D;
    llvm::LLVMContext C;
    std::unique_ptr<llvm::Module> llvm_module;
    {
 
        llvm_module = llvm::parseIRFile(src_llvm_file_pathname.string(), D, C);
        if (llvm_module == nullptr)
        {
            llvm::raw_os_ostream ros(std::cout);
            D.print(src_llvm_file_pathname.filename().string().c_str(), ros, false);
            ros.flush();
            return;
        }
    }

    eliminate_switch_instructions(llvm_module.get());
    eliminate_const_expressions(llvm_module.get());
    eliminate_select_instructions(llvm_module.get());

    {
        std::ofstream ostr(dst_llvm_file_pathname.string().c_str(), std::ios::binary);
        llvm::raw_os_ostream ros(ostr);
        llvm_module->print(ros, 0);
        ros.flush();
    }
}
