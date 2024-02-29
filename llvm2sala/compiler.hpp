#ifndef LLVM2SALA_COMPILER_HPP_INCLUDED
#   define LLVM2SALA_COMPILER_HPP_INCLUDED

#   include <sala/program.hpp>
#   include <utility/config.hpp>
#   if COMPILER() == COMPILER_VC()
#       pragma warning(push)
#       pragma warning(disable : 4624 4996 4146 4800 4996 4005 4355 4244 4267)
#   endif
#   include <llvm/IR/Instructions.h>
#   if COMPILER() == COMPILER_VC()
#       pragma warning(pop)
#   endif
#   include <filesystem>
#   include <unordered_map>
#   include <string>
#   include <cstdint>


struct Compiler
{
    Compiler(sala::Program& P, llvm::Module& M);

    void run();

    sala::Program& program() { return program_; }
    llvm::Module& module() { return module_; }

    struct MemoryObject
    {
        bool operator==(MemoryObject const& other) const { return index == other.index && descriptor == other.descriptor; }
        bool operator!=(MemoryObject const& other) const { return !operator==(other); }
        std::uint32_t index;
        sala::Instruction::Descriptor descriptor;
    };

    MemoryObject const& register_parameter(llvm::Value* llvm_value, sala::Variable const& sala_parameter);
    MemoryObject const& register_variable(llvm::Value* llvm_value, sala::Variable const& sala_variable);
    MemoryObject const& register_constant(llvm::Value* llvm_value, sala::Constant const& sala_constant);
    MemoryObject const& register_function(llvm::Value* llvm_value, sala::Function const& sala_function);
    MemoryObject const& update_variable(llvm::Value& llvm_value, llvm::Type* llvm_type, sala::SourceBackMapping const& back_mapping);
    MemoryObject const& memory_object(llvm::Value* llvm_value);
    MemoryObject const& memory_object(llvm::Value& llvm_value) { return memory_object(&llvm_value); }
    bool has_memory_object(llvm::Value* llvm_value) const { return created_memory_objects_.contains(llvm_value); }

    std::uint32_t moveptr_constant_index(std::int64_t ptr_move);

    template<typename T>
    std::uint32_t numeric_constant_index(T const value)
    { return numeric_constant_index_impl((std::uint8_t const*)&value, sizeof(value)); }

    sala::Function* compiled_function() const { return compiled_function_; }
    void set_compiled_function(sala::Function* sala_function = nullptr) { compiled_function_ = sala_function; }
    sala::BasicBlock* compiled_basic_block() const { return compiled_basic_block_; }
    void set_compiled_basic_block(sala::BasicBlock* sala_basic_block = nullptr) { compiled_basic_block_ = sala_basic_block; }
    bool uses_stacksave() const { return uses_stacksave_; };
    void set_uses_stacksave(bool state = false) { uses_stacksave_ = state; };

private:
    struct CompileConstantVariableIndices
    {
        std::uint32_t moveptr_variable_index{ std::numeric_limits<std::uint32_t>::max() };
        std::uint32_t address_variable_index{ std::numeric_limits<std::uint32_t>::max() };
    };

    std::uint32_t numeric_constant_index_impl(std::uint8_t const* const value_ptr, std::size_t const num_bytes);

    void compile_constant(
        llvm::Constant const* llvm_constant,
        sala::Constant& sala_constant,
        std::vector<std::pair<llvm::Value*, std::uint64_t> >& pointer_initializations
        );
    void compile_function(llvm::Function& llvm_function, sala::Function& sala_function);
    void compile_function_parameters(llvm::Function& llvm_function, sala::Function& sala_function);
    void compile_basic_block(llvm::BasicBlock& llvm_block, sala::BasicBlock& sala_block, std::vector<llvm::PHINode*>& phi_nodes);
    void compile_instruction(llvm::Instruction& llvm_instruction, sala::Instruction& sala_instruction, std::vector<llvm::PHINode*>& phi_nodes);
    void compile_instruction_unreachable(llvm::UnreachableInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_alloca(llvm::AllocaInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_bitcast(llvm::BitCastInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_store(llvm::StoreInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_load(llvm::LoadInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_add(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_sub(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_mul(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_div(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_rem(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_and(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_or(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_xor(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_shl(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_ashr(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_lshr(llvm::BinaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_fneg(llvm::UnaryOperator& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::CastInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::FPExtInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::TruncInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::FPTruncInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::SIToFPInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::UIToFPInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::FPToSIInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cast(llvm::FPToUIInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_ptrtoint(llvm::PtrToIntInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_inttoptr(llvm::IntToPtrInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_getelementptr(llvm::GetElementPtrInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_cmp(llvm::CmpInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_br(llvm::BranchInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_extractvalue(llvm::ExtractValueInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_insertvalue(llvm::InsertValueInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_call(llvm::CallInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_ret(llvm::ReturnInst& llvm_instruction, sala::Instruction& sala_instruction);
    void compile_instruction_vaarg(llvm::VAArgInst& llvm_instruction, sala::Instruction& sala_instruction);

    sala::Program& program_;
    llvm::Module& module_;
    std::unordered_map<llvm::Value*, MemoryObject> created_memory_objects_;
    std::unordered_map<std::int64_t, std::uint32_t> moveptr_constants_;
    std::unordered_map<std::string, std::uint32_t> numeric_constants_;
    CompileConstantVariableIndices compile_constant_variable_indices_;
    sala::Function* compiled_function_;
    sala::BasicBlock* compiled_basic_block_;
    bool uses_stacksave_;
};


void compile_llvm_file(std::filesystem::path const& llvm_file_pathname, sala::Program& P);


#endif
