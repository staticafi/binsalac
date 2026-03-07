#ifndef IR_TYPES_HPP_INCLUDED
#define IR_TYPES_HPP_INCLUDED
#include <optimizer/metadata/metadata.hpp>

#include <list>
#include <memory>
#include <variant>
#include <vector>

namespace optimizer::program
{
// Forward declarations.
class ConstantIR;
class VariableIR;
class InstructionIR;
class BasicBlockIR;
class FunctionIR;
class ProgramIR;

// Type aliases for pointers
using ConstantIR_sptr    = std::shared_ptr<ConstantIR>;
using VariableIR_sptr    = std::shared_ptr<VariableIR>;
using InstructionIR_sptr = std::shared_ptr<InstructionIR>;
using BasicBlockIR_sptr  = std::shared_ptr<BasicBlockIR>;
using FunctionIR_sptr    = std::shared_ptr<FunctionIR>;
using ProgramIR_sptr     = std::shared_ptr<ProgramIR>;
using OperandIR_sptr     = std::variant<VariableIR_sptr, ConstantIR_sptr, FunctionIR_sptr>;

using ConstantIR_raw    = ConstantIR*;
using VariableIR_raw    = VariableIR*;
using InstructionIR_raw = InstructionIR*;
using BasicBlockIR_raw  = BasicBlockIR*;
using FunctionIR_raw    = FunctionIR*;
using ProgramIR_raw     = ProgramIR*;
using OperandIR_raw     = std::variant<VariableIR*, ConstantIR*, FunctionIR*>;

using ConstantIR_csptr    = std::shared_ptr<const ConstantIR>;
using VariableIR_csptr    = std::shared_ptr<const VariableIR>;
using InstructionIR_csptr = std::shared_ptr<const InstructionIR>;
using BasicBlockIR_csptr  = std::shared_ptr<const BasicBlockIR>;
using FunctionIR_csptr    = std::shared_ptr<const FunctionIR>;
using ProgramIR_csptr     = std::shared_ptr<const ProgramIR>;
using OperandIR_csptr     = std::variant<VariableIR_csptr, ConstantIR_csptr, FunctionIR_csptr>;
using OperandIR_craw      = std::variant<const VariableIR*, const ConstantIR*, const FunctionIR*>;

using ConstantIR_wptr    = std::weak_ptr<ConstantIR>;
using VariableIR_wptr    = std::weak_ptr<VariableIR>;
using InstructionIR_wptr = std::weak_ptr<InstructionIR>;
using BasicBlockIR_wptr  = std::weak_ptr<BasicBlockIR>;
using FunctionIR_wptr    = std::weak_ptr<FunctionIR>;
using ProgramIR_wptr     = std::weak_ptr<ProgramIR>;
using OperandIR_wptr     = std::variant<VariableIR_wptr, ConstantIR_wptr, FunctionIR_wptr>;

using ConstantIR_cwptr    = std::weak_ptr<const ConstantIR>;
using VariableIR_cwptr    = std::weak_ptr<const VariableIR>;
using InstructionIR_cwptr = std::weak_ptr<const InstructionIR>;
using BasicBlockIR_cwptr  = std::weak_ptr<const BasicBlockIR>;
using FunctionIR_cwptr    = std::weak_ptr<const FunctionIR>;
using ProgramIR_cwptr     = std::weak_ptr<const ProgramIR>;

// Type aliases for containers.
using ConstantIRListS    = std::list<ConstantIR_sptr>;
using VariableIRListS    = std::list<VariableIR_sptr>;
using InstructionIRListS = std::list<InstructionIR_sptr>;
using BasicBlockIRListS  = std::list<BasicBlockIR_sptr>;
using FunctionIRListS    = std::list<FunctionIR_sptr>;

using ConstantIRListW    = std::list<ConstantIR_wptr>;
using VariableIRListW    = std::list<VariableIR_wptr>;
using InstructionIRListW = std::list<InstructionIR_wptr>;
using BasicBlockIRListW  = std::list<BasicBlockIR_wptr>;
using FunctionIRListW    = std::list<FunctionIR_wptr>;
using OperandIRVecR      = std::vector<OperandIR_raw>;

// Type aliases for iterators
using ConstantIRListS_iter    = ConstantIRListS::iterator;
using VariableIRListS_iter    = VariableIRListS::iterator;
using InstructionIRListS_iter = InstructionIRListS::iterator;
using BasicBlockIRListS_iter  = BasicBlockIRListS::iterator;
using FunctionIRListS_iter    = FunctionIRListS::iterator;

using ConstantIRListW_iter    = ConstantIRListW::iterator;
using VariableIRListW_iter    = VariableIRListW::iterator;
using InstructionIRListW_iter = InstructionIRListW::iterator;
using BasicBlockIRListW_iter  = BasicBlockIRListW::iterator;
using FunctionIRListW_iter    = FunctionIRListW::iterator;
using OperandIRVecR_iter      = OperandIRVecR::iterator;

// Metadata aliases
template <typename T>
using Metadata = metadata::Metadata<T>;

using MetaEntryI            = metadata::MetaEntryI;
using ProgramMetaEntryI     = metadata::ProgramMetaEntryI;
using FunctionMetaEntryI    = metadata::FunctionMetaEntryI;
using InstructionMetaEntryI = metadata::InstructionMetaEntryI;
using BasicBlockMetaEntryI  = metadata::BasicBlockMetaEntryI;
using VariableMetaEntryI    = metadata::VariableMetaEntryI;
using ConstantMetaEntryI    = metadata::ConstantMetaEntryI;

} // namespace optimizer::program
#endif
