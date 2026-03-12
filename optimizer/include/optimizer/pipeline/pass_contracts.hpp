#ifndef OPTIMIZER_PIPELINE_PASS_CONTRACTS_HPP_INCLUDED
#define OPTIMIZER_PIPELINE_PASS_CONTRACTS_HPP_INCLUDED

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/metadata/points_to.hpp>
#include <optimizer/metadata/translation.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/constant_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/program_repr.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/type_list.hpp>

#include <cstdint>
#include <type_traits>

namespace optimizer::pipeline
{

enum class Repr : uint8_t
{
    Sala,
    IR
};

struct AnalysisPass
{
};

struct TransformPass
{
};

struct TranslationPass
{
};

struct DebugPass
{
};

template <typename Owner, typename Entry>
struct MetaProduct
{
    using owner = Owner;
    using entry = Entry;
};

template <typename... Ts>
using ProductList = utils::TypeList<Ts...>;

template <Repr RequiredRepr, typename Requires = ProductList<>, typename Produces = ProductList<>,
          typename Preserves = ProductList<>, const char* Name = nullptr>
struct AnalysisSpec
{
    using kind         = AnalysisPass;
    using dependencies = Requires;
    using generates    = Produces;
    using preserves    = Preserves;

    static constexpr Repr        required_repr = RequiredRepr;
    static constexpr const char* name          = Name;
};

template <Repr RequiredRepr, typename Requires = ProductList<>, typename Produces = ProductList<>,
          typename Preserves = ProductList<>, const char* Name = nullptr>
struct TransformSpec
{
    using kind         = TransformPass;
    using dependencies = Requires;
    using generates    = Produces;
    using preserves    = Preserves;

    static constexpr Repr        required_repr = RequiredRepr;
    static constexpr const char* name          = Name;
};

template <Repr RequiredRepr, typename Requires = ProductList<>, typename Produces = ProductList<>,
          typename Preserves = ProductList<>, const char* Name = nullptr>
struct DebugSpec
{
    using kind         = DebugPass;
    using dependencies = Requires;
    using generates    = Produces;
    using preserves    = Preserves;

    static constexpr Repr        required_repr = RequiredRepr;
    static constexpr const char* name          = Name;
};

template <Repr From, Repr To, typename Requires = ProductList<>, typename Produces = ProductList<>,
          typename Preserves = ProductList<>, const char* Name = nullptr>
struct TranslationSpec
{
    using kind         = TranslationPass;
    using dependencies = Requires;
    using generates    = Produces;
    using preserves    = Preserves;

    static constexpr Repr        from_repr = From;
    static constexpr Repr        to_repr   = To;
    static constexpr const char* name      = Name;
};

template <typename Impl, typename Spec>
struct PassDef
{
    using impl = Impl;
    using spec = Spec;
};

template <typename T>
concept PipelinePassDef = requires {
    typename T::impl;
    typename T::spec;
    typename T::spec::kind;
};

namespace products
{
using TranslationProgram  = MetaProduct<program::ProgramIR, metadata::translation::ProgramMeta>;
using TranslationFunction = MetaProduct<program::FunctionIR, metadata::translation::FunctionMeta>;
using TranslationVariable = MetaProduct<program::VariableIR, metadata::translation::VariableMeta>;
using TranslationInstruction =
        MetaProduct<program::InstructionIR, metadata::translation::InstructionMeta>;

using TranslationAll = ProductList<TranslationProgram, TranslationFunction, TranslationVariable,
                                   TranslationInstruction>;

using GlobalPointsToProgram = MetaProduct<program::ProgramIR, metadata::points_to::ProgramMeta>;
using LocalPointsToFunction = MetaProduct<program::FunctionIR, metadata::points_to::FunctionMeta>;
using PointsToBasicBlock = MetaProduct<program::BasicBlockIR, metadata::points_to::BasicBlockMeta>;
using PointsToVariable   = MetaProduct<program::VariableIR, metadata::points_to::VariableMeta>;
using PointsToConstant   = MetaProduct<program::ConstantIR, metadata::points_to::ConstantMeta>;

using GlobalPointsToSet = ProductList<GlobalPointsToProgram, PointsToVariable, PointsToConstant>;
using LocalPointsToSet  = ProductList<LocalPointsToFunction, PointsToBasicBlock>;
using FullPointsToSet   = utils::set_union_unique_t<GlobalPointsToSet, LocalPointsToSet>;

using AvailableCopyFunction =
        MetaProduct<program::FunctionIR, metadata::available_copy::FunctionMeta>;
using AvailableCopyBasicBlock =
        MetaProduct<program::BasicBlockIR, metadata::available_copy::BasicBlockMeta>;
using AvailableCopyInstruction =
        MetaProduct<program::InstructionIR, metadata::available_copy::InstructionMeta>;
using AvailableCopyVariable =
        MetaProduct<program::VariableIR, metadata::available_copy::VariableMeta>;

using AvailableCopySet = ProductList<AvailableCopyFunction, AvailableCopyBasicBlock,
                                     AvailableCopyInstruction, AvailableCopyVariable>;
} // namespace products
} // namespace optimizer::pipeline

#endif
