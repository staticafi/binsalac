#ifndef OPTIMIZER_METADATA_ENTRY_HPP_INCLUDED
#define OPTIMIZER_METADATA_ENTRY_HPP_INCLUDED

#include <optimizer/metadata/meta_keys.hpp>

namespace optimizer::metadata
{

struct MetaEntryI
{
    virtual ~MetaEntryI() = default;

    MetaEntryI(const MetaEntryI&)            = delete;
    MetaEntryI(MetaEntryI&&)                 = delete;
    MetaEntryI& operator=(const MetaEntryI&) = delete;
    MetaEntryI& operator=(MetaEntryI&&)      = delete;

  protected:
    MetaEntryI() = default;
};

struct ProgramMetaEntryI : MetaEntryI
{
  protected:
    ProgramMetaEntryI() = default;
};
struct FunctionMetaEntryI : MetaEntryI
{
  protected:
    FunctionMetaEntryI() = default;
};
struct BasicBlockMetaEntryI : MetaEntryI
{
  protected:
    BasicBlockMetaEntryI() = default;
};
struct InstructionMetaEntryI : MetaEntryI
{
  protected:
    InstructionMetaEntryI() = default;
};
struct VariableMetaEntryI : MetaEntryI
{
  protected:
    VariableMetaEntryI() = default;
};
struct ConstantMetaEntryI : MetaEntryI
{
  protected:
    ConstantMetaEntryI() = default;
};

template <class IRLevelBase, MetaKey K>
struct ConcreteMetaEntry : IRLevelBase
{
    static constexpr MetaKey key = K;
};
} // namespace optimizer::metadata
#endif
