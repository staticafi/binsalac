#ifndef OPTIMIZER_METADATA_TRANSLATION_HPP_INCLUDED
#define OPTIMIZER_METADATA_TRANSLATION_HPP_INCLUDED

#include <optimizer/metadata/meta_entry.hpp>
#include <optimizer/metadata/meta_keys.hpp>

#include <sala/program.hpp>

#include <optional>

namespace optimizer::metadata::translation
{
template <typename BaseEntryI>
    requires std::derived_from<BaseEntryI, MetaEntryI>
using TranslationMetaEntry = ConcreteMetaEntry<BaseEntryI, MetaKey::TRANSLATION_SALA_TO_IR>;

struct InstructionMeta final : TranslationMetaEntry<InstructionMetaEntryI>
{
    sala::SourceBackMapping source_back_mapping;
};

struct VariableMeta final : TranslationMetaEntry<VariableMetaEntryI>
{
    sala::SourceBackMapping    source_back_mapping;
    std::optional<std::string> external_name;
};

struct FunctionMeta final : TranslationMetaEntry<FunctionMetaEntryI>
{
    sala::SourceBackMapping source_back_mapping;
    std::string             name;
};

struct ProgramMeta final : TranslationMetaEntry<ProgramMetaEntryI>
{
    std::string version;
    std::string system;
    std::string name;
};
} // namespace optimizer::metadata::translation

#endif
