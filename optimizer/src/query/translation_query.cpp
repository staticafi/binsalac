#include <optimizer/query/translation_query.hpp>

namespace optimizer::query
{

TranslationQuery::TranslationQuery(program::ProgramIR_csptr program) : program_(std::move(program))
{
}

std::string TranslationQuery::program_name() const
{
    if (program_ == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::ProgramMeta>(*program_);

    if (meta == nullptr)
        return {};

    return meta->name;
}

std::string TranslationQuery::program_system() const
{
    if (program_ == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::ProgramMeta>(*program_);

    if (meta == nullptr)
        return {};

    return meta->system;
}

std::string TranslationQuery::program_version() const
{
    if (program_ == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::ProgramMeta>(*program_);

    if (meta == nullptr)
        return {};

    return meta->version;
}

std::string TranslationQuery::function_name(const program::FunctionIR_sptr& function) const
{
    if (function == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::FunctionMeta>(*function);

    if (meta == nullptr)
        return {};

    return meta->name;
}

sala::SourceBackMapping
TranslationQuery::source_back_mapping(const program::FunctionIR_sptr& function) const
{
    if (function == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::FunctionMeta>(*function);

    if (meta == nullptr)
        return {};

    return meta->source_back_mapping;
}

sala::SourceBackMapping
TranslationQuery::source_back_mapping(const program::InstructionIR_sptr& instruction) const
{
    if (instruction == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::InstructionMeta>(*instruction);

    if (meta == nullptr)
        return {};

    return meta->source_back_mapping;
}

sala::SourceBackMapping
TranslationQuery::source_back_mapping(const program::VariableIR_sptr& variable) const
{
    if (variable == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::VariableMeta>(*variable);

    if (meta == nullptr)
        return {};

    return meta->source_back_mapping;
}

std::string TranslationQuery::external_name(const program::VariableIR_sptr& variable) const
{
    if (variable == nullptr)
        return {};

    const auto* meta = get_translation_meta<metadata::translation::VariableMeta>(*variable);

    if (meta == nullptr || !meta->external_name.has_value())
        return {};

    return *meta->external_name;
}

} // namespace optimizer::query
