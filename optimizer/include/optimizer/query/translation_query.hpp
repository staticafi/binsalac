#ifndef OPTIMIZER_QUERY_TRANSLATION_QUERY_HPP_INCLUDED
#define OPTIMIZER_QUERY_TRANSLATION_QUERY_HPP_INCLUDED

#include <optimizer/metadata/translation.hpp>

#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>

#include <sala/program.hpp>

#include <string>

namespace optimizer::query
{

class TranslationQuery
{
  public:
    explicit TranslationQuery(program::ProgramIR_csptr program);

    [[nodiscard]] std::string program_name() const;
    [[nodiscard]] std::string program_system() const;
    [[nodiscard]] std::string program_version() const;

    [[nodiscard]] std::string function_name(const program::FunctionIR_sptr& function) const;

    [[nodiscard]] sala::SourceBackMapping
    source_back_mapping(const program::FunctionIR_sptr& function) const;

    [[nodiscard]] sala::SourceBackMapping
    source_back_mapping(const program::InstructionIR_sptr& instruction) const;

    [[nodiscard]] sala::SourceBackMapping
    source_back_mapping(const program::VariableIR_sptr& variable) const;

    [[nodiscard]] std::string external_name(const program::VariableIR_sptr& variable) const;

  private:
    template <typename MetaT, typename EntityT>
    [[nodiscard]] static const MetaT* get_translation_meta(const EntityT& entity)
    {
        return entity.get_metadata().template get_raw<MetaT>();
    }

  private:
    program::ProgramIR_csptr program_;
};

} // namespace optimizer::query

#endif
