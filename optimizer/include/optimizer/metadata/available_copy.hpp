#ifndef OPTIMIZER_METADATA_AVAILABLE_COPY_HPP_INCLUDED
#define OPTIMIZER_METADATA_AVAILABLE_COPY_HPP_INCLUDED

#include <optimizer/metadata/metadata.hpp>
#include <optimizer/utils/available_copy/import.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace optimizer::metadata::available_copy
{

template <typename BaseEntryI>
    requires std::derived_from<BaseEntryI, MetaEntryI>
using AvailCopyMetaEntry = ConcreteMetaEntry<BaseEntryI, MetaKey::AVAIL_COPY>;

struct VariableMeta final : AvailCopyMetaEntry<VariableMetaEntryI>
{
    std::size_t id{0};
};

struct FunctionMeta final : AvailCopyMetaEntry<FunctionMetaEntryI>
{
    std::vector<utils::CopyFact>                            facts{};
    std::vector<std::vector<std::size_t>>                   facts_by_dest{};
    std::vector<std::vector<std::size_t>>                   facts_by_source{};
    utils::SparseMap<std::size_t, program::VariableIR_sptr> variables_by_id{};
};

struct BasicBlockMeta final : AvailCopyMetaEntry<BasicBlockMetaEntryI>
{
    std::vector<std::uint64_t> in_bits{};
};

} // namespace optimizer::metadata::available_copy

#endif
