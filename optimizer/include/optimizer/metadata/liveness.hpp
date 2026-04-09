#ifndef OPTIMIZER_METADATA_LIVENESS_HPP_INCLUDED
#define OPTIMIZER_METADATA_LIVENESS_HPP_INCLUDED

#include <optimizer/metadata/metadata.hpp>
#include <optimizer/programIR/ir_types.hpp>
#include <optimizer/utils/liveness/import.hpp>

#include <concepts>

namespace optimizer::metadata::liveness
{
template <typename BaseEntryI>
    requires std::derived_from<BaseEntryI, MetaEntryI>
using LivenessMetaEntry = ConcreteMetaEntry<BaseEntryI, MetaKey::LIVENESS>;

struct BasicBlockMeta final : LivenessMetaEntry<BasicBlockMetaEntryI>
{
    utils::LiveSet                                          live_out{};
    optimizer::utils::SparseSet<program::InstructionIR_raw> removable_instructions{};
};

} // namespace optimizer::metadata::liveness

#endif
