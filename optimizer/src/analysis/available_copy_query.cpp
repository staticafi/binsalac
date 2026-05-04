#include <iostream>
#include <optimizer/analysis/available_copy_query.hpp>

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/available_copy/import.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::analysis
{
namespace
{
using State = utils::DynamicBitset;

std::size_t compute_variable_count(const program::FunctionIR_csptr& function,
                                   const program::ProgramIR_csptr&  program_keepalive)
{
    std::size_t max_id = 0;
    bool        found  = false;

    const auto consider = [&](const auto& variable)
    {
        const auto* meta =
                variable->get_metadata().template get_raw<metadata::available_copy::VariableMeta>();
        if (meta == nullptr)
        {
            return;
        }

        if (!found || meta->id > max_id)
        {
            max_id = meta->id;
            found  = true;
        }
    };

    for (const auto& parameter : function->get_parameters())
    {
        consider(parameter);
    }

    for (const auto& local : function->get_local_variables())
    {
        consider(local);
    }

    for (const auto& variable : program_keepalive->get_static_vars())
    {
        consider(variable);
    }

    return found ? (max_id + 1U) : 0U;
}

} // namespace

AvailableCopyQueryFunction::AvailableCopyQueryFunction(program::FunctionIR_csptr function)
    : program_keepalive_{}, function_{std::move(function)}
{
    ASSUMPTION(function_ != nullptr);

    program_keepalive_ = function_->get_program();
    ASSUMPTION(program_keepalive_ != nullptr);

    function_meta_ = function_->get_metadata().get_raw<metadata::available_copy::FunctionMeta>();
    ASSUMPTION(function_meta_ != nullptr);

    cache_.bb    = nullptr;
    cache_.instr = nullptr;
    cache_.in_state.resize(function_meta_->facts.size());

    variable_ids_.clear();

    for (const auto& parameter : function_->get_parameters())
    {
        const auto* meta =
                parameter->get_metadata().get_raw<metadata::available_copy::VariableMeta>();
        if (meta != nullptr)
        {
            variable_ids_.emplace(parameter.get(), meta->id);
        }
    }

    for (const auto& local : function_->get_local_variables())
    {
        const auto* meta = local->get_metadata().get_raw<metadata::available_copy::VariableMeta>();
        if (meta != nullptr)
        {
            variable_ids_.emplace(local.get(), meta->id);
        }
    }

    for (const auto& variable : program_keepalive_->get_static_vars())
    {
        const auto* meta =
                variable->get_metadata().get_raw<metadata::available_copy::VariableMeta>();
        if (meta != nullptr)
        {
            variable_ids_.emplace(variable.get(), meta->id);
        }
    }

    fact_id_of_.clear();
    fact_id_of_.reserve(function_meta_->facts.size());

    for (const auto& fact : function_meta_->facts)
    {
        fact_id_of_.emplace(utils::CopyFactKey{fact.dest_id, fact.source_id}, fact.id);
    }

    const auto variable_count = compute_variable_count(function_, program_keepalive_);

    kill_masks_.assign(variable_count, utils::DynamicBitset{});
    for (auto& kill_mask : kill_masks_)
    {
        kill_mask.resize(function_meta_->facts.size());
    }

    for (const auto& fact : function_meta_->facts)
    {
        ASSUMPTION(fact.dest_id < kill_masks_.size());
        ASSUMPTION(fact.source_id < kill_masks_.size());

        kill_masks_[fact.dest_id].set(fact.id);
        kill_masks_[fact.source_id].set(fact.id);
    }
}

utils::TransferContext AvailableCopyQueryFunction::make_transfer_context() const
{
    return utils::TransferContext{
            .variable_ids    = variable_ids_,
            .facts           = function_meta_->facts,
            .facts_by_dest   = function_meta_->facts_by_dest,
            .facts_by_source = function_meta_->facts_by_source,
            .fact_id_of      = fact_id_of_,
            .kill_masks      = kill_masks_,
    };
}

std::size_t
AvailableCopyQueryFunction::get_instruction_index(const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);
    const auto bb = instruction->get_basic_block();
    ASSUMPTION(bb != nullptr);

    std::size_t idx = 0;
    for (const auto& current : bb->get_instructions())
    {
        if (current.get() == instruction.get())
        {
            return idx;
        }
        ++idx;
    }

    ASSUMPTION(false);
}

void AvailableCopyQueryFunction::reconstruct_before_state(
        const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    if (cache_.bb == bb_raw && cache_.instr == instruction.get())
    {
        return;
    }

    const auto* bb_meta =
            bb_raw->get_metadata().get_raw<metadata::available_copy::BasicBlockMeta>();
    ASSUMPTION(bb_meta != nullptr);

    cache_.bb    = bb_raw;
    cache_.instr = instruction.get();
    cache_.in_state.resize(function_meta_->facts.size());
    cache_.in_state.reset();

    const auto& words = bb_meta->in_bits;
    for (std::size_t wi = 0; wi < words.size(); ++wi)
    {
        const auto word = words[wi];
        for (std::size_t bi = 0; bi < utils::DynamicBitset::BITS_PER_WORD; ++bi)
        {
            if ((word & (1ULL << bi)) != 0ULL)
            {
                const auto bit = wi * utils::DynamicBitset::BITS_PER_WORD + bi;
                if (bit < function_meta_->facts.size())
                {
                    cache_.in_state.set(bit);
                }
            }
        }
    }

    std::size_t current_instr_index = 0;
    for (const auto& current_instruction : bb_raw->get_instructions())
    {
        if (current_instruction.get() == instruction.get())
        {
            break;
        }

        utils::apply_transfer(current_instruction, make_transfer_context(), cache_.in_state);
        ++current_instr_index;
    }
}

std::optional<std::size_t>
AvailableCopyQueryFunction::resolve_direct_source_id(const std::size_t variable_id,
                                                     const State&      state) const
{
    if (variable_id >= function_meta_->facts_by_dest.size())
    {
        return std::nullopt;
    }

    std::optional<std::size_t> result;
    for (const auto fact_id : function_meta_->facts_by_dest[variable_id])
    {
        if (!state.test(fact_id))
        {
            continue;
        }

        const auto& fact = function_meta_->facts[fact_id];
        if (result.has_value() && result.value() != fact.source_id)
        {
            return std::nullopt;
        }

        result = fact.source_id;
    }

    return result;
}

std::optional<std::size_t>
AvailableCopyQueryFunction::resolve_transitive_source_id(const std::size_t variable_id,
                                                         const State&      state) const
{
    std::vector<bool> visited(kill_masks_.size(), false);

    std::size_t current = variable_id;
    bool        moved   = false;

    while (current < visited.size() && !visited[current])
    {
        visited[current] = true;

        const auto next = resolve_direct_source_id(current, state);
        if (!next.has_value())
        {
            break;
        }

        current = next.value();
        moved   = true;
    }

    if (!moved || current == variable_id)
    {
        return std::nullopt;
    }

    return current;
}

std::vector<std::size_t> AvailableCopyQueryFunction::build_chain_ids(const std::size_t variable_id,
                                                                     const State&      state) const
{
    std::vector<std::size_t> chain;
    std::vector<bool>        visited(kill_masks_.size(), false);

    std::size_t current = variable_id;
    chain.push_back(current);

    while (current < visited.size() && !visited[current])
    {
        visited[current] = true;

        const auto next = resolve_direct_source_id(current, state);
        if (!next.has_value())
        {
            break;
        }

        current = next.value();
        chain.push_back(current);
    }

    return chain;
}

AvailableCopyResult
AvailableCopyQueryFunction::handle_cache_hit(const program::InstructionIR_sptr& instruction,
                                             const program::VariableIR& x, const bool before)
{
    ASSUMPTION(instruction != nullptr);
    ASSUMPTION(cache_.bb == instruction->get_basic_block_raw());
    ASSUMPTION(cache_.instr == instruction.get());

    State state = cache_.in_state;
    if (!before)
    {
        utils::apply_transfer(instruction, make_transfer_context(), state);
    }

    const auto variable_id = x.get_metadata().get<metadata::available_copy::VariableMeta>().id;

    AvailableCopyResult result{};

    if (const auto direct_id = resolve_direct_source_id(variable_id, state); direct_id.has_value())
    {
        result.direct_source = get_variable(direct_id.value());
    }

    if (const auto transitive_id = resolve_transitive_source_id(variable_id, state);
        transitive_id.has_value())
    {
        result.transitive_source = get_variable(transitive_id.value());
    }

    for (const auto id : build_chain_ids(variable_id, state))
    {
        if (const auto var = get_variable(id); var.has_value())
        {
            result.chain.push_back(var.value());
        }
    }

    return result;
}

AvailableCopyResult
AvailableCopyQueryFunction::handle_request(const program::InstructionIR_sptr& instruction,
                                           const program::VariableIR& x, const bool before)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    reconstruct_before_state(instruction);
    return handle_cache_hit(instruction, x, before);
}

AvailableCopyResult
AvailableCopyQueryFunction::before(const program::InstructionIR_sptr& instruction,
                                   const program::VariableIR&         x)
{
    return handle_request(instruction, x, true);
}

AvailableCopyResult
AvailableCopyQueryFunction::after(const program::InstructionIR_sptr& instruction,
                                  const program::VariableIR&         x)
{
    return handle_request(instruction, x, false);
}

std::optional<program::VariableIR_sptr>
AvailableCopyQueryFunction::get_variable(const std::size_t id) const
{
    const auto variable = function_meta_->variables_by_id.find(id);
    if (variable == function_meta_->variables_by_id.end())
    {
        return std::nullopt;
    }

    return variable->second;
}

} // namespace optimizer::analysis
