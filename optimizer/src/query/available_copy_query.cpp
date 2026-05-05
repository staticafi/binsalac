#include <optimizer/query/available_copy_query.hpp>

#include <optimizer/metadata/available_copy.hpp>
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>
#include <optimizer/programIR/program_ir.hpp>
#include <optimizer/programIR/variable_ir.hpp>
#include <optimizer/utils/available_copy/import.hpp>

#include <utility/assumptions.hpp>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace optimizer::query
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
        ASSUMPTION(variable != nullptr);

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

    cache_.after_valid = false;
    cache_.after_state.resize(function_meta_->facts.size());

    variable_ids_.clear();

    for (const auto& parameter : function_->get_parameters())
    {
        ASSUMPTION(parameter != nullptr);

        const auto* meta =
                parameter->get_metadata().get_raw<metadata::available_copy::VariableMeta>();

        if (meta != nullptr)
        {
            variable_ids_.emplace(parameter.get(), meta->id);
        }
    }

    for (const auto& local : function_->get_local_variables())
    {
        ASSUMPTION(local != nullptr);

        const auto* meta = local->get_metadata().get_raw<metadata::available_copy::VariableMeta>();

        if (meta != nullptr)
        {
            variable_ids_.emplace(local.get(), meta->id);
        }
    }

    for (const auto& variable : program_keepalive_->get_static_vars())
    {
        ASSUMPTION(variable != nullptr);

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
        fact_id_of_.emplace(
                utils::CopyFactKey{
                        .dest_id   = fact.dest_id,
                        .source_id = fact.source_id,
                },
                fact.id);
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

void AvailableCopyQueryFunction::apply_transfer_to_state(
        const program::InstructionIR_sptr& instruction, State& state) const
{
    ASSUMPTION(instruction != nullptr);

    utils::apply_transfer(instruction, make_transfer_context(), state);
}

void AvailableCopyQueryFunction::reset_cached_after_state()
{
    cache_.after_valid = false;
    cache_.after_state.resize(function_meta_->facts.size());
    cache_.after_state.reset();
}

void AvailableCopyQueryFunction::load_basic_block_in_state(program::BasicBlockIR_raw basic_block,
                                                           State&                    state) const
{
    ASSUMPTION(basic_block != nullptr);

    const auto* bb_meta =
            basic_block->get_metadata().get_raw<metadata::available_copy::BasicBlockMeta>();
    ASSUMPTION(bb_meta != nullptr);

    state.resize(function_meta_->facts.size());
    state.reset();

    const auto& words = bb_meta->in_bits;

    for (std::size_t wi = 0; wi < words.size(); ++wi)
    {
        const auto word = words[wi];

        for (std::size_t bi = 0; bi < utils::DynamicBitset::BITS_PER_WORD; ++bi)
        {
            if ((word & (1ULL << bi)) == 0ULL)
            {
                continue;
            }

            const auto bit = wi * utils::DynamicBitset::BITS_PER_WORD + bi;

            if (bit < function_meta_->facts.size())
            {
                state.set(bit);
            }
        }
    }
}

bool AvailableCopyQueryFunction::try_advance_cache_to(
        const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto target_bb = instruction->get_basic_block_raw();
    ASSUMPTION(target_bb != nullptr);

    if (cache_.bb != target_bb)
    {
        return false;
    }

    if (cache_.instr == nullptr)
    {
        return false;
    }

    const auto self_it = cache_.instr->get_self_it();
    ASSUMPTION(self_it.has_value());

    auto it = *self_it;

    while (it != cache_.bb->get_instructions().end())
    {
        ASSUMPTION(*it != nullptr);
        ASSUMPTION(cache_.instr == it->get());

        if (it->get() == instruction.get())
        {
            return true;
        }

        if (cache_.after_valid)
        {
            cache_.in_state = cache_.after_state;
        }
        else
        {
            apply_transfer_to_state(*it, cache_.in_state);
        }

        ++it;

        if (it == cache_.bb->get_instructions().end())
        {
            cache_.instr = nullptr;
        }
        else
        {
            ASSUMPTION(*it != nullptr);
            cache_.instr = it->get();
        }

        reset_cached_after_state();
    }

    return false;
}

void AvailableCopyQueryFunction::rebuild_cache_before(
        const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);

    const auto bb_raw = instruction->get_basic_block_raw();
    ASSUMPTION(bb_raw != nullptr);
    ASSUMPTION(bb_raw->get_function_raw() == function_.get());

    cache_.bb    = bb_raw;
    cache_.instr = instruction.get();

    load_basic_block_in_state(bb_raw, cache_.in_state);
    reset_cached_after_state();

    bool found = false;

    for (const auto& current_instruction : bb_raw->get_instructions())
    {
        ASSUMPTION(current_instruction != nullptr);

        if (current_instruction.get() == instruction.get())
        {
            found = true;
            break;
        }

        apply_transfer_to_state(current_instruction, cache_.in_state);
    }

    ASSUMPTION(found);
}

void AvailableCopyQueryFunction::populate_cache_before(
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

    if (try_advance_cache_to(instruction))
    {
        return;
    }

    rebuild_cache_before(instruction);
}

State AvailableCopyQueryFunction::compute_after_state_from_cache(
        const program::InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);
    ASSUMPTION(cache_.bb == instruction->get_basic_block_raw());
    ASSUMPTION(cache_.instr == instruction.get());

    if (!cache_.after_valid)
    {
        cache_.after_state = cache_.in_state;
        apply_transfer_to_state(instruction, cache_.after_state);
        cache_.after_valid = true;
    }

    return cache_.after_state;
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

    State state = before ? cache_.in_state : compute_after_state_from_cache(instruction);

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

    populate_cache_before(instruction);
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

} // namespace optimizer::query
