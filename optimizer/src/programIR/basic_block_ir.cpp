#include <optimizer/programIR/basic_block_ir.hpp>

#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/programIR/instruction_ir.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::program
{
void BasicBlockIR::assign_to_function(const FunctionIR_sptr& function)
{
    ASSUMPTION(function != nullptr);

    function->acquire_basic_block(shared_from_this());
}

FunctionIR_raw BasicBlockIR::get_function_raw() const
{
    return function_raw_;
}

FunctionIR_sptr BasicBlockIR::get_function() const
{
    return function_.lock();
}

const std::optional<BasicBlockIRListS_iter>& BasicBlockIR::get_self_it() const
{
    return self_it_;
}

const BasicBlockIR::Metadata& BasicBlockIR::get_metadata() const
{
    return metadata_;
}

std::optional<BasicBlockIRListS_iter>& BasicBlockIR::get_self_it()
{
    return self_it_;
}

BasicBlockIR::Metadata& BasicBlockIR::get_metadata()
{
    return metadata_;
}

void BasicBlockIR::set_function(FunctionIR_sptr function)
{
    function_raw_ = function.get();
    function_     = function;
}

void BasicBlockIR::add_successor(BasicBlockIR_sptr successor)
{
    ASSUMPTION(successor != nullptr);

    successors_.emplace_back(std::move(successor));
}

void BasicBlockIR::add_successor_front(BasicBlockIR_sptr successor)
{
    ASSUMPTION(successor != nullptr);

    successors_.emplace_front(std::move(successor));
}

void BasicBlockIR::add_predecessor(BasicBlockIR_sptr predecessor)
{
    ASSUMPTION(predecessor != nullptr);

    predecessors_.emplace_back(std::move(predecessor));
}

void BasicBlockIR::remove_predecessor(const BasicBlockIR_sptr& predecessor)
{
    ASSUMPTION(predecessor != nullptr);

    predecessors_.remove_if(
            [&](const std::weak_ptr<BasicBlockIR>& w)
            { return !w.owner_before(predecessor) && !predecessor.owner_before(w); });
}
void BasicBlockIR::remove_successor(const BasicBlockIR_sptr& successor)
{
    ASSUMPTION(successor != nullptr);

    successors_.remove_if([&](const std::weak_ptr<BasicBlockIR>& w)
                          { return !w.owner_before(successor) && !successor.owner_before(w); });
}

InstructionIRListS_iter BasicBlockIR::release_instruction(const InstructionIR_sptr& instruction)
{
    ASSUMPTION(instruction != nullptr);
    ASSUMPTION(instruction->get_basic_block_raw() == this);
    ASSUMPTION(instruction->get_basic_block().get() == this);
    ASSUMPTION(instruction->get_self_it().has_value());

    InstructionIRListS_iter next = instructions_.erase(instruction->get_self_it().value());
    instruction->set_basic_block(nullptr);
    instruction->get_self_it() = std::nullopt;
    return next;
}

void BasicBlockIR::acquire_instruction(InstructionIR_sptr instruction)
{
    ASSUMPTION(instruction != nullptr);

    if (const auto owner_before = instruction->get_basic_block(); owner_before != nullptr)
    {
        owner_before->release_instruction(instruction);
    }

    instruction->set_basic_block(shared_from_this());
    instructions_.push_back(std::move(instruction));
    instructions_.back()->get_self_it() = --instructions_.end();
}

void BasicBlockIR::acquire_instruction_front(InstructionIR_sptr instruction)
{
    ASSUMPTION(instruction != nullptr);

    if (const auto owner_before = instruction->get_basic_block(); owner_before != nullptr)
    {
        owner_before->release_instruction(instruction);
    }

    instruction->set_basic_block(shared_from_this());
    instructions_.push_front(std::move(instruction));
    instructions_.back()->get_self_it() = instructions_.begin();
}

const InstructionIRListS& BasicBlockIR::get_instructions() const
{
    return instructions_;
}

const BasicBlockIRListW& BasicBlockIR::get_successors() const
{
    return successors_;
}

const BasicBlockIRListW& BasicBlockIR::get_predecessors() const
{
    return predecessors_;
}
} // namespace optimizer::program
