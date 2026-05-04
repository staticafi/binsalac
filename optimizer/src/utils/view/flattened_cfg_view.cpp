#include <optimizer/utils/view/flattened_cfg_view.hpp>

#include <utility/assumptions.hpp>

#include <queue>
#include <utility>

namespace optimizer::utils::view
{

FlattenedCFGView::FlattenedCFGView(program::FunctionIR_sptr function)
    : function_{std::move(function)}
{
    ASSUMPTION(function_ != nullptr);
    build();
}

bool FlattenedCFGView::empty() const
{
    return blocks_.empty();
}

std::size_t FlattenedCFGView::size() const
{
    return blocks_.size();
}

std::size_t FlattenedCFGView::entry() const
{
    return entry_;
}

const program::BasicBlockIR_sptr& FlattenedCFGView::block(std::size_t index) const
{
    ASSUMPTION(index < blocks_.size());
    return blocks_[index];
}

const std::vector<program::BasicBlockIR_sptr>& FlattenedCFGView::blocks() const
{
    return blocks_;
}

const std::vector<std::size_t>& FlattenedCFGView::predecessors(std::size_t index) const
{
    ASSUMPTION(index < predecessors_.size());
    return predecessors_[index];
}

const std::vector<std::size_t>& FlattenedCFGView::successors(std::size_t index) const
{
    ASSUMPTION(index < successors_.size());
    return successors_[index];
}

const std::vector<std::size_t>& FlattenedCFGView::exit_blocks() const
{
    return exit_blocks_;
}

bool FlattenedCFGView::reachable(std::size_t index) const
{
    ASSUMPTION(index < reachable_.size());
    return reachable_[index];
}

bool FlattenedCFGView::is_exit_block(std::size_t index) const
{
    ASSUMPTION(index < successors_.size());
    return successors_[index].empty();
}

std::size_t FlattenedCFGView::index_of(program::BasicBlockIR_raw block) const
{
    ASSUMPTION(block != nullptr);

    const auto it = index_of_.find(block);
    ASSUMPTION(it != index_of_.end());

    return it->second;
}

std::size_t FlattenedCFGView::index_of(const program::BasicBlockIR_sptr& block) const
{
    ASSUMPTION(block != nullptr);
    return index_of(block.get());
}

void FlattenedCFGView::build()
{
    collect_blocks();
    build_block_index();
    build_edges();
    detect_entry_block();
    compute_reachable_blocks();
}

void FlattenedCFGView::collect_blocks()
{
    blocks_.clear();
    blocks_.reserve(function_->get_basic_blocks().size());

    for (const auto& block : function_->get_basic_blocks())
    {
        ASSUMPTION(block != nullptr);
        blocks_.push_back(block);
    }
}

void FlattenedCFGView::build_block_index()
{
    index_of_.clear();
    index_of_.reserve(blocks_.size());

    for (std::size_t index = 0; index < blocks_.size(); ++index)
    {
        index_of_.emplace(blocks_[index].get(), index);
    }
}

void FlattenedCFGView::build_edges()
{
    predecessors_.assign(blocks_.size(), {});
    successors_.assign(blocks_.size(), {});
    exit_blocks_.clear();

    for (std::size_t index = 0; index < blocks_.size(); ++index)
    {
        const auto& block = blocks_[index];

        for (const auto& weak_pred : block->get_predecessors())
        {
            const auto pred = weak_pred.lock();
            ASSUMPTION(pred != nullptr);

            predecessors_[index].push_back(index_of(pred));
        }

        for (const auto& weak_succ : block->get_successors())
        {
            const auto succ = weak_succ.lock();
            ASSUMPTION(succ != nullptr);

            successors_[index].push_back(index_of(succ));
        }

        if (successors_[index].empty())
        {
            exit_blocks_.push_back(index);
        }
    }
}

void FlattenedCFGView::detect_entry_block()
{
    entry_ = 0;

    if (blocks_.empty())
    {
        return;
    }

    const auto entry_block = function_->get_entry_basic_block();

    if (entry_block != nullptr)
    {
        entry_ = index_of(entry_block);
    }
}

void FlattenedCFGView::compute_reachable_blocks()
{
    reachable_.assign(blocks_.size(), false);

    if (blocks_.empty())
    {
        return;
    }

    std::queue<std::size_t> worklist;

    reachable_[entry_] = true;
    worklist.push(entry_);

    while (!worklist.empty())
    {
        const auto block = worklist.front();
        worklist.pop();

        for (const auto succ : successors_[block])
        {
            if (reachable_[succ])
            {
                continue;
            }

            reachable_[succ] = true;
            worklist.push(succ);
        }
    }
}
} // namespace optimizer::utils::view
