#ifndef OPTIMIZER_UTILS_VIEW_FLATTENED_CFG_VIEW_DEFINED
#define OPTIMIZER_UTILS_VIEW_FLATTENED_CFG_VIEW_DEFINED
#include <optimizer/programIR/basic_block_ir.hpp>
#include <optimizer/programIR/function_ir.hpp>
#include <optimizer/utils/sparse_map.hpp>

#include <cstddef>
#include <vector>

namespace optimizer::utils::view
{
class FlattenedCFGView
{
  public:
    explicit FlattenedCFGView(program::FunctionIR_sptr function);

    [[nodiscard]] bool        empty() const;
    [[nodiscard]] std::size_t size() const;

    [[nodiscard]] std::size_t entry() const;

    [[nodiscard]] const program::BasicBlockIR_sptr&              block(std::size_t index) const;
    [[nodiscard]] const std::vector<program::BasicBlockIR_sptr>& blocks() const;

    [[nodiscard]] const std::vector<std::size_t>& predecessors(std::size_t index) const;
    [[nodiscard]] const std::vector<std::size_t>& successors(std::size_t index) const;

    [[nodiscard]] const std::vector<std::size_t>& exit_blocks() const;

    [[nodiscard]] bool reachable(std::size_t index) const;
    [[nodiscard]] bool is_exit_block(std::size_t index) const;

    [[nodiscard]] std::size_t index_of(program::BasicBlockIR_raw block) const;
    [[nodiscard]] std::size_t index_of(const program::BasicBlockIR_sptr& block) const;

  private:
    void build();

    void collect_blocks();
    void build_block_index();
    void build_edges();
    void detect_entry_block();
    void compute_reachable_blocks();

  private:
    program::FunctionIR_sptr function_;

    std::size_t entry_{0};

    std::vector<program::BasicBlockIR_sptr> blocks_;

    std::vector<std::vector<std::size_t>> predecessors_;
    std::vector<std::vector<std::size_t>> successors_;

    std::vector<std::size_t> exit_blocks_;
    std::vector<bool>        reachable_;

    utils::SparseMap<program::BasicBlockIR_raw, std::size_t> index_of_;
};

} // namespace optimizer::utils::view
#endif
