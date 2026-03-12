#ifndef OPTIMIZER_UTILS_TRANSFER_MUST_HPP_INCLUDED
#define OPTIMIZER_UTILS_TRANSFER_MUST_HPP_INCLUDED
#include <optimizer/utils/points_to/defines.hpp>

#include <utility/assumptions.hpp>

namespace optimizer::utils::points_to
{
void apply_transfer_must(const MustTransferContextBundle& context);

void apply_transfer_must_free(const MustTransferContextBundle& context);
void apply_transfer_must_address(const MustTransferContextBundle& context);
void apply_transfer_must_copy(const MustTransferContextBundle& context);
void apply_transfer_must_load(const MustTransferContextBundle& context);
void apply_transfer_must_store(const MustTransferContextBundle& context);
void apply_transfer_must_alloca(const MustTransferContextBundle& context);
void apply_transfer_must_malloc(const MustTransferContextBundle& context);
void apply_transfer_must_i2p_p2i(const MustTransferContextBundle& context);
void apply_transfer_must_memcpy_memmove(const MustTransferContextBundle& context);
void apply_transfer_must_moveptr(const MustTransferContextBundle& context);
void apply_transfer_must_memset(const MustTransferContextBundle& context);
void apply_transfer_must_stackrestore(const MustTransferContextBundle& context);
void apply_transfer_must_stacksave(const MustTransferContextBundle& context);
void apply_transfer_must_va_start(const MustTransferContextBundle& context);
void apply_transfer_must_va_end(const MustTransferContextBundle& context);
void apply_transfer_must_va_arg(const MustTransferContextBundle& context);
void apply_transfer_must_va_copy(const MustTransferContextBundle& context);
void apply_transfer_must_call(const MustTransferContextBundle& context);

} // namespace optimizer::utils::points_to
#endif
