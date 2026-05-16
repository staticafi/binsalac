#ifndef OPTIMIZER_UTILS_TRANSFER_MAY_HPP_INCLUDED
#define OPTIMIZER_UTILS_TRANSFER_MAY_HPP_INCLUDED
#include <optimizer/utils/points_to/defines.hpp>

namespace optimizer::utils::points_to
{
void apply_transfer_may(const MayTransferContextBundle& context);

void apply_transfer_may_load(const MayTransferContextBundle& context);
void apply_transfer_may_store(const MayTransferContextBundle& context);
void apply_transfer_may_memcpy_memmove(const MayTransferContextBundle& context);
void apply_transfer_may_address(const MayTransferContextBundle& context);
void apply_transfer_may_copy(const MayTransferContextBundle& context);
void apply_transfer_may_alloca(const MayTransferContextBundle& context);
void apply_transfer_may_malloc(const MayTransferContextBundle& context);
void apply_transfer_may_i2p_p2i(const MayTransferContextBundle& context);
void apply_transfer_may_moveptr(const MayTransferContextBundle& context);
void apply_transfer_may_memset(const MayTransferContextBundle& context);
void apply_transfer_may_call(const MayTransferContextBundle& context);
void apply_transfer_may_stacksave(const MayTransferContextBundle& context);
void apply_transfer_may_stackrestore(const MayTransferContextBundle& context);
void apply_transfer_may_va_start(const MayTransferContextBundle& context);
void apply_transfer_may_va_end(const MayTransferContextBundle& context);
void apply_transfer_may_va_arg(const MayTransferContextBundle& context);
void apply_transfer_may_va_copy(const MayTransferContextBundle& context);
} // namespace optimizer::utils::points_to
#endif
