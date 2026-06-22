import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("zeros"),
    key=["n_elements"],
)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)
