from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# Must match the HiSparseMode enum in csrc/hisparse.cuh.
HISPARSE_MODE_FULL = 0
HISPARSE_MODE_PREFETCH = 1
HISPARSE_MODE_RESOLVE = 2


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    mode: int = HISPARSE_MODE_FULL,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla, mode)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla, mode
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    mode: int = HISPARSE_MODE_FULL,
    num_max_prefetch: int = 0,
    stats_out: torch.Tensor | None = None,
) -> None:
    """Run the HiSparse swap-in kernel.

    mode:
        HISPARSE_MODE_FULL     - Standalone swap-in (existing behavior).
        HISPARSE_MODE_PREFETCH - Stage B prefetch: cap H2D copies at num_max_prefetch and skip
                                 top_k_device_locs writes. Mutates device_buffer_tokens / lru_slots
                                 only for the entries that are actually copied.
        HISPARSE_MODE_RESOLVE  - Resolve the real top-k after a previous PREFETCH on the same layer.
                                 Same logic as FULL; the kernel naturally treats prefetched tokens
                                 as hits.
    """
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True, mode=mode
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    stats_tensor = stats_out if stats_out is not None else empty

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        int(num_max_prefetch),
        stats_tensor,
    )
