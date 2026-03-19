from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_alloc_module() -> Module:
    return load_jit(
        "alloc",
        cuda_files=["alloc.cu"],
        cuda_wrappers=[("alloc_hugepage_pinned", "alloc_hugepage_pinned")],
    )


def alloc_hugepage_pinned(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """Allocate a CUDA-pinned host tensor backed by Linux huge pages.

    Falls back to regular pages if huge pages are unavailable.
    The returned tensor lives on CPU but is registered with CUDA
    for efficient async DMA transfers.
    """
    numel = 1
    for s in shape:
        numel *= s
    element_size = torch.empty(0, dtype=dtype).element_size()
    num_bytes = numel * element_size

    module = _jit_alloc_module()
    raw = module.alloc_hugepage_pinned(num_bytes)

    tensor = torch.from_dlpack(raw)
    if dtype != torch.uint8:
        tensor = tensor.view(dtype)
    return tensor.reshape(shape)
