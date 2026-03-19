"""Benchmark for hugepage pinned memory allocation and H2D transfer.

Compares:
- JIT hugepage pinned: mmap(MAP_HUGETLB) + cudaHostRegister
- PyTorch pin_memory: torch.empty(..., pin_memory=True)
- Regular CPU: torch.empty(...)

Tests both allocation latency and host-to-device transfer throughput.

Note: Uses do_bench (not do_bench_cudagraph) since memory allocation
and CPU-GPU transfers are not CUDA graph compatible.
"""

from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.alloc import alloc_hugepage_pinned
from sglang.jit_kernel.benchmark.utils import DEFAULT_QUANTILES, get_benchmark_range

SIZE_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(20, 28)],  # 1MB to 128MB
    ci_range=[2**24],  # 16MB
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size_bytes"],
        x_vals=SIZE_RANGE,
        line_arg="provider",
        line_vals=["hugepage_pinned", "torch_pinned", "torch_cpu"],
        line_names=["Hugepage Pinned", "Torch Pinned", "Torch CPU"],
        styles=[("blue", "-"), ("orange", "--"), ("red", ":")],
        ylabel="us",
        plot_name="alloc-time",
        args={},
    )
)
def benchmark_alloc(
    size_bytes: int, provider: str
) -> Tuple[float, float, float]:
    """Allocation latency."""

    def fn():
        if provider == "hugepage_pinned":
            return alloc_hugepage_pinned((size_bytes,), dtype=torch.uint8)
        elif provider == "torch_pinned":
            return torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        else:
            return torch.empty(size_bytes, dtype=torch.uint8)

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, quantiles=DEFAULT_QUANTILES, warmup=3, rep=10
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size_bytes"],
        x_vals=SIZE_RANGE,
        line_arg="provider",
        line_vals=["hugepage_pinned", "torch_pinned", "torch_cpu"],
        line_names=["Hugepage Pinned", "Torch Pinned", "Torch CPU"],
        styles=[("blue", "-"), ("orange", "--"), ("red", ":")],
        ylabel="GB/s",
        plot_name="h2d-transfer-throughput",
        args={},
    )
)
def benchmark_transfer(
    size_bytes: int, provider: str
) -> Tuple[float, float, float]:
    """Host-to-device transfer throughput."""
    if provider == "hugepage_pinned":
        src = alloc_hugepage_pinned((size_bytes,), dtype=torch.uint8)
    elif provider == "torch_pinned":
        src = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    else:
        src = torch.empty(size_bytes, dtype=torch.uint8)

    dst = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    def fn():
        dst.copy_(src, non_blocking=True)

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, quantiles=DEFAULT_QUANTILES, warmup=5, rep=25
    )
    gb = size_bytes / 1e9
    return gb / (ms / 1e3), gb / (max_ms / 1e3), gb / (min_ms / 1e3)


if __name__ == "__main__":
    print("=" * 60)
    print("Allocation Latency")
    print("=" * 60)
    benchmark_alloc.run(print_data=True)

    print("\n" + "=" * 60)
    print("H2D Transfer Throughput")
    print("=" * 60)
    benchmark_transfer.run(print_data=True)
