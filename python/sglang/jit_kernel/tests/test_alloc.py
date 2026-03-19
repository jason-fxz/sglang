import pytest
import torch

from sglang.jit_kernel.alloc import alloc_hugepage_pinned


class TestAllocHugepagePinned:
    def test_basic_shape_and_dtype(self):
        t = alloc_hugepage_pinned((1024,), dtype=torch.float32)
        assert t.shape == (1024,)
        assert t.dtype == torch.float32
        assert t.device == torch.device("cpu")

    def test_multidim(self):
        t = alloc_hugepage_pinned((64, 128), dtype=torch.bfloat16)
        assert t.shape == (64, 128)
        assert t.dtype == torch.bfloat16

    def test_uint8_default(self):
        t = alloc_hugepage_pinned((4096,))
        assert t.dtype == torch.uint8
        assert t.shape == (4096,)

    def test_write_read_roundtrip(self):
        t = alloc_hugepage_pinned((1024,), dtype=torch.float32)
        t.fill_(42.0)
        assert (t == 42.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_h2d_transfer(self):
        src = alloc_hugepage_pinned((1024,), dtype=torch.float32)
        src.fill_(3.14)
        dst = src.cuda()
        assert torch.allclose(dst.cpu(), src)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_async_h2d_transfer(self):
        src = alloc_hugepage_pinned((2048,), dtype=torch.float32)
        src.fill_(2.718)
        dst = torch.empty(2048, dtype=torch.float32, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        assert torch.allclose(dst.cpu(), src)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.float32, torch.bfloat16, torch.int32, torch.int64],
    )
    def test_various_dtypes(self, dtype):
        t = alloc_hugepage_pinned((256,), dtype=dtype)
        assert t.dtype == dtype
        assert t.shape == (256,)

    def test_large_allocation(self):
        t = alloc_hugepage_pinned((4 * 1024 * 1024,), dtype=torch.float32)
        assert t.shape == (4 * 1024 * 1024,)
        t.fill_(1.0)
        assert t[0].item() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
