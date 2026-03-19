#include <sgl_kernel/ffi_tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <sys/mman.h>

namespace {

using namespace host;

inline constexpr size_t kHugePageSize = 1 << 30;

struct MmapPinnedDeleter {
  size_t alloc_size;
  void operator()(void* ptr) const {
    ::cudaHostUnregister(ptr);
    ::munmap(ptr, alloc_size);
  }
};

tvm::ffi::Tensor alloc_hugepage_pinned(int64_t num_bytes) {
  RuntimeCheck(num_bytes > 0, "num_bytes must be positive, got ", num_bytes);

  const size_t alloc_size =
      (static_cast<size_t>(num_bytes) + kHugePageSize - 1) & ~(kHugePageSize - 1);

  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | 30 << MAP_HUGE_SHIFT;
  void* ptr = ::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);

  if (ptr == MAP_FAILED) {
    flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
    ptr = ::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    RuntimeCheck(ptr != MAP_FAILED, "mmap failed for size ", alloc_size);
  }

  auto rc = ::cudaHostRegister(ptr, alloc_size, cudaHostRegisterDefault);
  if (rc != cudaSuccess) {
    ::munmap(ptr, alloc_size);
    RuntimeDeviceCheck(rc);
  }

  const auto shape = std::array<int64_t, 1>{num_bytes};
  constexpr DLDataType dtype = {.code = kDLUInt, .bits = 8, .lanes = 1};
  constexpr DLDevice device = {.device_type = kDLCPU, .device_id = 0};

  return host::ffi::from_blob(
      ptr, tvm::ffi::ShapeView(shape.data(), shape.size()),
      dtype, device, MmapPinnedDeleter{alloc_size});
}

}  // namespace
