// Traced dynamics through the robot contract: one FFI handler per (op, tier).
//
// Compiled per robot by dynamics/_contract_dynamics.py against a generated
// "_contract_robot_gen.cuh", which lists its ops in PYROFFI_CONTRACT_OPS(X) as
// X(name, NIN, NOUT) and defines, in pyroffi::traced::ops,
//
//   <name>_thread(x, y, ys)                 one config per thread
//   <name>_block(rank, x, y, s, ys)         32 configs per block, <name>_warps warps
//                                           splitting each level (warp-split tier)
//   <name>_scratch, <name>_smem             block scratch floats, and whether they fit
//                                           in shared memory (else global workspace)
//
// Buffers: input (B, NIN) row-major, in the generator's joint order (packed by the
// caller); output (NOUT, B), so every output is a coalesced store.

#include <string>

#include "_contract_robot_gen.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace ops = pyroffi::traced::ops;

static_assert(pyroffi::traced::contract_version == 2, "written against contract v2");

namespace {

using Buf = ffi::Buffer<ffi::DataType::F32>;
using Ret = ffi::Result<ffi::Buffer<ffi::DataType::F32>>;

ffi::Error CudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(what) + ": " + cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error CheckShape(const Buf& in, int nin) {
  if (in.dimensions().size() != 2 || in.dimensions()[1] != nin)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "contract op: input is not (B, NIN)");
  return ffi::Error::Success();
}

template <int NIN, void (*F)(const float*, float*, int)>
__global__ void ThreadKernel(const float* __restrict__ in, float* __restrict__ out, int batch) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  float x[NIN];
  for (int i = 0; i < NIN; ++i) x[i] = in[(long)b * NIN + i];
  F(x, out + b, batch);
}

template <int NIN, int SCRATCH, bool SMEM, void (*F)(int, const float*, float*, float*, int)>
__global__ void BlockKernel(const float* __restrict__ in, float* __restrict__ out, float* ws, int batch) {
  extern __shared__ float sdyn[];
  float* s = SMEM ? sdyn : ws + (long)blockIdx.x * SCRATCH;
  // Lanes past the batch recompute the last config (same values, same address): every
  // lane must reach every barrier.
  const int b = min(blockIdx.x * 32 + (int)(threadIdx.x & 31), batch - 1);
  float x[NIN];
  for (int i = 0; i < NIN; ++i) x[i] = in[(long)b * NIN + i];
  F((int)(threadIdx.x >> 5), x, out + b, s, batch);
}

template <int NIN, void (*F)(const float*, float*, int)>
ffi::Error RunThread(cudaStream_t stream, Buf in, Ret out) {
  if (auto err = CheckShape(in, NIN); err.failure()) return err;
  const int64_t batch = in.dimensions()[0];
  if (batch == 0) return ffi::Error::Success();
  constexpr int kThreads = 128;
  ThreadKernel<NIN, F><<<(batch + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      in.typed_data(), out->typed_data(), batch);
  return CudaCheck(cudaGetLastError(), "contract thread kernel");
}

template <int NIN, int SCRATCH, bool SMEM, int WARPS, void (*F)(int, const float*, float*, float*, int)>
ffi::Error RunBlock(cudaStream_t stream, ffi::ScratchAllocator scratch, Buf in, Ret out) {
  if (auto err = CheckShape(in, NIN); err.failure()) return err;
  const int64_t batch = in.dimensions()[0];
  if (batch == 0) return ffi::Error::Success();
  const int64_t blocks = (batch + 31) / 32;
  float* ws = nullptr;
  if (!SMEM) {
    auto mem = scratch.Allocate(sizeof(float) * SCRATCH * blocks, 16);
    if (!mem.has_value())
      return ffi::Error(ffi::ErrorCode::kResourceExhausted, "contract block kernel workspace");
    ws = static_cast<float*>(*mem);
  }
  constexpr size_t smem = SMEM ? sizeof(float) * SCRATCH : 0;
  static const cudaError_t attr = cudaFuncSetAttribute(
      BlockKernel<NIN, SCRATCH, SMEM, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (auto err = CudaCheck(attr, "contract block kernel smem attribute"); err.failure()) return err;
  BlockKernel<NIN, SCRATCH, SMEM, F><<<blocks, WARPS * 32, smem, stream>>>(
      in.typed_data(), out->typed_data(), ws, batch);
  return CudaCheck(cudaGetLastError(), "contract block kernel");
}

}  // namespace

#define PYROFFI_CONTRACT_HANDLERS(name, NIN, NOUT)                                              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                                 \
      Contract_##name##_thread, (RunThread<NIN, ops::name##_thread>),                            \
      ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<Buf>().Ret<Buf>());          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                                 \
      Contract_##name##_block,                                                                   \
      (RunBlock<NIN, ops::name##_scratch, ops::name##_smem, ops::name##_warps, ops::name##_block>), \
      ffi::Ffi::Bind()                                                                           \
          .Ctx<ffi::PlatformStream<cudaStream_t>>()                                              \
          .Ctx<ffi::ScratchAllocator>()                                                          \
          .Arg<Buf>()                                                                            \
          .Ret<Buf>());

PYROFFI_CONTRACT_OPS(PYROFFI_CONTRACT_HANDLERS)
