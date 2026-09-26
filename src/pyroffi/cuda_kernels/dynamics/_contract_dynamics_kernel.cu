// Forward dynamics through the traced-robot contract, one FFI handler per tier.
//
// Compiled per robot by dynamics/_contract_dynamics.py against a generated
// "_contract_robot_gen.cuh" (namespace pyroffi::traced). The kernels only see the
// contract: which generator implements each tier is decided when the header is made.
//
//   thread  one config per thread  pyroffi::traced::thread::forward_dynamics  (cricket)
//   block   one config per block   pyroffi::traced::block::forward_dynamics   (GRiD adapter)
//
// All buffers are (B, n_q) float32 in PyRoFFI's actuated joint order.

#include <mutex>
#include <unordered_map>

#include "_contract_robot_gen.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace traced = pyroffi::traced;

static_assert(traced::contract_version == 1, "written against contract v1");

namespace {

constexpr int N = traced::n_q;

ffi::Error CheckShape(int64_t n) {
  if (n != N)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "contract forward dynamics: n_q does not match the traced robot");
  return ffi::Error::Success();
}

ffi::Error CudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(what) + ": " + cudaGetErrorString(e));
  return ffi::Error::Success();
}

__global__ void FdThreadKernel(const float* __restrict__ q, const float* __restrict__ qd,
                               const float* __restrict__ tau, float* __restrict__ qdd, int batch) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  float sq[N], sqd[N], stau[N], sqdd[N];
  for (int i = 0; i < N; ++i) {
    sq[i] = q[b * N + i];
    sqd[i] = qd[b * N + i];
    stau[i] = tau[b * N + i];
  }
  traced::thread::forward_dynamics(sq, sqd, stau, sqdd);
  for (int i = 0; i < N; ++i) qdd[b * N + i] = sqdd[i];
}

ffi::Error FdThreadImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Buffer<ffi::DataType::F32> qd, ffi::Buffer<ffi::DataType::F32> tau,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> qdd) {
  const int64_t batch = q.dimensions()[0];
  if (auto err = CheckShape(q.dimensions()[1]); err.failure()) return err;
  if (batch == 0) return ffi::Error::Success();
  constexpr int kThreads = 128;
  FdThreadKernel<<<(batch + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      q.typed_data(), qd.typed_data(), tau.typed_data(), qdd->typed_data(), batch);
  return CudaCheck(cudaGetLastError(), "FdThreadKernel");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ContractFdThreadFfi, FdThreadImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());

#ifdef PYROFFI_CONTRACT_BLOCK_FORWARD_DYNAMICS
namespace {

__global__ void FdBlockKernel(const float* __restrict__ q, const float* __restrict__ qd,
                              const float* __restrict__ tau, float* __restrict__ qdd,
                              traced::block::Model model, float gravity, int batch) {
  // Kernel-owned shared memory is static: the block tier's provider owns the dynamic arena.
  __shared__ float s_io[4 * N];
  __shared__ float s_scratch[traced::block::forward_dynamics_scratch];
  float *s_q = s_io, *s_qd = s_io + N, *s_tau = s_io + 2 * N, *s_qdd = s_io + 3 * N;
  for (int b = blockIdx.x; b < batch; b += gridDim.x) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      s_q[i] = q[b * N + i];
      s_qd[i] = qd[b * N + i];
      s_tau[i] = tau[b * N + i];
    }
    __syncthreads();
    traced::block::forward_dynamics(s_q, s_qd, s_tau, s_qdd, s_scratch, model, gravity);
    for (int i = threadIdx.x; i < N; i += blockDim.x) qdd[b * N + i] = s_qdd[i];
    __syncthreads();
  }
}

ffi::Error FdBlockImpl(cudaStream_t stream, float gravity, ffi::Buffer<ffi::DataType::F32> q,
                       ffi::Buffer<ffi::DataType::F32> qd, ffi::Buffer<ffi::DataType::F32> tau,
                       ffi::Result<ffi::Buffer<ffi::DataType::F32>> qdd) {
  const int64_t batch = q.dimensions()[0];
  if (auto err = CheckShape(q.dimensions()[1]); err.failure()) return err;
  if (batch == 0) return ffi::Error::Success();
  // The provider's model lives on the device; one per device, made on first use.
  static std::mutex mu;
  static std::unordered_map<int, traced::block::Model> models;
  int device = 0;
  cudaGetDevice(&device);
  traced::block::Model model;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = models.find(device);
    if (it == models.end()) {
      it = models.emplace(device, traced::block::make_model()).first;
      if (auto err = CudaCheck(
              cudaFuncSetAttribute(FdBlockKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   traced::block::forward_dynamics_smem_bytes),
              "FdBlockKernel smem attribute");
          err.failure())
        return err;
    }
    model = it->second;
  }
  const int blocks = batch < 65535 ? static_cast<int>(batch) : 65535;
  FdBlockKernel<<<blocks, traced::block::forward_dynamics_threads,
                  traced::block::forward_dynamics_smem_bytes, stream>>>(
      q.typed_data(), qd.typed_data(), tau.typed_data(), qdd->typed_data(), model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "FdBlockKernel");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ContractFdBlockFfi, FdBlockImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("gravity")
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
#endif  // PYROFFI_CONTRACT_BLOCK_FORWARD_DYNAMICS
