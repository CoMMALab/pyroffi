// XLA FFI handlers wrapping the GRiD-generated rigid body dynamics kernels.
//
// This translation unit is compiled at runtime (per robot) by
// pyroffi.dynamics._grid_codegen: a robot-specific "grid.cuh" emitted by
// GRiDCodeGenerator is placed next to this file and nvcc builds the pair
// into a shared library, which is then registered as a set of JAX FFI
// targets (one .so per robot; symbol names below are constant, the Python
// side namespaces the *target* names).
//
// The generated __global__ kernels are launched directly on the XLA stream;
// GRiD's __host__ wrappers are bypassed because they perform their own
// host<->device transfers and synchronization, which XLA already manages.
//
// Batch mapping: the JAX batch dimension B is GRiD's NUM_TIMESTEPS. Inputs
// arrive as separate (B, n) buffers and are interleaved into GRiD's
// [q | qd (| u/qdd)] per-timestep layout by small pack kernels.

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "grid.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using T = float;

constexpr int kNq = grid::NUM_JOINTS;

// ---------------------------------------------------------------------------
// Per-device robot model cache (mirrors the FK kernel's ModelCache pattern).
// ---------------------------------------------------------------------------

grid::robotModel<T>* GetRobotModel() {
  static std::mutex mu;
  static std::unordered_map<int, grid::robotModel<T>*> models;
  int device = -1;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(mu);
  auto it = models.find(device);
  if (it != models.end()) return it->second;
  grid::robotModel<T>* model = grid::init_robotModel<T>();
  // Allow the gradient kernels to exceed the default dynamic shared memory
  // limit on large robots (replicates grid::init_grid's cudaFuncSetAttribute
  // calls for the kernel instantiations used here).
  void (*id_du_kern)(T*, const T*, int, const T*, const grid::robotModel<T>*,
                     const T, const int) =
      &grid::inverse_dynamics_gradient_kernel<T>;
  void (*fd_du_kern)(T*, const T*, int, const grid::robotModel<T>*, const T,
                     const int) = &grid::forward_dynamics_gradient_kernel<T>;
  cudaFuncSetAttribute(id_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grid::ID_DU_MAX_SHARED_MEM_COUNT * sizeof(T));
  cudaFuncSetAttribute(fd_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grid::FD_DU_MAX_SHARED_MEM_COUNT * sizeof(T));
  models[device] = model;
  return model;
}

// ---------------------------------------------------------------------------
// Pack kernels: (B, n) x k separate buffers -> interleaved stride-(k*n).
// ---------------------------------------------------------------------------

__global__ void Pack2Kernel(T* dst, const T* a, const T* b, int n, int batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * n) return;
  int k = idx / n, i = idx % n;
  dst[k * 2 * n + i] = a[idx];
  dst[k * 2 * n + n + i] = b[idx];
}

__global__ void Pack3Kernel(T* dst, const T* a, const T* b, const T* c, int n,
                            int batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * n) return;
  int k = idx / n, i = idx % n;
  dst[k * 3 * n + i] = a[idx];
  dst[k * 3 * n + n + i] = b[idx];
  dst[k * 3 * n + 2 * n + i] = c[idx];
}

inline dim3 GridDims(int batch) {
  // The GRiD kernels use a grid-stride loop over timesteps.
  return dim3(static_cast<unsigned>(batch < 65535 ? batch : 65535), 1, 1);
}

inline dim3 ThreadDims() {
  return dim3(static_cast<unsigned>(grid::SUGGESTED_THREADS), 1, 1);
}

inline int PackBlocks(int total) { return (total + 255) / 256; }

ffi::Error CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string(what) + ": " + cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

ffi::Error CheckDims(int64_t batch, int64_t n, int64_t elems) {
  if (n != kNq || elems != batch * n) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "buffer shape mismatch against NUM_JOINTS");
  }
  return ffi::Error::Success();
}

struct ScratchBuffer {
  T* ptr = nullptr;
  cudaStream_t stream = nullptr;
  ~ScratchBuffer() {
    if (ptr != nullptr) cudaFreeAsync(ptr, stream);
  }
  cudaError_t Alloc(size_t count, cudaStream_t s) {
    stream = s;
    return cudaMallocAsync(reinterpret_cast<void**>(&ptr), count * sizeof(T),
                           s);
  }
};

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

// Inverse dynamics: (q, qd, qdd) -> joint torques c.  All (B, n).
ffi::Error GridIdImpl(cudaStream_t stream, float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> qdd,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> c) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd;
  if (auto st = q_qd.Alloc(2 * kNq * batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(q_qd)");
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd.typed_data(), kNq, batch);
  grid::inverse_dynamics_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::ID_DYNAMIC_SHARED_MEM_COUNT * sizeof(T), stream>>>(
          c->typed_data(), q_qd.ptr, 2 * kNq, qdd.typed_data(), model, gravity,
          batch);
  return CudaCheck(cudaGetLastError(), "inverse_dynamics_kernel");
}

// Forward dynamics: (q, qd, u) -> joint accelerations qdd.  All (B, n).
ffi::Error GridFdImpl(cudaStream_t stream, float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> u,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> qdd) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u;
  if (auto st = q_qd_u.Alloc(3 * kNq * batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(q_qd_u)");
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  grid::forward_dynamics_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FD_DYNAMIC_SHARED_MEM_COUNT * sizeof(T), stream>>>(
          qdd->typed_data(), q_qd_u.ptr, 3 * kNq, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_kernel");
}

// Direct Minv: q -> inverse mass matrix, (B, n, n), SYMMETRIC_UPPER filled.
ffi::Error GridMinvImpl(cudaStream_t stream,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> minv) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  grid::direct_minv_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::MINV_DYNAMIC_SHARED_MEM_COUNT * sizeof(T), stream>>>(
          minv->typed_data(), q.typed_data(), kNq, model, batch);
  return CudaCheck(cudaGetLastError(), "direct_minv_kernel");
}

// Analytic inverse dynamics gradient: (q, qd, qdd) -> dc/d[q,qd],
// (B, 2n, n) with column-major n x 2n per timestep ([dq block | dqd block]).
ffi::Error GridIdGradImpl(cudaStream_t stream, float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> qdd,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> dc_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd;
  if (auto st = q_qd.Alloc(2 * kNq * batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(q_qd)");
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd.typed_data(), kNq, batch);
  grid::inverse_dynamics_gradient_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::ID_DU_DYNAMIC_SHARED_MEM_COUNT * sizeof(T), stream>>>(
          dc_du->typed_data(), q_qd.ptr, 2 * kNq, qdd.typed_data(), model,
          gravity, batch);
  return CudaCheck(cudaGetLastError(), "inverse_dynamics_gradient_kernel");
}

// Analytic forward dynamics gradient: (q, qd, u) -> dqdd/d[q,qd],
// (B, 2n, n) with column-major n x 2n per timestep.
ffi::Error GridFdGradImpl(cudaStream_t stream, float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> u,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> df_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u;
  if (auto st = q_qd_u.Alloc(3 * kNq * batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(q_qd_u)");
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  grid::forward_dynamics_gradient_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FD_DU_DYNAMIC_SHARED_MEM_COUNT * sizeof(T), stream>>>(
          df_du->typed_data(), q_qd_u.ptr, 3 * kNq, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_gradient_kernel");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridIdFfi, GridIdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridFdFfi, GridFdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridMinvFfi, GridMinvImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridIdGradFfi, GridIdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridFdGradFfi, GridFdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());
