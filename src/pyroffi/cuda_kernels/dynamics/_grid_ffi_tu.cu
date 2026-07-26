// XLA FFI handlers wrapping the GRiD-generated rigid body dynamics kernels.
//
// This translation unit is compiled at runtime (per robot) by
// pyroffi.dynamics._grid_codegen: a robot-specific "grid.cuh" emitted by
// grid_codegen.GRiDCodeGenerator (A2R-Lab/GRiD) is placed next to this file
// and nvcc builds the pair into a shared library, which is then registered as
// a set of JAX FFI targets (one .so per robot; symbol names below are
// constant, the Python side namespaces the *target* names).
//
// The generated __global__ kernels are launched directly on the XLA stream;
// GRiD's __host__ wrappers are bypassed because they perform their own
// host<->device transfers and synchronization, which XLA already manages.
// What we do copy from those wrappers is the *launch contract*, which the
// A2R-Lab fork changed in three ways:
//
//   * dynamic shared memory is now sized by the per-algo
//     <ALGO>_DYNAMIC_SHARED_MEM_BYTES<T>() helper (a packed arena that also
//     covers the linalg-helper scratch), not a bare float count;
//   * every kernel except inverse_dynamics takes a global spill workspace
//     (`unsigned char *d_workspace`), sized GRID_WORKSPACE_BYTES_PER_TIMESTEP
//     * GRID_WORKSPACE_SLOTS per timestep;
//   * the dynamics kernels take a `T *d_f_ext` external-wrench buffer, which
//     pyroffi passes as nullptr (no external wrenches on the GRiD path yet).
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
  // calls for the kernel instantiations used here). The overload set is
  // disambiguated by taking the address through an exactly-typed pointer.
  void (*id_du_kern)(T*, unsigned char*, const T*, int, const T*, T*,
                     const grid::robotModel<T>*, const T, const int) =
      &grid::inverse_dynamics_gradient_kernel<T>;
  void (*fd_du_kern)(T*, unsigned char*, const T*, int, T*,
                     const grid::robotModel<T>*, const T, const int) =
      &grid::forward_dynamics_gradient_kernel<T>;
  cudaFuncSetAttribute(id_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());
  cudaFuncSetAttribute(fd_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());
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
  return dim3(static_cast<unsigned>(grid::MAX_PERF_LEVEL_THREADS), 1, 1);
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

// The global spill workspace every non-inverse_dynamics kernel now takes.
// Sized exactly as grid::init_gridData does. It is scratch: the kernels only
// use it to spill what does not fit in the shared arena at the chosen resource
// tier, so a stream-ordered per-call allocation is correct (and at TIER_SHARED
// several kernels never touch it at all).
struct WorkspaceBuffer {
  unsigned char* ptr = nullptr;
  cudaStream_t stream = nullptr;
  ~WorkspaceBuffer() {
    if (ptr != nullptr) cudaFreeAsync(ptr, stream);
  }
  cudaError_t Alloc(int64_t batch, cudaStream_t s) {
    stream = s;
    const size_t bytes = grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() *
                         GRID_WORKSPACE_SLOTS * static_cast<size_t>(batch);
    if (bytes == 0) return cudaSuccess;
    return cudaMallocAsync(reinterpret_cast<void**>(&ptr), bytes, s);
  }
};

// ---------------------------------------------------------------------------
// CRBA-equivalent mass matrix kernel.
//
// GRiDCodeGenerator does not emit a CRBA; instead M(q) is assembled
// column-by-column as M e_j = ID(q, qd=0, qdd=e_j, g=0), reusing the
// generated (fully thread-parallel) inverse_dynamics_inner. XImats are
// loaded/updated once per timestep and shared across all n columns; blocks
// grid-stride over timesteps, so the batch dimension saturates the GPU and
// every column evaluation uses all threads of the block.
// ---------------------------------------------------------------------------

// GRiDCodeGenerator omits the s_topology_helpers parameter from the device
// helpers for serial chains with identical joint subspaces; these shims
// dispatch to whichever signature this robot's grid.cuh actually declares
// (the int/long dummy parameter makes the topology-helpers overload
// preferred when both could bind).

template <typename U>
__device__ auto LoadXImats(U* s_XImats, const U* s_q, int* s_top,
                           const grid::robotModel<U>* m, U* s_temp, int)
    -> decltype(grid::load_update_XImats_helpers<U>(s_XImats, s_q, s_top, m,
                                                    s_temp)) {
  grid::load_update_XImats_helpers<U>(s_XImats, s_q, s_top, m, s_temp);
}

template <typename U>
__device__ void LoadXImats(U* s_XImats, const U* s_q, int* /*s_top*/,
                           const grid::robotModel<U>* m, U* s_temp, long) {
  grid::load_update_XImats_helpers<U>(s_XImats, s_q, m, s_temp);
}

// d_f_ext is passed as nullptr throughout: the CRBA assembly below is a pure
// M(q) column sweep, which carries no external wrench by construction.
template <typename U>
__device__ auto IdInner(U* s_c, U* s_vaf, const U* s_q, const U* s_qd,
                        const U* s_qdd, U* s_XImats, int* s_top, U* s_temp,
                        const U gravity, int)
    -> decltype(grid::inverse_dynamics_inner<U>(s_c, s_vaf, s_q, s_qd, s_qdd,
                                                s_XImats, s_top, s_temp,
                                                nullptr, gravity)) {
  grid::inverse_dynamics_inner<U>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats,
                                  s_top, s_temp, nullptr, gravity);
}

template <typename U>
__device__ void IdInner(U* s_c, U* s_vaf, const U* s_q, const U* s_qd,
                        const U* s_qdd, U* s_XImats, int* /*s_top*/, U* s_temp,
                        const U gravity, long) {
  grid::inverse_dynamics_inner<U>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats,
                                  s_temp, nullptr, gravity);
}

__global__ void CrbaKernel(T* d_M, const T* d_q,
                           const grid::robotModel<T>* d_robotModel,
                           const int NUM_TIMESTEPS) {
  __shared__ T s_q[kNq];
  __shared__ T s_qd[kNq];   // zero
  __shared__ T s_qdd[kNq];  // unit column
  __shared__ T s_c[kNq];
  __shared__ T s_vaf[18 * kNq];
  // Upper bound on gen_topology_helpers_size() (6n+1; 0 for serial chains).
  __shared__ int s_topology_helpers[6 * kNq + 2];
  extern __shared__ T s_XITemp[];
  T* s_XImats = s_XITemp;          // 72n floats (X and I mats per joint)
  T* s_temp = &s_XITemp[72 * kNq];
  for (int k = blockIdx.x; k < NUM_TIMESTEPS; k += gridDim.x) {
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < kNq;
         i += blockDim.x * blockDim.y) {
      s_q[i] = d_q[k * kNq + i];
      s_qd[i] = T(0);
    }
    __syncthreads();
    LoadXImats<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp, 0);
    __syncthreads();
    for (int col = 0; col < kNq; ++col) {
      for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < kNq;
           i += blockDim.x * blockDim.y) {
        s_qdd[i] = (i == col) ? T(1) : T(0);
      }
      __syncthreads();
      // qd = 0 and g = 0 kill every bias term, so s_c = M(q) @ e_col.
      IdInner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers,
                 s_temp, T(0), 0);
      __syncthreads();
      for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < kNq;
           i += blockDim.x * blockDim.y) {
        d_M[(k * kNq + col) * kNq + i] = s_c[i];  // [t, col, row]
      }
      __syncthreads();
    }
  }
}

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
         grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          c->typed_data(), q_qd.ptr, 2 * kNq, qdd.typed_data(),
          /*d_f_ext=*/nullptr, model, gravity, batch);
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
  WorkspaceBuffer ws;
  if (auto st = ws.Alloc(batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(workspace)");
  grid::forward_dynamics_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          qdd->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq, /*d_f_ext=*/nullptr,
          model, gravity, batch);
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

  WorkspaceBuffer ws;
  if (auto st = ws.Alloc(batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(workspace)");
  grid::minv_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          minv->typed_data(), ws.ptr, q.typed_data(), kNq, model, batch);
  return CudaCheck(cudaGetLastError(), "minv_kernel");
}

// Mass matrix M(q): q -> (B, n, n), [t, col, row] fully populated.
ffi::Error GridCrbaImpl(cudaStream_t stream,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> m) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  CrbaKernel<<<GridDims(batch), ThreadDims(),
               grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
      m->typed_data(), q.typed_data(), model, batch);
  return CudaCheck(cudaGetLastError(), "CrbaKernel");
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
  WorkspaceBuffer ws;
  if (auto st = ws.Alloc(batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(workspace)");
  grid::inverse_dynamics_gradient_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
         stream>>>(dc_du->typed_data(), ws.ptr, q_qd.ptr, 2 * kNq,
                   qdd.typed_data(), /*d_f_ext=*/nullptr, model, gravity,
                   batch);
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
  WorkspaceBuffer ws;
  if (auto st = ws.Alloc(batch, stream); st != cudaSuccess)
    return CudaCheck(st, "cudaMallocAsync(workspace)");
  grid::forward_dynamics_gradient_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
         stream>>>(df_du->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq,
                   /*d_f_ext=*/nullptr, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_gradient_kernel");
}

}  // namespace

#ifdef PYROFFI_GRID_RUNTIME_INERTIA
// Runtime-mutable inertia table (grid_codegen runtime_inertia=True).
//
// These are NOT FFI handlers and are deliberately not stream-ordered: upstream's
// set_inertia_params is a blocking host->device memcpy into device-resident
// *model* state, which is not a traceable JAX value. It is safe only at
// grasp-topology boundaries (pick / place / handoff), and the Python side
// enforces that by refusing to run under a tracer. See GridModelState.
extern "C" int GridInertiaParamsSize() { return 10 * grid::NUM_JOINTS; }

extern "C" void GridSetInertiaParams(const float* h_params) {
  grid::robotModel<T>* model = GetRobotModel();
  cudaDeviceSynchronize();  // the table is read by any in-flight kernel launch
  grid::set_inertia_params<T>(model, h_params);
}
#endif  // PYROFFI_GRID_RUNTIME_INERTIA

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridCrbaFfi, GridCrbaImpl,
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
