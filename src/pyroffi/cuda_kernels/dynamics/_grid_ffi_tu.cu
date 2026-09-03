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

// Call-scoped device scratch, sourced from XLA's own allocator.
//
// These used to live on private `cudaMallocAsync`/`cudaFreeAsync` calls on
// the XLA stream. That put them in a separate memory pool from JAX/XLA, so
// with XLA preallocating the device up front the FFI mallocs competed for
// the sliver XLA left behind and eventually OOM'd -- the same failure mode
// already found and fixed for the collision kernels (see the comment above
// `scratch_alloc_void` in _robogpu_collision_host.cu). Instead we draw them
// from `ffi::ScratchAllocator`, which allocates stream-ordered from XLA's
// device allocator and reclaims everything when the handler returns.
struct ScratchBuffer {
  T* ptr = nullptr;
  ffi::Error err = ffi::Error::Success();
  ScratchBuffer(ffi::ScratchAllocator& scratch, size_t count) {
    auto p = scratch.Allocate(count * sizeof(T), alignof(T));
    if (!p.has_value()) {
      err = ffi::Error(ffi::ErrorCode::kResourceExhausted,
                       "scratch.Allocate(q_qd) failed");
      return;
    }
    ptr = reinterpret_cast<T*>(*p);
  }
};

// The global spill workspace every non-inverse_dynamics kernel now takes.
// Sized exactly as grid::init_gridData does. It is scratch: the kernels only
// use it to spill what does not fit in the shared arena at the chosen resource
// tier, so a stream-ordered per-call allocation is correct (and at TIER_SHARED
// several kernels never touch it at all).
struct WorkspaceBuffer {
  unsigned char* ptr = nullptr;
  ffi::Error err = ffi::Error::Success();
  WorkspaceBuffer(ffi::ScratchAllocator& scratch, int64_t batch) {
    const size_t bytes = grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() *
                         GRID_WORKSPACE_SLOTS * static_cast<size_t>(batch);
    if (bytes == 0) return;
    auto p = scratch.Allocate(bytes, alignof(std::max_align_t));
    if (!p.has_value()) {
      err = ffi::Error(ffi::ErrorCode::kResourceExhausted,
                       "scratch.Allocate(workspace) failed");
      return;
    }
    ptr = reinterpret_cast<unsigned char*>(*p);
  }
};

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

// Inverse dynamics: (q, qd, qdd) -> joint torques c.  All (B, n).
ffi::Error GridIdImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                      float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> qdd,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> c) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
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
ffi::Error GridFdImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                      float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> u,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> qdd) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u(scratch, 3 * kNq * batch);
  if (q_qd_u.err.failure()) return q_qd_u.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grid::forward_dynamics_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          qdd->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq, /*d_f_ext=*/nullptr,
          model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_kernel");
}

// Direct Minv: q -> inverse mass matrix, (B, n, n), SYMMETRIC_UPPER filled.
ffi::Error GridMinvImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> minv) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grid::minv_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          minv->typed_data(), ws.ptr, q.typed_data(), kNq, model, batch);
  return CudaCheck(cudaGetLastError(), "minv_kernel");
}

// Mass matrix M(q): q -> (B, n, n), [t, col, row] fully populated.
// Mass matrix M(q) via GRiD's own generated CRBA kernel (BFS-parallel
// composite-inertia accumulation), rather than the old n-column ID sweep.
// crba_kernel still takes a [q|qd] interleaved input (stride 2n) and a
// gravity scalar even though M(q) does not depend on either; qd is packed
// as zero here to match ID/FD's packing convention with no extra kernel.
ffi::Error GridCrbaImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        float gravity,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> m) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer qd_zero(scratch, kNq * batch);
  if (qd_zero.err.failure()) return qd_zero.err;
  if (auto err = CudaCheck(
          cudaMemsetAsync(qd_zero.ptr, 0, kNq * batch * sizeof(T), stream),
          "crba qd memset");
      err.failure())
    return err;
  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd_zero.ptr, kNq, batch);

  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grid::crba_kernel<T><<<GridDims(batch), ThreadDims(),
                        grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
      m->typed_data(), ws.ptr, q_qd.ptr, 2 * kNq, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "crba_kernel");
}

// Analytic inverse dynamics gradient: (q, qd, qdd) -> dc/d[q,qd],
// (B, 2n, n) with column-major n x 2n per timestep ([dq block | dqd block]).
ffi::Error GridIdGradImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                          float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> qdd,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> dc_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
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
ffi::Error GridFdGradImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                          float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> u,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> df_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u(scratch, 3 * kNq * batch);
  if (q_qd_u.err.failure()) return q_qd_u.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grid::forward_dynamics_gradient_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
         stream>>>(df_du->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq,
                   /*d_f_ext=*/nullptr, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_gradient_kernel");
}

// Second-order inverse dynamics (GRiD idsva_so, body-frame fixed-base
// dispatch): (q, qd, qdd) -> 4 flattened (n,n,n) tensors per timestep,
// concatenated as [d2tau_dq | d2tau_dqd | d2tau_cross | dM_dq], each n^3
// floats (SECOND_ORDER_TENSOR_SIZE = 4*n^3 total). Same [q|qd|qdd] packing
// as GridFdGradImpl (stride 3n = Q_QD_U_STRIDE).
ffi::Error GridIdsvaSoImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                           float gravity,
                           ffi::Buffer<ffi::DataType::F32> q,
                           ffi::Buffer<ffi::DataType::F32> qd,
                           ffi::Buffer<ffi::DataType::F32> qdd,
                           ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grid::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_qdd(scratch, 3 * kNq * batch);
  if (q_qd_qdd.err.failure()) return q_qd_qdd.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_qdd.ptr, q.typed_data(), qd.typed_data(), qdd.typed_data(), kNq,
      batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grid::idsva_so_body_frame_kernel<T>
      <<<GridDims(batch), ThreadDims(),
         grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          out->typed_data(), ws.ptr, q_qd_qdd.ptr, 3 * kNq, model, gravity,
          batch);
  return CudaCheck(cudaGetLastError(), "idsva_so_body_frame_kernel");
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
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridFdFfi, GridFdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridMinvFfi, GridMinvImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridCrbaFfi, GridCrbaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridIdGradFfi, GridIdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridFdGradFfi, GridFdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GridIdsvaSoFfi, GridIdsvaSoImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());
