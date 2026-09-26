/**
 * JAX FFI binding for the PBA+ 3D Euclidean Distance Transform
 * (pyroffi faithful-V2 ESDF port, Piece 2 -- see pba3d_kernel.cuh for the
 * ported device kernels and pyroffi.collision._esdf for how this plugs into
 * the seed -> propagate -> distance pipeline as a selectable
 * edt_solver="pba" alternative to the pure-JAX edt_solver="jfa" baseline).
 *
 * Forward-only custom call, deliberately with NO jax.custom_jvp anywhere on
 * the Python side wrapping it -- see pyroffi.collision._esdf's module
 * docstring for the full reasoning (grid construction has no meaningful
 * gradient w.r.t. sdf_grid by construction: distance magnitude comes from
 * an integer nearest-site index, sign comes from a step function; verified
 * empirically that jax.grad through the plain composite needs no bridging
 * at all, unlike CUDADifferentiableSDFCollisionChecker's kernels which DO
 * carry a real float tangent across the FFI boundary).
 *
 * Deliberately NOT CUDA-graph-cached (unlike _collision_cuda_kernel.cu's
 * world/self kernels) -- this is the "baseline (v1), get it correct first"
 * pass; graph-caching is a later optimization if this becomes a hot path,
 * matching the project's own tiered approach.
 *
 * Interface: one FFI target, pba_propagate_sites, taking a packed int32
 * site-index grid [nx, ny, nz] (see site_encoding.cuh -- empty voxels are
 * negative int32, i.e. bit 31 set) and returning the same grid after the
 * 5-launch PBA propagation. Packing seed coordinates into that format and
 * unpacking the result into (site_xyz, has_site) -- matching
 * jfa_propagate_jax's return contract -- is pure-JAX glue, not CUDA (see
 * pyroffi.collision._esdf.pba_propagate_jax).
 *
 * Build: bash build_kernels/build_pba_cuda.sh
 */

#include "xla/ffi/api/ffi.h"
#include "pba3d_kernel.cuh"

namespace ffi = xla::ffi;
using namespace pyroffi::pba;

namespace {

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

// ── PBA propagate: packed site-index grid in -> propagated grid out ─────────
//
// Two Results are declared: `scratch` (an internal ping-pong workspace,
// XLA-allocated so we don't need a raw cudaMalloc/cudaFree per call, value
// discarded by the Python wrapper) and `out` (the final propagated grid).
// The 5-launch sequence always ends in the *second* ping-pong buffer (see
// the phase-by-phase comments below), so `out` is wired to receive it
// directly -- no extra copy needed.

static ffi::Error PbaPropagateSitesImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> site_index_in,      // [nx, ny, nz] packed
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> scratch, // workspace, discarded
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out,      // [nx, ny, nz] result
    int64_t nx, int64_t ny, int64_t nz, int64_t m3)
{
    const int NX = static_cast<int>(nx);
    const int NY = static_cast<int>(ny);
    const int NZ = static_cast<int>(nz);
    const int M3 = static_cast<int>(m3);

    if (NX <= 0 || NY <= 0 || NZ <= 0)
        return ffi::Error::Success();
    if (NX > 1024 || NY > 1024 || NZ > 1024)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "PBA site coordinates are packed into 10 bits each (site_encoding.cuh) "
            "-- every grid dimension must be <= 1024.");

    const size_t n_voxels = static_cast<size_t>(NX) * NY * NZ;

    int32_t* buf_a = scratch->typed_data();
    int32_t* buf_b = out->typed_data();

    cudaError_t e = cudaMemcpyAsync(
        buf_a, site_index_in.typed_data(), n_voxels * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

    // PBA axis mapping (matches cuRobo): sx=nz, sy=ny, sz=nx -- see
    // pba3d_kernel.cuh's "Grid convention" note for why this requires no
    // data transposition for our own [nx,ny,nz] row-major storage.
    const int sx = NZ, sy = NY, sz = NX;

    const dim3 flood_block(32, 4);
    const dim3 flood_grid(cdiv(sx, 32), cdiv(sy, 4));
    const dim3 maurer_block(32, 4);
    const dim3 color_block(32, M3);

    // Phase 1: Flood Z          (buf_a -> buf_b)
    kernel_flood_z<<<flood_grid, flood_block, 0, stream>>>(buf_a, buf_b, sx, sy, sz);

    // Phase 2a: Maurer Y        (buf_b -> buf_a, as a stack)
    const dim3 maurer_grid_2(cdiv(sx, 32), cdiv(sz, 4));
    kernel_maurer_axis<<<maurer_grid_2, maurer_block, 0, stream>>>(buf_b, buf_a, sx, sy, sz);

    // Phase 2b: Color Y + transpose   (buf_a -> buf_b, output dims: sy,sx,sz)
    const dim3 color_grid_2(cdiv(sx, 32), sz);
    kernel_color_axis<<<color_grid_2, color_block, 0, stream>>>(buf_a, buf_b, sx, sy, sz);

    // Phase 3a: Maurer on transposed data   (buf_b -> buf_a)
    const dim3 maurer_grid_3(cdiv(sy, 32), cdiv(sz, 4));
    kernel_maurer_axis<<<maurer_grid_3, maurer_block, 0, stream>>>(buf_b, buf_a, sy, sx, sz);

    // Phase 3b: Color + transpose back   (buf_a -> buf_b, restores original layout)
    const dim3 color_grid_3(cdiv(sy, 32), sz);
    kernel_color_axis<<<color_grid_3, color_block, 0, stream>>>(buf_a, buf_b, sy, sx, sz);

    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));

    // Final result is already in buf_b == out->typed_data() -- no copy needed.
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PbaPropagateSitesFfi, PbaPropagateSitesImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // site_index_in [nx,ny,nz]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // scratch (discarded)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out     [nx,ny,nz]
        .Attr<int64_t>("nx")
        .Attr<int64_t>("ny")
        .Attr<int64_t>("nz")
        .Attr<int64_t>("m3"));
