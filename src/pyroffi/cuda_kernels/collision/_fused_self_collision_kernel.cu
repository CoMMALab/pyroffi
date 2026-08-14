/**
 * Fused FK + self-collision kernel.
 *
 * Computes self-collision distances directly from joint configurations in ONE
 * launch, with the forward kinematics never leaving the thread. The existing
 * path instead runs FK as a separate XLA op, materialises a padded
 * [B, S, N, 3] sphere-position tensor to global memory, and then reads it back
 * in the collision kernel.
 *
 * Why fuse
 * --------
 * Three separate optimisations of the *existing* kernel were measured and all
 * were near-null: compacting the pair enumeration (13,608 -> 450 evaluations)
 * gave 1.35x, switching to a compact float4 sphere layout gave 1.00x, and
 * eliminating the duplicate FK caps out around 1.5x since FK is ~51% of the
 * cost and is real work. Meanwhile the collision arithmetic in isolation runs
 * at 27.8M configs/s against 1.5M for the full JAX path -- an 18x gap that sits
 * in neither the arithmetic nor the layout.
 *
 * What is left is the round trip: XLA materialises intermediates between ops,
 * so every sphere position is written to global memory and read back. Fusing
 * removes that traffic entirely -- link transforms stay in shared memory,
 * sphere positions are formed in registers on demand and never stored.
 *
 * Memory strategy
 * ---------------
 * Storing all K sphere positions per thread is not viable: 59 spheres x 4
 * floats is 236 registers, past the 255/thread limit, and in shared memory it
 * caps occupancy hard. Instead only the N link transforms are kept (7 floats
 * each as quaternion + translation = 364 B/config), and a sphere is transformed
 * from its link-local pose at the moment it is needed. That trades arithmetic
 * for memory traffic, which is the correct direction here precisely because the
 * measurements show this workload is not arithmetic-bound.
 *
 * Thread tier
 * -----------
 * One thread per configuration. FK is a sequential chain walk with nothing to
 * parallelise inside it, and at pyroffi's batch sizes the batch dimension
 * already supplies all the parallelism needed -- consistent with the tier
 * selection finding that thread-level wins at large batch even at high DOF.
 * GLASS's block/warp reductions therefore do not apply. The thread-level
 * geometry comes from pyroffi's own `_collision_cuda_helpers.cuh` rather than
 * GLASS's `base/geom/sphere.cuh`: it carries the wider primitive vocabulary
 * (sphere/capsule/box/half-space in every combination, plus SDF margin
 * handling) and is already shared by ls_ik, hjcd_ik, sqp_ik and the analytic-IK
 * kernel, so this kernel stays consistent with the rest of the suite and gains
 * capsule/box world geometry for free when that is added. `apply_se3_point`
 * also consumes exactly the [wxyz_xyz] layout `fk_single` emits.
 *
 * Build with:  bash build_kernels/build_fused_self_collision_cuda.sh
 */

#include "../_fk_cuda_helpers.cuh"
#include "../_collision_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cfloat>
#include <cmath>

namespace ffi = xla::ffi;

#ifndef FUSED_MAX_LINKS
#define FUSED_MAX_LINKS 32
#endif

/**
 * One thread per configuration.
 *
 * cfg           [B, n_act]
 * sph_local     [K, 4]   link-local (x, y, z, r), grouped by link
 * link_start    [N + 1]  CSR offsets into the per-link sphere runs
 * pair_i/pair_j [P]      active self-collision link pairs
 * out           [B, P]   minimum signed distance per pair
 * min_z         [B]      lowest point on any sphere, min over K of (z - r)
 *
 * ``min_z`` exists so a floor-clearance test does not need a second FK. The
 * caller's alternative is a separate FK pass in JAX purely to place spheres and
 * reduce their lowest point, which measured 3.47 ms against this kernel's total
 * 1.93 ms at B=61440 -- the redundant FK cost more than the collision check it
 * accompanied. Here the transforms are already in shared memory and every
 * sphere is already being walked, so the extra reduction is close to free.
 */
static __global__ __launch_bounds__(64, 4)
void fused_self_collision_kernel(
    const float* __restrict__ cfg,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    float*       __restrict__ out,
    float*       __restrict__ min_z,
    int B, int n_joints, int n_act, int N_links, int P)
{
    // Link transforms for this thread's configuration, wxyz_xyz per link.
    // Shared rather than register: 7 floats x N links exceeds a sensible
    // register budget, but is small enough that occupancy stays reasonable.
    // fk_single writes per-JOINT transforms, so the buffer is sized by joint
    // count; collision spheres are attached to LINKS, and `link_joint` maps
    // between them (a link is posed by its parent joint's transform).
    // Stride by the ACTUAL joint count, not FUSED_MAX_LINKS. Sizing by the
    // compile-time bound requested 64 x 32 x 7 x 4 = 56 KB, past the 48 KB
    // shared-memory limit, and the launch failed with a bare "invalid
    // argument" rather than anything naming shared memory.
    extern __shared__ float s_T[];
    float* T = s_T + (size_t)threadIdx.x * n_joints * 7;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // --- FK once, in-thread. Reuses pyroffi's tested chain walk rather than
    // reimplementing it, so this kernel cannot drift from the other backends.
    fk_single(cfg + (size_t)b * n_act, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, T, n_joints, n_act);

    // --- Self-collision via the shared helper, so this kernel and the IK
    // solvers evaluate byte-identical geometry.
    for (int p = 0; p < P; ++p)
        out[(size_t)b * P + p] = self_collision_pair_dist(
            T, sph_local, link_start, link_joint, pair_i[p], pair_j[p]);

    // --- Lowest sphere point, from the transforms already in shared memory.
    // Links with no spheres contribute nothing (their CSR run is empty), and a
    // model with no spheres at all leaves +inf, which no floor test rejects.
    const float IDENTITY_TF_S[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lowest = INFINITY;
    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF_S;
        for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
            float c[3];
            apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
            lowest = fminf(lowest, c[2] - sph_local[(size_t)a * 4 + 3]);
        }
    }
    min_z[b] = lowest;
}

// ---------------------------------------------------------------------------
// Fused FK + WORLD collision
// ---------------------------------------------------------------------------
// Same structure as the self-collision kernel: FK once per thread, link
// transforms in shared memory, sphere positions formed in registers and never
// stored. Only the inner comparison changes -- robot spheres against world
// primitives instead of against each other.
//
// The robot side is spheres-only by construction (RobotCollisionSpherized *is*
// a sphere model), so only sphere-vs-X is ever needed. All four world types are
// supported because `_collision_cuda_helpers.cuh` already provides them; the
// world buffers use the same row layouts every other CUDA IK kernel takes, so a
// world built for ls_ik/hjcd_ik/sqp_ik works here unchanged:
//   spheres    (Ms, 4)   capsules (Mc, 7)
//   boxes      (Mb, 15)  halfspaces (Mh, 6)
//
// Output is [B, N, M] with M = Ms + Mc + Mb + Mh in that order, matching
// `compute_world_collision_distance`'s (link x object) contract. Per link the
// value is the MINIMUM over that link's spheres -- note the JAX docstring says
// "maximum", but `collide_link_vs_world` reduces with `.min`, and min is the
// correct conservative choice for a distance field.

static __global__ __launch_bounds__(64, 4)
void fused_world_collision_kernel(
    const float* __restrict__ cfg,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sph_local,
    const int*   __restrict__ link_start,
    const int*   __restrict__ link_joint,
    const float* __restrict__ w_sph,
    const float* __restrict__ w_cap,
    const float* __restrict__ w_box,
    const float* __restrict__ w_hs,
    float*       __restrict__ out,
    int B, int n_joints, int n_act, int N_links,
    int n_ws, int n_wc, int n_wb, int n_wh)
{
    extern __shared__ float s_Tw[];
    float* T = s_Tw + (size_t)threadIdx.x * n_joints * 7;

    const float IDENTITY_TF[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    fk_single(cfg + (size_t)b * n_act, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, T, n_joints, n_act);

    const int M = n_ws + n_wc + n_wb + n_wh;

    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF;
        float* orow = out + ((size_t)b * N_links + n) * M;

        for (int m = 0; m < M; ++m) orow[m] = 1e9f;

        for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
            float c[3];
            apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
            const float r = sph_local[(size_t)a * 4 + 3];
            int m = 0;

            for (int k = 0; k < n_ws; ++k, ++m) {
                const float* o = w_sph + k * 4;
                orow[m] = fminf(orow[m], sphere_sphere_dist(
                    c[0], c[1], c[2], r, o[0], o[1], o[2], o[3]));
            }
            for (int k = 0; k < n_wc; ++k, ++m) {
                const float* o = w_cap + k * 7;
                orow[m] = fminf(orow[m], sphere_capsule_dist(
                    c[0], c[1], c[2], r, o[0], o[1], o[2], o[3], o[4], o[5], o[6]));
            }
            for (int k = 0; k < n_wb; ++k, ++m) {
                const float* o = w_box + k * 15;
                orow[m] = fminf(orow[m], sphere_box_dist(
                    c[0], c[1], c[2], r, o[0], o[1], o[2], o[3], o[4], o[5],
                    o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14]));
            }
            for (int k = 0; k < n_wh; ++k, ++m) {
                const float* o = w_hs + k * 6;
                orow[m] = fminf(orow[m], sphere_halfspace_dist(
                    c[0], c[1], c[2], r, o[0], o[1], o[2], o[3], o[4], o[5]));
            }
        }
    }
}

// ---------------------------------------------------------------------------

static ffi::Error FusedSelfCollisionImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfg,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::F32> sph_local,
    ffi::Buffer<ffi::DataType::S32> link_start,
    ffi::Buffer<ffi::DataType::S32> link_joint,
    ffi::Buffer<ffi::DataType::S32> pair_i,
    ffi::Buffer<ffi::DataType::S32> pair_j,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> min_z)
{
    const auto d = cfg.dimensions();
    if (d.size() != 2)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_self_collision: cfg must be [B, n_act]");

    const int B = static_cast<int>(d[0]);
    const int n_act = static_cast<int>(d[1]);
    const int n_joints = static_cast<int>(parent_idx.dimensions()[0]);
    const int N_links = static_cast<int>(link_start.dimensions()[0]) - 1;
    const int P = static_cast<int>(pair_i.dimensions()[0]);

    if (n_joints > FUSED_MAX_LINKS)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_self_collision: link count exceeds "
                          "FUSED_MAX_LINKS; rebuild with a larger bound");

    const int threads = 64;
    const int blocks = (B + threads - 1) / threads;
    const size_t shmem = (size_t)threads * n_joints * 7 * sizeof(float);
    if (shmem > 48 * 1024)
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "fused_self_collision: shared memory exceeds 48KB; "
                          "reduce the block size or joint count");

    fused_self_collision_kernel<<<blocks, threads, shmem, stream>>>(
        cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(), mimic_mul.typed_data(),
        mimic_off.typed_data(), mimic_act_idx.typed_data(),
        topo_inv.typed_data(), sph_local.typed_data(),
        link_start.typed_data(), link_joint.typed_data(),
        pair_i.typed_data(), pair_j.typed_data(),
        out->typed_data(), min_z->typed_data(),
        B, n_joints, n_act, N_links, P);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedSelfCollisionFfi, FusedSelfCollisionImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sph_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // link_start
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // link_joint
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_j
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out   [B, P]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // min_z [B]
);

// ---------------------------------------------------------------------------

static ffi::Error FusedWorldCollisionImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfg,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::F32> sph_local,
    ffi::Buffer<ffi::DataType::S32> link_start,
    ffi::Buffer<ffi::DataType::S32> link_joint,
    ffi::Buffer<ffi::DataType::F32> w_sph,
    ffi::Buffer<ffi::DataType::F32> w_cap,
    ffi::Buffer<ffi::DataType::F32> w_box,
    ffi::Buffer<ffi::DataType::F32> w_hs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out)
{
    const auto d = cfg.dimensions();
    if (d.size() != 2)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_collision: cfg must be [B, n_act]");

    const int B = static_cast<int>(d[0]);
    const int n_act = static_cast<int>(d[1]);
    const int n_joints = static_cast<int>(parent_idx.dimensions()[0]);
    const int N_links = static_cast<int>(link_start.dimensions()[0]) - 1;
    const int n_ws = static_cast<int>(w_sph.dimensions()[0]);
    const int n_wc = static_cast<int>(w_cap.dimensions()[0]);
    const int n_wb = static_cast<int>(w_box.dimensions()[0]);
    const int n_wh = static_cast<int>(w_hs.dimensions()[0]);

    const int threads = 64;
    const int blocks = (B + threads - 1) / threads;
    const size_t shmem = (size_t)threads * n_joints * 7 * sizeof(float);
    if (shmem > 48 * 1024)
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "fused_world_collision: shared memory exceeds 48KB");

    fused_world_collision_kernel<<<blocks, threads, shmem, stream>>>(
        cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(), mimic_mul.typed_data(),
        mimic_off.typed_data(), mimic_act_idx.typed_data(),
        topo_inv.typed_data(), sph_local.typed_data(),
        link_start.typed_data(), link_joint.typed_data(),
        w_sph.typed_data(), w_cap.typed_data(), w_box.typed_data(),
        w_hs.typed_data(), out->typed_data(),
        B, n_joints, n_act, N_links, n_ws, n_wc, n_wb, n_wh);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedWorldCollisionFfi, FusedWorldCollisionImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sph_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // link_start
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // link_joint
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world halfspaces
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out [B, N, M]
);
