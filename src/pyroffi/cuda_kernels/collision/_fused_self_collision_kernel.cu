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


#ifdef PYROFFI_TRACED_ROBOT
#include "../_traced_robot.cuh"
// Traced build: FK from cricket's straight-line code (self kernel), the model's sphere and
// pair tables baked in so the pair/link loops are constant, and (world kernel) the scene's
// obstacle counts as constants so per-link minima accumulate in registers.
#define FUSED_FK(b, T)                                                                      \
    do {                                                                                    \
        const float frz_[1] = {0.f};                                                        \
        pyroffi::traced::frame_poses(cfg + (size_t)(b) * pyroffi::traced::n_q, frz_, (T));  \
    } while (0)
#define FUSED_BIND_TABLES()                                                                 \
    sph_local  = pyroffi::traced::kSelfSph;                                                 \
    link_start = pyroffi::traced::kSelfLinkStart;                                           \
    link_joint = pyroffi::traced::kSelfLinkJoint
#else
#define FUSED_FK(b, T)                                                                      \
    fk_single(cfg + (size_t)(b) * n_act, twists, parent_tf, parent_idx, act_idx,            \
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, (T), n_joints, n_act)
#define FUSED_BIND_TABLES() ((void)0)
#endif

// Launch sizing. Each thread keeps its configuration's joint transforms in dynamic shared
// memory (n_joints * 7 floats), so the block is sized to the robot: 64 threads while that
// fits the device's opt-in shared-memory limit (99 KB on sm_86 -- enough for ~55 joints),
// halving beyond it. There is no fixed joint cap; a 43-DOF humanoid is just another robot.
template <typename Kernel>
static ffi::Error fused_launch_config(Kernel kernel, int n_joints, int* threads, size_t* shmem)
{
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    const size_t row = (size_t)n_joints * 7 * sizeof(float);
    int t = 64;
    while (t > 1 && (size_t)t * row > (size_t)optin) t /= 2;
    if (row > (size_t)optin)
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "fused collision: one configuration's joint transforms exceed the "
                          "device's shared memory");
    *threads = t;
    *shmem = (size_t)t * row;
    if (*shmem > 48 * 1024 &&
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)*shmem)
            != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(cudaGetLastError()));
    return ffi::Error::Success();
}

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
    int B, int n_joints, int n_act, int N_links_arg, int P_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    constexpr int N_links = pyroffi::traced::n_self_links, P = pyroffi::traced::n_self_pairs;
    (void)N_links_arg; (void)P_arg;
    pair_i = pyroffi::traced::kSelfPairI;
    pair_j = pyroffi::traced::kSelfPairJ;
#else
    const int N_links = N_links_arg, P = P_arg;
#endif
    FUSED_BIND_TABLES();
    // Link transforms for this thread's configuration, wxyz_xyz per link.
    // Shared rather than register: 7 floats x N links exceeds a sensible
    // register budget, but is small enough that occupancy stays reasonable.
    // fk_single writes per-JOINT transforms, so the buffer is sized by joint
    // count; collision spheres are attached to LINKS, and `link_joint` maps
    // between them (a link is posed by its parent joint's transform).
    // Stride by the ACTUAL joint count; the host sizes the block to fit it
    // (fused_launch_config).
    extern __shared__ float s_T[];
    float* T = s_T + (size_t)threadIdx.x * n_joints * 7;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // --- FK once, in-thread. Reuses pyroffi's tested chain walk rather than
    // reimplementing it, so this kernel cannot drift from the other backends.
    FUSED_FK(b, T);

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
    int B, int n_joints, int n_act, int N_links_arg,
    int n_ws_arg, int n_wc_arg, int n_wb_arg, int n_wh_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    // The scene's obstacle set is fixed, so M is a compile-time constant and each link's
    // per-obstacle minima live in registers, written to global memory once per link instead
    // of read-modify-written once per (sphere, obstacle).
    constexpr int N_links = pyroffi::traced::n_self_links;
    constexpr int n_ws = pyroffi::traced::n_world_spheres, n_wc = pyroffi::traced::n_world_capsules;
    constexpr int n_wb = pyroffi::traced::n_world_boxes, n_wh = pyroffi::traced::n_world_halfspaces;
    (void)N_links_arg; (void)n_ws_arg; (void)n_wc_arg; (void)n_wb_arg; (void)n_wh_arg;
    FUSED_BIND_TABLES();
#else
    const int N_links = N_links_arg;
    const int n_ws = n_ws_arg, n_wc = n_wc_arg, n_wb = n_wb_arg, n_wh = n_wh_arg;
#endif
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
        float* orow_out = out + ((size_t)b * N_links + n) * M;
#ifdef PYROFFI_TRACED_ROBOT
        float orow[M > 0 ? M : 1];
#else
        float* orow = orow_out;
#endif

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
#ifdef PYROFFI_TRACED_ROBOT
        for (int m = 0; m < M; ++m) orow_out[m] = orow[m];
#endif
    }
}


// ---------------------------------------------------------------------------
// Fused FK + ESDF world collision
// ---------------------------------------------------------------------------
// Same structure again, against a dense signed-distance grid instead of primitives: each
// sphere samples the grid trilinearly at its centre (edge-clamped, exactly as
// collision/_esdf.py::esdf_query_jax) and subtracts its radius; the per-link minimum over
// spheres is written once. A voxel grid has a fixed shape and resolution, so a traced build
// with PYROFFI_TRACED_ESDF takes them as compile-time constants (grid values and origin stay
// runtime inputs).
//
// grid [nx, ny, nz] row-major (C order), origin [3] = world position of voxel (0, 0, 0)'s
// centre, voxel = cubic voxel edge. out [B, N_links].

static __device__ __forceinline__ float esdf_trilinear(
    const float* __restrict__ grid, const float* __restrict__ origin, float voxel,
    int nx, int ny, int nz, const float p[3])
{
    const int dims[3] = {nx, ny, nz};
    int i0[3], i1[3];
    float f[3];
    for (int k = 0; k < 3; ++k) {
        float idx = (p[k] - origin[k]) / voxel;
        idx = fminf(fmaxf(idx, 0.f), (float)dims[k] - 1.f - 1e-6f);
        i0[k] = (int)floorf(idx);
        f[k] = idx - (float)i0[k];
        i1[k] = min(i0[k] + 1, dims[k] - 1);
    }
    auto g = [&](int x, int y, int z) { return grid[((size_t)x * ny + y) * nz + z]; };
    const float c00 = g(i0[0], i0[1], i0[2]) * (1.f - f[0]) + g(i1[0], i0[1], i0[2]) * f[0];
    const float c01 = g(i0[0], i0[1], i1[2]) * (1.f - f[0]) + g(i1[0], i0[1], i1[2]) * f[0];
    const float c10 = g(i0[0], i1[1], i0[2]) * (1.f - f[0]) + g(i1[0], i1[1], i0[2]) * f[0];
    const float c11 = g(i0[0], i1[1], i1[2]) * (1.f - f[0]) + g(i1[0], i1[1], i1[2]) * f[0];
    const float c0 = c00 * (1.f - f[1]) + c10 * f[1];
    const float c1 = c01 * (1.f - f[1]) + c11 * f[1];
    return c0 * (1.f - f[2]) + c1 * f[2];
}

static __global__ __launch_bounds__(64, 4)
void fused_world_esdf_kernel(
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
    const float* __restrict__ grid,
    const float* __restrict__ origin,
    float*       __restrict__ out,
    int B, int n_joints, int n_act, int N_links_arg,
    int nx_arg, int ny_arg, int nz_arg, float voxel_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    constexpr int N_links = pyroffi::traced::n_self_links;
    (void)N_links_arg;
    FUSED_BIND_TABLES();
#else
    const int N_links = N_links_arg;
#endif
#ifdef PYROFFI_TRACED_ESDF
    constexpr int nx = pyroffi::traced::esdf_nx, ny = pyroffi::traced::esdf_ny, nz = pyroffi::traced::esdf_nz;
    constexpr float voxel = pyroffi::traced::esdf_voxel;
    (void)nx_arg; (void)ny_arg; (void)nz_arg; (void)voxel_arg;
#else
    const int nx = nx_arg, ny = ny_arg, nz = nz_arg;
    const float voxel = voxel_arg;
#endif
    extern __shared__ float s_Te[];
    float* T = s_Te + (size_t)threadIdx.x * n_joints * 7;

    const float IDENTITY_TF[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    fk_single(cfg + (size_t)b * n_act, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, T, n_joints, n_act);

    const float o[3] = {origin[0], origin[1], origin[2]};
    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF;
        float d = INFINITY;  // links without spheres report +inf, as the JAX path does
        for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
            float c[3];
            apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
            d = fminf(d, esdf_trilinear(grid, o, voxel, nx, ny, nz, c) - sph_local[(size_t)a * 4 + 3]);
        }
        out[(size_t)b * N_links + n] = d;
    }
}


// ---------------------------------------------------------------------------
// Analytic Jacobians of the fused outputs
// ---------------------------------------------------------------------------
// Each kernel below recomputes its op and also writes d(output)/dq, one row of n_act per
// output, so autodiff can take tangents (and, by transposition, cotangents) from the GPU
// instead of re-running the whole computation in JAX. Every output is a minimum over
// spheres, so its derivative is that of the witness sphere (pair): the separation
// direction pushed through the joint chain by collision_point_grad_chain, which carries
// the twist and mimic handling the IK kernels already use. Rows are overwritten.

#define FUSED_CHAIN_ARGS twists, parent_idx, act_idx, mimic_mul, mimic_act_idx

static __global__ __launch_bounds__(64, 4)
void fused_self_collision_jac_kernel(
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
    float*       __restrict__ out,     // [B, P]
    float*       __restrict__ out_jac, // [B, P, n_act]
    float*       __restrict__ min_z,   // [B]
    float*       __restrict__ z_jac,   // [B, n_act]
    int B, int n_joints, int n_act, int N_links_arg, int P_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    constexpr int N_links = pyroffi::traced::n_self_links, P = pyroffi::traced::n_self_pairs;
    (void)N_links_arg; (void)P_arg;
    pair_i = pyroffi::traced::kSelfPairI;
    pair_j = pyroffi::traced::kSelfPairJ;
#else
    const int N_links = N_links_arg, P = P_arg;
#endif
    FUSED_BIND_TABLES();
    extern __shared__ float s_Tj[];
    float* T = s_Tj + (size_t)threadIdx.x * n_joints * 7;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    FUSED_FK(b, T);

    for (int p = 0; p < P; ++p) {
        float* g = out_jac + ((size_t)b * P + p) * n_act;
        for (int a = 0; a < n_act; ++a) g[a] = 0.f;
        out[(size_t)b * P + p] = self_collision_pair_dist_grad(
            T, sph_local, link_start, link_joint, pair_i[p], pair_j[p],
            FUSED_CHAIN_ARGS, n_act, INFINITY, g);
    }

    const float IDENTITY_TF_S[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lowest = INFINITY, c_low[3] = {0.f, 0.f, 0.f};
    int j_low = -1;
    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF_S;
        for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
            float c[3];
            apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
            const float z = c[2] - sph_local[(size_t)a * 4 + 3];
            if (z < lowest) { lowest = z; c_low[0] = c[0]; c_low[1] = c[1]; c_low[2] = c[2]; j_low = jn; }
        }
    }
    min_z[b] = lowest;
    float* gz = z_jac + (size_t)b * n_act;
    for (int a = 0; a < n_act; ++a) gz[a] = 0.f;
    const float up[3] = {0.f, 0.f, 1.f};
    if (j_low >= 0) collision_point_grad_chain(T, FUSED_CHAIN_ARGS, j_low, c_low, up, 1.f, gz);
}

static __global__ __launch_bounds__(64, 4)
void fused_world_collision_jac_kernel(
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
    float*       __restrict__ out,      // [B, N, M]
    float*       __restrict__ out_jac,  // [B, N, M, n_act]
    int B, int n_joints, int n_act, int N_links_arg,
    int n_ws_arg, int n_wc_arg, int n_wb_arg, int n_wh_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    constexpr int N_links = pyroffi::traced::n_self_links;
    constexpr int n_ws = pyroffi::traced::n_world_spheres, n_wc = pyroffi::traced::n_world_capsules;
    constexpr int n_wb = pyroffi::traced::n_world_boxes, n_wh = pyroffi::traced::n_world_halfspaces;
    (void)N_links_arg; (void)n_ws_arg; (void)n_wc_arg; (void)n_wb_arg; (void)n_wh_arg;
    FUSED_BIND_TABLES();
#else
    const int N_links = N_links_arg;
    const int n_ws = n_ws_arg, n_wc = n_wc_arg, n_wb = n_wb_arg, n_wh = n_wh_arg;
#endif
    extern __shared__ float s_Twj[];
    float* T = s_Twj + (size_t)threadIdx.x * n_joints * 7;
    const float IDENTITY_TF[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    fk_single(cfg + (size_t)b * n_act, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, T, n_joints, n_act);

    const float* base[4] = {w_sph, w_cap, w_box, w_hs};
    const int count[4] = {n_ws, n_wc, n_wb, n_wh};
    const int stride[4] = {4, 7, 15, 6};
    const int M = n_ws + n_wc + n_wb + n_wh;
    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF;
        int m = 0;
        for (int kind = 0; kind < 4; ++kind) {
            for (int k = 0; k < count[kind]; ++k, ++m) {
                const float* o = base[kind] + (size_t)k * stride[kind];
                float best = 1e9f, cb[3] = {0.f, 0.f, 0.f}, rb = 0.f;
                for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
                    float c[3];
                    apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
                    const float r = sph_local[(size_t)a * 4 + 3];
                    const float d = world_prim_dist(c, r, kind, o);
                    if (d < best) { best = d; cb[0] = c[0]; cb[1] = c[1]; cb[2] = c[2]; rb = r; }
                }
                const size_t row = ((size_t)b * N_links + n) * M + m;
                out[row] = best;
                float* g = out_jac + row * n_act;
                for (int a = 0; a < n_act; ++a) g[a] = 0.f;
                if (jn < 0 || best >= 1e9f) continue;
                // Closed-form directions where they exist (float32 central differences leave
                // ~3e-4 in the unit vector); capsules and boxes keep the shared FD helper.
                float u[3];
                bool ok = true;
                if (kind == kWorldSphere) {
                    const float dx = cb[0] - o[0], dy = cb[1] - o[1], dz = cb[2] - o[2];
                    const float nrm = sqrtf(dx * dx + dy * dy + dz * dz);
                    ok = nrm > 1e-9f;
                    if (ok) { u[0] = dx / nrm; u[1] = dy / nrm; u[2] = dz / nrm; }
                } else if (kind == kWorldHalfspace) {
                    u[0] = o[0]; u[1] = o[1]; u[2] = o[2];
                } else {
                    ok = world_prim_grad_dir(cb, rb, kind, o, u);
                }
                if (ok) collision_point_grad_chain(T, FUSED_CHAIN_ARGS, jn, cb, u, 1.f, g);
            }
        }
    }
}

// d/dp of esdf_trilinear, matching jax.grad of collision/_esdf.py::esdf_query_jax: the clamp
// has zero derivative outside the grid and floor has none, so inside a cell it is the
// trilinear partials over the voxel size.
static __device__ __forceinline__ float esdf_trilinear_grad(
    const float* __restrict__ grid, const float* __restrict__ origin, float voxel,
    int nx, int ny, int nz, const float p[3], float g[3])
{
    const int dims[3] = {nx, ny, nz};
    int i0[3], i1[3];
    float f[3], inside[3];
    for (int k = 0; k < 3; ++k) {
        const float raw = (p[k] - origin[k]) / voxel;
        const float hi = (float)dims[k] - 1.f - 1e-6f;
        const float idx = fminf(fmaxf(raw, 0.f), hi);
        inside[k] = (raw >= 0.f && raw <= hi) ? 1.f : 0.f;
        i0[k] = (int)floorf(idx);
        f[k] = idx - (float)i0[k];
        i1[k] = min(i0[k] + 1, dims[k] - 1);
    }
    auto G = [&](int x, int y, int z) { return grid[((size_t)x * ny + y) * nz + z]; };
    const float c000 = G(i0[0], i0[1], i0[2]), c100 = G(i1[0], i0[1], i0[2]);
    const float c010 = G(i0[0], i1[1], i0[2]), c110 = G(i1[0], i1[1], i0[2]);
    const float c001 = G(i0[0], i0[1], i1[2]), c101 = G(i1[0], i0[1], i1[2]);
    const float c011 = G(i0[0], i1[1], i1[2]), c111 = G(i1[0], i1[1], i1[2]);
    const float fx = f[0], fy = f[1], fz = f[2];
    const float c00 = c000 * (1.f - fx) + c100 * fx, c01 = c001 * (1.f - fx) + c101 * fx;
    const float c10 = c010 * (1.f - fx) + c110 * fx, c11 = c011 * (1.f - fx) + c111 * fx;
    const float c0 = c00 * (1.f - fy) + c10 * fy, c1 = c01 * (1.f - fy) + c11 * fy;
    const float dx0 = (c100 - c000) * (1.f - fy) + (c110 - c010) * fy;
    const float dx1 = (c101 - c001) * (1.f - fy) + (c111 - c011) * fy;
    g[0] = (dx0 * (1.f - fz) + dx1 * fz) * inside[0] / voxel;
    g[1] = ((c10 - c00) * (1.f - fz) + (c11 - c01) * fz) * inside[1] / voxel;
    g[2] = (c1 - c0) * inside[2] / voxel;
    return c0 * (1.f - fz) + c1 * fz;
}

static __global__ __launch_bounds__(64, 4)
void fused_world_esdf_jac_kernel(
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
    const float* __restrict__ grid,
    const float* __restrict__ origin,
    float*       __restrict__ out,      // [B, N]
    float*       __restrict__ out_jac,  // [B, N, n_act]
    int B, int n_joints, int n_act, int N_links_arg,
    int nx_arg, int ny_arg, int nz_arg, float voxel_arg)
{
#ifdef PYROFFI_TRACED_ROBOT
    constexpr int N_links = pyroffi::traced::n_self_links;
    (void)N_links_arg;
    FUSED_BIND_TABLES();
#else
    const int N_links = N_links_arg;
#endif
#ifdef PYROFFI_TRACED_ESDF
    constexpr int nx = pyroffi::traced::esdf_nx, ny = pyroffi::traced::esdf_ny, nz = pyroffi::traced::esdf_nz;
    constexpr float voxel = pyroffi::traced::esdf_voxel;
    (void)nx_arg; (void)ny_arg; (void)nz_arg; (void)voxel_arg;
#else
    const int nx = nx_arg, ny = ny_arg, nz = nz_arg;
    const float voxel = voxel_arg;
#endif
    extern __shared__ float s_Tej[];
    float* T = s_Tej + (size_t)threadIdx.x * n_joints * 7;
    const float IDENTITY_TF[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    fk_single(cfg + (size_t)b * n_act, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv, T, n_joints, n_act);

    const float o[3] = {origin[0], origin[1], origin[2]};
    for (int n = 0; n < N_links; ++n) {
        const int jn = link_joint[n];
        const float* Tn = (jn >= 0) ? T + (size_t)jn * 7 : IDENTITY_TF;
        float best = INFINITY, gb[3] = {0.f, 0.f, 0.f}, cb[3] = {0.f, 0.f, 0.f};
        for (int a = link_start[n]; a < link_start[n + 1]; ++a) {
            float c[3], gc[3];
            apply_se3_point(Tn, sph_local + (size_t)a * 4, c);
            const float d = esdf_trilinear_grad(grid, o, voxel, nx, ny, nz, c, gc) - sph_local[(size_t)a * 4 + 3];
            if (d < best) {
                best = d;
                gb[0] = gc[0]; gb[1] = gc[1]; gb[2] = gc[2];
                cb[0] = c[0]; cb[1] = c[1]; cb[2] = c[2];
            }
        }
        out[(size_t)b * N_links + n] = best;
        float* g = out_jac + ((size_t)b * N_links + n) * n_act;
        for (int a = 0; a < n_act; ++a) g[a] = 0.f;
        // The ESDF gradient is not unit length (it is the field's own slope), so push it
        // through the chain as a direction with its magnitude kept.
        if (jn >= 0 && best < INFINITY) collision_point_grad_chain(T, FUSED_CHAIN_ARGS, jn, cb, gb, 1.f, g);
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

#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links || P != pyroffi::traced::n_self_pairs)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_self_collision (traced): launch does not match the robot and "
                          "collision model this build was traced for.");
#endif

    int threads = 0;
    size_t shmem = 0;
    if (ffi::Error cfg_err = fused_launch_config(fused_self_collision_kernel, n_joints, &threads, &shmem);
        cfg_err.failure())
        return cfg_err;
    const int blocks = (B + threads - 1) / threads;

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
#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links || n_ws != pyroffi::traced::n_world_spheres ||
        n_wc != pyroffi::traced::n_world_capsules || n_wb != pyroffi::traced::n_world_boxes ||
        n_wh != pyroffi::traced::n_world_halfspaces)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_collision (traced): launch does not match the robot and "
                          "scene structure this build was traced for.");
#endif

    int threads = 0;
    size_t shmem = 0;
    if (ffi::Error cfg_err = fused_launch_config(fused_world_collision_kernel, n_joints, &threads, &shmem);
        cfg_err.failure())
        return cfg_err;
    const int blocks = (B + threads - 1) / threads;

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

// ---------------------------------------------------------------------------

static ffi::Error FusedWorldEsdfImpl(
    cudaStream_t stream,
    float voxel,
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
    ffi::Buffer<ffi::DataType::F32> grid,
    ffi::Buffer<ffi::DataType::F32> origin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out)
{
    const auto d = cfg.dimensions();
    const auto g = grid.dimensions();
    if (d.size() != 2 || g.size() != 3 || origin.element_count() != 3)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_esdf: cfg must be [B, n_act], grid [nx, ny, nz], origin [3]");

    const int B = static_cast<int>(d[0]);
    const int n_act = static_cast<int>(d[1]);
    const int n_joints = static_cast<int>(parent_idx.dimensions()[0]);
    const int N_links = static_cast<int>(link_start.dimensions()[0]) - 1;
    const int nx = static_cast<int>(g[0]), ny = static_cast<int>(g[1]), nz = static_cast<int>(g[2]);
#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_esdf (traced): launch does not match the robot and "
                          "collision model this build was traced for.");
#endif
#ifdef PYROFFI_TRACED_ESDF
    if (nx != pyroffi::traced::esdf_nx || ny != pyroffi::traced::esdf_ny ||
        nz != pyroffi::traced::esdf_nz || voxel != pyroffi::traced::esdf_voxel)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_esdf (traced): grid shape or voxel size differs from the "
                          "one this build was traced for.");
#endif

    int threads = 0;
    size_t shmem = 0;
    if (ffi::Error cfg_err = fused_launch_config(fused_world_esdf_kernel, n_joints, &threads, &shmem);
        cfg_err.failure())
        return cfg_err;
    const int blocks = (B + threads - 1) / threads;

    fused_world_esdf_kernel<<<blocks, threads, shmem, stream>>>(
        cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(), mimic_mul.typed_data(),
        mimic_off.typed_data(), mimic_act_idx.typed_data(),
        topo_inv.typed_data(), sph_local.typed_data(),
        link_start.typed_data(), link_joint.typed_data(),
        grid.typed_data(), origin.typed_data(), out->typed_data(),
        B, n_joints, n_act, N_links, nx, ny, nz, voxel);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedWorldEsdfFfi, FusedWorldEsdfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("voxel")
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
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // grid [nx, ny, nz]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // origin [3]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out [B, N]
);

// ---------------------------------------------------------------------------
// Jacobian variants: the same launches with an extra [.., n_act] output per value.

#define FUSED_ROBOT_ARGS_DECL                                                                   \
    ffi::Buffer<ffi::DataType::F32> cfg, ffi::Buffer<ffi::DataType::F32> twists,               \
    ffi::Buffer<ffi::DataType::F32> parent_tf, ffi::Buffer<ffi::DataType::S32> parent_idx,     \
    ffi::Buffer<ffi::DataType::S32> act_idx, ffi::Buffer<ffi::DataType::F32> mimic_mul,        \
    ffi::Buffer<ffi::DataType::F32> mimic_off, ffi::Buffer<ffi::DataType::S32> mimic_act_idx,  \
    ffi::Buffer<ffi::DataType::S32> topo_inv, ffi::Buffer<ffi::DataType::F32> sph_local,       \
    ffi::Buffer<ffi::DataType::S32> link_start, ffi::Buffer<ffi::DataType::S32> link_joint
#define FUSED_ROBOT_ARGS_PASS                                                                   \
    cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(), parent_idx.typed_data(),    \
    act_idx.typed_data(), mimic_mul.typed_data(), mimic_off.typed_data(),                      \
    mimic_act_idx.typed_data(), topo_inv.typed_data(), sph_local.typed_data(),                 \
    link_start.typed_data(), link_joint.typed_data()
#define FUSED_ROBOT_ARGS_BIND                                                                   \
    .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()              \
    .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::S32>>()              \
    .Arg<ffi::Buffer<ffi::DataType::S32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()              \
    .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::S32>>()              \
    .Arg<ffi::Buffer<ffi::DataType::S32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()              \
    .Arg<ffi::Buffer<ffi::DataType::S32>>().Arg<ffi::Buffer<ffi::DataType::S32>>()

template <typename Kernel, typename Launch>
static ffi::Error fused_launch(Kernel kernel, int B, int n_joints, Launch launch)
{
    int threads = 0;
    size_t shmem = 0;
    if (ffi::Error err = fused_launch_config(kernel, n_joints, &threads, &shmem); err.failure())
        return err;
    launch((B + threads - 1) / threads, threads, shmem);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

static ffi::Error FusedSelfCollisionJacImpl(
    cudaStream_t stream, FUSED_ROBOT_ARGS_DECL,
    ffi::Buffer<ffi::DataType::S32> pair_i, ffi::Buffer<ffi::DataType::S32> pair_j,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_jac,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> min_z, ffi::Result<ffi::Buffer<ffi::DataType::F32>> z_jac)
{
    const int B = (int)cfg.dimensions()[0], n_act = (int)cfg.dimensions()[1];
    const int n_joints = (int)parent_idx.dimensions()[0];
    const int N_links = (int)link_start.dimensions()[0] - 1, P = (int)pair_i.dimensions()[0];
#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links || P != pyroffi::traced::n_self_pairs)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_self_collision_jac (traced): launch does not match the traced model.");
#endif
    return fused_launch(fused_self_collision_jac_kernel, B, n_joints, [&](int blocks, int threads, size_t shmem) {
        fused_self_collision_jac_kernel<<<blocks, threads, shmem, stream>>>(
            FUSED_ROBOT_ARGS_PASS, pair_i.typed_data(), pair_j.typed_data(), out->typed_data(),
            out_jac->typed_data(), min_z->typed_data(), z_jac->typed_data(), B, n_joints, n_act, N_links, P);
    });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedSelfCollisionJacFfi, FusedSelfCollisionJacImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() FUSED_ROBOT_ARGS_BIND
        .Arg<ffi::Buffer<ffi::DataType::S32>>().Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>().Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>().Ret<ffi::Buffer<ffi::DataType::F32>>());

static ffi::Error FusedWorldCollisionJacImpl(
    cudaStream_t stream, FUSED_ROBOT_ARGS_DECL,
    ffi::Buffer<ffi::DataType::F32> w_sph, ffi::Buffer<ffi::DataType::F32> w_cap,
    ffi::Buffer<ffi::DataType::F32> w_box, ffi::Buffer<ffi::DataType::F32> w_hs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_jac)
{
    const int B = (int)cfg.dimensions()[0], n_act = (int)cfg.dimensions()[1];
    const int n_joints = (int)parent_idx.dimensions()[0], N_links = (int)link_start.dimensions()[0] - 1;
    const int n_ws = (int)w_sph.dimensions()[0], n_wc = (int)w_cap.dimensions()[0];
    const int n_wb = (int)w_box.dimensions()[0], n_wh = (int)w_hs.dimensions()[0];
#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links || n_ws != pyroffi::traced::n_world_spheres ||
        n_wc != pyroffi::traced::n_world_capsules || n_wb != pyroffi::traced::n_world_boxes ||
        n_wh != pyroffi::traced::n_world_halfspaces)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_collision_jac (traced): launch does not match the traced scene.");
#endif
    return fused_launch(fused_world_collision_jac_kernel, B, n_joints, [&](int blocks, int threads, size_t shmem) {
        fused_world_collision_jac_kernel<<<blocks, threads, shmem, stream>>>(
            FUSED_ROBOT_ARGS_PASS, w_sph.typed_data(), w_cap.typed_data(), w_box.typed_data(),
            w_hs.typed_data(), out->typed_data(), out_jac->typed_data(),
            B, n_joints, n_act, N_links, n_ws, n_wc, n_wb, n_wh);
    });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedWorldCollisionJacFfi, FusedWorldCollisionJacImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() FUSED_ROBOT_ARGS_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>().Ret<ffi::Buffer<ffi::DataType::F32>>());

static ffi::Error FusedWorldEsdfJacImpl(
    cudaStream_t stream, float voxel, FUSED_ROBOT_ARGS_DECL,
    ffi::Buffer<ffi::DataType::F32> grid, ffi::Buffer<ffi::DataType::F32> origin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_jac)
{
    const auto g = grid.dimensions();
    if (g.size() != 3 || origin.element_count() != 3)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "fused_world_esdf_jac: grid [nx, ny, nz], origin [3]");
    const int B = (int)cfg.dimensions()[0], n_act = (int)cfg.dimensions()[1];
    const int n_joints = (int)parent_idx.dimensions()[0], N_links = (int)link_start.dimensions()[0] - 1;
    const int nx = (int)g[0], ny = (int)g[1], nz = (int)g[2];
#ifdef PYROFFI_TRACED_ROBOT
    if (n_act != pyroffi::traced::n_q || n_joints != pyroffi::traced::n_frames ||
        N_links != pyroffi::traced::n_self_links)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_esdf_jac (traced): launch does not match the traced model.");
#endif
#ifdef PYROFFI_TRACED_ESDF
    if (nx != pyroffi::traced::esdf_nx || ny != pyroffi::traced::esdf_ny ||
        nz != pyroffi::traced::esdf_nz || voxel != pyroffi::traced::esdf_voxel)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "fused_world_esdf_jac (traced): grid shape or voxel size differs from the traced one.");
#endif
    return fused_launch(fused_world_esdf_jac_kernel, B, n_joints, [&](int blocks, int threads, size_t shmem) {
        fused_world_esdf_jac_kernel<<<blocks, threads, shmem, stream>>>(
            FUSED_ROBOT_ARGS_PASS, grid.typed_data(), origin.typed_data(), out->typed_data(),
            out_jac->typed_data(), B, n_joints, n_act, N_links, nx, ny, nz, voxel);
    });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedWorldEsdfJacFfi, FusedWorldEsdfJacImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<float>("voxel") FUSED_ROBOT_ARGS_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>().Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>().Ret<ffi::Buffer<ffi::DataType::F32>>());
