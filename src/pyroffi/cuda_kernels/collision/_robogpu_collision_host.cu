/**
 * RoboGPU: GPU-accelerated sphere-octree collision checking via NVIDIA OptiX.
 *
 * Implements the RoboGPU architecture (arXiv:2603.01517) adapted for
 * sphere-based robot representations used in pyroffi.  OBB-AABB SAT is
 * replaced by sphere-sphere occupancy queries against an OptiX BVH built from
 * the environment point cloud, retaining the massive parallelism and early-
 * exit benefits of the original design.
 *
 *   Stage 1 (CUDA kernel — robogpu_prepare_kernel):
 *     FK → link world transforms → transform robot collision spheres to world
 *     frame → check against regular world geometry (spheres/capsules/boxes/
 *     halfspaces) → self-collision check.  Outputs: world-frame robot spheres
 *     [B*K, 4] and per-config free flags [B].
 *
 *   Stage 2 (OptiX — same CUDA stream, no host sync needed):
 *     For each config still marked free, fires K "query rays" (one per robot
 *     sphere) into the env sphere BVH.  The custom intersection program does
 *     the sphere-sphere proximity test; any-hit terminates BVH traversal on
 *     the first hit (early exit at both the per-env-sphere and per-robot-sphere
 *     levels).
 *
 * Build: bash build_kernels/build_robogpu_collision.sh
 *
 * Requires NVIDIA OptiX SDK 7.x (headers; runtime loaded via optixInit()).
 */

// optix_function_table_definition.h must appear in exactly one TU — it provides
// the storage for the OptiX function table that optixInit() populates.
// optix_stubs.h provides the thin wrapper functions (optixInit, optixLaunch, …).
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "xla/ffi/api/ffi.h"
#include "_collision_cuda_helpers.cuh"  // includes _fk_cuda_helpers.cuh; all dist prims + fk_single

#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>     // dladdr (Linux)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Multi-GPU safety
//
// Under jax.pmap each visible GPU is driven by its own host thread running this
// handler concurrently. The OptiX pipeline, BVH cache, and scratch buffer are
// all bound to a specific CUDA device/context, so they must be kept per-device
// and indexed by the current device ordinal — sharing them across devices would
// launch one device's pipeline/BVH against another device's stream and memory.
// ---------------------------------------------------------------------------

static constexpr int PYROFFI_MAX_GPUS = 16;

// Current CUDA device ordinal, or -1 if it exceeds PYROFFI_MAX_GPUS / errors.
static int robogpu_current_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    if (dev < 0 || dev >= PYROFFI_MAX_GPUS) return -1;
    return dev;
}

// ---------------------------------------------------------------------------
// Error-checking macros (return ffi::Error on failure)
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            return ffi::Error(ffi::ErrorCode::kInternal,                    \
                              cudaGetErrorString(_e));                      \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_VOID(call)                                               \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess)                                              \
            fprintf(stderr, "CUDA %s:%d  %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
    } while (0)

#define OPTIX_CHECK(call)                                                   \
    do {                                                                    \
        OptixResult _r = (call);                                            \
        if (_r != OPTIX_SUCCESS) {                                          \
            return ffi::Error(ffi::ErrorCode::kInternal,                    \
                              "OptiX call failed (code=" +                  \
                              std::to_string((int)_r) + ")");               \
        }                                                                   \
    } while (0)

#define OPTIX_CHECK_VOID(call)                                              \
    do {                                                                    \
        OptixResult _r = (call);                                            \
        if (_r != OPTIX_SUCCESS)                                            \
            fprintf(stderr, "OptiX %s:%d  code=%d\n",                      \
                    __FILE__, __LINE__, (int)_r);                           \
    } while (0)

// ---------------------------------------------------------------------------
// Shared data structures (must match _robogpu_optix_programs.cu exactly)
// ---------------------------------------------------------------------------

struct RoboGPULaunchParams {
    OptixTraversableHandle handle;
    const float4*          robot_spheres; // [B * K, 4] world-frame
    int32_t*               out_free;      // [B] 1=free, 0=collision (in/out)
    int                    B;
    int                    K;
};

struct HitGroupData {
    const float4* env_spheres; // [Mp, 4] (cx, cy, cz, r_env)
};

// SBT record wrappers — must be OPTIX_SBT_RECORD_ALIGNMENT-aligned.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
    char         header[OPTIX_SBT_RECORD_HEADER_SIZE];
    HitGroupData data;
};

// ---------------------------------------------------------------------------
// CUDA prepare kernel constants
// ---------------------------------------------------------------------------

#define RGB_MAX_JOINTS 64
#define RGB_MAX_LINKS  64
#define RGB_THREADS    64

// ---------------------------------------------------------------------------
// World-geometry hit test (mirrors _collision_binary_cuda_kernel.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool sphere_world_hit(
    float px, float py, float pz, float r,
    const float* __restrict__ ws, int Ms,
    const float* __restrict__ wc, int Mc,
    const float* __restrict__ wb, int Mb,
    const float* __restrict__ wh, int Mh)
{
    for (int i = 0; i < Ms; i++) {
        const float* o = ws + i * 4;
        if (sphere_sphere_dist(px, py, pz, r, o[0], o[1], o[2], o[3]) < 0.0f) return true;
    }
    for (int i = 0; i < Mc; i++) {
        const float* o = wc + i * 7;
        if (sphere_capsule_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5], o[6]) < 0.0f) return true;
    }
    for (int i = 0; i < Mb; i++) {
        const float* o = wb + i * 15;
        if (sphere_box_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8],
                o[9], o[10], o[11], o[12], o[13], o[14]) < 0.0f) return true;
    }
    for (int i = 0; i < Mh; i++) {
        const float* o = wh + i * 6;
        if (sphere_halfspace_dist(px, py, pz, r,
                o[0], o[1], o[2], o[3], o[4], o[5]) < 0.0f) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Point-cloud → sphere + AABB conversion kernel
//
// For each env point p_i:  sphere = (p_i, r_env),  AABB expanded by r_total
// (= r_env + r_robot_max) so that any robot sphere centre within collision
// range of p_i falls inside the AABB and triggers BVH traversal.
// ---------------------------------------------------------------------------

__global__ void build_env_spheres_kernel(
    const float* __restrict__ pc,      // [Mp, 3]
    float4*      __restrict__ sph,     // [Mp, 4] → env sphere (x,y,z,r_env)
    OptixAabb*   __restrict__ aabb,    // [Mp]
    float r_env, float r_total,        // r_total = r_env + r_robot_max
    int Mp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Mp) return;
    const float x = pc[i * 3 + 0];
    const float y = pc[i * 3 + 1];
    const float z = pc[i * 3 + 2];
    sph[i]  = make_float4(x, y, z, r_env);
    aabb[i] = { x - r_total, y - r_total, z - r_total,
                x + r_total, y + r_total, z + r_total };
}

// ---------------------------------------------------------------------------
// Stage-1 CUDA kernel: FK + sphere transform + world geometry + self-collision
//
// One block per configuration (blockIdx.x = b).  RGB_THREADS threads cooperate
// over the K robot spheres and Pf self-collision pairs.
//
// Outputs:
//   robot_spheres_world[b*K .. b*K+K-1]  — world-frame (x,y,z,r) per sphere
//   out_free[b]                           — 1 if free, 0 if collision
// ---------------------------------------------------------------------------

__global__ void robogpu_prepare_kernel(
    const float* __restrict__ cfg,               // [B, n_act]
    const float* __restrict__ twists,            // [J, 6]
    const float* __restrict__ parent_tf,         // [J, 7]
    const int*   __restrict__ parent_idx,        // [J]
    const int*   __restrict__ act_idx,           // [J]
    const float* __restrict__ mimic_mul,         // [J]
    const float* __restrict__ mimic_off,         // [J]
    const int*   __restrict__ mimic_act_idx,     // [J]
    const int*   __restrict__ topo_inv,          // [J]
    const int*   __restrict__ link_parent_joint, // [NL]
    const float* __restrict__ f_local,           // [K, 4]  k = s*NL + n
    const int*   __restrict__ f_pair_i,          // [Pf]
    const int*   __restrict__ f_pair_j,          // [Pf]
    const float* __restrict__ ws, int Ms,
    const float* __restrict__ wc, int Mc,
    const float* __restrict__ wb, int Mb,
    const float* __restrict__ wh, int Mh,
    float4*      __restrict__ robot_spheres_world, // [B*K, 4]  output
    int*         __restrict__ out_free,             // [B]        output
    int B, int n_act, int J, int NL, int K, int Pf)
{
    const int b  = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x; // == RGB_THREADS

    __shared__ float         Tw[RGB_MAX_JOINTS * 7]; // joint world transforms
    __shared__ float         Tl[RGB_MAX_LINKS  * 7]; // link  world transforms
    __shared__ volatile int  cc;                      // collision flag

    const int NL_cap = min(NL, RGB_MAX_LINKS);
    // Spheres per link: layout is k = s*NL + n  →  link n has Sf spheres
    const int Sf = (NL > 0) ? (K / NL) : 0;

    // ── FK (thread 0 walks the topological order) ────────────────────────────
    if (tid == 0) {
        fk_single(cfg + (long long)b * n_act,
                  twists, parent_tf, parent_idx, act_idx,
                  mimic_mul, mimic_off, mimic_act_idx, topo_inv,
                  Tw, J, n_act);
        cc = 0;
    }
    __syncthreads();

    // ── Compute per-link world transforms from joint transforms ──────────────
    for (int l = tid; l < NL_cap; l += nt) {
        const int pj = link_parent_joint[l];
        float* dst = Tl + l * 7;
        if (pj < 0) {
            dst[0]=1.f; dst[1]=dst[2]=dst[3]=dst[4]=dst[5]=dst[6]=0.f;
        } else {
            #pragma unroll
            for (int i = 0; i < 7; i++) dst[i] = Tw[pj * 7 + i];
        }
    }
    __syncthreads();

    // ── Transform all robot spheres to world frame; store in global mem ──────
    // We write all spheres even if a collision is found, so that OptiX's
    // raygen can use the buffer (it skips configs where out_free[b]=0 anyway).
    float4* base = robot_spheres_world + (long long)b * K;
    for (int k = tid; k < K; k += nt) {
        const int n = k % NL;
        const float* lp = f_local + k * 4;
        if (lp[3] < 0.0f || n >= NL_cap) {
            base[k] = make_float4(0.f, 0.f, 0.f, -1.f); // padding
            continue;
        }
        float p[3] = { lp[0], lp[1], lp[2] }, w[3];
        apply_se3_point(Tl + n * 7, p, w);
        base[k] = make_float4(w[0], w[1], w[2], lp[3]);
    }
    __syncthreads();

    // ── World geometry check (early exit via shared cc flag) ─────────────────
    for (int k = tid; k < K; k += nt) {
        if (cc) break;
        const float4 s = base[k];
        if (s.w < 0.0f) continue;
        if (sphere_world_hit(s.x, s.y, s.z, s.w,
                             ws, Ms, wc, Mc, wb, Mb, wh, Mh))
            cc = 1;
    }
    __syncthreads();
    if (cc) { if (tid == 0) out_free[b] = 0; return; }

    // ── Self-collision check (fine sphere pairs) ─────────────────────────────
    for (int p_idx = tid; p_idx < Pf; p_idx += nt) {
        if (cc) break;
        const int li = f_pair_i[p_idx];
        const int lj = f_pair_j[p_idx];
        for (int si = 0; si < Sf && !cc; ++si) {
            const int ki = si * NL + li;
            if (ki >= K) continue;
            const float4 wi = base[ki];
            if (wi.w < 0.0f) continue;
            for (int sj = 0; sj < Sf && !cc; ++sj) {
                const int kj = sj * NL + lj;
                if (kj >= K) continue;
                const float4 wj = base[kj];
                if (wj.w < 0.0f) continue;
                if (sphere_sphere_dist(wi.x, wi.y, wi.z, wi.w,
                                       wj.x, wj.y, wj.z, wj.w) < 0.0f)
                    cc = 1;
            }
        }
    }
    __syncthreads();

    if (tid == 0) out_free[b] = (cc ? 0 : 1);
}

// ---------------------------------------------------------------------------
// OptiX pipeline (process-lifetime singleton)
// ---------------------------------------------------------------------------

struct OptiXPipeline {
    OptixDeviceContext ctx       = nullptr;
    OptixModule        module    = nullptr;
    OptixProgramGroup  pg_rg    = nullptr;  // raygen
    OptixProgramGroup  pg_ms    = nullptr;  // miss
    OptixProgramGroup  pg_hg    = nullptr;  // hit group
    OptixPipeline      pipeline = nullptr;
    bool               ready    = false;
};

static OptiXPipeline g_pipe[PYROFFI_MAX_GPUS];
static std::mutex    g_pipe_mtx[PYROFFI_MAX_GPUS];

// ---------------------------------------------------------------------------
// Locate the PTX file alongside the running shared library
// ---------------------------------------------------------------------------

static std::string ptx_file_path() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&ptx_file_path), &info)
            && info.dli_fname) {
        std::string path(info.dli_fname);
        auto slash = path.rfind('/');
        std::string dir = (slash == std::string::npos) ? "." : path.substr(0, slash);
        return dir + "/_robogpu_optix_programs.ptx";
    }
    return "_robogpu_optix_programs.ptx";
}

// ---------------------------------------------------------------------------
// Initialise OptiX pipeline (idempotent, mutex-protected)
// ---------------------------------------------------------------------------

static ffi::Error ensure_optix_pipeline(int dev) {
    OptiXPipeline& gp = g_pipe[dev];
    std::lock_guard<std::mutex> lk(g_pipe_mtx[dev]);
    if (gp.ready) return ffi::Error::Success();

    OPTIX_CHECK(optixInit());

    CUcontext cu_ctx = nullptr;
    OptixDeviceContextOptions ctx_opts = {};
    ctx_opts.logCallbackLevel = 1; // errors only
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_opts, &gp.ctx));

    // Load PTX from file.
    std::string ptx_path = ptx_file_path();
    std::ifstream ifs(ptx_path, std::ios::binary);
    if (!ifs)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "RoboGPU: cannot open PTX: " + ptx_path);
    std::ostringstream ss; ss << ifs.rdbuf();
    std::string ptx = ss.str();

    OptixModuleCompileOptions mco = {};
    mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues                 = 2; // p0=robot_r, p1=hit_flag
    pco.numAttributeValues               = 0;
    pco.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    char log[2048]; size_t lsz = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(gp.ctx, &mco, &pco,
                                  ptx.c_str(), ptx.size(),
                                  log, &lsz, &gp.module));

    OptixProgramGroupOptions pgo = {};

    // Raygen
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = gp.module;
        d.raygen.entryFunctionName = "__raygen__sphere_query";
        OPTIX_CHECK(optixProgramGroupCreate(gp.ctx, &d, 1, &pgo,
                                            log, &lsz, &gp.pg_rg));
    }
    // Miss
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module = gp.module;
        d.miss.entryFunctionName = "__miss__sphere";
        OPTIX_CHECK(optixProgramGroupCreate(gp.ctx, &d, 1, &pgo,
                                            log, &lsz, &gp.pg_ms));
    }
    // Hit group (intersection + any-hit; no closest-hit)
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleIS            = gp.module;
        d.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        d.hitgroup.moduleAH            = gp.module;
        d.hitgroup.entryFunctionNameAH = "__anyhit__sphere";
        OPTIX_CHECK(optixProgramGroupCreate(gp.ctx, &d, 1, &pgo,
                                            log, &lsz, &gp.pg_hg));
    }

    OptixProgramGroup pgs[] = { gp.pg_rg, gp.pg_ms, gp.pg_hg };
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OPTIX_CHECK(optixPipelineCreate(gp.ctx, &pco, &plo,
                                    pgs, 3, log, &lsz, &gp.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(gp.pipeline,
        2048, 2048, 2048, 1));

    gp.ready = true;
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// BVH cache entry (one per unique point cloud + r_env + r_robot_max)
// ---------------------------------------------------------------------------

struct BVHEntry {
    CUdeviceptr            d_gas = 0;       // GAS output buffer (see `updatable`)
    size_t                 d_gas_size = 0;
    OptixTraversableHandle handle = {};

    CUdeviceptr d_env_spheres = 0;  // [Mp, 4] float4 on device
    CUdeviceptr d_aabbs       = 0;  // [Mp] OptixAabb on device
    int         Mp = 0;

    // Dynamic-refit support.  When `updatable` is true the GAS was built with
    // OPTIX_BUILD_FLAG_ALLOW_UPDATE and is *not* compacted (a compacted GAS
    // cannot be refit), so `d_gas` is the raw build-output buffer that
    // optixAccelBuild(UPDATE) rewrites in place.  `d_update_tmp` is the scratch
    // buffer reused across refits, and the radii are kept so a refit can rerun
    // build_env_spheres_kernel with the same AABB expansion.
    bool        updatable        = false;
    CUdeviceptr d_update_tmp      = 0;
    size_t      d_update_tmp_size = 0;
    float       r_env       = 0.f;
    float       r_robot_max = 0.f;

    CUdeviceptr d_launch_params = 0; // per-call (updated each call)

    // SBT device buffers
    CUdeviceptr d_sbt_rg = 0;
    CUdeviceptr d_sbt_ms = 0;
    CUdeviceptr d_sbt_hg = 0;
    OptixShaderBindingTable sbt = {};
};

static std::unordered_map<std::string, BVHEntry*> g_bvh_cache[PYROFFI_MAX_GPUS];
static std::mutex                                   g_bvh_mtx[PYROFFI_MAX_GPUS];

// ---------------------------------------------------------------------------
// Build (or return cached) BVH for a given point cloud
// ---------------------------------------------------------------------------

// Fill the OptixBuildInput for a custom-primitive (AABB) GAS from an entry.
// `geo_flags` and `aabb_ptr` must outlive the returned struct (OptiX reads
// through the pointers), so the caller owns them.
static OptixBuildInput make_aabb_build_input(
    const BVHEntry* e, unsigned int& geo_flags, CUdeviceptr& aabb_ptr)
{
    geo_flags = OPTIX_GEOMETRY_FLAG_NONE;
    aabb_ptr  = e->d_aabbs;
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers   = &aabb_ptr;
    bi.customPrimitiveArray.numPrimitives = static_cast<unsigned>(e->Mp);
    bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    bi.customPrimitiveArray.flags         = &geo_flags;
    bi.customPrimitiveArray.numSbtRecords = 1;
    return bi;
}

// Free every device buffer owned by an entry and the entry itself.
static void destroy_bvh(BVHEntry* e) {
    if (!e) return;
    if (e->d_gas)           cudaFree(reinterpret_cast<void*>(e->d_gas));
    if (e->d_env_spheres)   cudaFree(reinterpret_cast<void*>(e->d_env_spheres));
    if (e->d_aabbs)         cudaFree(reinterpret_cast<void*>(e->d_aabbs));
    if (e->d_update_tmp)    cudaFree(reinterpret_cast<void*>(e->d_update_tmp));
    if (e->d_launch_params) cudaFree(reinterpret_cast<void*>(e->d_launch_params));
    if (e->d_sbt_rg)        cudaFree(reinterpret_cast<void*>(e->d_sbt_rg));
    if (e->d_sbt_ms)        cudaFree(reinterpret_cast<void*>(e->d_sbt_ms));
    if (e->d_sbt_hg)        cudaFree(reinterpret_cast<void*>(e->d_sbt_hg));
    delete e;
}

// Refit (OPTIX_BUILD_OPERATION_UPDATE) an existing updatable GAS in place.
//
// Reuses the BVH topology and only refits node AABBs to the new point cloud —
// far cheaper than a rebuild and, crucially, fully stream-asynchronous: there
// is no compacted-size readback, so no cudaStreamSynchronize is needed.  The
// env-sphere buffer is rewritten in place, so the SBT (which points at it) is
// unchanged.
static bool refit_bvh(cudaStream_t stream, const float* d_pc, BVHEntry* e, int dev) {
    OptiXPipeline& gp = g_pipe[dev];

    const int blk = 256;
    build_env_spheres_kernel<<<(e->Mp + blk - 1) / blk, blk, 0, stream>>>(
        d_pc,
        reinterpret_cast<float4*>(e->d_env_spheres),
        reinterpret_cast<OptixAabb*>(e->d_aabbs),
        e->r_env, e->r_env + e->r_robot_max, e->Mp);

    unsigned int geo_flags; CUdeviceptr aabb_ptr;
    OptixBuildInput bi = make_aabb_build_input(e, geo_flags, aabb_ptr);

    OptixAccelBuildOptions abo = {};
    abo.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE
                   | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    abo.operation  = OPTIX_BUILD_OPERATION_UPDATE;

    // In-place update: source and output GAS buffer are the same.
    OptixResult r = optixAccelBuild(gp.ctx, stream, &abo, &bi, 1,
                                    e->d_update_tmp, e->d_update_tmp_size,
                                    e->d_gas, e->d_gas_size,
                                    &e->handle, nullptr, 0);
    if (r != OPTIX_SUCCESS) {
        fprintf(stderr, "RoboGPU refit failed (code=%d)\n", (int)r);
        return false;
    }
    return true;
}

static BVHEntry* build_bvh(
    cudaStream_t stream,
    const float* d_pc,    // [Mp, 3] device pointer (from JAX buffer)
    int  Mp,
    float r_env,
    float r_robot_max,
    int  dev,
    bool updatable)
{
    OptiXPipeline& gp = g_pipe[dev];
    BVHEntry* e = new BVHEntry{};
    e->Mp          = Mp;
    e->updatable   = updatable;
    e->r_env       = r_env;
    e->r_robot_max = r_robot_max;

    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_env_spheres),
                               (size_t)Mp * sizeof(float4)));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_aabbs),
                               (size_t)Mp * sizeof(OptixAabb)));

    const int blk = 256;
    build_env_spheres_kernel<<<(Mp + blk - 1) / blk, blk, 0, stream>>>(
        d_pc,
        reinterpret_cast<float4*>(e->d_env_spheres),
        reinterpret_cast<OptixAabb*>(e->d_aabbs),
        r_env, r_env + r_robot_max, Mp);

    CUDA_CHECK_VOID(cudaStreamSynchronize(stream));

    // GAS build
    unsigned int geo_flags; CUdeviceptr aabb_ptr;
    OptixBuildInput bi = make_aabb_build_input(e, geo_flags, aabb_ptr);

    OptixAccelBuildOptions abo = {};
    // An updatable GAS is kept uncompacted so it can be refit in place; a
    // static GAS is compacted to minimise traversal memory.
    abo.buildFlags = (updatable ? OPTIX_BUILD_FLAG_ALLOW_UPDATE
                                : OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
                   | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    abo.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bs = {};
    OPTIX_CHECK_VOID(optixAccelComputeMemoryUsage(gp.ctx, &abo, &bi, 1, &bs));

    CUdeviceptr d_tmp = 0, d_out = 0;
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_tmp), bs.tempSizeInBytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_out), bs.outputSizeInBytes));

    if (updatable) {
        // No compaction: keep the build-output buffer as the live GAS and a
        // persistent refit-scratch buffer for subsequent UPDATE operations.
        e->d_gas             = d_out;
        e->d_gas_size        = bs.outputSizeInBytes;
        e->d_update_tmp_size = bs.tempUpdateSizeInBytes;
        CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_update_tmp),
                                   e->d_update_tmp_size));

        OPTIX_CHECK_VOID(optixAccelBuild(gp.ctx, stream, &abo, &bi, 1,
                                         d_tmp, bs.tempSizeInBytes,
                                         d_out, bs.outputSizeInBytes,
                                         &e->handle, nullptr, 0));
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream));
        CUDA_CHECK_VOID(cudaFree(reinterpret_cast<void*>(d_tmp)));
    } else {
        CUdeviceptr d_compact_sz = 0;
        CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_compact_sz), sizeof(size_t)));

        OptixAccelEmitDesc emit = {};
        emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = d_compact_sz;

        OptixTraversableHandle raw_handle = {};
        OPTIX_CHECK_VOID(optixAccelBuild(gp.ctx, stream, &abo,
                                         &bi, 1,
                                         d_tmp, bs.tempSizeInBytes,
                                         d_out, bs.outputSizeInBytes,
                                         &raw_handle, &emit, 1));
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream));

        size_t compact_sz = 0;
        CUDA_CHECK_VOID(cudaMemcpy(&compact_sz,
                                   reinterpret_cast<void*>(d_compact_sz),
                                   sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_gas), compact_sz));
        e->d_gas_size = compact_sz;
        OPTIX_CHECK_VOID(optixAccelCompact(gp.ctx, stream, raw_handle,
                                           e->d_gas, compact_sz, &e->handle));
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream));

        CUDA_CHECK_VOID(cudaFree(reinterpret_cast<void*>(d_tmp)));
        CUDA_CHECK_VOID(cudaFree(reinterpret_cast<void*>(d_out)));
        CUDA_CHECK_VOID(cudaFree(reinterpret_cast<void*>(d_compact_sz)));
    }

    // Build SBT
    RaygenRecord   rg_rec = {};
    MissRecord     ms_rec = {};
    HitGroupRecord hg_rec = {};
    OPTIX_CHECK_VOID(optixSbtRecordPackHeader(gp.pg_rg, &rg_rec));
    OPTIX_CHECK_VOID(optixSbtRecordPackHeader(gp.pg_ms, &ms_rec));
    OPTIX_CHECK_VOID(optixSbtRecordPackHeader(gp.pg_hg, &hg_rec));
    hg_rec.data.env_spheres = reinterpret_cast<const float4*>(e->d_env_spheres);

    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_sbt_rg), sizeof(rg_rec)));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_sbt_ms), sizeof(ms_rec)));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_sbt_hg), sizeof(hg_rec)));
    CUDA_CHECK_VOID(cudaMemcpy(reinterpret_cast<void*>(e->d_sbt_rg), &rg_rec, sizeof(rg_rec), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(reinterpret_cast<void*>(e->d_sbt_ms), &ms_rec, sizeof(ms_rec), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(reinterpret_cast<void*>(e->d_sbt_hg), &hg_rec, sizeof(hg_rec), cudaMemcpyHostToDevice));

    e->sbt.raygenRecord                = e->d_sbt_rg;
    e->sbt.missRecordBase              = e->d_sbt_ms;
    e->sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    e->sbt.missRecordCount             = 1;
    e->sbt.hitgroupRecordBase          = e->d_sbt_hg;
    e->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    e->sbt.hitgroupRecordCount         = 1;

    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&e->d_launch_params),
                               sizeof(RoboGPULaunchParams)));
    return e;
}

static BVHEntry* get_or_build_bvh(
    cudaStream_t stream,
    const float* d_pc, int Mp,
    float r_env, float r_robot_max,
    const std::string& key,
    int dev)
{
    {
        std::lock_guard<std::mutex> lk(g_bvh_mtx[dev]);
        auto it = g_bvh_cache[dev].find(key);
        if (it != g_bvh_cache[dev].end()) return it->second;
    }
    BVHEntry* e = build_bvh(stream, d_pc, Mp, r_env, r_robot_max, dev,
                            /*updatable=*/false);
    if (e) {
        std::lock_guard<std::mutex> lk(g_bvh_mtx[dev]);
        g_bvh_cache[dev].emplace(key, e);
    }
    return e;
}

// ---------------------------------------------------------------------------
// Dynamic BVH: a single updatable GAS per device, refit on every call.
//
// Intended for streaming point clouds (e.g. a depth camera) where the cloud
// changes every frame but its size and the query radii stay fixed.  Instead of
// rebuilding (and recompacting) the BVH each frame — which the pointer-keyed
// static cache would do, since every frame is a fresh JAX buffer — we keep one
// persistent updatable GAS and refit it in place.  The refit reuses the
// existing tree topology, so the per-frame cost drops to an AABB refit and is
// fully asynchronous on the stream.
//
// If the point count or radii change, the topology assumption breaks, so we
// tear the GAS down and rebuild it (the next frame refits the new tree).
// ---------------------------------------------------------------------------

static BVHEntry* g_dyn_bvh[PYROFFI_MAX_GPUS] = {};

static BVHEntry* get_or_update_dynamic_bvh(
    cudaStream_t stream,
    const float* d_pc, int Mp,
    float r_env, float r_robot_max,
    int dev)
{
    std::lock_guard<std::mutex> lk(g_bvh_mtx[dev]);
    BVHEntry* e = g_dyn_bvh[dev];

    const bool reusable = e && e->updatable && e->Mp == Mp
                       && e->r_env == r_env && e->r_robot_max == r_robot_max;
    if (reusable) {
        refit_bvh(stream, d_pc, e, dev);
        return e;
    }

    if (e) { destroy_bvh(e); g_dyn_bvh[dev] = nullptr; }
    e = build_bvh(stream, d_pc, Mp, r_env, r_robot_max, dev, /*updatable=*/true);
    g_dyn_bvh[dev] = e;
    return e;
}

// ---------------------------------------------------------------------------
// Persistent scratch buffer for the world-frame robot spheres [B*K, 4].
//
// Allocating this per call with cudaMallocAsync forces synchronization against
// JAX's own caching GPU allocator (they manage separate pools), adding a fixed
// ~0.5 ms stall to every check.  Instead we keep one buffer that grows
// monotonically and is reused across calls.  JAX serialises FFI calls on a
// single stream, so sequential reuse is safe; the mutex guards the rare grow.
// ---------------------------------------------------------------------------

struct ScratchBuffer {
    float4* ptr      = nullptr;
    size_t  capacity = 0;   // in float4 elements
};
static ScratchBuffer g_scratch[PYROFFI_MAX_GPUS];
static std::mutex    g_scratch_mtx[PYROFFI_MAX_GPUS];

// Returns a device buffer of at least `n` float4 elements (nullptr on failure).
static float4* get_scratch(size_t n, int dev) {
    ScratchBuffer& sc = g_scratch[dev];
    std::lock_guard<std::mutex> lk(g_scratch_mtx[dev]);
    if (n <= sc.capacity) return sc.ptr;
    if (sc.ptr) cudaFree(sc.ptr);
    // Over-allocate (1.5x) to amortise growth across increasing batch sizes.
    size_t want = n + n / 2;
    if (cudaMalloc(reinterpret_cast<void**>(&sc.ptr),
                   want * sizeof(float4)) != cudaSuccess) {
        sc.ptr = nullptr;
        sc.capacity = 0;
        return nullptr;
    }
    sc.capacity = want;
    return sc.ptr;
}

// ---------------------------------------------------------------------------
// XLA FFI implementation
// ---------------------------------------------------------------------------

static ffi::Error RoboGPUCheckImpl(
    cudaStream_t                    stream,
    ffi::Buffer<ffi::DataType::F32> cfg,               // [B, n_act]
    ffi::Buffer<ffi::DataType::F32> twists,            // [J, 6]
    ffi::Buffer<ffi::DataType::F32> parent_tf,         // [J, 7]
    ffi::Buffer<ffi::DataType::S32> parent_idx,        // [J]
    ffi::Buffer<ffi::DataType::S32> act_idx,           // [J]
    ffi::Buffer<ffi::DataType::F32> mimic_mul,         // [J]
    ffi::Buffer<ffi::DataType::F32> mimic_off,         // [J]
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,     // [J]
    ffi::Buffer<ffi::DataType::S32> topo_inv,          // [J]
    ffi::Buffer<ffi::DataType::S32> link_parent_joint, // [NL]
    ffi::Buffer<ffi::DataType::F32> f_local,           // [K, 4]
    ffi::Buffer<ffi::DataType::S32> f_pair_i,          // [Pf]
    ffi::Buffer<ffi::DataType::S32> f_pair_j,          // [Pf]
    ffi::Buffer<ffi::DataType::F32> world_spheres,     // [Ms, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,    // [Mc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,       // [Mb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,  // [Mh, 6]
    ffi::Buffer<ffi::DataType::F32> point_cloud,       // [Mp, 3]
    float r_env,
    float r_robot_max,
    int32_t dynamic,                                   // 1 = refit, 0 = rebuild
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out   // [B]
) {
    const int B     = static_cast<int>(cfg.dimensions()[0]);
    const int n_act = static_cast<int>(cfg.dimensions()[1]);
    const int J     = static_cast<int>(twists.dimensions()[0]);
    const int NL    = static_cast<int>(link_parent_joint.dimensions()[0]);
    const int K     = static_cast<int>(f_local.dimensions()[0]);
    const int Pf    = static_cast<int>(f_pair_i.dimensions()[0]);
    const int Ms    = static_cast<int>(world_spheres.dimensions()[0]);
    const int Mc    = static_cast<int>(world_capsules.dimensions()[0]);
    const int Mb    = static_cast<int>(world_boxes.dimensions()[0]);
    const int Mh    = static_cast<int>(world_halfspaces.dimensions()[0]);
    const int Mp    = static_cast<int>(point_cloud.dimensions()[0]);

    if (B <= 0) return ffi::Error::Success();

    const int dev = robogpu_current_device();
    if (dev < 0)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "RoboGPU: failed to query CUDA device, or device "
                          "ordinal exceeds PYROFFI_MAX_GPUS (rebuild with a "
                          "larger limit).");

    if (J > RGB_MAX_JOINTS || NL > RGB_MAX_LINKS)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "RoboGPU: J or NL exceeds compile-time bounds "
                          "(RGB_MAX_JOINTS=" + std::to_string(RGB_MAX_JOINTS) +
                          ", RGB_MAX_LINKS=" + std::to_string(RGB_MAX_LINKS) +
                          "); rebuild with larger values.");

    // ── Stage 1: CUDA prepare (FK + world geom + self-collision) ─────────────

    float4* d_spheres = get_scratch(static_cast<size_t>(B) * K, dev);
    if (!d_spheres)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "RoboGPU: scratch allocation failed");

    robogpu_prepare_kernel<<<B, RGB_THREADS, 0, stream>>>(
        cfg.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(),
        mimic_mul.typed_data(), mimic_off.typed_data(),
        mimic_act_idx.typed_data(), topo_inv.typed_data(),
        link_parent_joint.typed_data(), f_local.typed_data(),
        f_pair_i.typed_data(), f_pair_j.typed_data(),
        world_spheres.typed_data(), Ms,
        world_capsules.typed_data(), Mc,
        world_boxes.typed_data(), Mb,
        world_halfspaces.typed_data(), Mh,
        d_spheres, out->typed_data(),
        B, n_act, J, NL, K, Pf);

    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // ── Stage 2: OptiX BVH traversal for point cloud ─────────────────────────
    if (Mp > 0) {
        {
            auto err = ensure_optix_pipeline(dev);
            if (err.failure()) return err;
        }

        // Key the BVH cache on the point-cloud *device pointer* (+ Mp + radii).
        // The Python checker captures the point cloud as a constant in its jitted
        // closure, so the buffer — and hence this pointer — is stable across
        // calls that reuse the same cloud, and a different cloud yields a
        // different buffer.  This avoids the per-call D2H copy + stream sync that
        // a content hash would require, keeping the whole check asynchronous.
        char keybuf[96];
        snprintf(keybuf, sizeof(keybuf), "%p_%d_%g_%g",
                 reinterpret_cast<const void*>(point_cloud.typed_data()),
                 Mp, (double)r_env, (double)r_robot_max);
        std::string key(keybuf);

        BVHEntry* bvh = dynamic
            ? get_or_update_dynamic_bvh(stream, point_cloud.typed_data(), Mp,
                                        r_env, r_robot_max, dev)
            : get_or_build_bvh(stream, point_cloud.typed_data(), Mp,
                               r_env, r_robot_max, key, dev);
        if (!bvh)
            return ffi::Error(ffi::ErrorCode::kInternal, "RoboGPU: BVH build failed");

        // Update launch params (stream-ordered ahead of optixLaunch).
        RoboGPULaunchParams hp = {};
        hp.handle        = bvh->handle;
        hp.robot_spheres = d_spheres;
        hp.out_free      = out->typed_data();
        hp.B             = B;
        hp.K             = K;
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(bvh->d_launch_params),
                                   &hp, sizeof(hp),
                                   cudaMemcpyHostToDevice, stream));

        // OptiX launch — one raygen thread per config.
        OPTIX_CHECK(optixLaunch(g_pipe[dev].pipeline, stream,
                                bvh->d_launch_params, sizeof(hp),
                                &bvh->sbt,
                                static_cast<unsigned>(B), 1, 1));
    }

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// XLA FFI handler symbol (loaded by Python via ctypes.CDLL)
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RoboGPUCollisionFfi, RoboGPUCheckImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // link_parent_joint
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // f_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // f_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // f_pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // point_cloud [Mp, 3]
        .Attr<float>("r_env")
        .Attr<float>("r_robot_max")
        .Attr<int32_t>("dynamic")                 // 1=refit BVH, 0=rebuild/cache
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // out [B]
