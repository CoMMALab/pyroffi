/**
 * OptiX device programs for RoboGPU sphere-octree collision checking.
 *
 * Compiled to PTX (NOT linked into the host .so):
 *   nvcc --ptx -arch=sm_XX -I${OPTIX_SDK}/include \
 *        -o _robogpu_optix_programs.ptx _robogpu_optix_programs.cu
 *
 * The environment point cloud is represented as a BVH of spheres (one per
 * point, radius = r_env expanded by r_robot_max for AABB coverage).  Robot
 * collision spheres query the BVH via degenerate "rays" whose origin is the
 * robot sphere centre.  The custom intersection program performs the exact
 * sphere-sphere proximity test; the any-hit program terminates traversal on
 * the first hit (early exit — the key RoboGPU contribution over plain CUDA).
 *
 * Per RoboGPU §IV "P-Sphere" approach: environment points are the spheres in
 * the BVH; robot sphere centres become query points (degenerate rays with
 * direction=(0,0,1), tmax=1).  The intersection program ignores the ray
 * equation entirely and just tests sphere-sphere overlap, reporting a hit at
 * t=0.5 (within [tmin=0, tmax=1]).  This lets the OptiX BVH provide the
 * tree-traversal acceleration (§III-B early-exit support) while the actual
 * primitive test is a simple sphere-sphere distance check.
 *
 * Payload registers:
 *   p0 — robot sphere radius (float bits, set by raygen, read by intersection)
 *   p1 — hit flag (0 = miss, 1 = hit; written by any-hit, read by raygen)
 */

#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Launch parameters (set via optixLaunch params buffer; __constant__ in PTX)
// ---------------------------------------------------------------------------

struct RoboGPULaunchParams {
    OptixTraversableHandle handle;        // BVH over environment spheres
    const float4*          robot_spheres; // [B * K, 4] world-frame (x,y,z,r)
    int32_t*               out_free;      // [B] in/out: 1=free, 0=collision
    int                    B;             // batch size
    int                    K;             // robot spheres per config (incl. padding)
};

extern "C" {
    __constant__ RoboGPULaunchParams params;
}

// ---------------------------------------------------------------------------
// Per-primitive SBT hit-group data (one record covers all env sphere prims)
// ---------------------------------------------------------------------------

struct HitGroupData {
    const float4* env_spheres; // [Mp, 4] (cx, cy, cz, r_env)
};

// ---------------------------------------------------------------------------
// Ray generation — one OptiX thread per configuration
//
// Loops over K robot collision spheres.  For each active sphere, fires a
// single optixTrace into the env BVH.  Breaks immediately on the first hit
// (per-config early exit complementing the per-sphere early exit in any-hit).
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__sphere_query() {
    const int b = static_cast<int>(optixGetLaunchIndex().x);
    if (b >= params.B) return;

    // Skip configs already marked as in-collision by the CUDA prepare stage.
    if (params.out_free[b] == 0) return;

    for (int k = 0; k < params.K; ++k) {
        const float4 s = params.robot_spheres[b * params.K + k];
        // Padding spheres have negative radius — skip.
        if (s.w < 0.0f) continue;

        // p0 = robot sphere radius (read by intersection program)
        // p1 = hit flag, 0 initially; set to 1 by any-hit on first collision
        unsigned int p0 = __float_as_uint(s.w);
        unsigned int p1 = 0u;

        // Degenerate point query: a near-zero-length ray.  The robot sphere
        // centre lies inside every expanded env-sphere AABB it could collide
        // with, so a tiny tmax still visits all relevant AABBs while avoiding
        // the spurious candidates a long ray would sweep up on dense clouds.
        optixTrace(
            params.handle,
            make_float3(s.x, s.y, s.z),    // ray origin = robot sphere centre
            make_float3(0.0f, 0.0f, 1.0f), // dummy direction (ignored by isect)
            0.0f,                            // tmin
            1.0e-3f,                         // tmax  (intersection reports t=5e-4)
            0.0f,                            // ray time
            OptixVisibilityMask(0xFF),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,                         // SBT offset, stride, miss SBT idx
            p0, p1
        );

        if (p1 != 0u) {
            // First environment sphere hit → collision for this config.
            params.out_free[b] = 0;
            return;  // early exit: no need to test remaining robot spheres
        }
    }
    // All robot spheres clear of the point cloud — out_free[b] stays 1.
}

// ---------------------------------------------------------------------------
// Intersection program — custom sphere-sphere proximity test
//
// The ray "origin" encodes the robot sphere centre; payload word 0 carries
// the robot sphere radius.  We test overlap with the BVH primitive's env
// sphere and report a hit at t=0.5 if they overlap.
// ---------------------------------------------------------------------------

extern "C" __global__ void __intersection__sphere() {
    const HitGroupData* hg =
        reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const int prim_idx = optixGetPrimitiveIndex();
    const float4 env   = hg->env_spheres[prim_idx]; // (cx, cy, cz, r_env)

    // Robot sphere centre from ray origin; robot sphere radius from payload.
    const float3 o       = optixGetWorldRayOrigin();
    const float  r_robot = __uint_as_float(optixGetPayload_0());

    const float dx    = o.x - env.x;
    const float dy    = o.y - env.y;
    const float dz    = o.z - env.z;
    const float r_sum = r_robot + env.w;

    if (dx*dx + dy*dy + dz*dz < r_sum * r_sum) {
        // Spheres overlap: report hit at t=5e-4, within [tmin=0, tmax=1e-3].
        optixReportIntersection(5.0e-4f, 0u);
    }
}

// ---------------------------------------------------------------------------
// Any-hit program — terminate ray on first overlap (RoboGPU early-exit)
// ---------------------------------------------------------------------------

extern "C" __global__ void __anyhit__sphere() {
    optixSetPayload_1(1u);  // signal collision back to raygen
    optixTerminateRay();     // stop BVH traversal immediately
}

// ---------------------------------------------------------------------------
// Miss program — no environment sphere hit for this robot sphere
// ---------------------------------------------------------------------------

extern "C" __global__ void __miss__sphere() {
    // p1 remains 0: robot sphere is clear of the point cloud
}
