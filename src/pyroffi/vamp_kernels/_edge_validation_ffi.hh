// Generic JAX FFI handlers for VAMP's CPU collision checker.
//
// This header is robot-agnostic: include it from a small translation unit that
// first defines a `vamp::robots::<Robot>` struct (e.g. one emitted by cricket's
// `generate_robot_source`) and the macros below, and it will emit two XLA FFI
// custom-call handlers specialised for that robot:
//
//   * validate_configs_<robot>  — per-configuration validity (fused FK + CC).
//     a : F32 [B, dim]                          configurations (raw joint values)
//     -> r : PRED [B]                           true == collision-free
//
//   * validate_edges_<robot>    — batch edge validation (the gtmp branch's
//     `validate_motion_batch`, but exposed through the JAX FFI).
//     a : F32 [E, dim]                          edge start configs
//     b : F32 [E, dim]                          edge goal configs
//     -> r : PRED [E]                           true == whole edge collision-free
//
// Both handlers take the world geometry as flat float buffers so the buffer
// layout matches pyroffi's existing CUDA binary checker
// (`_extract_world_arrays`), plus a CAPT point-cloud buffer so point-cloud
// obstacles are checked too:
//
//   spheres  : F32 [Ms, 4]    (cx, cy, cz, r)
//   capsules : F32 [Mc, 7]    (ax, ay, az, bx, by, bz, r)         endpoints
//   cuboids  : F32 [Mb, 15]   (cx,cy,cz, axis1(3), axis2(3), axis3(3), half(3))
//   points   : F32 [Mp, 3]    (x, y, z)                            CAPT cloud
//   attrs    : r_min, r_max, r_point (CAPT query/point radii)
//
// HalfSpace obstacles are intentionally not handled here: VAMP's
// `collision::Environment` has no half-space primitive.  Callers that need a
// ground plane should pass a large flat cuboid instead.

#pragma once

#if defined(VAMP_JAX_ROBOT) && defined(VAMP_JAX_ROBOT_NAME)

#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <cstddef>
#include <vector>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/shapes.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/validate.hh>

namespace ffi = xla::ffi;

namespace vamp::binding::jax
{
    static constexpr const std::size_t rake = vamp::FloatVectorWidth;
    using EnvironmentF = vamp::collision::Environment<float>;
    using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

    // Reconstruct a scalar (float) Environment from the flat FFI buffers.
    //
    // The CAPT point cloud is built only when at least one finite point is
    // supplied.  Building the affordance tree is O(n log n); for repeated edge
    // batches against a static cloud the caller should keep the cloud fixed so
    // JAX caches the buffer on device, but the tree itself is rebuilt per call —
    // a future optimisation could memoise it keyed by the points pointer.
    inline auto build_environment(
        const float *spheres, std::size_t n_spheres,
        const float *capsules, std::size_t n_capsules,
        const float *cuboids, std::size_t n_cuboids,
        const float *points, std::size_t n_points,
        float capt_r_min, float capt_r_max, float capt_r_point) -> EnvironmentF
    {
        EnvironmentF env;

        env.spheres.reserve(n_spheres);
        for (std::size_t i = 0; i < n_spheres; ++i)
        {
            const float *s = &spheres[i * 4];
            env.spheres.emplace_back(vamp::collision::factory::sphere::flat(s[0], s[1], s[2], s[3]));
        }

        env.capsules.reserve(n_capsules);
        for (std::size_t i = 0; i < n_capsules; ++i)
        {
            const float *c = &capsules[i * 7];
            env.capsules.emplace_back(vamp::collision::factory::capsule::endpoints::flat(
                c[0], c[1], c[2], c[3], c[4], c[5], c[6]));
        }

        env.cuboids.reserve(n_cuboids);
        for (std::size_t i = 0; i < n_cuboids; ++i)
        {
            const float *b = &cuboids[i * 15];
            // (cx,cy,cz, axis1(3), axis2(3), axis3(3), half(3)) maps directly to
            // the Cuboid(center, axis1, axis2, axis3, half-extents) constructor.
            env.cuboids.emplace_back(vamp::collision::Cuboid<float>(
                b[0], b[1], b[2],
                b[3], b[4], b[5],
                b[6], b[7], b[8],
                b[9], b[10], b[11],
                b[12], b[13], b[14]));
        }

        if (n_points > 0)
        {
            std::vector<vamp::collision::Point> cloud;
            cloud.reserve(n_points);
            for (std::size_t i = 0; i < n_points; ++i)
            {
                const float *p = &points[i * 3];
                cloud.push_back(vamp::collision::Point{p[0], p[1], p[2]});
            }
            env.pointclouds.emplace_back(cloud, capt_r_min, capt_r_max, capt_r_point);
        }

        env.sort();
        return env;
    }

    // Build a robot Configuration from a dense [dim] row.
    //
    // We must NOT construct directly from the raw row pointer: a FloatVector
    // rounds the dimension up to the SIMD width and loads all rounded lanes, so
    // an unaligned load off the final row reads out-of-bounds padding, which
    // perturbs l2_norm and hence the edge sample count.  Mirror VAMP's
    // validate_motion_batch: zero-initialise an aligned buffer, copy `dimension`
    // scalars, then load.
    template <typename Robot>
    inline auto make_configuration(const float *row) noexcept -> typename Robot::Configuration
    {
        typename Robot::ConfigurationBuffer buf{};  // zero-initialised padding
        for (std::size_t d = 0; d < Robot::dimension; ++d)
        {
            buf[d] = row[d];
        }
        return typename Robot::Configuration(buf.data());
    }

    template <typename Robot>
    inline auto validate_configs_impl(
        ffi::Buffer<ffi::F32> a,
        ffi::Buffer<ffi::F32> spheres,
        ffi::Buffer<ffi::F32> capsules,
        ffi::Buffer<ffi::F32> cuboids,
        ffi::Buffer<ffi::F32> points,
        float capt_r_min,
        float capt_r_max,
        float capt_r_point,
        ffi::ResultBuffer<ffi::PRED> r) noexcept -> ffi::Error
    {
        const auto a_d = a.dimensions();
        const std::size_t B = a_d[0];
        const float *a_data = a.typed_data();
        bool *r_data = r->typed_data();

        const EnvironmentF env_f = build_environment(
            spheres.typed_data(), spheres.dimensions()[0],
            capsules.typed_data(), capsules.dimensions()[0],
            cuboids.typed_data(), cuboids.dimensions()[0],
            points.typed_data(), points.dimensions()[0],
            capt_r_min, capt_r_max, capt_r_point);
        const EnvironmentVector env(env_f);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1000)
#endif
        for (std::size_t i = 0; i < B; ++i)
        {
            const auto c = make_configuration<Robot>(&a_data[i * Robot::dimension]);
            // A zero-length motion reduces to a single fused FK + CC pass.
            r_data[i] = vamp::planning::validate_motion<Robot, rake, Robot::resolution>(c, c, env);
        }

        return ffi::Error::Success();
    }

    template <typename Robot>
    inline auto validate_edges_impl(
        ffi::Buffer<ffi::F32> a,
        ffi::Buffer<ffi::F32> b,
        ffi::Buffer<ffi::F32> spheres,
        ffi::Buffer<ffi::F32> capsules,
        ffi::Buffer<ffi::F32> cuboids,
        ffi::Buffer<ffi::F32> points,
        float capt_r_min,
        float capt_r_max,
        float capt_r_point,
        ffi::ResultBuffer<ffi::PRED> r) noexcept -> ffi::Error
    {
        const auto a_d = a.dimensions();
        const auto b_d = b.dimensions();
        const std::size_t E = a_d[0];
        if (b_d[0] != a_d[0])
        {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "validate_edges expects a and b to have the same number of rows");
        }

        const float *a_data = a.typed_data();
        const float *b_data = b.typed_data();
        bool *r_data = r->typed_data();

        const EnvironmentF env_f = build_environment(
            spheres.typed_data(), spheres.dimensions()[0],
            capsules.typed_data(), capsules.dimensions()[0],
            cuboids.typed_data(), cuboids.dimensions()[0],
            points.typed_data(), points.dimensions()[0],
            capt_r_min, capt_r_max, capt_r_point);
        const EnvironmentVector env(env_f);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1000)
#endif
        for (std::size_t i = 0; i < E; ++i)
        {
            const auto ca = make_configuration<Robot>(&a_data[i * Robot::dimension]);
            const auto cb = make_configuration<Robot>(&b_data[i * Robot::dimension]);
            // validate_motion samples the open interval (0, 1] — it checks the
            // goal but assumes the start is already valid (the usual planner
            // contract).  We additionally validate the start config so a "valid"
            // edge means *both* endpoints and the interior are collision-free,
            // matching the CUDA check_edges_collision_free semantics.
            r_data[i] =
                vamp::planning::validate_motion<Robot, rake, Robot::resolution>(ca, ca, env) and
                vamp::planning::validate_motion<Robot, rake, Robot::resolution>(ca, cb, env);
        }

        return ffi::Error::Success();
    }
}  // namespace vamp::binding::jax

#define VAMP_PASTE(A, B) A##B
#define VAMP_XSTRING(A) VAMP_STRING(A)
#define VAMP_STRING(A) #A
#define VAMP_CONFIGS_SYMBOL(robot_name) VAMP_PASTE(validate_configs_, robot_name)
#define VAMP_EDGES_SYMBOL(robot_name) VAMP_PASTE(validate_edges_, robot_name)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VAMP_CONFIGS_SYMBOL(VAMP_JAX_ROBOT_NAME),
    vamp::binding::jax::validate_configs_impl<VAMP_JAX_ROBOT>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()    // a [B, dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // spheres  [Ms, 4]
        .Arg<ffi::Buffer<ffi::F32>>()    // capsules [Mc, 7]
        .Arg<ffi::Buffer<ffi::F32>>()    // cuboids  [Mb, 15]
        .Arg<ffi::Buffer<ffi::F32>>()    // points   [Mp, 3]
        .Attr<float>("capt_r_min")
        .Attr<float>("capt_r_max")
        .Attr<float>("capt_r_point")
        .Ret<ffi::Buffer<ffi::PRED>>()   // [B] validity
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VAMP_EDGES_SYMBOL(VAMP_JAX_ROBOT_NAME),
    vamp::binding::jax::validate_edges_impl<VAMP_JAX_ROBOT>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()    // a [E, dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // b [E, dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // spheres  [Ms, 4]
        .Arg<ffi::Buffer<ffi::F32>>()    // capsules [Mc, 7]
        .Arg<ffi::Buffer<ffi::F32>>()    // cuboids  [Mb, 15]
        .Arg<ffi::Buffer<ffi::F32>>()    // points   [Mp, 3]
        .Attr<float>("capt_r_min")
        .Attr<float>("capt_r_max")
        .Attr<float>("capt_r_point")
        .Ret<ffi::Buffer<ffi::PRED>>()   // [E] validity
);

#endif  // VAMP_JAX_ROBOT && VAMP_JAX_ROBOT_NAME
