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
//   * project_configs_<robot>   — MPPI-style projection of configurations onto
//     the collision-free set (e.g. to repair IK seeds before handing them to
//     QuIK).  VAMP's fkcc is a boolean oracle (no signed distances), so the
//     projection is a derivative-free particle optimization: per iteration a
//     population of antithetic (+/-delta) particle pairs at stratified
//     log-spaced radius scales is drawn around the running mean, feasible
//     particles are softmax-weighted by exp(-||q - seed||^2 / temperature)
//     (the information-theoretic MPPI update with cost = squared distance to
//     the original seed), and the mean moves to the weighted average.  The
//     sampling scale shrinks when feasible particles are found and grows when
//     none are (all-colliding iterations carry no cost signal, only the need
//     for more exploration).  The best feasible particle is retained across
//     iterations and finally bisected back toward the seed so the result lands
//     near the constraint-manifold boundary, close to the seed.
//     q     : F32 [B, dim]                      configurations to project
//     lower : F32 [dim]                         joint lower bounds (clamped)
//     upper : F32 [dim]                         joint upper bounds (clamped)
//     -> qp : F32 [B, dim]                      projected configurations
//     -> ok : PRED [B]                          true == qp is collision-free
//     attrs: rng seed, num_particles / num_iters (population), temperature
//     (MPPI weighting), sigma0 / sigma_shrink / sigma_grow (scale adaptation),
//     bisect_iters.  Failures return q unchanged with ok == false.
//
//     Parallelization: one OpenMP thread runs one batch element's entire
//     optimization (schedule(dynamic, 1) — elements finish at very different
//     times since free seeds return immediately).  Particle evaluations within
//     an element are deliberately serial: VAMP's fkcc already saturates the
//     SIMD units, so threads split across problems and vectors within a check,
//     with no nested parallelism to oversubscribe cores.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
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

    // FNV-1a over a raw byte range, chained into a running hash.
    inline auto hash_bytes(const void *data, std::size_t nbytes, std::uint64_t h) noexcept
        -> std::uint64_t
    {
        const auto *p = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < nbytes; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Content hash of the world buffers (obstacle counts, raw float payloads, and
    // CAPT radii).  Two calls with byte-identical worlds hash equal, so the
    // memoised Environment below is reused; the 64-bit FNV-1a collision risk
    // (~2^-64) is negligible for a perf cache over fixed-layout buffers.
    inline auto world_hash(
        const float *spheres, std::size_t n_spheres,
        const float *capsules, std::size_t n_capsules,
        const float *cuboids, std::size_t n_cuboids,
        const float *points, std::size_t n_points,
        float capt_r_min, float capt_r_max, float capt_r_point) noexcept -> std::uint64_t
    {
        std::uint64_t h = 1469598103934665603ULL;  // FNV offset basis
        const std::size_t sizes[4] = {n_spheres, n_capsules, n_cuboids, n_points};
        h = hash_bytes(sizes, sizeof(sizes), h);
        h = hash_bytes(spheres, n_spheres * 4 * sizeof(float), h);
        h = hash_bytes(capsules, n_capsules * 7 * sizeof(float), h);
        h = hash_bytes(cuboids, n_cuboids * 15 * sizeof(float), h);
        h = hash_bytes(points, n_points * 3 * sizeof(float), h);
        const float attrs[3] = {capt_r_min, capt_r_max, capt_r_point};
        h = hash_bytes(attrs, sizeof(attrs), h);
        return h;
    }

    // Build (or reuse) the SIMD Environment for these world buffers.
    //
    // The JIT-compiled handler is called repeatedly against the *same* static
    // world during planning / benchmarking, yet each call would otherwise rebuild
    // the obstacle vectors, re-sort, and reconstruct the CAPT affordance tree
    // (O(n log n)) before a single config is checked — pure per-call overhead.
    // We memoise the last-built EnvironmentVector keyed by ``world_hash`` so an
    // unchanged world is free after the first call.
    //
    // The cached Environment is heap-allocated and *intentionally never freed*:
    // (1) a previously-returned reference may still be in use by a concurrent
    //     kernel when the world changes, and (2) it keeps the cache statics
    //     trivially destructible — a function-local ``static`` with a non-trivial
    //     destructor (e.g. a ``shared_ptr``) makes the compiler emit an
    //     ``__cxa_atexit(__dso_handle)`` registration that cricket's ORC JIT
    //     cannot relocate.  Worlds change rarely, so the leak is bounded.
    inline auto environment_for(
        const float *spheres, std::size_t n_spheres,
        const float *capsules, std::size_t n_capsules,
        const float *cuboids, std::size_t n_cuboids,
        const float *points, std::size_t n_points,
        float capt_r_min, float capt_r_max, float capt_r_point) -> const EnvironmentVector &
    {
        const std::uint64_t h = world_hash(
            spheres, n_spheres, capsules, n_capsules, cuboids, n_cuboids,
            points, n_points, capt_r_min, capt_r_max, capt_r_point);

        static std::mutex mtx;                          // trivially destructible
        static std::uint64_t cached_hash = 0;
        static const EnvironmentVector *cached = nullptr;  // leaked; never freed

        std::lock_guard<std::mutex> lock(mtx);
        if (cached != nullptr and cached_hash == h)
        {
            return *cached;
        }

        const EnvironmentF env_f = build_environment(
            spheres, n_spheres, capsules, n_capsules, cuboids, n_cuboids,
            points, n_points, capt_r_min, capt_r_max, capt_r_point);
        cached = new EnvironmentVector(env_f);
        cached_hash = h;
        return *cached;
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

        const EnvironmentVector &env = environment_for(
            spheres.typed_data(), spheres.dimensions()[0],
            capsules.typed_data(), capsules.dimensions()[0],
            cuboids.typed_data(), cuboids.dimensions()[0],
            points.typed_data(), points.dimensions()[0],
            capt_r_min, capt_r_max, capt_r_point);

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

        const EnvironmentVector &env = environment_for(
            spheres.typed_data(), spheres.dimensions()[0],
            capsules.typed_data(), capsules.dimensions()[0],
            cuboids.typed_data(), cuboids.dimensions()[0],
            points.typed_data(), points.dimensions()[0],
            capt_r_min, capt_r_max, capt_r_point);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1000)
#endif
        for (std::size_t i = 0; i < E; ++i)
        {
            const auto ca = make_configuration<Robot>(&a_data[i * Robot::dimension]);
            const auto cb = make_configuration<Robot>(&b_data[i * Robot::dimension]);
            // validate_motion samples the open interval (0, 1]: it checks the goal
            // and the interior but assumes the start is already valid (the usual
            // planner contract, and what VAMP's own validate_motion[_batch] does).
            // A "valid" edge therefore guarantees the goal endpoint and interior
            // are collision-free; callers needing the start checked should
            // validate it separately (check_collision_free).
            r_data[i] =
                vamp::planning::validate_motion<Robot, rake, Robot::resolution>(ca, cb, env);
        }

        return ffi::Error::Success();
    }
    // ── Collision-free projection ───────────────────────────────────────────

    // SplitMix64: tiny counter-based PRNG, seeded per batch element so results
    // are deterministic and independent of the OpenMP schedule.
    struct SplitMix64
    {
        std::uint64_t state;

        inline auto next() noexcept -> std::uint64_t
        {
            std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

        // Uniform in [0, 1).
        inline auto uniform() noexcept -> float
        {
            return static_cast<float>(next() >> 40) * 0x1.0p-24f;
        }

        // Standard normal via Box-Muller.
        inline auto normal() noexcept -> float
        {
            float u1 = uniform();
            const float u2 = uniform();
            if (u1 < 1e-12f)
            {
                u1 = 1e-12f;
            }
            return std::sqrt(-2.0f * std::log(u1)) *
                   std::cos(6.28318530717958647692f * u2);
        }
    };

    template <typename Robot>
    inline auto is_free(
        const typename Robot::ConfigurationBuffer &buf,
        const EnvironmentVector &env) noexcept -> bool
    {
        const typename Robot::Configuration c(buf.data());
        return vamp::planning::validate_motion<Robot, rake, Robot::resolution>(c, c, env);
    }

    template <typename Robot>
    inline auto project_configs_impl(
        ffi::Buffer<ffi::F32> q,
        ffi::Buffer<ffi::F32> lower,
        ffi::Buffer<ffi::F32> upper,
        ffi::Buffer<ffi::F32> spheres,
        ffi::Buffer<ffi::F32> capsules,
        ffi::Buffer<ffi::F32> cuboids,
        ffi::Buffer<ffi::F32> points,
        float capt_r_min,
        float capt_r_max,
        float capt_r_point,
        std::int64_t seed,
        std::int64_t num_particles,
        std::int64_t num_iters,
        float temperature,
        float sigma0,
        float sigma_shrink,
        float sigma_grow,
        std::int64_t bisect_iters,
        ffi::ResultBuffer<ffi::F32> qp,
        ffi::ResultBuffer<ffi::PRED> ok) noexcept -> ffi::Error
    {
        constexpr std::size_t dim = Robot::dimension;
        const std::size_t B = q.dimensions()[0];
        const float *q_data = q.typed_data();
        const float *lo = lower.typed_data();
        const float *hi = upper.typed_data();
        float *qp_data = qp->typed_data();
        bool *ok_data = ok->typed_data();

        const EnvironmentVector &env = environment_for(
            spheres.typed_data(), spheres.dimensions()[0],
            capsules.typed_data(), capsules.dimensions()[0],
            cuboids.typed_data(), cuboids.dimensions()[0],
            points.typed_data(), points.dimensions()[0],
            capt_r_min, capt_r_max, capt_r_point);

        // Antithetic pairs: P pairs -> 2P particle evaluations per iteration.
        const std::size_t P = static_cast<std::size_t>(std::max<std::int64_t>(
            1, num_particles / 2));
        // Stratified log-spaced radius scales across pairs: pair j samples at
        // sigma * 2^(spread * (2(j+0.5)/P - 1)), covering [sigma/2, 2*sigma]
        // deterministically so one iteration probes several length scales
        // instead of clumping at one (i.i.d. Gaussian radii concentrate hard
        // around sigma*sqrt(dim) in high dimension).
        constexpr float radius_spread = 1.0f;
        const float inv_temp = 1.0f / std::max(temperature, 1e-8f);

#ifdef _OPENMP
        // One thread owns one element's whole optimization: elements finish at
        // wildly different times (free seeds return after one check), so chunk
        // size 1 load-balances without nested parallelism — fkcc already uses
        // the SIMD units within a thread.
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::size_t b = 0; b < B; ++b)
        {
            const float *row = &q_data[b * dim];
            float *out = &qp_data[b * dim];

            typename Robot::ConfigurationBuffer q0{};  // zero-initialised padding
            for (std::size_t d = 0; d < dim; ++d)
            {
                q0[d] = row[d];
            }

            if (is_free<Robot>(q0, env))
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    out[d] = q0[d];
                }
                ok_data[b] = true;
                continue;
            }

            SplitMix64 rng{static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ULL +
                           static_cast<std::uint64_t>(b) + 1ULL};

            // MPPI state: running mean (starts at the seed), adaptive scale,
            // elite (best feasible particle so far, by distance to the seed).
            typename Robot::ConfigurationBuffer mu = q0;
            typename Robot::ConfigurationBuffer elite{};
            float elite_d2 = std::numeric_limits<float>::infinity();
            bool have_elite = false;
            float sigma = sigma0;

            typename Robot::ConfigurationBuffer delta{};
            typename Robot::ConfigurationBuffer cand{};
            // Feasible population of one iteration (flat [count, dim] + costs);
            // allocated once per element, reused across iterations.
            std::vector<float> feas;
            std::vector<float> feas_d2;
            feas.reserve(2 * P * dim);
            feas_d2.reserve(2 * P);

            for (std::int64_t t = 0; t < num_iters; ++t)
            {
                feas.clear();
                feas_d2.clear();
                const float prev_elite_d2 = elite_d2;

                // Rollout pass: evaluate the antithetic, radius-stratified
                // population and keep the feasible particles with their cost
                // (cost = ||q - seed||^2 — infeasible particles are the
                // infinite-cost limit, i.e. weight 0).
                for (std::size_t j = 0; j < P; ++j)
                {
                    const float scale =
                        sigma * std::exp2(radius_spread *
                                          (2.0f * (static_cast<float>(j) + 0.5f) /
                                               static_cast<float>(P) -
                                           1.0f));
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        delta[d] = scale * rng.normal();
                    }
                    for (int sign = 0; sign < 2; ++sign)  // antithetic +/- pair
                    {
                        const float s = (sign == 0) ? 1.0f : -1.0f;
                        float d2 = 0.0f;
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            const float v = mu[d] + s * delta[d];
                            cand[d] = std::min(std::max(v, lo[d]), hi[d]);
                            const float dq = cand[d] - q0[d];
                            d2 += dq * dq;
                        }
                        if (not is_free<Robot>(cand, env))
                        {
                            continue;
                        }
                        feas.insert(feas.end(), cand.begin(), cand.begin() + dim);
                        feas_d2.push_back(d2);
                        if (d2 < elite_d2)
                        {
                            elite = cand;
                            elite_d2 = d2;
                            have_elite = true;
                        }
                    }
                }

                if (not feas_d2.empty())
                {
                    // MPPI update: softmax-weight the feasible population
                    // against its best cost, move the mean to the weighted
                    // average, and pull the sampling scale in around it.
                    float w_sum = 0.0f;
                    std::array<float, dim> acc{};
                    for (std::size_t i = 0; i < feas_d2.size(); ++i)
                    {
                        const float w =
                            std::exp(-(feas_d2[i] - elite_d2) * inv_temp);
                        w_sum += w;
                        const float *qi = &feas[i * dim];
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            acc[d] += w * qi[d];
                        }
                    }
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        mu[d] = acc[d] / w_sum;
                    }
                    sigma *= sigma_shrink;
                    // Converged: feasible evidence stopped improving the elite
                    // meaningfully; the bisection polish takes it from here.
                    if (std::isfinite(prev_elite_d2) and
                        prev_elite_d2 - elite_d2 <= 0.01f * prev_elite_d2)
                    {
                        break;
                    }
                }
                else
                {
                    // No feasible particle carries no cost signal — only the
                    // need for a wider search.
                    sigma *= sigma_grow;
                }
            }

            if (not have_elite)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    out[d] = q0[d];
                }
                ok_data[b] = false;
                continue;
            }

            // Bisect [seed (colliding), elite (free)] toward the seed; the free
            // endpoint is maintained by construction, so the result is a free
            // point near the constraint-manifold boundary closest to the seed.
            float t_lo = 0.0f;  // colliding side (seed)
            float t_hi = 1.0f;  // free side (elite)
            typename Robot::ConfigurationBuffer mid{};
            for (std::int64_t it = 0; it < bisect_iters; ++it)
            {
                const float t = 0.5f * (t_lo + t_hi);
                for (std::size_t d = 0; d < dim; ++d)
                {
                    mid[d] = q0[d] + t * (elite[d] - q0[d]);
                }
                if (is_free<Robot>(mid, env))
                {
                    t_hi = t;
                }
                else
                {
                    t_lo = t;
                }
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                out[d] = q0[d] + t_hi * (elite[d] - q0[d]);
            }
            ok_data[b] = true;
        }

        return ffi::Error::Success();
    }
}  // namespace vamp::binding::jax

#define VAMP_PASTE(A, B) A##B
#define VAMP_XSTRING(A) VAMP_STRING(A)
#define VAMP_STRING(A) #A
#define VAMP_CONFIGS_SYMBOL(robot_name) VAMP_PASTE(validate_configs_, robot_name)
#define VAMP_EDGES_SYMBOL(robot_name) VAMP_PASTE(validate_edges_, robot_name)
#define VAMP_PROJECT_SYMBOL(robot_name) VAMP_PASTE(project_configs_, robot_name)

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VAMP_PROJECT_SYMBOL(VAMP_JAX_ROBOT_NAME),
    vamp::binding::jax::project_configs_impl<VAMP_JAX_ROBOT>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()    // q     [B, dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // lower [dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // upper [dim]
        .Arg<ffi::Buffer<ffi::F32>>()    // spheres  [Ms, 4]
        .Arg<ffi::Buffer<ffi::F32>>()    // capsules [Mc, 7]
        .Arg<ffi::Buffer<ffi::F32>>()    // cuboids  [Mb, 15]
        .Arg<ffi::Buffer<ffi::F32>>()    // points   [Mp, 3]
        .Attr<float>("capt_r_min")
        .Attr<float>("capt_r_max")
        .Attr<float>("capt_r_point")
        .Attr<std::int64_t>("seed")
        .Attr<std::int64_t>("num_particles")
        .Attr<std::int64_t>("num_iters")
        .Attr<float>("temperature")
        .Attr<float>("sigma0")
        .Attr<float>("sigma_shrink")
        .Attr<float>("sigma_grow")
        .Attr<std::int64_t>("bisect_iters")
        .Ret<ffi::Buffer<ffi::F32>>()    // qp [B, dim]
        .Ret<ffi::Buffer<ffi::PRED>>()   // ok [B]
);

#endif  // VAMP_JAX_ROBOT && VAMP_JAX_ROBOT_NAME
