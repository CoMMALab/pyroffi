// Generic JAX FFI handler for VAMP's end-effector forward kinematics (CPU).
//
// Robot-agnostic, mirroring _edge_validation_ffi.hh: include it from a tiny
// translation unit that first defines a `vamp::robots::<Robot>` struct (emitted
// by cricket's `generate_robot_source`) plus the VAMP_JAX_ROBOT[_NAME] macros,
// and it emits one XLA FFI custom-call handler:
//
//   * eefk_<robot>  — batched end-effector forward kinematics.
//     a : F32 [B, dim]     configurations (raw joint values)
//     -> r : F32 [B, 4, 4] row-major world<-EE homogeneous transforms
//
// VAMP generates a scalar `Robot::eefk(std::array<float, dim>) -> Isometry3f`
// straight-line kernel per robot; we batch it with OpenMP.  This is the CPU-
// accelerated counterpart to pyroffi's JAX / CUDA forward kinematics, intended
// for CPU-only planning (JAX_PLATFORMS=cpu).

#pragma once

#if defined(VAMP_JAX_ROBOT) && defined(VAMP_JAX_ROBOT_NAME)

#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <array>
#include <cstddef>

#include <Eigen/Geometry>

namespace ffi = xla::ffi;

namespace vamp::binding::jax
{
    template <typename Robot>
    inline auto eefk_impl(
        ffi::Buffer<ffi::F32> a,
        ffi::ResultBuffer<ffi::F32> r) noexcept -> ffi::Error
    {
        const auto a_d = a.dimensions();
        const std::size_t B = a_d[0];
        constexpr std::size_t dim = Robot::dimension;
        const float *a_data = a.typed_data();
        float *r_data = r->typed_data();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
        for (std::size_t i = 0; i < B; ++i)
        {
            std::array<float, dim> q;
            for (std::size_t d = 0; d < dim; ++d)
                q[d] = a_data[i * dim + d];
            const Eigen::Isometry3f T = Robot::eefk(q);
            const Eigen::Matrix4f M = T.matrix();
            // Row-major [4,4] to match JAX's default layout.
            for (int rr = 0; rr < 4; ++rr)
                for (int cc = 0; cc < 4; ++cc)
                    r_data[i * 16 + rr * 4 + cc] = M(rr, cc);
        }
        return ffi::Error::Success();
    }
}  // namespace vamp::binding::jax

#define VAMP_PASTE(A, B) A##B
#define VAMP_EEFK_SYMBOL(robot_name) VAMP_PASTE(eefk_, robot_name)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    VAMP_EEFK_SYMBOL(VAMP_JAX_ROBOT_NAME),
    vamp::binding::jax::eefk_impl<VAMP_JAX_ROBOT>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()    // a [B, dim]
        .Ret<ffi::Buffer<ffi::F32>>()    // r [B, 4, 4]
);

#endif  // VAMP_JAX_ROBOT && VAMP_JAX_ROBOT_NAME
