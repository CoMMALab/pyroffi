// JAX FFI handler wrapping the QuIK C++ inverse-kinematics solver (CPU).
//
// QuIK (S. Lloyd et al., "Fast and Robust Inverse Kinematics for Serial Robots
// using Halley's Method", IEEE T-RO 2022; external/QuIK) is a standard-DH,
// batched serial-chain IK solver.  Its templates support a *dynamic* DOF, so a
// single compiled translation unit works for any serial robot: the DH table,
// base/tool transforms and joint types are passed in as runtime FFI buffers
// rather than baked in at compile time.  pyroffi derives those buffers from a
// robot's product-of-exponentials model via ``kinematics._dh.extract_dh``.
//
// Handler ``quik_ik_solve``:
//   q0        : F32 [B, DOF]     seeds (chain order)
//   twt       : F32 [B, 4, 4]    desired end-effector poses (row-major homog.)
//   dh        : F32 [DOF, 4]     (a, alpha, d, theta) rows
//   link_type : F32 [DOF]        1.0 == prismatic, 0.0 == revolute
//   qsign     : F32 [DOF]        +/-1 joint direction
//   tbase     : F32 [4, 4]       world -> DH frame 0
//   ttool     : F32 [4, 4]       DH frame N -> end-effector
//   attrs     : algorithm, iter_max, exit_tol, min_step, rel_improve_tol,
//               max_grad_fails, max_grad_fails_total, lambda2,
//               max_lin_step, max_ang_step
//   -> qstar  : F32 [B, DOF]     solved joints
//   -> enorm  : F32 [B]          final pose-error norm
//   -> iters  : S32 [B]          iterations taken
//
// The batch is solved column-by-column so it can be parallelised across
// configurations with OpenMP (QuIK's own loop is serial per column).

#pragma once

#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>

#include "Eigen/Dense"
#include "Robot.hpp"
#include "IKOptions.hpp"
#include "IK.hpp"

namespace ffi = xla::ffi;

namespace pyroffi::quik
{
    using Eigen::Dynamic;
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::Matrix4d;

    // Read a row-major [4,4] f32 buffer into a column-major Eigen Matrix4d.
    inline Matrix4d read_mat4(const float *p) noexcept
    {
        Matrix4d M;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                M(r, c) = static_cast<double>(p[r * 4 + c]);
        return M;
    }

    inline ffi::Error quik_ik_solve_impl(
        ffi::Buffer<ffi::F32> q0,
        ffi::Buffer<ffi::F32> twt,
        ffi::Buffer<ffi::F32> dh,
        ffi::Buffer<ffi::F32> link_type,
        ffi::Buffer<ffi::F32> qsign,
        ffi::Buffer<ffi::F32> tbase,
        ffi::Buffer<ffi::F32> ttool,
        int algorithm,
        int iter_max,
        double exit_tol,
        double min_step,
        double rel_improve_tol,
        int max_grad_fails,
        int max_grad_fails_total,
        double lambda2,
        double max_lin_step,
        double max_ang_step,
        ffi::ResultBuffer<ffi::F32> qstar,
        ffi::ResultBuffer<ffi::F32> enorm,
        ffi::ResultBuffer<ffi::S32> iters) noexcept
    {
        const auto q0_d = q0.dimensions();
        const std::size_t B = q0_d[0];
        const int dof = static_cast<int>(q0_d[1]);

        // Build the (const, shared) Robot from the DH buffers.
        Eigen::Array<double, Dynamic, 4> DH(dof, 4);
        const float *dh_p = dh.typed_data();
        for (int i = 0; i < dof; ++i)
            for (int j = 0; j < 4; ++j)
                DH(i, j) = static_cast<double>(dh_p[i * 4 + j]);

        Eigen::Vector<bool, Dynamic> linkTypes(dof);
        Eigen::Vector<double, Dynamic> Qsign(dof);
        const float *lt_p = link_type.typed_data();
        const float *qs_p = qsign.typed_data();
        for (int i = 0; i < dof; ++i)
        {
            linkTypes(i) = lt_p[i] > 0.5f;
            Qsign(i) = static_cast<double>(qs_p[i]);
        }

        const Robot<Dynamic> R(
            DH, linkTypes, Qsign,
            read_mat4(tbase.typed_data()),
            read_mat4(ttool.typed_data()));

        const IKOptions opt(
            iter_max, algorithm, exit_tol, min_step, rel_improve_tol,
            max_grad_fails, max_grad_fails_total, lambda2,
            max_lin_step, max_ang_step);

        const float *q0_p = q0.typed_data();
        const float *twt_p = twt.typed_data();
        float *qstar_p = qstar->typed_data();
        float *enorm_p = enorm->typed_data();
        std::int32_t *iters_p = iters->typed_data();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
        for (std::size_t b = 0; b < B; ++b)
        {
            // One-column problem for this batch element.
            Matrix<double, Dynamic, Dynamic> Q0(dof, 1);
            for (int i = 0; i < dof; ++i)
                Q0(i, 0) = static_cast<double>(q0_p[b * dof + i]);

            Matrix<double, Dynamic, 4> Twt(4, 4);
            Twt = read_mat4(&twt_p[b * 16]);

            Matrix<double, Dynamic, Dynamic> Qs(dof, 1);
            Matrix<double, 6, Dynamic> Es(6, 1);
            Eigen::VectorXi it(1), br(1);

            IK<Dynamic>(R, Twt, Q0, opt, Qs, Es, it, br);

            for (int i = 0; i < dof; ++i)
                qstar_p[b * dof + i] = static_cast<float>(Qs(i, 0));
            enorm_p[b] = static_cast<float>(Es.col(0).norm());
            iters_p[b] = it(0);
        }

        return ffi::Error::Success();
    }
}  // namespace pyroffi::quik

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    quik_ik_solve,
    pyroffi::quik::quik_ik_solve_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()   // q0        [B, DOF]
        .Arg<ffi::Buffer<ffi::F32>>()   // twt       [B, 4, 4]
        .Arg<ffi::Buffer<ffi::F32>>()   // dh        [DOF, 4]
        .Arg<ffi::Buffer<ffi::F32>>()   // link_type [DOF]
        .Arg<ffi::Buffer<ffi::F32>>()   // qsign     [DOF]
        .Arg<ffi::Buffer<ffi::F32>>()   // tbase     [4, 4]
        .Arg<ffi::Buffer<ffi::F32>>()   // ttool     [4, 4]
        .Attr<int>("algorithm")
        .Attr<int>("iter_max")
        .Attr<double>("exit_tol")
        .Attr<double>("min_step")
        .Attr<double>("rel_improve_tol")
        .Attr<int>("max_grad_fails")
        .Attr<int>("max_grad_fails_total")
        .Attr<double>("lambda2")
        .Attr<double>("max_lin_step")
        .Attr<double>("max_ang_step")
        .Ret<ffi::Buffer<ffi::F32>>()   // qstar [B, DOF]
        .Ret<ffi::Buffer<ffi::F32>>()   // enorm [B]
        .Ret<ffi::Buffer<ffi::S32>>()   // iters [B]
);
