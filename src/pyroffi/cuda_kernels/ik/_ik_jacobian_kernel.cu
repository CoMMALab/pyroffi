/*
 * Analytic task Jacobian at a given configuration.
 *
 * This is the IK counterpart of GRiD's gradient kernels: GRiD pairs each solve
 * kernel (inverse_dynamics) with a SEPARATE analytic-derivative kernel
 * (inverse_dynamics_gradient), and the JAX side wires the latter into a
 * custom_jvp tangent rule. The same split is used here, for the same reason --
 * one kernel then serves every IK solver (ls, sqp, mppi, hjcd, analytic)
 * instead of five near-identical output paths, and none of the working solve
 * kernels have to be touched to get the derivative.
 *
 * Why a second pass rather than returning the solver's in-loop J:
 *
 *   - the solvers MUTATE their J -- it is scaled by the per-DOF pose weights
 *     W[k] and then column-equilibrated by col_scale[a] before the normal
 *     equations are formed, so what survives the loop is not dr/dq;
 *   - it is evaluated at the iterate's START point, not at the returned
 *     winner, and for a multi-seed solve the winner is chosen afterwards.
 *
 * Recomputing costs one FK + Jacobian per problem against a whole solve, which
 * is the same trade GRiD makes.
 *
 * CONVENTION -- this matters for correctness, not style. The residual here is
 * the one `compute_multi_ee_residual_and_jacobian` already implements:
 * world-frame position difference `p_ee - p_tgt` stacked with a quaternion
 * error. That is NOT the SE(3) local log-map residual `(T_a^-1 T_t).log()`.
 * The two differ by an invertible 6x6 A (its position block carries a
 * rotation), so their Jacobians differ elementwise. Since
 *
 *     pinv(A J_q) (A J_t) = J_q^+ J_t
 *
 * for full-row-rank J_q, the resulting gradient is invariant to the choice --
 * but ONLY if J_q, J_t and J_theta all use the SAME residual. The JAX side is
 * therefore moved onto this convention (`_ik_residual_kernel_convention`)
 * rather than this kernel being bent to the log-map. Mixing them silently
 * yields a wrong gradient with no error anywhere.
 */

#include "_ik_cuda_helpers.cuh"

#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

// One thread per problem: the work is a single FK sweep plus a Jacobian fill,
// which is exactly the thread-tier body the solvers already use at their
// innermost level. No shared memory and no cross-lane cooperation, so there is
// nothing to tier here.
__global__ void ik_task_jacobian_kernel(
    const float* __restrict__ cfgs,            // (n_problems, n_act)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,     // (n_ee,)
    const int*   __restrict__ ancestor_masks,  // (n_ee, n_joints)
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7)
    int n_problems, int n_joints, int n_act, int n_ee,
    float* __restrict__ out_r,                 // (n_problems, 6*n_ee)
    float* __restrict__ out_J)                 // (n_problems, 6*n_ee, n_act)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_problems) return;

    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    compute_multi_ee_residual_and_jacobian(
        cfgs + (size_t)p * n_act, T_world,
        twists, parent_tf, parent_idx, act_idx,
        mimic_mul, mimic_off, mimic_act_idx, topo_inv,
        target_jnts, ancestor_masks,
        target_Ts + (size_t)p * n_ee * 7,
        n_joints, n_act, n_ee, r, J);

    const int rows = 6 * n_ee;
    for (int k = 0; k < rows; k++) out_r[(size_t)p * rows + k] = r[k];
    for (int k = 0; k < rows * n_act; k++)
        out_J[(size_t)p * rows * n_act + k] = J[k];
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error IkTaskJacobianImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfgs,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> target_jnts,
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_r,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_J)
{
    const int n_problems = static_cast<int>(cfgs.dimensions()[0]);
    const int n_act      = static_cast<int>(cfgs.dimensions()[1]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);
    const int n_ee       = static_cast<int>(target_jnts.dimensions()[0]);

    if (n_act > MAX_ACT || n_joints > MAX_JOINTS || n_ee > MAX_EE)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "ik_task_jacobian: robot exceeds compiled capacity "
                          "(rebuild with --max-act / --max-joints)");

    const int block = 128;
    const int grid  = (n_problems + block - 1) / block;

    ik_task_jacobian_kernel<<<grid, block, 0, stream>>>(
        cfgs.typed_data(), twists.typed_data(), parent_tf.typed_data(),
        parent_idx.typed_data(), act_idx.typed_data(),
        mimic_mul.typed_data(), mimic_off.typed_data(),
        mimic_act_idx.typed_data(), topo_inv.typed_data(),
        target_jnts.typed_data(), ancestor_masks.typed_data(),
        target_Ts.typed_data(),
        n_problems, n_joints, n_act, n_ee,
        out_r->typed_data(), out_J->typed_data());

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IkTaskJacobianFfi, IkTaskJacobianImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfgs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // target_jnts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_r
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out_J
