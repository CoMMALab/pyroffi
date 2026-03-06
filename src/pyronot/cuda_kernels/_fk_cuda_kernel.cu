/**
 * Forward kinematics CUDA kernel with XLA FFI binding.
 *
 * Computes SE(3) world-frame transforms for every joint given the actuated
 * joint configuration.  Mirrors the JAX implementation in _robot.py exactly:
 *
 *   1. Expand actuated config to full joint config (handles fixed / mimic joints).
 *   2. Compute delta transform  delta_T[j] = SE3.exp(twists[j] * q_full[j]).
 *   3. Combine with constant parent offset: T_pc[j] = parent_tf[j] @ delta_T[j].
 *   4. Walk joints in topological order to accumulate world transforms.
 *
 * Tangent convention (matches jaxlie SE3):
 *   tangent = [v_x, v_y, v_z, omega_x, omega_y, omega_z]
 *   i.e., linear velocity first, angular velocity last.
 *
 * Output format: SE(3) as quaternion + translation, [w, x, y, z, tx, ty, tz].
 *
 * Build with:  bash src/pyronot/build_fk_cuda.sh
 */

#include <cmath>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// SE(3) math helpers
// ---------------------------------------------------------------------------

/**
 * Hamilton quaternion product.
 * q1, q2, out are all in wxyz order.
 */
__device__ __forceinline__
void quat_mul(const float* __restrict__ q1,
              const float* __restrict__ q2,
              float* __restrict__ out)
{
    const float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    const float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

/**
 * Rotate vector v by unit quaternion q (wxyz). Result written to out.
 * Uses the cross-product form:  v' = v + 2w*(q_vec x v) + 2*(q_vec x (q_vec x v))
 */
__device__ __forceinline__
void quat_rotate(const float* __restrict__ q,
                 const float* __restrict__ v,
                 float* __restrict__ out)
{
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    // first cross: q_vec x v
    const float cx = y*v[2] - z*v[1];
    const float cy = z*v[0] - x*v[2];
    const float cz = x*v[1] - y*v[0];
    // second cross: q_vec x (q_vec x v)
    const float ccx = y*cz - z*cy;
    const float ccy = z*cx - x*cz;
    const float ccz = x*cy - y*cx;
    out[0] = v[0] + 2.0f*w*cx + 2.0f*ccx;
    out[1] = v[1] + 2.0f*w*cy + 2.0f*ccy;
    out[2] = v[2] + 2.0f*w*cz + 2.0f*ccz;
}

/**
 * SE(3) composition:  T_out = T1 @ T2.
 * Layout: [w, x, y, z, tx, ty, tz]
 */
__device__ __forceinline__
void se3_compose(const float* __restrict__ T1,
                 const float* __restrict__ T2,
                 float* __restrict__ T_out)
{
    // Quaternion part: q_out = q1 * q2
    quat_mul(T1, T2, T_out);
    // Translation part: t_out = R(q1) * t2 + t1
    float rotated[3];
    quat_rotate(T1, T2 + 4, rotated);
    T_out[4] = rotated[0] + T1[4];
    T_out[5] = rotated[1] + T1[5];
    T_out[6] = rotated[2] + T1[6];
}

/**
 * SE(3) exponential map.
 *
 * Tangent convention (jaxlie):
 *   tangent[0:3] = v      (linear)
 *   tangent[3:6] = omega  (angular)
 *
 * Output: [w, x, y, z, tx, ty, tz]
 *
 * For theta = ||omega|| near zero the result approaches (I, v) -- a pure
 * translation.  For larger angles we use the closed-form Rodrigues formula.
 */
__device__ __forceinline__
void se3_exp(const float* __restrict__ tangent,
             float* __restrict__ T_out)
{
    constexpr float EPS = 1e-6f;

    const float vx = tangent[0], vy = tangent[1], vz = tangent[2];
    const float ox = tangent[3], oy = tangent[4], oz = tangent[5];

    const float theta2 = ox*ox + oy*oy + oz*oz;
    const float theta  = sqrtf(theta2);

    if (theta < EPS) {
        // Near-zero rotation: identity quaternion, translation = v.
        T_out[0] = 1.0f; T_out[1] = 0.0f; T_out[2] = 0.0f; T_out[3] = 0.0f;
        T_out[4] = vx;   T_out[5] = vy;   T_out[6] = vz;
    } else {
        // ------------------------------------------------------------------
        // Rotation quaternion via half-angle Rodrigues.
        // ------------------------------------------------------------------
        const float half_theta = theta * 0.5f;
        const float s = sinf(half_theta) / theta;
        T_out[0] = cosf(half_theta);
        T_out[1] = s * ox;
        T_out[2] = s * oy;
        T_out[3] = s * oz;

        // ------------------------------------------------------------------
        // Translation:  t = V * v  where
        //   V = A*I + B*Omega + C*Omega^2
        //   A = sin(theta)/theta
        //   B = (1 - cos(theta)) / theta^2
        //   C = (theta - sin(theta)) / theta^3
        //
        // Omega * v  = omega x v
        // Omega^2 * v = omega x (omega x v) = omega*(omega.v) - v*theta^2
        // ------------------------------------------------------------------
        const float sin_t = sinf(theta);
        const float cos_t = cosf(theta);
        const float A = sin_t / theta;
        const float B = (1.0f - cos_t) / theta2;
        const float C = (theta - sin_t) / (theta2 * theta);

        // omega x v
        const float omv_x = oy*vz - oz*vy;
        const float omv_y = oz*vx - ox*vz;
        const float omv_z = ox*vy - oy*vx;

        // omega*(omega.v) - v*theta^2
        const float om_dot_v = ox*vx + oy*vy + oz*vz;
        const float om2v_x = ox*om_dot_v - theta2*vx;
        const float om2v_y = oy*om_dot_v - theta2*vy;
        const float om2v_z = oz*om_dot_v - theta2*vz;

        T_out[4] = A*vx + B*omv_x + C*om2v_x;
        T_out[5] = A*vy + B*omv_y + C*om2v_y;
        T_out[6] = A*vz + B*omv_z + C*om2v_z;
    }
}

// ---------------------------------------------------------------------------
// FK kernel
// ---------------------------------------------------------------------------

/**
 * One CUDA thread handles one batch element.
 * Joints are traversed sequentially in topological order so every parent
 * transform is ready before any child reads it.
 *
 * @param cfg            (batch, n_act)        float32  actuated config
 * @param twists         (n_joints, 6)         float32  Lie-algebra twist / joint
 * @param parent_tf      (n_joints, 7)         float32  constant T_parent_joint [wxyz_xyz]
 * @param parent_idx     (n_joints,)           int32    original parent joint idx, -1 for roots
 * @param act_idx        (n_joints,)           int32    actuated joint source idx, -1 if fixed
 * @param mimic_mul      (n_joints,)           float32  mimic multiplier (1.0 for non-mimic)
 * @param mimic_off      (n_joints,)           float32  mimic offset (0.0 for non-mimic)
 * @param mimic_act_idx  (n_joints,)           int32    mimicked actuated idx, -1 if not mimic
 * @param topo_inv       (n_joints,)           int32    topo_sort_inv: sorted_i -> orig_j
 * @param out            (batch, n_joints, 7)  float32  world transforms, orig-joint-indexed
 */
__global__
void fk_kernel(const float* __restrict__ cfg,
               const float* __restrict__ twists,
               const float* __restrict__ parent_tf,
               const int*   __restrict__ parent_idx,
               const int*   __restrict__ act_idx,
               const float* __restrict__ mimic_mul,
               const float* __restrict__ mimic_off,
               const int*   __restrict__ mimic_act_idx,
               const int*   __restrict__ topo_inv,
               float*       __restrict__ out,
               int batch, int n_joints, int n_act)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    const float* cfg_b = cfg + (long long)b * n_act;
    float*       out_b = out + (long long)b * n_joints * 7;

    for (int i = 0; i < n_joints; ++i) {
        const int j = topo_inv[i];  // original joint index

        // ------------------------------------------------------------------
        // Expand actuated config to full joint value q_j.
        //
        // Mirrors JointInfo._map_to_full_joint_space():
        //   src = mimic_act_idx[j] if mimic, else act_idx[j]
        //   q_ref = cfg[src]  (0 when src == -1, i.e. fixed joint)
        //   q_j   = q_ref * mimic_mul[j] + mimic_off[j]
        //
        // For non-mimic joints: mimic_mul=1, mimic_off=0, so q_j = cfg[act_idx[j]].
        // For fixed joints:     act_idx=-1,  src=-1,       so q_j = 0.
        // ------------------------------------------------------------------
        const int m_idx = mimic_act_idx[j];
        const int a_idx = act_idx[j];
        const int src   = (m_idx != -1) ? m_idx : a_idx;
        const float q_ref = (src == -1) ? 0.0f : cfg_b[src];
        const float q_j   = q_ref * mimic_mul[j] + mimic_off[j];

        // ------------------------------------------------------------------
        // tangent = twists[j] * q_j,  then delta_T = SE3.exp(tangent)
        // ------------------------------------------------------------------
        float tangent[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k)
            tangent[k] = twists[j * 6 + k] * q_j;

        float delta_T[7];
        se3_exp(tangent, delta_T);

        // ------------------------------------------------------------------
        // T_parent_child = parent_tf[j] @ delta_T
        // ------------------------------------------------------------------
        float T_pc[7];
        se3_compose(parent_tf + j * 7, delta_T, T_pc);

        // ------------------------------------------------------------------
        // T_world_child = T_world[parent_idx[j]] @ T_pc   (root: = T_pc)
        // ------------------------------------------------------------------
        float* dst = out_b + j * 7;
        const int p = parent_idx[j];
        if (p == -1) {
            #pragma unroll
            for (int k = 0; k < 7; ++k) dst[k] = T_pc[k];
        } else {
            se3_compose(out_b + p * 7, T_pc, dst);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error FkCudaImpl(
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
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out)
{
    // Derive dimensions from buffer shapes — avoids passing redundant
    // attributes across the Python/C++ boundary.
    // cfg:    (batch, n_act),  twists: (n_joints, 6)
    const int batch    = static_cast<int>(cfg.dimensions()[0]);
    const int n_act    = static_cast<int>(cfg.dimensions()[1]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);

    constexpr int THREADS = 256;
    const int blocks = (batch + THREADS - 1) / THREADS;

    fk_kernel<<<blocks, THREADS, 0, stream>>>(
        cfg.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        out->typed_data(),
        batch,
        n_joints,
        n_act);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FkCudaFfi, FkCudaImpl,
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out
