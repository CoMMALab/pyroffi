"""All pyroffi interop for spasm-pyroffi (replaces original kinematics/).

Loads SPaSM's own sphere URDF into pyroffi so the collision-sphere set is
identical to the original. q layout: 7-dof arm (fingers visual-only).
"""
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy
import pyroffi as pk

# pyroffi's IK primitives force jax_enable_x64 globally; SPaSM runs in f32.
# The port only uses pyroffi's pure-JAX paths, which are f32-safe.
jax.config.update("jax_enable_x64", False)

from spasm.paths import SPASM_URDF, require

require(SPASM_URDF, 'stock SPaSM checkout (commalab/spasm)')
EE_LINK = 'panda_grasptarget'

_urdf = yourdfpy.URDF.load(SPASM_URDF, load_collision_meshes=True)
ROBOT = pk.Robot.from_urdf(_urdf)
ROBOT_COLL = pk.collision.RobotCollisionSpherized.from_urdf(_urdf)

_N_ACT = len(ROBOT.joints.actuated_names)  # 8 (7 arm + finger1; finger2 mimics)
_EE_IDX = ROBOT.links.names.index(EE_LINK)

# Static mask of real (non-padding) spheres; padding radius is -1e9 (config-
# independent). Computed from at_config output so the flatten order matches
# fk's (at_config yields batch (S, N); the stored coll is (N, S)).
_RADII_ALL = np.asarray(
    ROBOT_COLL.at_config(ROBOT, jnp.zeros(len(ROBOT.joints.actuated_names))).radius
).reshape(-1)
_VALID_IDX = jnp.array(np.nonzero(_RADII_ALL > 0.0)[0])
NUM_SPHERES = int(_VALID_IDX.shape[0])

# Lean-FK static tables: each collision sphere's link-local center, radius, and
# parent-link index, with padding dropped. Lets _fk transform points by link
# poses directly instead of building jaxlie SE3 over every link x sphere (incl.
# padding) as at_config does — ~1.4x faster and it's the FK-heavy trajopt
# stage's hot path. coll.coll batch order is (N_links, S_spheres).
_NL, _NS = ROBOT_COLL.coll.get_batch_axes()
_LOC_ALL = np.asarray(ROBOT_COLL.coll.pose.translation()).reshape(-1, 3)
_RAD_LEAN_ALL = np.asarray(ROBOT_COLL.coll.radius).reshape(-1)
_LINK_ALL = np.repeat(np.arange(_NL), _NS)
_LEAN_VALID = _RAD_LEAN_ALL > 0.0
_SPH_LOCAL = jnp.array(_LOC_ALL[_LEAN_VALID])          # (K, 3) link-frame centers
_SPH_RADII = jnp.array(_RAD_LEAN_ALL[_LEAN_VALID])     # (K,)
_SPH_LINK = jnp.array(_LINK_ALL[_LEAN_VALID])          # (K,) parent-link index


def _to_actuated(q):
    """Accept 7- or 9-dof q (original convention), map to pyroffi's 8 actuated."""
    q = q[:7]
    return jnp.concatenate([q, jnp.zeros(_N_ACT - 7, dtype=q.dtype)])


def _fk(q):
    """World-frame robot collision spheres. Returns (positions (K,3), radii (K,)).

    Transforms precomputed link-local sphere centers by the FK link poses
    directly (lean path); matches the at_config result to <1e-7. Sphere order
    is link-major and differs from at_config's, but every consumer treats the
    set order-agnostically (sums over spheres)."""
    Ts = jaxlie.SE3(ROBOT.forward_kinematics(_to_actuated(q)))  # (N,7)
    R = Ts.rotation().as_matrix()          # (N,3,3)
    t = Ts.translation()                   # (N,3)
    pos = jnp.einsum('kij,kj->ki', R[_SPH_LINK], _SPH_LOCAL) + t[_SPH_LINK]
    return pos, _SPH_RADII


fk = jax.jit(_fk)
fk_batched = jax.jit(jax.vmap(_fk))


def _fk_ee(q):
    """Collision spheres AND the grasp-link translation from ONE forward-
    kinematics call. Returns (pos (K,3), radii (K,), ee_xyz (3,)).

    `ROBOT.forward_kinematics` already computes every link pose, so the EE
    translation is just another index into the same result -- this fuses what
    `_fk` and `_get_ee_pose` otherwise recompute independently (matches both to
    <1e-7: same chain, different link indices)."""
    Ts = jaxlie.SE3(ROBOT.forward_kinematics(_to_actuated(q)))  # (N,7)
    R = Ts.rotation().as_matrix()          # (N,3,3)
    t = Ts.translation()                   # (N,3)
    pos = jnp.einsum('kij,kj->ki', R[_SPH_LINK], _SPH_LOCAL) + t[_SPH_LINK]
    return pos, _SPH_RADII, t[_EE_IDX]


fk_ee = jax.jit(_fk_ee)


def _transform_coll_one(Ts_link_world_wxyz_xyz):
    """Apply one unbatched (N,7) set of link poses to ROBOT_COLL.coll (N,S),
    replicating RobotCollisionSpherized.at_config's own vmap-over-links/
    swapaxes trick (see pyroffi/src/pyroffi/collision/_robot_collision.py,
    the `at_config` on RobotCollisionSpherized) -- needed because we can't
    call at_config() itself (it doesn't take use_cuda) and jaxlie broadcasts
    from the right, which would misalign coll's (N, S) batch axes against a
    naive (N, 7) transform otherwise."""
    Ts_link_world = jaxlie.SE3(Ts_link_world_wxyz_xyz)
    coll_n_s = jax.vmap(lambda ts, c: c.transform(ts), in_axes=(-2, 0), out_axes=0)(
        Ts_link_world, ROBOT_COLL.coll)
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), coll_n_s)


def fk_cuda(q_batch):
    """Leading-dim-batched CUDA FK. q_batch: (B, 7 or 9) f32 -> (positions
    (B,K,3), radii (B,K)) f32, the same sphere contract as fk_batched.

    Dispatches to pyroffi's CUDA FFI FK kernel (Robot.forward_kinematics(...,
    use_cuda=True)) instead of the pure-JAX joint-chain. That kernel call
    itself must NOT be vmapped (the FFI target has no batching rule -- only
    leading-dim batches are supported, verified: vmap raises
    NotImplementedError: vmap_method=None). We batch it directly on the
    leading axis, then vmap only the (pure-JAX, vmap-safe) coll.transform
    step over that same axis to build the sphere geometry, matching
    RobotCollisionSpherized.at_config's own logic exactly.

    Casts to f32 at this boundary: backend.py runs with jax_enable_x64=False
    so pyroffi's CUDA FK kernel already computes in f32 here (verified
    empirically -- output dtype is float32, matching fk_batched to <3e-7).
    The explicit casts below are defensive in case x64 gets re-enabled
    transiently elsewhere in the process (see PORT_NOTES.md's f64 trap)."""
    q_batch = jnp.asarray(q_batch, dtype=jnp.float32)
    cfg = jax.vmap(_to_actuated)(q_batch)
    Ts = ROBOT.forward_kinematics(cfg, use_cuda=True)  # (B, N, 7), FFI, leading-dim batch
    Ts = Ts.astype(jnp.float32)
    geom = jax.vmap(_transform_coll_one)(Ts)  # pure-JAX vmap over B, safe
    B = q_batch.shape[0]
    pos = geom.pose.wxyz_xyz[..., 4:].reshape(B, -1, 3)[:, _VALID_IDX].astype(jnp.float32)
    radii = geom.radius.reshape(B, -1)[:, _VALID_IDX].astype(jnp.float32)
    return pos, radii


fk_cuda = jax.jit(fk_cuda)


def _get_ee_pose(q):
    """4x4 world pose of the grasp-target link."""
    wxyz_xyz = ROBOT.forward_kinematics(_to_actuated(q))[_EE_IDX]
    return jaxlie.SE3(wxyz_xyz).as_matrix()


get_ee_pose = jax.jit(_get_ee_pose)


def get_joint_limits():
    """(lower (7,), upper (7,)) for the arm joints."""
    return ROBOT.joints.lower_limits[:7], ROBOT.joints.upper_limits[:7]


def ik_numeric(target_pose_4x4, q_ref=None, solver='ls'):
    """General-robot numeric IK via pyroffi. Returns q (7,)."""
    target = jaxlie.SE3.from_matrix(target_pose_4x4)
    fixed_mask = jnp.array([n.startswith('panda_finger') for n in ROBOT.joints.actuated_names])
    prev = _to_actuated(q_ref) if q_ref is not None else None
    q = ROBOT.inverse_kinematics(
        target_link_name=EE_LINK, target_pose=target,
        rng_key=jax.random.PRNGKey(0), previous_cfg=prev,
        solver=solver, fixed_joint_mask=fixed_mask,
    )
    return q[:7]


def jax_cache_on():
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_spasm_pyroffi")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ---------------------------------------------------------------------------
# Analytic Franka IK, copied verbatim from the original kinematics/ik.py
# (He & Liu 2021 geometric solver). Pure JAX, backend-free.
# ---------------------------------------------------------------------------

def analytic_ik(O_T_EE, q7):
    invalid_solution = jnp.full((4,), False)
    q_all = jnp.full((4, 7), jnp.nan)

    assert O_T_EE.shape == (4, 4)

    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = a4**2 + d3**2
    LL46 = a4**2 + d5**2
    L24 = jnp.sqrt(LL24)
    L46 = jnp.sqrt(LL46)

    thetaH46 = jnp.arctan(d5 / a4)
    theta342 = jnp.arctan(d3 / a4)
    theta46H = jnp.arctan(a4 / d5)

    q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    invalid_solution |= (q7 <= q_min[6]) | (q7 >= q_max[6])

    q_all = q_all.at[:, 6].set(q7)

    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE

    x_EE_6 = jnp.array([jnp.cos(q7 - jnp.pi / 4), -jnp.sin(q7 - jnp.pi / 4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 /= jnp.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = jnp.array([0.0, 0.0, d1])
    V26 = p_6 - p_2
    LL26 = jnp.dot(V26, V26)
    L26 = jnp.sqrt(LL26)

    invalid_solution |= (L24 + L46 < L26) | (L24 + L26 < L46) | (L26 + L46 < L24)

    theta246 = jnp.arccos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * jnp.pi

    invalid_solution |= (q4 <= q_min[3]) | (q4 >= q_max[3])

    q_all = q_all.at[:, 3].set(q4)

    theta462 = jnp.arccos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * jnp.cos(theta26H)

    Z_6 = jnp.cross(z_EE, x_6)
    Y_6 = jnp.cross(Z_6, x_6)
    R_6 = jnp.vstack([x_6, Y_6 / jnp.linalg.norm(Y_6), Z_6 / jnp.linalg.norm(Z_6)]).T
    V_6_62 = R_6.T @ (-V26)

    Phi6 = jnp.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = jnp.arcsin(D26 / jnp.sqrt(V_6_62[0]**2 + V_6_62[1]**2))

    q6_0 = jnp.pi - Theta6 - Phi6
    q6_1 = Theta6 - Phi6
    q6 = jnp.array([q6_0, q6_1])

    for i in range(2):
        q6 = q6.at[i].set(jnp.where(q6[i] <= q_min[5], q6[i] + 2.0 * jnp.pi, q6[i]))
        q6 = q6.at[i].set(jnp.where(q6[i] >= q_max[5], q6[i] - 2.0 * jnp.pi, q6[i]))

        invalid = (q6[i] <= q_min[5]) | (q6[i] >= q_max[5])
        invalid_solution = invalid_solution.at[2*i].set(invalid_solution[2*i] | invalid)
        invalid_solution = invalid_solution.at[2*i + 1].set(invalid_solution[2*i + 1] | invalid)

        q_all = q_all.at[2 * i, 5].set(q6[i])
        q_all = q_all.at[2 * i + 1, 5].set(q6[i])

    invalid_solution |= ~jnp.isfinite(q_all[0, 5])
    invalid_solution |= ~jnp.isfinite(q_all[2, 5])

    thetaP26 = 3.0 * jnp.pi / 2.0 - theta462 - theta246 - theta342
    thetaP = jnp.pi - thetaP26 - theta26H
    LP6 = L26 * jnp.sin(thetaP26) / jnp.sin(thetaP)

    z_5_all = jnp.empty((4, 3))
    V2P_all = jnp.empty((4, 3))

    for i in range(2):
        z_6_5 = jnp.array([jnp.sin(q6[i]), jnp.cos(q6[i]), 0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - LP6 * z_5 - p_2

        z_5_all = z_5_all.at[2 * i].set(z_5)
        z_5_all = z_5_all.at[2 * i + 1].set(z_5)

        V2P_all = V2P_all.at[2 * i].set(V2P)
        V2P_all = V2P_all.at[2 * i + 1].set(V2P)

        L2P = jnp.linalg.norm(V2P)

        invalid = jnp.abs(V2P[2] / L2P) > 0.999
        invalid_solution = invalid_solution.at[2*i].set(invalid_solution[2*i] | invalid)
        invalid_solution = invalid_solution.at[2*i + 1].set(invalid_solution[2*i + 1] | invalid)

        q_all = q_all.at[2 * i, 0].set(jnp.atan2(V2P[1], V2P[0]))
        q_all = q_all.at[2*i, 1].set(jnp.arccos(V2P[2] / L2P))

        q_all = q_all.at[2*i+1, 0].set(jnp.where(q_all[2*i, 0] < 0,
                                                 q_all[2*i, 0] + jnp.pi,
                                                 q_all[2*i, 0] - jnp.pi))

        q_all = q_all.at[2 * i + 1, 1].set(-q_all[2 * i, 1])

    for i in range(4):
        invalid = (q_all[i, 0] <= q_min[0]) | (q_all[i, 0] >= q_max[0]) | \
                  (q_all[i, 1] <= q_min[1]) | (q_all[i, 1] >= q_max[1])

        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)

        z_3 = V2P_all[i] / jnp.linalg.norm(V2P_all[i])
        Y_3 = -jnp.cross(V26, V2P_all[i])
        y_3 = Y_3 / jnp.linalg.norm(Y_3)
        x_3 = jnp.cross(y_3, z_3)

        c1 = jnp.cos(q_all[i, 0])
        s1 = jnp.sin(q_all[i, 0])
        c2 = jnp.cos(q_all[i, 1])
        s2 = jnp.sin(q_all[i, 1])

        R_1 = jnp.array([[c1, -s1, 0.0],
                         [s1,  c1, 0.0],
                         [0.0, 0.0, 1.0]])
        R_1_2 = jnp.array([[ c2, -s2, 0.0],
                           [0.0, 0.0, 1.0],
                           [-s2, -c2, 0.0]])
        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3
        q_all = q_all.at[i, 2].set(jnp.atan2(x_2_3[2], x_2_3[0]))

        invalid = (q_all[i, 2] <= q_min[2]) | (q_all[i, 2] >= q_max[2])
        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)

        VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i]
        c6 = jnp.cos(q_all[i, 5])
        s6 = jnp.sin(q_all[i, 5])
        R_5_6 = jnp.array([[c6, -s6,   0.0],
                           [0.0, 0.0, -1.0],
                           [s6,  c6,   0.0]])
        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4

        q_all = q_all.at[i, 4].set(-jnp.arctan2(V_5_H4[1], V_5_H4[0]))

        invalid = (q_all[i, 4] <= q_min[4]) | (q_all[i, 4] >= q_max[4])
        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)

    return q_all, invalid_solution


def analytic_ik_case_consistent(O_T_EE, q7, q_actual):
    """Case-consistent variant (verbatim port of the original)."""
    q = jnp.full(7, jnp.nan)
    invalid = False

    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = a4**2 + d3**2
    LL46 = a4**2 + d5**2
    L24 = jnp.sqrt(LL24)
    L46 = jnp.sqrt(LL46)

    thetaH46 = jnp.arctan(d5/a4)
    theta342 = jnp.arctan(d3/a4)
    theta46H = jnp.arctan(a4/d5)

    q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = jnp.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    invalid |= (q7 <= q_min[6]) | (q7 >= q_max[6])

    q = q.at[6].set(q7)

    c1_a = jnp.cos(q_actual[0]); s1_a = jnp.sin(q_actual[0])
    c2_a = jnp.cos(q_actual[1]); s2_a = jnp.sin(q_actual[1])
    c3_a = jnp.cos(q_actual[2]); s3_a = jnp.sin(q_actual[2])
    c4_a = jnp.cos(q_actual[3]); s4_a = jnp.sin(q_actual[3])
    c5_a = jnp.cos(q_actual[4]); s5_a = jnp.sin(q_actual[4])
    c6_a = jnp.cos(q_actual[5]); s6_a = jnp.sin(q_actual[5])

    As_a = jnp.zeros((7, 4, 4))
    As_a = As_a.at[0].set(jnp.array([[   c1_a, -s1_a,  0.0,  0.0],
                                     [   s1_a,  c1_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,   d1],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[1].set(jnp.array([[   c2_a, -s2_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,  0.0],
                                     [  -s2_a, -c2_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[2].set(jnp.array([[   c3_a, -s3_a,  0.0,  0.0],
                                     [    0.0,   0.0, -1.0,  -d3],
                                     [   s3_a,  c3_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[3].set(jnp.array([[   c4_a, -s4_a,  0.0,   a4],
                                     [    0.0,   0.0, -1.0,  0.0],
                                     [   s4_a,  c4_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[4].set(jnp.array([[    1.0,   0.0,  0.0,  -a4],
                                     [    0.0,   1.0,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[5].set(jnp.array([[   c5_a, -s5_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,   d5],
                                     [  -s5_a, -c5_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[6].set(jnp.array([[   c6_a, -s6_a,  0.0,  0.0],
                                     [    0.0,   0.0, -1.0,  0.0],
                                     [   s6_a,  c6_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))

    Ts_a = jnp.zeros((7, 4, 4))
    Ts_a = Ts_a.at[0].set(As_a[0])
    for j in range(1, 7):
        Ts_a = Ts_a.at[j].set(Ts_a[j - 1] @ As_a[j])

    V62_a = Ts_a[1][:3, 3] - Ts_a[6][:3, 3]
    V6H_a = Ts_a[4][:3, 3] - Ts_a[6][:3, 3]
    Z6_a = Ts_a[6][:3, 2]
    is_case6_0 = (jnp.dot(jnp.cross(V6H_a, V62_a), Z6_a) <= 0)

    is_case1_1 = (q_actual[1] < 0)

    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE

    x_EE_6 = jnp.array([jnp.cos(q7 - jnp.pi/4), -jnp.sin(q7 - jnp.pi/4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 /= jnp.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = jnp.array([0.0, 0.0, d1])
    V26 = p_6 - p_2

    LL26 = jnp.dot(V26, V26)
    L26 = jnp.sqrt(LL26)

    invalid |= (L24 + L46 < L26) | (L24 + L26 < L46) | (L26 + L46 < L24)

    theta246 = jnp.arccos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * jnp.pi
    invalid |= (q4 <= q_min[3]) | (q4 >= q_max[3])
    q = q.at[3].set(q4)

    theta462 = jnp.arccos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * jnp.cos(theta26H)

    Z_6 = jnp.cross(z_EE, x_6)
    Y_6 = jnp.cross(Z_6, x_6)
    R_6 = jnp.vstack([x_6, Y_6 / jnp.linalg.norm(Y_6), Z_6 / jnp.linalg.norm(Z_6)]).T
    V_6_62 = R_6.T @ (-V26)

    Phi6 = jnp.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = jnp.arcsin(D26 / jnp.sqrt(V_6_62[0]**2 + V_6_62[1]**2))

    q6 = jnp.where(is_case6_0, jnp.pi - Theta6 - Phi6, Theta6 - Phi6)

    q6 = jnp.where(q6 <= q_min[5], q6 + 2.0 * jnp.pi, q6)
    q6 = jnp.where(q6 >= q_max[5], q6 - 2.0 * jnp.pi, q6)

    invalid |= (q6 <= q_min[5]) | (q6 >= q_max[5])
    q = q.at[5].set(q6)

    thetaP26 = 3.0 * jnp.pi / 2.0 - theta462 - theta246 - theta342
    thetaP = jnp.pi - thetaP26 - theta26H
    LP6 = L26 * jnp.sin(thetaP26) / jnp.sin(thetaP)

    z_6_5 = jnp.array([jnp.sin(q[5]), jnp.cos(q[5]), 0.0])
    z_5 = R_6 @ z_6_5
    V2P = p_6 - LP6 * z_5 - p_2

    L2P = jnp.linalg.norm(V2P)

    singular = jnp.abs(V2P[2] / L2P) > 0.999
    q0 = jnp.where(singular, q_actual[0], jnp.arctan2(V2P[1], V2P[0]))
    q1 = jnp.where(singular, 0.0, jnp.arccos(V2P[2] / L2P))

    q0_final = jnp.where(is_case1_1, jnp.where(q0 < 0.0, q0 + jnp.pi, q0 - jnp.pi), q0)
    q1_final = jnp.where(is_case1_1, -q1, q1)

    q = q.at[0].set(q0_final)
    q = q.at[1].set(q1_final)

    invalid |= (q[0] <= q_min[0]) | (q[0] >= q_max[0]) | (q[1] <= q_min[1]) | (q[1] >= q_max[1])

    z_3 = V2P / jnp.linalg.norm(V2P)
    Y_3 = -jnp.cross(V26, V2P)
    y_3 = Y_3 / jnp.linalg.norm(Y_3)
    x_3 = jnp.cross(y_3, z_3)

    c1 = jnp.cos(q[0]); s1 = jnp.sin(q[0])
    R_1 = jnp.array([[ c1, -s1, 0.0], [ s1,  c1, 0.0], [0.0, 0.0, 1.0]])

    c2 = jnp.cos(q[1]); s2 = jnp.sin(q[1])
    R_1_2 = jnp.array([[ c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])

    R_2 = R_1 @ R_1_2
    x_2_3 = R_2.T @ x_3
    q2_val = jnp.arctan2(x_2_3[2], x_2_3[0])
    q = q.at[2].set(q2_val)

    invalid |= (q[2] <= q_min[2]) | (q[2] >= q_max[2])

    VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5
    c6 = jnp.cos(q[5]); s6 = jnp.sin(q[5])
    R_5_6 = jnp.array([[ c6, -s6, 0.0], [0.0, 0.0, -1.0], [ s6,  c6, 0.0]])
    R_5 = R_6 @ R_5_6.T
    V_5_H4 = R_5.T @ VH4

    q4_val = -jnp.arctan2(V_5_H4[1], V_5_H4[0])
    q = q.at[4].set(q4_val)

    invalid |= (q[4] <= q_min[4]) | (q[4] >= q_max[4])

    return q, invalid


def ik(target_pose_4x4, q_ref):
    """Analytic IK entry point: best valid solution nearest q_ref. Returns q (7,)."""
    q7s = jnp.linspace(-2.8970, 2.8970, 9)
    qs, invalids = jax.vmap(analytic_ik, in_axes=(None, 0))(target_pose_4x4, q7s)
    qs = qs.reshape(-1, 7)
    invalids = invalids.reshape(-1)
    dist = jnp.linalg.norm(qs - q_ref[None, :7], axis=-1)
    dist = jnp.where(invalids, jnp.inf, dist)
    return qs[jnp.argmin(dist)]
