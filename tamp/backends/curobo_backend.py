"""cuRobo geometric oracle for PDDLStream — the state-of-the-art GPU baseline.

cuRobo (NVIDIA) is the GPU robot-kinematics library cuTAMP uses for its own
geometry: ``CudaRobotModel`` for kinematics, ``IKSolver`` for inverse
kinematics, ``SelfCollisionCost`` and ``MotionGen`` for collision and motion.
Extracting "cuTAMP's oracle" therefore means using cuRobo directly, which is
both cleaner to describe and honest about provenance — cuTAMP's own
contribution is the particle optimiser above it, not the geometry beneath.

This is the baseline pyroffi's claim is really measured against: both are
GPU-accelerated, batched, differentiable robot-kinematics libraries, so the
comparison is like-for-like in a way pybullet is not.

Implements the same three primitives as the other validators::

    ik_topdown(pose, grasp_yaw, approach) -> (q7, reachable)
    arm_path_valid(q_path)                -> bool
    interpolate(q1, q2, n)                -> (n, 7)

Expect a batch-size effect
--------------------------
cuRobo is engineered for *batched* queries — ``solve_batch`` over thousands of
targets is where its design pays off. PDDLStream's ``s-ik`` and ``s-motion``
streams issue exactly one query at a time, so this backend runs in the regime
it is least suited to, and per-call latency will be dominated by launch and
transfer overhead rather than by solve time. That is a real property of the
*interface*, not a defect in cuRobo, and it is worth reporting as such: the
same effect showed up for pyroffi's own CUDA kernel, whose crossover against
the JAX path sat near batch 20.

:func:`ik_batch` is exposed precisely so that effect can be measured rather
than assumed — it is the entry point for a parallel-scaling study.

Runs in cuRobo's own conda environment; it cannot share an interpreter with the
JAX stack, so the benchmark harness invokes it in a subprocess.
"""
from __future__ import annotations

import functools

import numpy as np

TOP_DOWN_APPROACH = 0.10

#: Table height, matching ``geometry.FLOOR_Z``.
FLOOR_Z = -0.035

REST_POSE = (0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4)

#: cuRobo ships a Franka config; using it rather than converting SPaSM's URDF
#: keeps the backend in its supported configuration. The collision geometry
#: therefore differs from the sphere model the other validators share — noted
#: rather than hidden, since it means cuRobo's collision results are not
#: sphere-for-sphere identical to pyroffi's.
ROBOT_CONFIG = "franka.yml"


@functools.lru_cache(maxsize=1)
def _solver():
    """Build the IK solver and collision model once (both are expensive)."""
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    cfg = load_yaml(join_path(get_robot_configs_path(), ROBOT_CONFIG))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(cfg, tensor_args)

    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, rotation_threshold=0.05, position_threshold=0.005,
        num_seeds=20, self_collision_check=True, self_collision_opt=True,
        tensor_args=tensor_args,
        # CUDA graphs lock the batch size: a solver captured at batch 24 raises
        # "changing goal type, cuda graph reset not available" when handed a
        # single query. PDDLStream issues batch 1 while the scaling study sweeps
        # many sizes, so the graph is disabled to let one solver serve both.
        # This costs cuRobo some launch-overhead advantage, and that is worth
        # stating: a deployment fixed at one batch size could re-enable it.
        use_cuda_graph=False,
    )
    ik = IKSolver(ik_cfg)

    world_cfg = RobotWorldConfig.load_from_config(
        robot_cfg, None, collision_activation_distance=0.0,
        tensor_args=tensor_args)
    world = RobotWorld(world_cfg)
    return dict(ik=ik, world=world, tensor_args=tensor_args)


def _topdown_quat(yaw):
    """Top-down grasp orientation as ``(w, x, y, z)``, cuRobo's convention.

    Roll = pi (z-axis down) composed with the requested yaw about world z.
    """
    half = 0.5 * yaw
    # q = Rz(yaw) * Rx(pi)
    return np.array([0.0, np.cos(half), np.sin(half), 0.0], dtype=np.float32)


def ik_batch(poses, grasp_yaws=None, approach=TOP_DOWN_APPROACH):
    """Batched top-down IK. ``poses`` is ``(B, 4)``; returns ``(q[B,7], ok[B])``.

    The batched entry point exists so the parallel-scaling behaviour can be
    measured directly. cuRobo's per-call cost is nearly flat in batch size over
    a wide range, so single-query use throws away most of what it offers.
    """
    import torch
    from curobo.types.math import Pose

    s = _solver()
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    yaws = (poses[:, 3] if grasp_yaws is None
            else poses[:, 3] + np.asarray(grasp_yaws, dtype=float))

    pos = np.stack([poses[:, 0], poses[:, 1], poses[:, 2] + approach], axis=-1)
    quat = np.stack([_topdown_quat(float(y)) for y in yaws], axis=0)

    goal = Pose(
        position=torch.as_tensor(pos, dtype=torch.float32, device="cuda"),
        quaternion=torch.as_tensor(quat, dtype=torch.float32, device="cuda"),
    )
    result = s["ik"].solve_batch(goal)

    q = result.solution.squeeze(1).detach().cpu().numpy()[:, :7]
    ok = result.success.squeeze(-1).detach().cpu().numpy().astype(bool)
    return q, ok


def ik_topdown(pose, grasp_yaw=0.0, approach=TOP_DOWN_APPROACH):
    """Single-query top-down IK, the form PDDLStream's ``s-ik`` needs."""
    q, ok = ik_batch(np.asarray(pose)[None, :],
                     np.asarray([grasp_yaw]), approach)
    return q[0], bool(ok[0])


def arm_path_valid(q_path, floor_z=FLOOR_Z):
    """Joint limits, self-collision and floor clearance along a path.

    The whole path goes to the GPU in one call rather than waypoint by
    waypoint — this is the one place PDDLStream's interface happens to hand a
    GPU backend a batch, and not exploiting it would understate cuRobo.
    """
    import torch

    s = _solver()
    q_path = np.asarray(q_path)[:, :7]

    lower, upper = _joint_limits()
    if np.any(q_path < lower - 1e-3) or np.any(q_path > upper + 1e-3):
        return False

    q = torch.as_tensor(q_path, dtype=torch.float32, device="cuda")
    d_self = s["world"].get_self_collision_distance(
        s["world"].get_kinematics(q).link_spheres_tensor.unsqueeze(1))
    if float(d_self.max()) > 0.0:            # positive == penetrating
        return False

    spheres = s["world"].get_kinematics(q).link_spheres_tensor
    lowest = float((spheres[..., 2] - spheres[..., 3]).min())
    return lowest >= floor_z


@functools.lru_cache(maxsize=1)
def _joint_limits():
    s = _solver()
    lim = s["ik"].robot_config.kinematics.get_joint_limits()
    lower = lim.position[0].detach().cpu().numpy()[:7]
    upper = lim.position[1].detach().cpu().numpy()[:7]
    return lower, upper


def arm_paths_valid(paths, floor_z=FLOOR_Z):
    """Batched path validation: ``[N, T, 7]`` -> ``[N]`` bool.

    All N*T waypoints go to the GPU as ONE kinematics call. Validating paths one
    at a time -- as an interface like PDDLStream's forces -- hides cuRobo's
    entire design advantage: its per-call cost is nearly flat in batch size, so
    N separate calls cost N times a fixed overhead that one call would pay once.
    This entry point exists so the comparison measures cuRobo batched against
    pyroffi batched, rather than cuRobo serial against pyroffi batched.
    """
    import torch

    s = _solver()
    paths = np.asarray(paths)[:, :, :7]
    N, T = paths.shape[:2]

    lower, upper = _joint_limits()
    in_lim = np.all((paths >= lower - 1e-3) & (paths <= upper + 1e-3), axis=(1, 2))

    q = torch.as_tensor(paths.reshape(N * T, 7), dtype=torch.float32,
                        device="cuda")
    spheres = s["world"].get_kinematics(q).link_spheres_tensor   # [N*T, K, 4]

    d_self = s["world"].get_self_collision_distance(spheres.unsqueeze(1))
    self_ok = (d_self.view(N, T).max(dim=1).values <= 0.0).cpu().numpy()

    lowest = (spheres[..., 2] - spheres[..., 3]).min(dim=-1).values
    floor_ok = (lowest.view(N, T).min(dim=1).values >= floor_z).cpu().numpy()

    return in_lim & self_ok & floor_ok


def interpolate(q1, q2, n=20):
    """Straight-line joint path, identical to the other validators'."""
    q1 = np.asarray(q1)[:7]
    q2 = np.asarray(q2)[:7]
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - ts) * q1[None] + ts * q2[None]
