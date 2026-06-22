"""Interactive: drag the Panda through a point-cloud field, RoboGPU vs CAPT.

Grab the transform gizmo and sweep the arm (via IK) through a wall of points.
The robot is drawn as its collision-sphere model; whenever a link's spheres
overlap the point cloud the whole link turns **red**, and the offending points
turn red too.  Two collision backends are evaluated every frame and reported in
the GUI panel so you can visually confirm they agree with what you see:

  * **RoboGPU**  — OptiX ray-tracing sphere-octree checker (GPU).  Uses the same
    pyroffi spherized model that drives the red colouring, so its verdict should
    match the "any link red" reference exactly.
  * **CAPT**     — VAMP's Collision-Affording Point Tree checker (CPU).  Uses
    VAMP's *own* internal spherization, so its verdict may differ slightly near
    grazing contacts — that divergence is exactly what this tool lets you see.

The brute-force per-sphere test (NumPy) is the ground-truth reference for the
red colouring.

Multi-GPU (pmap)
  RoboGPU's CUDA/OptiX kernels are device-safe under ``jax.pmap``: each visible
  GPU keeps its own pipeline/BVH/graph caches, so pmap shards a batch of
  configurations across all devices automatically.  To show this off, every
  frame we also fan a batch of perturbed candidate configurations out across all
  visible GPUs via ``jax.pmap`` and report the aggregate free-count + timing in
  the GUI.  Candidate ``(device 0, slot 0)`` is the exact dragged config, so its
  pmap verdict must match the single-device RoboGPU verdict (sanity check).

Run (inside the `pyroffi` conda env):
    # build the kernels first:
    #   bash build_kernels/build_robogpu_collision.sh
    #   bash build_kernels/build_cricket_jit.sh        (optional, for CAPT)
    python examples/13_02_robogpu_vs_capt_pointcloud.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import trimesh
import viser
import yourdfpy

import pyroffi as pk
from pyroffi.collision import RobotCollisionSpherized, RoboGPUCollisionChecker, Sphere
from pyroffi.collision._cuda_collision import _spherized_local_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
SPHERIZED_URDF = REPO_ROOT / "resources" / "panda" / "panda_spherized.urdf"
SRDF = REPO_ROOT / "resources" / "panda" / "panda.srdf"

TARGET_LINK = "panda_hand"
R_ENV = 0.02              # environment point sphere radius
N_POINTS = 1200           # point-cloud size
BLOB_CENTER = np.array([0.45, 0.0, 0.55], np.float32)  # small sphere location
BLOB_RADIUS = 0.10        # small sphere radius

# Multi-GPU pmap demo: candidate configs evaluated per GPU each frame.
N_PER_DEVICE = 256        # perturbed candidates checked on every visible GPU
PERTURB_STD = 0.15        # std-dev (rad) of the candidate perturbation


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_point_field(rng) -> np.ndarray:
    """A small dense sphere of points the arm sweeps through."""
    # Uniformly sample points inside a small ball so the robot isn't engulfed.
    dirs = rng.normal(size=(N_POINTS, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = BLOB_RADIUS * np.cbrt(rng.uniform(0.0, 1.0, N_POINTS))[:, None]
    return (BLOB_CENTER + dirs * radii).astype(np.float32)


def main() -> None:
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(SPHERIZED_URDF)))
    coll = RobotCollisionSpherized.from_urdf(yourdfpy.URDF.load(str(SPHERIZED_URDF)))

    NL = coll.num_links
    f_local = np.asarray(_spherized_local_geometry(coll))     # [K, 4]  k = s*NL + n
    K = f_local.shape[0]
    sphere_link = np.arange(K) % NL                            # link index per sphere
    sphere_valid = f_local[:, 3] > 0.0
    radii_all = f_local[:, 3].copy()
    r_robot_max = float(radii_all[sphere_valid].max())

    # Map each collision-model link to its forward-kinematics link index.
    fk_idx = np.array(
        [robot.links.names.index(name) for name in coll.link_names], dtype=np.int32
    )
    sphere_fk = fk_idx[sphere_link]                            # FK link idx per sphere

    f_local_j = jnp.asarray(f_local[:, :3])
    sphere_fk_j = jnp.asarray(sphere_fk)

    @jax.jit
    def world_spheres(cfg):
        """Return [K, 3] world-frame sphere centres for a single config."""
        link_poses = robot.forward_kinematics(cfg)            # [NL, 7] wxyz_xyz
        T = jaxlie.SE3(link_poses[sphere_fk_j])               # [K] SE3
        return T.apply(f_local_j)                              # [K, 3]

    # ── Point cloud + checkers ───────────────────────────────────────────────
    rng = np.random.default_rng(0)
    points = make_point_field(rng)
    points_j = jnp.asarray(points)
    far = Sphere.from_center_and_radius(
        center=jnp.array([[100.0, 100.0, 100.0]]), radius=jnp.array([0.01]))

    print("Building RoboGPU checker ...")
    robogpu = RoboGPUCollisionChecker(coll)
    # Disable self-collision so RoboGPU's verdict reflects ONLY point-cloud
    # contact — exactly what the red-link reference colouring shows.  (Self-
    # collision still works in production; we switch it off here so the
    # "RoboGPU == reference" indicator is a clean point-cloud comparison.)
    robogpu._f_pair_i = jnp.zeros((0,), dtype=jnp.int32)
    robogpu._f_pair_j = jnp.zeros((0,), dtype=jnp.int32)
    robogpu._cached_robot_id = None
    robogpu._jit_fn = None
    robogpu.set_world(far, point_cloud=points_j, r_env=R_ENV)

    # ── Multi-GPU pmap setup ─────────────────────────────────────────────────
    # A single warmup call builds the checker's jitted FFI function (which closes
    # over the static robot/world geometry).  We then wrap it in jax.pmap so a
    # config batch shaped [n_devices, N_PER_DEVICE, n_act] is sharded across every
    # visible GPU — one shard per device, each running on its own CUDA stream and
    # per-device kernel caches.
    n_devices = jax.device_count()
    n_act = robot.joints.lower_limits.shape[0]
    mid_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2.0
    robogpu.check_collision_free(robot, mid_cfg[None, :]).block_until_ready()
    pmap_check = jax.pmap(robogpu._jit_fn)  # [D, P, n_act] -> [D, P] int32 (1=free)
    pmap_key = jax.random.PRNGKey(0)
    print(f"pmap RoboGPU over {n_devices} GPU(s): "
          f"{n_devices * N_PER_DEVICE} candidates/frame")

    capt = None
    try:
        from pyroffi.collision import VAMPCPUCollisionChecker
        print("Building CAPT (VAMP) checker ... (first run JIT-compiles)")
        capt = VAMPCPUCollisionChecker(SPHERIZED_URDF, srdf_path=SRDF)
        capt.set_world(
            far, point_cloud=points_j,
            capt_r_min=0.0, capt_r_max=r_robot_max, capt_r_point=R_ENV,
        )
        print("  CAPT ready.")
    except Exception as exc:
        print(f"  CAPT unavailable ({exc}); continuing with RoboGPU only.")

    # ── Viser scene ──────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0)

    # One icosphere mesh, instanced once per collision sphere.
    unit = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    verts = np.asarray(unit.vertices, dtype=np.float32)
    faces = np.asarray(unit.faces, dtype=np.uint32)

    n_show = int(sphere_valid.sum())
    show_idx = np.where(sphere_valid)[0]
    sphere_handle = server.scene.add_batched_meshes_simple(
        "/robot_spheres",
        vertices=verts,
        faces=faces,
        batched_positions=np.zeros((n_show, 3), np.float32),
        batched_wxyzs=np.tile(np.array([1, 0, 0, 0], np.float32), (n_show, 1)),
        batched_scales=radii_all[show_idx].astype(np.float32),
        batched_colors=np.tile(np.array([90, 200, 255], np.uint8), (n_show, 1)),
    )

    pc_colors = np.tile(np.array([160, 160, 160], np.uint8), (len(points), 1))
    pc_handle = server.scene.add_point_cloud(
        "/point_cloud", points=points, colors=pc_colors, point_size=R_ENV,
        point_shape="circle",
    )

    # IK drag target.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.45, 0.0, 0.55), wxyz=(0, 1, 0, 0)
    )

    # GUI readouts.
    g_ref = server.gui.add_text("Reference (red links)", "—", disabled=True)
    g_rg = server.gui.add_text("RoboGPU verdict", "—", disabled=True)
    g_capt = server.gui.add_text("CAPT verdict", "—", disabled=True)
    g_agree = server.gui.add_text("RoboGPU == reference", "—", disabled=True)
    g_t_rg = server.gui.add_number("RoboGPU (us)", 0.0, disabled=True)
    g_t_capt = server.gui.add_number("CAPT (us)", 0.0, disabled=True)
    g_pmap = server.gui.add_text(
        f"pmap free / total ({n_devices} GPUs)", "—", disabled=True)
    g_t_pmap = server.gui.add_number("pmap batch (us)", 0.0, disabled=True)
    g_pmap_agree = server.gui.add_text("pmap[0,0] == single", "—", disabled=True)

    RED = np.array([220, 40, 40], np.uint8)
    BLUE = np.array([90, 200, 255], np.uint8)
    GREY = np.array([160, 160, 160], np.uint8)
    PT_RED = np.array([240, 60, 60], np.uint8)

    target_idx = robot.links.names.index(TARGET_LINK)
    ik_solve = jax.jit(
        lambda pose, key, prev: robot.inverse_kinematics(
            target_link_name=TARGET_LINK, target_pose=pose,
            rng_key=key, previous_cfg=prev,
        )
    )
    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    while True:
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target.wxyz)),
            translation=jnp.array(ik_target.position),
        )
        rng_key, subkey = jax.random.split(rng_key)
        solution = ik_solve(target_pose, subkey, solution)
        solution.block_until_ready()
        cfg = solution

        # World-frame collision spheres + brute-force reference vs the cloud.
        centers = np.asarray(world_spheres(cfg))               # [K, 3]
        d2 = ((centers[show_idx][:, None, :] - points[None, :, :]) ** 2).sum(-1)
        rsum = (radii_all[show_idx][:, None] + R_ENV) ** 2      # [n_show, 1]
        sphere_hit = np.any(d2 < rsum, axis=1)                  # [n_show] bool

        # Per-link: a link is red if any of its spheres collide.
        link_hit = np.zeros(NL, dtype=bool)
        hit_links = sphere_link[show_idx][sphere_hit]
        link_hit[hit_links] = True
        # Colour each shown sphere by whether its LINK is in collision.
        link_of_show = sphere_link[show_idx]
        col = np.where(link_hit[link_of_show][:, None], RED, BLUE).astype(np.uint8)

        # Colliding points → red.
        pt_hit = np.any(d2 < rsum, axis=0)                      # [Mp] bool
        pcol = np.where(pt_hit[:, None], PT_RED, GREY).astype(np.uint8)

        sphere_handle.batched_positions = centers[show_idx].astype(np.float32)
        sphere_handle.batched_colors = col
        pc_handle.colors = pcol

        ref_collision = bool(link_hit.any())
        g_ref.value = ("COLLISION" if ref_collision else "free") + \
            f"  ({int(link_hit.sum())} links)"

        # RoboGPU verdict (1 = free).
        t0 = time.perf_counter()
        rg_free = bool(np.asarray(
            robogpu.check_collision_free(robot, cfg[None, :])).reshape(()))
        g_t_rg.value = (time.perf_counter() - t0) * 1e6
        g_rg.value = "free" if rg_free else "COLLISION"
        g_agree.value = "yes" if (rg_free != ref_collision) else "NO — mismatch!"

        # ── Multi-GPU pmap batch: perturbed candidates fanned across all GPUs ──
        # Build [n_devices, N_PER_DEVICE, n_act].  Slot (0, 0) is the exact dragged
        # config (zero perturbation) so its verdict must match the single-device
        # RoboGPU result above — a live correctness check of the pmap path.
        pmap_key, sub = jax.random.split(pmap_key)
        noise = PERTURB_STD * jax.random.normal(
            sub, (n_devices, N_PER_DEVICE, n_act), dtype=jnp.float32)
        noise = noise.at[0, 0].set(0.0)
        cands = cfg[None, None, :] + noise
        t0 = time.perf_counter()
        verdicts = pmap_check(cands)           # [n_devices, N_PER_DEVICE] int32
        verdicts.block_until_ready()
        g_t_pmap.value = (time.perf_counter() - t0) * 1e6
        verdicts_np = np.asarray(verdicts)
        n_total = verdicts_np.size
        n_free = int(verdicts_np.sum())
        g_pmap.value = f"{n_free} / {n_total}"
        pmap_ref_free = bool(verdicts_np[0, 0])
        g_pmap_agree.value = "yes" if (pmap_ref_free == rg_free) else "NO — mismatch!"

        # CAPT verdict.
        if capt is not None:
            t0 = time.perf_counter()
            capt_free = bool(np.asarray(
                capt.check_collision_free(None, cfg[None, :])).reshape(()))
            g_t_capt.value = (time.perf_counter() - t0) * 1e6
            g_capt.value = "free" if capt_free else "COLLISION"
        else:
            g_capt.value = "n/a"

        time.sleep(0.02)


if __name__ == "__main__":
    main()
