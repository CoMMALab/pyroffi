"""Every path, task constant and solver default in one place.

Before this module these lived wherever they were first needed, and the
consequences were the kind that waste an afternoon: `study3` imported the URDF
paths from `study0_segment_ablation`, the ground-truth weights from
`recovery_bench`, and the held-out scene offsets from `generalization_check` --
so three "experiments" had to be importable, and stay importable, for any
fourth one to run at all.  Nothing here imports anything from `iosp`.

`THETA_IK_STAR` / `Z_TRAJOPT_STAR` are the DEMONSTRATOR's cost: every synthetic
demonstration in this package is a rollout of the composed model at these
values, which is what makes "did recovery work" a question with an exact
answer rather than a judgement call.
"""

import pathlib

import jax.numpy as jnp

# -- resources --------------------------------------------------------------
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
URDF_PATH = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF_PATH = RESOURCE_ROOT / "panda" / "panda.srdf"
MESH_DIR = RESOURCE_ROOT / "panda" / "meshes"

# XLA compile on this composed chain is MEASURED at ~1486s for a single module,
# and every experiment here shares the approach/grasp/place subgraphs, so the
# persistent cache is the difference between a 25-minute rerun and a 2-minute
# one.  `enable_compilation_cache()` must be called before the first trace.
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "data" / "jax_cache"


def enable_compilation_cache(min_compile_secs=5.0):
    """Point JAX at the shared on-disk compile cache.  Call before any jit."""
    import jax
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)


# -- the demonstrator's cost (ground truth) ---------------------------------
THETA_IK_STAR = jnp.array([0.06, 0.04], dtype=jnp.float32)      # grasp/place standoff, m
Z_TRAJOPT_STAR = jnp.array([1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0], dtype=jnp.float32)
Z_FULL_STAR = jnp.array([1.0, 2.0, 0.5, 1.5], dtype=jnp.float32)  # refine: smooth, clearance, upright, skeleton

# -- the canonical task ------------------------------------------------------
Q_START = jnp.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=jnp.float32)
PICK_POS = jnp.array([0.4, 0.2, 0.3], dtype=jnp.float32)
PLACE_POS = jnp.array([0.4, -0.2, 0.3], dtype=jnp.float32)
OBS_CENTER = jnp.array([0.3, 0.0, 0.4], dtype=jnp.float32)
OBS_RADIUS = jnp.array([0.05], dtype=jnp.float32)

# Held-out scene B: displacements from scene A, scaled by `scene_b(scale)`.
# Chosen large enough that B is a genuine generalization probe and not a
# near-duplicate of A; see `iosp.model.scenes.scene_b`.
SCENE_B_Q_START_OFFSET = jnp.array([0.15, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0], dtype=jnp.float32)
SCENE_B_PICK_OFFSET = jnp.array([0.05, 0.08, -0.03], dtype=jnp.float32)
SCENE_B_PLACE_OFFSET = jnp.array([-0.05, -0.06, 0.04], dtype=jnp.float32)

# -- outer-loop defaults -----------------------------------------------------
# 40 steps, not 12: at 12 neither the wide fit nor the refit is near a minimum
# (path A: init 0.0500 -> wide 0.0371), so a 12-step comparison measures step
# efficiency inside a tiny budget rather than the quality of the identifiable
# subspace, which is the claim these experiments are written to make.
N_STEPS = 40
LR = 0.05
N_ITERS = 60          # inner trajopt iterations
TRACE_FRAC = 0.95     # rank rule for `ioc.identifiability.select_rank(rule="trace")`

# `THETA_IK_STAR` is [0.06, 0.04] m while the trajopt logits are O(1), so the
# raw parameter vector spans two orders of magnitude and a single Adam step
# size cannot serve both blocks.  See `iosp.fit.params.z_scale`.
STANDOFF_SCALE = 0.05  # metres
