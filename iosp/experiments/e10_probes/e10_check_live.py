"""Is the anchoring actually LIVE inside the multistart forward map?

The bug this guards against: `batched_paths` calls `_ik_batch` directly rather
than `grasp_ik`/`place_ik`, so an orientation added to the latter was inert in
the fit and an anchored run computed the same thing as an unanchored one.  This
compares q_pick/q_place produced BY THE FIT'S OWN MAP, at u=0, both ways.
"""
import numpy as np, jax.numpy as jnp
from iosp import config
config.enable_compilation_cache()
from iosp.fit import multistart as ms
from iosp.model import pickplace as pp

out = {}
for anchor in (False, True):
    b = ms.build_from_demos(n_fit=1, max_episodes=1, n_branches=1,
                            anchor_grasp=anchor)
    u0 = jnp.zeros((1, b["K"]), jnp.float32)
    q = b["batched_paths"](u0, b["refs"][0][None], "joint")[0]   # (M, T, dof)
    d = np.asarray(b["demo"])
    out[anchor] = np.asarray(q)
    print(f"anchor={str(anchor):5s}  rollout row7 vs demo "
          f"{np.linalg.norm(np.asarray(q)[:, pp.SKELETON_PICK[0]] - d[:, pp.SKELETON_PICK[0]], axis=-1)[0]:.3f}"
          f"   row19 vs demo "
          f"{np.linalg.norm(np.asarray(q)[:, pp.SKELETON_PLACE[0]] - d[:, pp.SKELETON_PLACE[0]], axis=-1)[0]:.3f} rad",
          flush=True)
print(f"\nrollouts differ by {np.abs(out[True] - out[False]).max():.4f} rad max "
      f"-- anything near 0 means anchoring is STILL inert")
