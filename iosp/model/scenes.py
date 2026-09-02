"""The scenes every experiment is run on: the fit scene, the held-out scene,
and the multi-scene contexts.

Scene A and scene B differ only in placement -- start configuration, pick and
place targets -- and share the obstacle.  Both are rolled out from the SAME
ground-truth cost (`config.THETA_IK_STAR` / `Z_TRAJOPT_STAR`), so both have an
exact correct answer; A is where the outer loss is computed and B never enters
the loss or the gradient.

Cao, Cohen & Szpruch (arXiv:2106.03498) prove the reward in IRL is recoverable
only given demonstrations under sufficiently DIFFERENT environments, which is
why `scenes_multi` exists at all: a single-scene fit has an identifiability
ceiling that is a property of the scene set, not of the optimizer.
"""

import jax
import jax.numpy as jnp
import numpy as np

from iosp.config import (
    OBS_CENTER, OBS_RADIUS, PICK_POS, PLACE_POS, Q_START,
    SCENE_B_PICK_OFFSET, SCENE_B_PLACE_OFFSET, SCENE_B_Q_START_OFFSET,
)
from iosp.model import pickplace as pp

# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def _scene(q_start, pick, place):
    s = pp.PickPlaceScene(q_start=q_start, pick_pos=pick, place_pos=place,
                          obs_center=OBS_CENTER, obs_radius=OBS_RADIUS)
    return jax.tree.map(lambda a: a[None], s)


def scene_a():
    """`recovery_bench`'s fitting scene -- the UNANCHORED one."""
    return _scene(Q_START, PICK_POS, PLACE_POS)


def scene_b(scale=1.0):
    """`generalization_check`'s held-out scene.

    `scale` multiplies all three offsets, so `scale=1.0` is byte-for-byte every
    recorded result and larger values push the held-out scene further from the
    fit scene -- a stress test for generalization rather than a near-duplicate.
    The offsets are displacements from scene A, so scaling them is a pure
    extrapolation along the same direction: nothing about the task changes, only
    how far the fitted cost has to carry.  It is NOT unbounded -- past some
    scale the pick/place targets leave the arm's reachable set and the IK
    stage's residual stops being a generalization signal at all, so a run at a
    large scale must be read together with `screen_stationarity`.
    """
    return _scene(Q_START + scale * SCENE_B_Q_START_OFFSET,
                  PICK_POS + scale * SCENE_B_PICK_OFFSET,
                  PLACE_POS + scale * SCENE_B_PLACE_OFFSET)


# ---------------------------------------------------------------------------
# multi-scene contexts (fix 2: identifiability needs several environments)
# ---------------------------------------------------------------------------
#
# Cao, Cohen & Szpruch (arXiv:2106.03498) prove the reward in IRL is not
# identifiable even given perfect knowledge of optimal behaviour, and is
# recoverable only up to a constant -- and only given demonstrations under
# sufficiently DIFFERENT ENVIRONMENTS (or distinct discount factors).  Every
# result in this study so far fit a single scene, so part of the rank
# deficiency the whole three-stage refit exists to manage is an artifact of
# that choice rather than a property of the cost.
#
# Note this applies to Path A exactly as much as to Path B: A's measured
# `captured_frac = 0.859` is a ONE-SCENE ceiling.
#
# Scene 0 is the canonical `scene_a()` unchanged, so results stay anchored to
# every earlier run; the rest are jittered around it.  `obs_center` is jittered
# TOO -- the original A/B pair moved only `q_start`/`pick`/`place` and left the
# obstacle fixed, which is precisely the direction that makes a clearance cost
# identifiable.  A demo set that never moves the obstacle cannot distinguish
# "avoid that obstacle" from "prefer that region of joint space".

_SCENE_SCALE = dict(q_start=0.15, pick=0.08, place=0.06, obs=0.06)


def _scenes_multi(n_fit=3, n_gen=2, seed=0):
    """(batched PickPlaceScene, n_fit) -- first `n_fit` rows are the fit set,
    the remaining `n_gen` are held out.  Row 0 is exactly `scene_a()`."""
    rng = np.random.default_rng(seed)
    qs, pk, pl, oc, orad = [], [], [], [], []
    for i in range(n_fit + n_gen):
        z = (lambda d, s: np.zeros(d) if i == 0 else rng.normal(size=d) * s)
        qs.append(np.asarray(Q_START) + z(7, _SCENE_SCALE["q_start"]))
        pk.append(np.asarray(PICK_POS) + z(3, _SCENE_SCALE["pick"]))
        pl.append(np.asarray(PLACE_POS) + z(3, _SCENE_SCALE["place"]))
        oc.append(np.asarray(OBS_CENTER) + z(3, _SCENE_SCALE["obs"]))
        orad.append(np.asarray(OBS_RADIUS))
    f32 = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return pp.PickPlaceScene(q_start=f32(qs), pick_pos=f32(pk), place_pos=f32(pl),
                             obs_center=f32(oc), obs_radius=f32(orad))




def scenes_ab(scene_b_scale=1.0):
    """Scenes A and B as ONE batch of 2.

    `PickPlaceProblem.solve` already vmaps over the leading batch axis, so the
    fit scene and the held-out scene become one executable and one batched GPU
    solve instead of two separately-compiled chains -- worth ~25 min of XLA
    compile per run on this model.
    """
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0),
                        scene_a(), scene_b(scene_b_scale))


def scenes_multi(n_fit=3, n_gen=2, seed=0):
    """`n_fit` fit contexts followed by `n_gen` held-out ones, as one batch."""
    return _scenes_multi(n_fit=n_fit, n_gen=n_gen, seed=seed)


def sample_pickplace_scenes(rng, n, jitter_pos=0.03, jitter_q=0.05):
    """Sample `n` pick-and-place contexts by jittering the nominal scene.

    Mirrors `ioc.robot.problem.RobotProblem.sample_scenes`'s SPIRIT (small,
    per-context jitter around a nominal configuration, so recovery is measured
    across genuinely different contexts rather than one pinned scene) rather
    than reusing its code directly: `PickPlaceScene`'s fields (pick/place EE
    positions plus a start config) don't match `Scene`'s (paired joint-space
    start/goal), so there is no single function to share -- the jitter
    magnitudes (~3cm on the pick/place targets, ~0.05 rad on q_start) are
    chosen to be large enough to make each context a genuinely different IK/
    trajopt problem without moving the pick/place points outside the arm's
    reach or through the fixed obstacle.
    """
    starts, picks, places = [], [], []
    for _ in range(n):
        starts.append(np.asarray(Q_START) + rng.normal(scale=jitter_q, size=Q_START.shape[0]))
        picks.append(np.asarray(PICK_POS) + rng.normal(scale=jitter_pos, size=3))
        places.append(np.asarray(PLACE_POS) + rng.normal(scale=jitter_pos, size=3))
    return pp.PickPlaceScene(
        q_start=jnp.asarray(np.stack(starts), dtype=jnp.float32),
        pick_pos=jnp.asarray(np.stack(picks), dtype=jnp.float32),
        place_pos=jnp.asarray(np.stack(places), dtype=jnp.float32),
        obs_center=jnp.asarray(np.tile(np.asarray(OBS_CENTER), (n, 1)), dtype=jnp.float32),
        obs_radius=jnp.asarray(np.tile(np.asarray(OBS_RADIUS), (n, 1)), dtype=jnp.float32),
    )

