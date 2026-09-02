"""The outer parameterization: how a dimensionless vector `u` becomes cost
weights, and the two gauge facts that follow from using a softmax.

Every experiment optimizes `u`, not `theta`, for one measured reason recorded
in `z_scale` below.  `gauge_vector` names the exact null direction the softmax
introduces, so a rank result can say which of its zero eigenvalues is a real
identifiability failure and which is arithmetic.
"""

import jax.numpy as jnp
import numpy as np

from iosp.config import STANDOFF_SCALE

# ---------------------------------------------------------------------------
# coordinates: a dimensionally-homogeneous, gauge-fixed `u`
# ---------------------------------------------------------------------------
#
# CONFOUND 1 (units).  The natural parameter vector `z = [theta_ik(2) |
# z_trajopt(7)]` concatenates standoff distances in METRES (~0.06) with softmax
# LOGITS (~O(1)).  Eigendecomposition is not invariant to per-coordinate
# rescaling, so in raw `z` the question "which directions are identifiable?" is
# partly answered by the choice of units.  `identifiability_check.py` measured
# how lopsided that is here: gradient norm 223.3 for `grasp.standoff` and 8.1
# for `place.standoff` against <= 1.4 for every trajopt feature.  Left alone,
# the two `theta_ik` directions swamp the spectrum and `U_r` collapses onto
# them -- so the refit would silently never learn a single trajopt weight.
#
# Fix: optimize in `u`, with `z = Z_SCALE * u` and `Z_SCALE` a characteristic
# magnitude per coordinate, so every coordinate of `u` is O(1) and a unit step
# means the same thing everywhere.  This also repairs the fit itself, not just
# the diagnosis: at `lr=0.05` in raw `z`, one Adam step moved a standoff by
# 0.05m against a ground-truth value of 0.06m.
#
# CONFOUND 2 (gauge).  `softmax` is invariant to adding a constant to all of
# its logits, so `(0,0,1,...,1)/sqrt(7)` is an EXACT null direction of `G` by
# construction, independent of the demonstration.  It is one of the two exactly
# -zero eigenvalues measured on Path A.  It also makes `||z_hat - z_star||` and
# `captured_frac` ill-defined, since `z_star` and `z_star + c*1` are the same
# cost.  Fix: `gauge_fix` centres the logit block, and every parameter-space
# metric is computed on gauge-fixed vectors.




def z_scale(K, n_ik):
    s = np.ones(K, dtype=np.float32)
    s[:n_ik] = STANDOFF_SCALE
    return jnp.asarray(s)


def gauge_fix(z, n_ik):
    """Centre the softmax logit block: the canonical representative of the
    equivalence class `{z + c * (0..0,1..1)}`, all of which are one cost."""
    z = np.asarray(z, dtype=np.float64).copy()
    z[n_ik:] -= z[n_ik:].mean()
    return z


def gauge_vector(K, n_ik):
    """Unit vector along the softmax gauge direction, in `u` coordinates."""
    g = np.zeros(K)
    g[n_ik:] = 1.0
    return g / np.linalg.norm(g)


# ---------------------------------------------------------------------------
# stages 1-4 (path-agnostic) live in `ioc.identifiability`; this module keeps
# only what's specific to the pickplace composition: scenes, the u/z gauge
# reparametrization, and the two path builders.
# ---------------------------------------------------------------------------

def _proj_norm(delta, V):
    return float(np.linalg.norm(V.T @ delta)) if V.shape[1] else 0.0


