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

# z = [theta_ik | z_trajopt] mixes metres (~0.06) with logits (~O(1)).
# Without rescaling, theta_ik swamps the eigenspectrum.
# Fix: optimize u with z = Z_SCALE * u so all coordinates are O(1).
#
# softmax is gauge-invariant (z + c*1 is the same cost), introducing an exact
# null direction. gauge_fix centres the logit block to remove it.




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


