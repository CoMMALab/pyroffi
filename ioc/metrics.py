"""Scoring a recovered cost against ground truth."""

import jax.numpy as jnp


def cosine(a, b, eps=1e-300):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps))


def simplex_metrics(theta_hat, theta_star):
    """L1 error and cosine similarity between two points of the simplex.

    L1 is reported alongside *regret* everywhere because the two answer
    different questions: L1 asks whether the weights were recovered, regret asks
    whether the behaviour was.  They come apart whenever a feature is weakly
    identifiable -- a large weight error along a direction the demonstrations do
    not excite costs nothing in behaviour, which is the point of the collinear
    control and of the Gram-matrix certificate.
    """
    l1 = float(jnp.sum(jnp.abs(theta_hat - theta_star)))
    return l1, cosine(theta_hat, theta_star)
