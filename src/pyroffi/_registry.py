"""Concrete copies of robots and collision models, for resolving traced ones.

Inside ``jax.jit`` a robot passed as an argument has tracers for arrays, so its kinematic
constants cannot be baked into a compiled kernel. Its pytree structure survives tracing,
though, and includes identity-hashed static fields (a robot's backend holder), so it
names the concrete object it came from. Constructors register that object here.
"""

from __future__ import annotations

import jax
import numpy as np

# ponytail: strong references, never evicted; a process builds a handful of robots.
_CONCRETE: dict = {}


def register(obj):
    """Record a concrete robot / collision model; returns it unchanged."""
    bucket = _CONCRETE.setdefault(jax.tree.structure(obj), [])
    if not any(o is obj for o in bucket):
        bucket.append(obj)
    return obj


def _equal(a, b) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def resolve(obj):
    """``obj`` with concrete NumPy leaves; a traced object resolves through the registry."""
    leaves = jax.tree.leaves(obj)
    if not any(isinstance(x, jax.core.Tracer) for x in leaves):
        return jax.tree.map(np.asarray, obj)
    distinct: list = []
    for c in _CONCRETE.get(jax.tree.structure(obj), []):
        if not any(_equal(c, d) for d in distinct):
            distinct.append(c)
    if len(distinct) != 1:
        raise ValueError(
            f"A traced {type(obj).__name__} reached a compiled problem, and "
            + ("no concrete one with its structure was created through a pyroffi constructor"
               if not distinct else f"{len(distinct)} different concrete ones match it")
            + ". Build the problem outside jax.jit, or pass the model in as a closure.")
    try:
        return jax.tree.map(lambda c, t: np.broadcast_to(np.asarray(c), t.shape), distinct[0], obj)
    except ValueError as e:
        raise ValueError(f"A traced {type(obj).__name__} does not match its concrete original "
                         "in shape; build the problem outside jax.jit.") from e
