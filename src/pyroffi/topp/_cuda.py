"""Batched TOPP-RA on the GPU (``cuda_kernels/topp/topp_ra_pairs.cu``): boundary-speed pairs
(:func:`topp_ra_pairs`) and reachability jobs (:func:`topp_ra_reach`).

For a batch of paths and, per path, V start and V end path speeds, :func:`topp_ra_pairs` returns
the time-optimal duration and feasibility of all V x V boundary-condition pairs. It takes the
same canonical :class:`~pyroffi.topp.Constraints` (and grid) as :func:`~pyroffi.topp.solve_topp_ra`
and solves the same discretisation -- V backward passes (one controllable-set sweep per end
speed, shared by every start speed) and V^2 greedy forward passes -- so each pair matches
``solve_topp_ra(path, constraints, sd_start, sd_end)``. The kernel is robot-independent (the
dynamics are already in the constraint rows), built once and cached on disk.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._constraints import Constraints


class PairsResult(NamedTuple):
    feasible: jax.Array    # [E, V, V] bool: start speed a -> end speed b
    duration: jax.Array    # [E, V, V] seconds (meaningless where infeasible)
    k_lo: jax.Array        # [E, V, N] controllable sets per end speed (x = sdot^2)
    k_hi: jax.Array


@lru_cache(maxsize=1)
def _targets() -> tuple[str, str]:
    from ..cuda_kernels._build import build_shared_library, register_cuda_targets

    src = Path(__file__).resolve().parents[1] / "cuda_kernels" / "topp" / "topp_ra_pairs.cu"
    so = build_shared_library("topp_ra_pairs", src, {}, ["-O3", "-shared", "-Xcompiler", "-fPIC", "-std=c++17",
                                                          "-arch=native"])
    return register_cuda_targets(so, ("ToppRaPairsFfi", "ToppRaReachFfi"), "pyroffi")


def topp_ra_pairs(constraints: Constraints, delta, sd_start, sd_end) -> PairsResult:
    """Durations of every (start, end) path-speed pair.

    Args:
        constraints: batched canonical constraints (``A [E, N, m, 2]``, ``h [E, N, m]``,
            ``x_upper [E, N]``), as built for :func:`~pyroffi.topp.solve_topp_ra`.
        delta: ``[E]`` grid spacing (``GeometricPath.delta``).
        sd_start, sd_end: ``[E, V]`` start / end path speeds ``sdot``.
    """
    A = jnp.asarray(constraints.A, jnp.float32)
    E, N = A.shape[:2]
    V = jnp.shape(sd_start)[-1]
    f = jax.ffi.ffi_call(_targets()[0], (jax.ShapeDtypeStruct((E, V, V), jnp.int32),
                                     jax.ShapeDtypeStruct((E, V, V), jnp.float32),
                                     jax.ShapeDtypeStruct((E, V, N), jnp.float32),
                                     jax.ShapeDtypeStruct((E, V, N), jnp.float32),
                                     jax.ShapeDtypeStruct((E, V), jnp.int32)))
    xs = jnp.square(jnp.asarray(sd_start, jnp.float32))
    xe = jnp.square(jnp.asarray(sd_end, jnp.float32))
    feas, dur, lo, hi, _ = f(A, jnp.asarray(constraints.h, jnp.float32), jnp.asarray(constraints.x_upper, jnp.float32),
                             jnp.asarray(delta, jnp.float32), xs, xe)
    return PairsResult(feas.astype(bool), dur, lo, hi)


class ReachResult(NamedTuple):
    k0: jax.Array        # [J] fastest feasible start sdot (-1: the end interval is unreachable)
    sd_end: jax.Array    # [J, S] end sdot of each start's time-optimal sweep
    duration: jax.Array  # [J, S] seconds
    feasible: jax.Array  # [J, S] bool


def topp_ra_reach(constraints: Constraints, delta, job_edge, sd_end_max, sd_start) -> ReachResult:
    """Reachability jobs for the interval lift. Job j runs on path ``job_edge[j]`` (an index into
    the batched ``constraints`` / ``delta``) and must end with ``sdot`` in ``[0, sd_end_max[j]]``:
    one backward sweep of that interval's controllable sets (``k0[j]``: the fastest start that
    can still make it), then TOPP-RA's greedy forward sweep from every ``sd_start[j, s]`` inside
    them -- the time-optimal traversal that lands in the end interval, its exact end speed and time.
    """
    A = jnp.asarray(constraints.A, jnp.float32)
    N = A.shape[1]
    J, S = jnp.shape(sd_start)
    f = jax.ffi.ffi_call(_targets()[1], (jax.ShapeDtypeStruct((J,), jnp.float32), jax.ShapeDtypeStruct((J, S), jnp.float32),
                                         jax.ShapeDtypeStruct((J, S), jnp.float32), jax.ShapeDtypeStruct((J, S), jnp.int32),
                                         jax.ShapeDtypeStruct((J, N), jnp.float32), jax.ShapeDtypeStruct((J, N), jnp.float32),
                                         jax.ShapeDtypeStruct((J,), jnp.int32)))
    k0, xe, dur, feas, _, _, _ = f(A, jnp.asarray(constraints.h, jnp.float32), jnp.asarray(constraints.x_upper, jnp.float32),
                                   jnp.asarray(delta, jnp.float32), jnp.asarray(job_edge, jnp.int32),
                                   jnp.square(jnp.asarray(sd_end_max, jnp.float32)),
                                   jnp.square(jnp.asarray(sd_start, jnp.float32)))
    return ReachResult(jnp.where(k0 >= 0, jnp.sqrt(jnp.maximum(k0, 0.0)), -1.0), jnp.sqrt(jnp.maximum(xe, 0.0)),
                       dur, feas.astype(bool))
