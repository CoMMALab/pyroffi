"""Route ``vmap`` over a single-problem CUDA solver into its batched kernel.

The CUDA IK kernels already carry a problem axis: they are launched over
``(n_problems, n_seeds, n_act)`` and solve every problem in one launch. The
single-problem entry points are that same kernel with ``n_problems == 1``.

So ``vmap`` has an obvious right answer here -- fold the mapped axis into
``n_problems`` and make ONE launch -- but ``jax.ffi.ffi_call`` cannot know that.
Left to itself it refuses outright::

    NotImplementedError: vmap is only supported for the ffi_call primitive when
    vmap_method is one of 'sequential', ... Got vmap_method=None.

and the options it does offer are all wrong for these kernels: ``sequential``
serialises what the kernel was built to do in parallel, while ``broadcast_all``
and ``expand_dims`` would hand the handler a rank it does not accept, since the
batch axis is already spoken for.

The rule therefore lives one level up, in Python, where the batched entry point
is in scope. ``vmap`` over the single-problem solver dispatches to the batched
solver; the mapped axis becomes ``n_problems``.
"""

from __future__ import annotations

import os as _os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float


def dispatch_vmap_to_batched(single_call: Callable, batched_call: Callable) -> Callable:
    """Wrap ``single_call`` so that ``vmap`` over it runs ``batched_call`` once.

    Args:
        single_call: ``f(target_wxyz_xyz, previous_cfg) -> pytree``, solving ONE
            problem. Everything else (robot, collision buffers, solver options)
            is closed over -- those are identical across a mapped axis by
            construction, since they describe the robot and the world rather
            than the problem.
        batched_call: ``f(target_wxyz_xyz[B, 7], previous_cfgs[B, n_act]) ->
            pytree`` with a leading batch axis on every output.

    Returns:
        A callable with ``single_call``'s signature and a custom batching rule.
    """

    @jax.custom_batching.custom_vmap
    def solve(target_wxyz_xyz, previous_cfg):
        return single_call(target_wxyz_xyz, previous_cfg)

    @solve.def_vmap
    def _rule(axis_size, in_batched, target_wxyz_xyz, previous_cfg):
        target_batched, prev_batched = in_batched

        # An argument may be mapped or closed over independently -- vmapping
        # targets while holding one seed configuration fixed is a normal thing
        # to want. Broadcast whichever is unmapped so the batched kernel sees a
        # full problem axis either way.
        if not target_batched:
            target_wxyz_xyz = jnp.broadcast_to(
                target_wxyz_xyz, (axis_size, *jnp.shape(target_wxyz_xyz)))
        if not prev_batched:
            previous_cfg = jnp.broadcast_to(
                previous_cfg, (axis_size, *jnp.shape(previous_cfg)))

        out = batched_call(target_wxyz_xyz, previous_cfg)
        # Every output carries the problem axis the batched kernel produced.
        return out, jax.tree.map(lambda _: True, out)

    return solve


# ---------------------------------------------------------------------------
# Multi-GPU problem-axis sharding
# ---------------------------------------------------------------------------
#
# One more level of the same idea. vmap folds a mapped axis into the kernel's
# problem axis so ONE device solves all of it in one launch; this splits that
# problem axis across DEVICES so each solves its share. Both are launch
# strategies over an axis the kernel already understands, which is why they
# compose: shard first, and each device then runs the batched kernel it would
# have run anyway.
#
# Lifted out of _sqp_ik, where it worked but was reachable by exactly one
# solver. Nothing in it is SQP-specific -- the solver-specific part is the pmap
# body, which each caller supplies.

#: Below this many problems the pmap's split/gather costs more than the parallel
#: solve saves. Overridable per solver by env var; see `sharding_enabled`.
#:
#: MEASURED on 4x A5000 with sqp_ik, against the same solve on one device:
#:     B=512     73.9 -> 64.0 ms   1.15x
#:     B=4096   523.9 -> 254.2 ms  2.06x
#:     B=16384 1971.4 -> 720.0 ms  2.74x   (8.3 -> 22.8 kIK/s)
#: The gain grows with batch size because the fixed pmap cost -- padding,
#: splitting, the cross-device gather -- is amortised over more work, and it
#: never reaches Nx because that cost does not shrink. 512 is where the win
#: becomes worth the machinery; the inherited value of 64 predates any
#: measurement and would have shared a batch that gains almost nothing.
PMAP_MIN_PROBLEMS_DEFAULT = 512


def sharding_enabled(n_problems: int, n_devices: int, env_var: str | None = None) -> bool:
    """Shard only when it actually helps: >1 GPU and a compute-bound batch.

    A small batch spends more time padding, splitting and gathering than it
    saves by dividing the solve, so the threshold is a real cutoff rather than
    caution. `env_var` names an override so a solver whose per-problem cost
    differs can be tuned without touching this.
    """
    if n_devices <= 1:
        return False
    min_problems = PMAP_MIN_PROBLEMS_DEFAULT
    if env_var is not None:
        try:
            min_problems = int(_os.environ.get(env_var, PMAP_MIN_PROBLEMS_DEFAULT))
        except ValueError:
            pass
    return n_problems >= max(n_devices, min_problems)


def make_sharded_pmap(body: Callable, n_broadcast: int) -> Callable:
    """``pmap`` a solver body over the problem axis.

    The first three arguments (rng key, previous configurations, target poses)
    are MAPPED across devices; the remaining ``n_broadcast`` are shared -- robot
    model, collision buffers, per-EE masks, constraint weights. Those describe
    the robot and the world rather than the problem, so replicating them is
    correct and splitting them would be a bug.
    """
    return jax.pmap(body, in_axes=(0, 0, 0) + (None,) * n_broadcast)


def run_sharded(
    pmapped: Callable,
    target_poses_batch: jaxlie.SE3,
    rng_key: Array,
    previous_cfgs: Float[Array, "n_problems n_act"],
    n_devices: int,
    *broadcast_args,
) -> Float[Array, "n_problems n_act"]:
    """Split the problem axis across ``n_devices`` GPUs via a cached pmap.

    Pads the problem axis up to a multiple of ``n_devices`` by REPEATING THE
    LAST target -- padding with zeros would hand a device a degenerate problem
    whose solve time is unrepresentative, and pmap requires equal shares.
    Winners for the padding are discarded on the way out.
    """
    n_problems, n_act = previous_cfgs.shape
    pad = (-n_problems) % n_devices

    wxyz_xyz = target_poses_batch.wxyz_xyz
    if pad:
        prev_pad = jnp.broadcast_to(previous_cfgs[-1:], (pad, n_act))
        wxyz_pad = jnp.broadcast_to(wxyz_xyz[-1:], (pad, wxyz_xyz.shape[-1]))
        previous_cfgs = jnp.concatenate([previous_cfgs, prev_pad], axis=0)
        wxyz_xyz = jnp.concatenate([wxyz_xyz, wxyz_pad], axis=0)

    per_device = (n_problems + pad) // n_devices
    prev_sh = previous_cfgs.reshape(n_devices, per_device, n_act)
    wxyz_sh = wxyz_xyz.reshape(n_devices, per_device, wxyz_xyz.shape[-1])
    keys = jax.random.split(rng_key, n_devices)

    winners = pmapped(keys, prev_sh, wxyz_sh, *broadcast_args)
    return winners.reshape(n_devices * per_device, n_act)[:n_problems]
