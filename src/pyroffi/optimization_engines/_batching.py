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

from collections.abc import Callable

import jax
import jax.numpy as jnp


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
