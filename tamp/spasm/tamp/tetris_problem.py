"""PDDLStream tetris-packing problem, matching cuTAMP's ``tetris_N`` task.

The three systems in the benchmark have to solve the *same* problem for the
comparison to mean anything, and they did not natively: cuTAMP ships tetris
packing, SPaSM's headline task is tetris packing, but our PDDLStream domain did
tabletop rearrangement of cubes. This module closes that gap on the PDDLStream
side.

Nothing here re-implements geometry. The tetromino sphere sets, their initial
poses, the walled goal bin and both collision tests are taken directly from
:mod:`spasm.tetris.env` and :mod:`spasm.tetris.solve` — the same functions the
stock SPaSM solver optimises against. So the PDDLStream configurations are
solving SPaSM's own problem instance, scored by SPaSM's own predicates, and any
difference against stock SPaSM is the *planner*, not the task.

Why this is harder than the rearrangement domain, and why that is the point:

* **Non-convex pieces.** Tetrominoes are sphere unions, not boxes, so a valid
  packing depends on relative orientation. Rejection sampling over ``(x, y, yaw)``
  succeeds far less often than it does for cubes.
* **A tight bin with walls.** The goal region is sized to the pieces plus a
  one-radius buffer, so admissible placements occupy a thin sliver of the
  sample space. This is exactly the regime the cuTAMP paper argues sampling-based
  planners fail in — their SAMPLING baseline scores 0/50 on tetris-5 — so it is
  the regime the comparison should be run in rather than avoided.

The symbolic layer is unchanged: the same ``domain.pddl`` and ``stream.pddl``
the rearrangement problem uses. Only the geometry and the region test differ,
which keeps the two PDDLStream configurations comparable to each other.
"""
from __future__ import annotations

import numpy as np
from pddlstream.language.constants import And, Equal, PDDLProblem, TOTAL_COST
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import get_file_path, read

import jax.numpy as jnp

from . import _setup  # noqa: F401
from . import geometry as g
from .problems import MANIPULATE_COST, ROBOT, World
from .streams_pyroffi import make_stream_map

from spasm.tetris.env import Simulation, _block_pose_to_spheres
from spasm.tetris.solve import sphere_sphere_penetration, sphere_wall_penetration

#: Block counts SPaSM's tetris Simulation supports.
SUPPORTED = (1, 3, 5)

#: Penetration beyond the margin baked into SPaSM's predicates that counts as a
#: collision. Matches the epsilon ``geometry.blocks_collide`` uses.
EPS = 1e-3

#: Margin folded into SPaSM's penetration predicates: contact reads as this
#: value, not 0. Both the wall and sphere-sphere tests carry it.
MARGIN = 0.010


class TetrisWorld(World):
    """A :class:`World` carrying the SPaSM tetris simulation.

    The bin's walls have no counterpart in the rearrangement problem, so the
    ``sim`` is kept alongside for the wall test rather than being flattened into
    the region rectangle — ``sphere_wall_penetration`` needs the real geometry.
    """

    def __init__(self, sim, **kw):
        super().__init__(**kw)
        object.__setattr__(self, "sim", sim)


def make_tetris_world(num_blocks=5, seed=0):
    """Build the PDDLStream instance of SPaSM's tetris-``N`` packing task."""
    if num_blocks not in SUPPORTED:
        raise ValueError(
            f"num_blocks={num_blocks}; SPaSM's tetris Simulation supports "
            f"{SUPPORTED} (cuTAMP exposes tetris_1/2/3/5)")

    sim = Simulation(num_blocks=num_blocks)
    names = [f"b{i}" for i in range(sim.num_blocks)]

    blocks = {n: sim.block_spheres[i] for i, n in enumerate(names)}
    initial = {n: np.asarray(sim.block_poses[i], dtype=float)
               for i, n in enumerate(names)}

    # The goal "region" is the bin footprint, with the SAME bounds SPaSM's own
    # `sample_particles` draws from: xy uniform in goal_dims/2 about
    # goal_position, and z pinned to `sim.block_z`.
    #
    # That z matters more than it looks. Placing pieces at the table surface
    # (z=0) instead puts every sphere above the goal volume, so
    # `sphere_wall_penetration` reports ~0.1m for *every* sample and the
    # rejection sampler accepts nothing — 0/2000 measured. The predicate was
    # right; the height was not.
    gp = np.asarray(sim.goal_position, dtype=float)
    gd = np.asarray(sim.goal_dims, dtype=float)
    regions = {"goal": dict(cx=float(gp[0]), cy=float(gp[1]),
                            z=float(sim.block_z),
                            hx=float(gd[0]) / 2.0, hy=float(gd[1]) / 2.0)}

    halfz = {n: float(jnp.max(sim.block_spheres[i][:, 2]
                              + sim.block_spheres[i][:, 3]))
             for i, n in enumerate(names)}

    return TetrisWorld(
        sim=sim,
        seed=seed,
        blocks=blocks,
        block_halfz=halfz,
        regions=regions,
        initial_poses=initial,
        goal_region={n: "goal" for n in names},
    )


def block_in_bin(world, block, pose, eps=EPS):
    """True if the piece at ``pose`` sits inside the bin without hitting a wall.

    Uses SPaSM's own ``sphere_wall_penetration`` rather than a rectangle test:
    a tetromino's footprint depends on its yaw, so the piece can be
    centre-inside and still clip a wall.

    Note the ``MARGIN`` offset. ``sphere_wall_penetration`` folds a 0.010 m
    margin into its result, so a *perfectly contained* piece scores 0.010 and
    not 0 — the same convention ``geometry.blocks_collide`` compensates for.
    Comparing against ``eps`` alone rejects every placement; the best sample
    measured 0.0099, just under the margin it should have been credited.
    """
    spheres = _block_pose_to_spheres(world.blocks[block],
                                     jnp.asarray(pose, jnp.float32))
    pen = float(jnp.max(sphere_wall_penetration(spheres, world.sim)))
    return pen <= MARGIN + eps


def make_tetris_stream_map(world, motion_backend="linear", motion_params=None):
    """Stream map for tetris packing.

    Delegates to the shared :func:`~spasm.tamp.streams_pyroffi.make_stream_map`
    and overrides only the two streams the bin changes — placement sampling and
    the region test — so grasp/IK/motion/collision stay byte-identical to the
    rearrangement configuration.
    """
    base = make_stream_map(world, motion_backend=motion_backend,
                           motion_params=motion_params)
    rng = np.random.default_rng(world.seed)
    region = world.regions["goal"]

    def s_region_gen(b, r):
        """Sample a placement in the bin, unbounded.

        Yaw matters here in a way it does not for cubes: a tetromino only fits
        in particular orientation modes, so yaw is sampled over the full circle
        and most draws are rejected. Measured acceptance is ~0.05% (1/2000) on
        tetris-1 — the bin is 0.15m square and the piece spans ~0.12m, so
        admissible (x, y, yaw) is a thin sliver of the sample space.

        That is why this generator is unbounded rather than capped at a few
        hundred tries as the rearrangement sampler is: at 0.05% a 200-try cap
        yields nothing at all. PDDLStream generators are expected to be
        infinite, with the planner's ``max_time`` doing the stopping.

        The low rate is the task, not a defect — it is precisely the regime the
        cuTAMP paper argues sampling-based TAMP fails in, and the reason this
        comparison is worth running.
        """
        while True:
            pose = np.array([
                rng.uniform(region["cx"] - region["hx"], region["cx"] + region["hx"]),
                rng.uniform(region["cy"] - region["hy"], region["cy"] + region["hy"]),
                region["z"],
                rng.uniform(-np.pi, np.pi),
            ], dtype=float)
            if block_in_bin(world, b, pose):
                yield (pose,)

    def t_region(b, p, r):
        return block_in_bin(world, b, p)

    def t_cfree(b1, p1, b2, p2):
        """Piece-vs-piece penetration, via SPaSM's own predicate."""
        s1 = _block_pose_to_spheres(world.blocks[b1], jnp.asarray(p1, jnp.float32))
        s2 = _block_pose_to_spheres(world.blocks[b2], jnp.asarray(p2, jnp.float32))
        return float(jnp.max(sphere_sphere_penetration(s1, s2))) <= MARGIN + EPS

    base["s-region"] = from_gen_fn(s_region_gen)
    base["t-region"] = from_test(t_region)
    base["t-cfree"] = from_test(t_cfree)
    return base


def pddlstream_from_tetris_world(world, motion_backend="linear",
                                 motion_params=None):
    """Wire a :class:`TetrisWorld` into a ``PDDLProblem``.

    Reuses the rearrangement domain and stream declarations unchanged — the task
    differs in geometry, not in symbolic structure — so the two PDDLStream
    configurations remain directly comparable.
    """
    domain_pddl = read(get_file_path(__file__, "domain.pddl"))
    stream_pddl = read(get_file_path(__file__, "stream.pddl"))
    stream_map = make_tetris_stream_map(world, motion_backend=motion_backend,
                                        motion_params=motion_params)

    init = [
        Equal(("Cost",), MANIPULATE_COST),
        Equal((TOTAL_COST,), 0),
        ("Robot", ROBOT),
        ("CanMove", ROBOT),
        ("Conf", world.conf0),
        ("AtConf", ROBOT, world.conf0),
        ("HandEmpty", ROBOT),
        ("Region", "goal"),
    ]
    for b, p in world.initial_poses.items():
        init += [("Block", b), ("Pose", b, p), ("AtPose", b, p),
                 ("Placeable", b, "goal")]
    goal = And(*[("In", b, "goal") for b in world.initial_poses])
    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
