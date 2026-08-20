"""Rearrangement problem instances + PDDLProblem construction.

A :class:`World` holds the sphere geometry (per-block local spheres), the table
regions, initial block poses and the goal assignment. :func:`pddlstream_from_world`
turns it into a ``PDDLProblem`` wired to the pyroffi-backed streams.

The default suite is *tabletop rearrangement*: N cubes start scattered in a
"start" region and must be packed, collision-free, into a "goal" box — the same
geometric regime as SPaSM's tetris packing, on a Panda, so the differentiable
solver and this classical baseline attack an identical task.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pddlstream.language.constants import And, Equal, PDDLProblem, TOTAL_COST
from pddlstream.utils import read, get_file_path

from . import geometry as g
from .streams_pyroffi import make_stream_map

TABLE_Z = 0.0
CUBE_HALF = 0.025          # 5cm cubes
# Effective collision half-footprint: the conservative sphere enclosure
# (corner spheres of radius ~0.03) makes two 5cm cubes collision-free only at
# center spacing >= ~0.09m, i.e. they behave like a 0.045 half-footprint. Box
# sizing and region insets use THIS, not CUBE_HALF, so instances are feasible.
EFF_HALF = 0.045
MANIPULATE_COST = 1.0
ROBOT = "panda"


@dataclass
class World:
    seed: int
    blocks: dict                       # name -> local spheres (K,4) jnp
    block_halfz: dict                  # name -> half-height (m)
    regions: dict                      # name -> {cx,cy,z,hx,hy}
    initial_poses: dict                # name -> pose (4,) np
    goal_region: dict                  # name -> region name
    conf0: np.ndarray = field(default=None)

    def block_half_height(self, b):
        return self.block_halfz[b]

    def __post_init__(self):
        if self.conf0 is None:
            self.conf0 = np.asarray(g.NEUTRAL_Q[:7], dtype=float)


def make_rearrange_world(num_blocks, seed=0, cube_half=CUBE_HALF, tightness=0.45):
    """N-cube tabletop rearrangement: scattered start -> packed goal box.

    ``tightness`` in (0,1] is the target packing fraction (total cube footprint /
    goal-box area). Higher = tighter box = harder collision-free packing. At
    ``tightness`` near 1 the box barely fits the cubes and rejection sampling
    (PDDLStream's s-region) needs exponentially many tries; a gradient packer
    descends the overlap directly. 0.45 (default) is a loose warm-up; the sweep
    dials it up to 0.75+.
    """
    rng = np.random.default_rng(seed)
    names = [f"b{i}" for i in range(num_blocks)]

    local = g.cuboid_spheres((cube_half, cube_half, cube_half))
    blocks = {n: local for n in names}
    block_halfz = {n: cube_half for n in names}

    # Start region: front-right strip; goal box sized by target packing fraction.
    start = dict(cx=0.50, cy=-0.28, z=TABLE_Z, hx=0.14, hy=0.12)
    # Effective-footprint packing: box_area = N*(2*EFF_HALF)^2 / tightness ->
    # square half-extent EFF_HALF*sqrt(N/t). tightness->1 => cubes just tile it.
    half = EFF_HALF * float(np.sqrt(num_blocks / max(1e-3, tightness)))
    goal = dict(cx=0.45, cy=0.22, z=TABLE_Z, hx=half, hy=half)
    regions = {"start": start, "goal": goal}

    # Non-overlapping initial poses inside the start region (rejection sample).
    initial = {}
    placed = []
    for n in names:
        for _ in range(500):
            x = rng.uniform(start["cx"] - start["hx"], start["cx"] + start["hx"])
            y = rng.uniform(start["cy"] - start["hy"], start["cy"] + start["hy"])
            pose = np.array([x, y, TABLE_Z + cube_half, 0.0])
            if all(not g.blocks_collide(local, pose, local, p) for p in placed):
                initial[n] = pose
                placed.append(pose)
                break
        else:
            raise RuntimeError("could not place initial cubes without overlap")

    goal_region = {n: "goal" for n in names}
    return World(seed=seed, blocks=blocks, block_halfz=block_halfz, regions=regions,
                 initial_poses=initial, goal_region=goal_region)


def _init_and_goal(world):
    init = [
        Equal(("Cost",), MANIPULATE_COST),
        Equal((TOTAL_COST,), 0),
        ("Robot", ROBOT),
        ("CanMove", ROBOT),
        ("Conf", world.conf0),
        ("AtConf", ROBOT, world.conf0),
        ("HandEmpty", ROBOT),
    ]
    for r in world.regions:
        init.append(("Region", r))
    for b, p in world.initial_poses.items():
        init += [("Block", b), ("Pose", b, p), ("AtPose", b, p)]
        init.append(("Placeable", b, world.goal_region[b]))
    goal = And(*[("In", b, world.goal_region[b]) for b in world.initial_poses])
    return init, goal


def pddlstream_from_world(world, collisions=True,
                          motion_backend="linear", motion_params=None):
    """Wire a :class:`World` into a ``PDDLProblem``.

    ``motion_backend`` / ``motion_params`` are forwarded to
    :func:`~spasm.tamp.streams_pyroffi.make_stream_map`; everything else about
    the problem is held fixed across backends.
    """
    domain_pddl = read(get_file_path(__file__, "domain.pddl"))
    stream_pddl = read(get_file_path(__file__, "stream.pddl"))
    constant_map = {}
    stream_map = make_stream_map(world, collisions=collisions,
                                 motion_backend=motion_backend,
                                 motion_params=motion_params)
    init, goal = _init_and_goal(world)
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


# --------------------------------------------------------------------------- #
# Stacking task: N cubes scattered on the table -> a single tower.
# --------------------------------------------------------------------------- #

def make_stack_world(num_blocks=3, seed=0, cube_half=CUBE_HALF, base_xy=(0.45, 0.0)):
    """N cubes scattered on the table; b0's start pose is the tower base."""
    rng = np.random.default_rng(seed)
    names = [f"b{i}" for i in range(num_blocks)]
    local = g.cuboid_spheres((cube_half, cube_half, cube_half))
    blocks = {n: local for n in names}
    block_halfz = {n: cube_half for n in names}

    start = dict(cx=0.50, cy=-0.05, z=TABLE_Z, hx=0.16, hy=0.30)
    regions = {"table": start}

    initial, placed = {}, []
    for i, n in enumerate(names):
        if i == 0:
            pose = np.array([base_xy[0], base_xy[1], TABLE_Z + cube_half, 0.0])
        else:
            for _ in range(500):
                x = rng.uniform(start["cx"] - start["hx"], start["cx"] + start["hx"])
                y = rng.uniform(start["cy"] - start["hy"], start["cy"] + start["hy"])
                pose = np.array([x, y, TABLE_Z + cube_half, 0.0])
                if all(not g.blocks_collide(local, pose, local, p) for p in placed):
                    break
            else:
                raise RuntimeError("could not scatter cubes without overlap")
        initial[n] = pose
        placed.append(pose)

    # goal_region reused only to carry the stacking order (b_{i} on b_{i-1}).
    goal_region = {names[i]: names[i - 1] for i in range(1, num_blocks)}
    return World(seed=seed, blocks=blocks, block_halfz=block_halfz, regions=regions,
                 initial_poses=initial, goal_region=goal_region)


def _stack_init_and_goal(world):
    names = list(world.initial_poses)
    init = [
        Equal(("Cost",), MANIPULATE_COST),
        Equal((TOTAL_COST,), 0),
        ("Robot", ROBOT),
        ("CanMove", ROBOT),
        ("Conf", world.conf0),
        ("AtConf", ROBOT, world.conf0),
        ("HandEmpty", ROBOT),
    ]
    for b, p in world.initial_poses.items():
        init += [("Block", b), ("Pose", b, p), ("AtPose", b, p),
                 ("OnTable", b), ("Clear", b)]
    goal = And(*[("On", b, bu) for b, bu in world.goal_region.items()])
    return init, goal


def pddlstream_from_stack_world(world, motion_backend="linear", motion_params=None):
    from .streams_pyroffi import make_stack_stream_map
    domain_pddl = read(get_file_path(__file__, "domain_stack.pddl"))
    stream_pddl = read(get_file_path(__file__, "stream_stack.pddl"))
    stream_map = make_stack_stream_map(world, motion_backend=motion_backend,
                                       motion_params=motion_params)
    init, goal = _stack_init_and_goal(world)
    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
