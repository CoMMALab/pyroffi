"""PyRoFFI vs upstream PyRoki trajopt, head to head.

Both solvers get the same robot (robot_descriptions Panda), sphere model
(PyRoki's panda_spheres.json), world (ground + wall box, PyRoki example 07 and
randomized variants), timestep count, dt and start/goal configurations (from
PyRoki's own collision-aware IK, shared). One evaluator scores every
trajectory:

  tau   max |tau| / URDF effort limit (RNEA, central differences at dt)
  vel   max |qd| / URDF velocity limit
  world min clearance to the world, waypoints upsampled 8x (PyRoki constrains
        swept volumes; PyRoFFI checks 4 sub-steps, so 8x is stricter than
        either optimizer), excluding the fixed base link
  self  min self-collision clearance, excluding pairs that overlap at every
        configuration in this sphere model
  lim   max joint-limit violation (rad)
  acc   sum of squared accelerations (smoothness)
  time  warm wall-clock of the trajopt call only (endpoints precomputed)

Run: CUDA_VISIBLE_DEVICES=<free gpu> python tests/bench_pyroki_trajopt.py
Requires the upstream checkout at baselines/pyroki (its examples provide
PyRoki's reference `solve_trajopt`).
"""

import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "pyroki" / "examples"))

import pyroki  # noqa: E402
import pyroki_snippets._trajopt as pks_trajopt  # noqa: E402

import pyroffi as pk  # noqa: E402
from pyroffi.optimization_engines import (  # noqa: E402
    DynamicsAwareTrajOptConfig,
    ScoTrajOptConfig,
    dynamics_aware_trajopt,
    effort_limits,
    sco_trajopt,
)
from pyroffi.optimization_engines._contact_trajopt import _fd_vel_acc  # noqa: E402

DOWN_WXYZ = onp.array([0.0, 0.0, 1.0, 0.0])
EE = "panda_hand"


def make_problems(n: int, seed: int) -> list[dict]:
    """PyRoki example 07 first, then randomized wall/endpoint variants."""
    problems = [dict(wall_x=0.5, wall_h=0.4, start=(0.5, -0.35, 0.2), end=(0.5, 0.35, 0.2))]
    rng = onp.random.default_rng(seed)
    while len(problems) < n:
        x, h = rng.uniform(0.45, 0.6), rng.uniform(0.25, 0.45)
        y0, y1 = rng.uniform(0.3, 0.4), rng.uniform(0.3, 0.4)
        z0, z1 = rng.uniform(0.15, 0.3, size=2)
        problems.append(dict(wall_x=x, wall_h=h, start=(x, -y0, z0), end=(x, y1, z1)))
    return problems


def world_for(lib, p):
    ground = lib.collision.HalfSpace.from_point_and_normal(
        onp.zeros(3), onp.array([0.0, 0.0, 1.0]))
    wall = lib.collision.Box.from_extent(
        extent=onp.array([0.4, 0.1, p["wall_h"]]),
        position=onp.array([p["wall_x"], 0.0, p["wall_h"] / 2]))
    return [ground, wall]


class Evaluator:
    def __init__(self, robot, rc, world, dt):
        self.robot, self.rc, self.world, self.dt = robot, rc, world, dt
        self.eff = effort_limits(robot)
        self.vel = robot.joints.velocity_limits
        self.lo, self.hi = robot.joints.lower_limits, robot.joints.upper_limits
        base = robot.links.names.index("panda_link0")
        self.world_links = jnp.arange(rc.num_links) != base
        # Pairs overlapping at both the zero and the default config never separate.
        d0 = rc.compute_self_collision_distance(robot, jnp.zeros_like(robot.default_cfg))
        d1 = rc.compute_self_collision_distance(robot, robot.default_cfg)
        self.self_pairs = ~((d0 < 0) & (d1 < 0))
        self.n_permanent = int((~self.self_pairs).sum())

    def __call__(self, traj) -> dict:
        traj = jnp.asarray(traj, jnp.float32)
        qd, qdd = _fd_vel_acc(traj, self.dt)
        tau = self.robot.inverse_dynamics(traj, qd, qdd)
        a = jnp.linspace(0.0, 1.0, 9)[:-1, None, None]
        dense = ((1 - a) * traj[None, :-1] + a * traj[None, 1:]).reshape(-1, traj.shape[1])
        dense = jnp.concatenate([dense, traj[-1:]])
        world = min(
            float(jnp.min(jnp.where(
                self.world_links[None, :, None],
                self.rc.compute_world_collision_distance(self.robot, dense, w), jnp.inf)))
            for w in self.world)
        selfd = float(jnp.min(jnp.where(
            self.self_pairs, self.rc.compute_self_collision_distance(self.robot, dense), jnp.inf)))
        return dict(
            tau=float(jnp.max(jnp.abs(tau) / self.eff)),
            vel=float(jnp.max(jnp.abs(qd) / self.vel)),
            world=world,
            self=selfd,
            lim=float(jnp.max(jnp.maximum(traj - self.hi, self.lo - traj).clip(0.0))),
            acc=float(jnp.sum(qdd**2) * self.dt),
        )


import jax_dataclasses as jdc  # noqa: E402
import jaxls  # noqa: E402


@jaxls.Cost.factory(kind="constraint_eq_zero", name="endpoint_constraint")
def _endpoint_constraint(vals, var, target):
    return (vals[var] - target).flatten()


def _world_coll_residual(vals, robot, robot_coll, world_coll_obj, prev_vars, curr_vars):
    coll = robot_coll.get_swept_capsules(robot, vals[prev_vars], vals[curr_vars])
    dist = pyroki.collision.collide(coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1)))
    return dist.flatten() - 0.05


@jdc.jit
def pyroki_trajopt_jit(robot, robot_coll, world_coll, start_cfg, end_cfg,
                       timesteps: jdc.Static[int], dt: jdc.Static[float]):
    """PyRoki's `solve_trajopt` (pyroki_snippets/_trajopt.py), cost for cost, with the
    endpoints passed as data instead of closed over, so it compiles once. The
    shipped snippet redefines its endpoint constraints per call and recompiles."""
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    robot = jax.tree.map(lambda x: x[None], robot)
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)
    costs = [
        pyroki.costs.rest_cost(traj_vars, traj_vars.default_factory()[None], jnp.array([0.01])[None]),
        pyroki.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)), jnp.array([1])[None]),
        pyroki.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt, jnp.array([0.1])[None]),
        pyroki.costs.self_collision_cost(robot, robot_coll, traj_vars, 0.02, 5.0),
        pyroki.costs.limit_constraint(robot, traj_vars),
        _endpoint_constraint(robot.joint_var_cls(jnp.arange(0, 2)), start_cfg[None]),
        _endpoint_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)), end_cfg[None]),
        pyroki.costs.limit_velocity_constraint(
            robot, robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)), dt),
    ]
    for w in world_coll:
        costs.append(jaxls.Cost(
            _world_coll_residual,
            (robot, robot_coll, jax.tree.map(lambda x: x[None], w),
             robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
             robot.joint_var_cls(jnp.arange(1, timesteps))),
            kind="constraint_geq_zero", name="world Collision (sweep)"))
    sol = jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars]).analyze().solve(
        initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)), verbose=False)
    return sol[traj_vars]


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def main(
    n_problems: int = 12,
    seeds: int = 6,
    timesteps: int = 50,
    dt: float = 0.02,
    batch: int = 8,
    noise: float = 0.05,
    out: Path = ROOT / "resources" / "bench_pyroki_trajopt.csv",
) -> None:
    urdf = load_robot_description("panda_description")
    sd = json.loads((ROOT / "resources" / "panda_spheres.json").read_text())
    up_robot = pyroki.Robot.from_urdf(urdf)
    up_rc = pyroki.collision.RobotCollision.from_sphere_decomposition(sd, urdf)
    robot = pk.Robot.from_urdf(urdf)
    rc = pk.collision.RobotCollision.from_sphere_decomposition(sd, urdf)
    ee_idx = up_robot.links.names.index(EE)

    # Match PyRoki's world-collision safety margin (0.05 m).
    dyn_cfg = DynamicsAwareTrajOptConfig(dt=dt, collision_margin=0.05)
    sco_cfg = ScoTrajOptConfig(collision_margin=0.05)

    rows = []
    compiled = set()
    problems = make_problems(n_problems, seed=0)
    for pi, p in enumerate(problems):
        up_world, world = world_for(pyroki, p), tuple(world_for(pk, p))
        start_cfg, end_cfg = pks_trajopt.solve_iks_with_collision(
            up_robot, up_rc, up_world, ee_idx,
            jnp.asarray(p["start"]), jnp.asarray(DOWN_WXYZ),
            jnp.asarray(p["end"]), jnp.asarray(DOWN_WXYZ))
        start_cfg, end_cfg = jnp.asarray(start_cfg), jnp.asarray(end_cfg)
        evaluate = Evaluator(robot, rc, world, dt)
        if pi == 0:
            print(f"excluded permanent self-overlap pairs: {evaluate.n_permanent}")

        # PyRoki's solve_trajopt with its IK stubbed to the shared endpoints.
        pks_trajopt.solve_iks_with_collision = lambda *a, **k: (start_cfg, end_cfg)
        run_pyroki = lambda: pks_trajopt.solve_trajopt(
            up_robot, up_rc, up_world, EE, onp.asarray(p["start"]), DOWN_WXYZ,
            onp.asarray(p["end"]), DOWN_WXYZ, timesteps, dt)

        line = jnp.linspace(start_cfg, end_cfg, timesteps)
        methods = {
            "pyroki_snippet": [(0, run_pyroki)],
            "pyroki_jit": [(0, lambda: pyroki_trajopt_jit(
                up_robot, up_rc, tuple(up_world), start_cfg, end_cfg, timesteps, dt))],
        }
        # Like-for-like: one straight-line seed, as PyRoki uses.
        methods["pyroffi_dyn_B1"] = [(0, lambda: dynamics_aware_trajopt(
            line[None], start_cfg, end_cfg, robot, rc, world, dyn_cfg)[0])]
        # Batched noisy seeds (the motion generator's mode), several seeds.
        noisy = lambda s: line[None] + noise * jax.random.normal(
            jax.random.PRNGKey(s), (batch, *line.shape))
        methods[f"pyroffi_dyn_B{batch}"] = [
            (s, (lambda s=s: dynamics_aware_trajopt(
                noisy(s), start_cfg, end_cfg, robot, rc, world, dyn_cfg)[0]))
            for s in range(seeds)]
        methods[f"pyroffi_sco_B{batch}"] = [
            (s, (lambda s=s: sco_trajopt(
                noisy(s), start_cfg, end_cfg, robot, rc, world, sco_cfg)[0]))
            for s in range(seeds)]

        trajs = {}
        for name, runs in methods.items():
            for seed, fn in runs:
                if name not in compiled:
                    _, t_compile = timed(fn)
                    compiled.add(name)
                    print(f"  [{name}] first call (compile) {t_compile:.1f}s")
                traj, t = timed(fn)
                m = evaluate(traj)
                rows.append(dict(problem=pi, method=name, seed=seed, time=t, **m))
                trajs[name] = traj
        port_err = float(jnp.abs(jnp.asarray(trajs["pyroki_jit"]) - trajs["pyroki_snippet"]).max())
        print(f"problem {pi + 1}/{len(problems)} done (jit port vs snippet max |dq| = {port_err:.2e} rad)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    def med(xs):
        return float(onp.median(xs))

    print(f"\n{'method':18s} {'time_s':>7s} {'tau<=1':>7s} {'tau_med':>8s} {'vel<=1':>7s} "
          f"{'world>=0':>9s} {'self>=0':>8s} {'all_ok':>7s} {'acc_med':>9s}")
    for name in methods:
        r = [x for x in rows if x["method"] == name]
        ok = [x["tau"] <= 1 and x["vel"] <= 1 and x["world"] >= 0 and x["self"] >= 0
              and x["lim"] <= 1e-3 for x in r]
        print(f"{name:18s} {med([x['time'] for x in r]):7.2f} "
              f"{onp.mean([x['tau'] <= 1 for x in r]):7.0%} {med([x['tau'] for x in r]):8.2f} "
              f"{onp.mean([x['vel'] <= 1 for x in r]):7.0%} "
              f"{onp.mean([x['world'] >= 0 for x in r]):9.0%} "
              f"{onp.mean([x['self'] >= 0 for x in r]):8.0%} {onp.mean(ok):7.0%} "
              f"{med([x['acc'] for x in r]):9.1f}")


if __name__ == "__main__":
    tyro.cli(main)
