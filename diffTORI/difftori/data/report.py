"""Quality gates for a demonstration dataset.

"Good demonstrations" is not a vibe; each gate below is a property that, if
violated, makes the dataset actively misleading to train on.  Run this before
committing GPU time:

    PYTHONPATH=diffTORI python -m difftori.data.report --data <path.zarr>

Gates
-----
``no_penetration``   No waypoint of any episode has a robot collision sphere
                     inside the obstacle.  A demonstration that passes through
                     the obstacle teaches the policy to do the same.
``stationary``       Every kept context's inner solve reached ``||grad_x C|| <
                     1e-4``.  A non-stationary solve is not an optimum; it is
                     wherever the solver's budget ran out.
``obstacle_active``  The straight-line seed violates the clearance margin in
                     most scenes, so the demonstrated detour is caused by the
                     obstacle rather than being a straight line that happens to
                     miss it.  Without this the dataset teaches nothing about
                     avoidance.
``actions_in_box``   Normalised actions lie in [-1, 1]; DiffTORI's inner problem
                     carries a unit-box barrier and anything outside it is
                     unreachable for the policy.
``chunks_available`` Enough ``(n_obs_steps, horizon)`` windows exist to train on.

``diversity`` is reported, not gated: there is no threshold that is right for
every task, but a collapse in end-effector spread is the first sign the scene
sampler has stopped covering anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from difftori.data.dataset import ReplayBuffer, SequenceDataset
from difftori.data.visualize import _resolve, unpack_episodes

CLEARANCE_MARGIN = 0.05   # ioc.robot.problem.CLEARANCE_MARGIN


def _seed_clearance(problem, scenes_like):
    """Min clearance of the straight-line seed, per episode."""
    from ioc.robot.problem import Scene

    out = []
    for ep in scenes_like:
        s = Scene(q_start=jnp.asarray(ep["q"][0]),
                  q_goal=jnp.asarray(ep["q_goal"]),
                  obs_center=jnp.asarray(ep["obs_center"]),
                  obs_radius=jnp.asarray([ep["obs_radius"]]))
        q = problem.unpack(problem.seed(s), s)
        coll = problem.robot_coll.at_config(problem.robot, q)
        d = (jnp.linalg.norm(coll.pose.translation() - s.obs_center, axis=-1)
             - coll.radius - s.obs_radius[0])
        out.append(float(jnp.min(d.reshape(d.shape[0], -1))))
    return np.array(out)


def quality_report(data: str, n_obs_steps: int = 2, horizon: int = 4,
                   verbose: bool = True) -> dict:
    from ioc.robot import problem as prob

    buf = ReplayBuffer.load(_resolve(data))
    meta = buf.meta
    problem = prob.RobotProblem.load(
        _resolve("resources/panda/panda_spherized.urdf"),
        _resolve("resources/panda/panda.srdf"),
        _resolve("resources/panda/meshes"),
        int(meta.get("n_timesteps", 16)))

    eps = unpack_episodes(buf, problem)
    clearance = np.array([ep["clearance"].min() for ep in eps])
    seed_clr = _seed_clearance(problem, eps)
    ee_start = np.stack([ep["ee"][0] for ep in eps])
    ee_goal = np.stack([ep["ee"][-1] for ep in eps])
    ds = SequenceDataset(buf, n_obs_steps=n_obs_steps, horizon=horizon)

    gates = {
        "no_penetration": bool((clearance >= 0).all()),
        "stationary": float(meta.get("max_stationarity_kept", np.inf)) < 1e-4,
        "obstacle_active": float((seed_clr < CLEARANCE_MARGIN).mean()) > 0.9,
        "actions_in_box": bool(np.abs(buf.action).max() <= 1.0 + 1e-6),
        "chunks_available": len(ds) >= 1000,
    }
    stats = {
        "episodes": buf.n_episodes,
        "rows": len(buf.state),
        "train_windows": len(ds),
        "min_clearance_worst_m": float(clearance.min()),
        "min_clearance_mean_m": float(clearance.mean()),
        "episodes_penetrating": int((clearance < 0).sum()),
        "seed_active_frac": float((seed_clr < CLEARANCE_MARGIN).mean()),
        "seed_penetrating_frac": float((seed_clr < 0).mean()),
        "ee_start_spread_m": float(ee_start.std(axis=0).mean()),
        "ee_goal_spread_m": float(ee_goal.std(axis=0).mean()),
        "max_stationarity": float(meta.get("max_stationarity_kept", np.nan)),
        "action_scale_rad": float(meta.get("action_scale", np.nan)),
    }

    if verbose:
        print(f"dataset: {data}")
        print(f"  task {meta.get('task')}  teacher {meta.get('teacher')}")
        for k, v in stats.items():
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float)
                  else f"  {k:24s} {v}")
        print("  gates:")
        for k, ok in gates.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"  => {'READY TO TRAIN' if all(gates.values()) else 'NOT READY'}")
    return {"gates": gates, "stats": stats, "ready": all(gates.values())}


def main(data: str = "diffTORI/data/panda_reach_expert_v2.zarr",
         n_obs_steps: int = 2, horizon: int = 4):
    report = quality_report(data, n_obs_steps, horizon)
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
