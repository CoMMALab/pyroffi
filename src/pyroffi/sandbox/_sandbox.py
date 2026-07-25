"""The execution sandbox: a stepped MuJoCo world an agent acts on.

This is deliberately a **different adapter** from :mod:`pyroffi.mcp`. That
server computes and never commands anything; the boundary is structural rather
than a convention, and the way to keep it structural while still having
something to execute against is to put execution in its own package with its
own entry point. Nothing here is imported by ``pyroffi.mcp``.

What the sandbox is honest about:

* **Nothing is welded or teleported.** Blocks are free bodies with real
  collision geometry and mass, held by a tendon-driven gripper closing on
  them. A tower that gets knocked over falls over.
* **The reference is tracked, not applied.** ``execute_path`` interpolates the
  commanded waypoints onto the control grid and reports what the arm actually
  did. A discontinuous reference rips the block out of the gripper — measured,
  not hypothesised — which is why an un-retimed path is accepted but flagged.
* **Observation is of the simulator, not of the plan.** ``observe`` reads
  ``MjData``. If a place went 2 cm wrong, that is what comes back, and
  replanning against it is the agent's job.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger

from ._scene import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    MENAGERIE_ARM_JOINTS,
    build_scene,
    viewer_geometry,
)

ARM_DOF = 7


@dataclasses.dataclass
class ExecutionRecord:
    """What one commanded motion actually did."""

    kind: str
    duration_s: float
    max_tracking_error_rad: float
    mean_tracking_error_rad: float
    n_waypoints: int
    retimed: bool


class Sandbox:
    """A block-manipulation world, stepped in MuJoCo and rendered through viser.

    The agent's contract is small on purpose — move the arm, work the gripper,
    look at the result — because everything *decided* (where to grasp, what
    order, how to get there) belongs to the orchestrator and to the pyroffi
    planning server, not here.
    """

    def __init__(
        self,
        task: Mapping[str, Any],
        variant: str = "wall",
        viewer_port: int = 8080,
        start_viewer: bool = True,
        realtime: bool = True,
        control_hz: float = 500.0,
    ) -> None:
        import mujoco

        self.task = dict(task)
        self.variant = variant
        self.blocks = list(task["blocks"])
        self.obstacles = list(task["variants"][variant]["obstacles"])
        self.block_size = np.asarray(task["block_size_m"], dtype=np.float64)
        self.joint_names = tuple(task["robot_setup"]["start_config"])
        self.realtime = bool(realtime)
        self._mujoco = mujoco

        self.scene = build_scene(
            self.blocks, self.obstacles, block_size=self.block_size
        )
        self.model, self.data = self.scene.model, self.scene.data
        self.control_dt = 1.0 / float(control_hz)
        self._steps_per_control = max(
            1, int(round(self.control_dt / float(self.model.opt.timestep)))
        )

        self._lock = threading.Lock()
        self._history: list[ExecutionRecord] = []
        self._gripper = GRIPPER_OPEN
        self._t_start = time.time()

        self.reset()

        self.viewer = None
        if start_viewer:
            self._start_viewer(viewer_port)

    # ── viewer ────────────────────────────────────────────────────────────

    def _start_viewer(self, port: int) -> None:
        """Bind the official pyroffi render layer to this simulation.

        The source is a :class:`~pyroffi.viewer.MuJoCoSource` reading ``MjData``
        in place, so what is drawn is what the physics did. Swapping that source
        for a perception stack is the only change needed to point the same
        viewer at a real cell.
        """
        from ..toolbox import load_urdf
        from ..viewer import MuJoCoSource, RenderViewer

        # The robot *drawn* is pyroffi's planning URDF, driven by joint values
        # read out of the Menagerie model — which spells the same joints
        # differently. The mapping is explicit rather than a prefix rule,
        # because reading the wrong joint would draw a plausible robot that is
        # not the one being simulated.
        source = MuJoCoSource(
            self.model,
            self.data,
            joint_names=self.joint_names,
            mujoco_joint_names=MENAGERIE_ARM_JOINTS,
            object_bodies={name: name for name in self.scene.block_bodies},
            geometry=viewer_geometry(self.blocks, self.obstacles, self.block_size),
            urdf=load_urdf(self.task["robot"]),
        )
        self.viewer = RenderViewer(source, port=port, rate_hz=30.0).start()

    def render(self, viewpoint: str | None = "iso", width: int = 960, height: int = 720):
        """Base64 PNG through the viser client. Raises when nobody is looking."""
        if self.viewer is None:
            raise RuntimeError("this sandbox was started with start_viewer=False")
        return self.viewer.capture_base64(viewpoint, width=width, height=height)

    # ── state ─────────────────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Put the world back to the task's initial state."""
        mujoco = self._mujoco
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            start = self.task["robot_setup"]["start_config"]
            q0 = np.array([float(start[n]) for n in self.joint_names])
            self.data.qpos[:ARM_DOF] = q0
            self.data.qpos[ARM_DOF : ARM_DOF + 2] = 0.04
            for block in self.blocks:
                adr = self.scene.block_qpos_adr[block["name"]]
                self.data.qpos[adr : adr + 3] = block["position"]
                self.data.qpos[adr + 3 : adr + 7] = block.get(
                    "wxyz", (1.0, 0.0, 0.0, 0.0)
                )
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:ARM_DOF] = q0
            self.data.ctrl[ARM_DOF] = GRIPPER_OPEN
            self._gripper = GRIPPER_OPEN
            self._history.clear()
            self._t_start = time.time()
            # Let the blocks settle onto the floor before anything is measured,
            # so a "block moved" report later is about the agent, not about
            # 2 mm of initial penetration relaxing.
            self._advance(200)
        return self.observe()

    def _advance(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self._mujoco.mj_step(self.model, self.data)

    def arm_config(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:ARM_DOF], dtype=np.float64).copy()

    def observe(self) -> dict[str, Any]:
        """The simulator's actual state — the agent's only ground truth."""
        with self._lock:
            q = self.arm_config()
            opening = float(self.data.qpos[ARM_DOF] + self.data.qpos[ARM_DOF + 1])
            held = self._grasped_block()
            blocks = {}
            for name, body in self.scene.block_bodies.items():
                blocks[name] = {
                    "position": [round(float(v), 5) for v in self.data.xpos[body]],
                    "wxyz": [round(float(v), 5) for v in self.data.xquat[body]],
                    "resting_on": self._support_of(name),
                }
            hand = self.data.xpos[self.scene.hand_body]
            return {
                "sim_time_s": round(float(self.data.time), 3),
                "joint_values": {
                    n: round(float(v), 6) for n, v in zip(self.joint_names, q)
                },
                "gripper_opening_m": round(opening, 5),
                "gripper_state": "open" if self._gripper >= 128 else "closed",
                "held_block": held,
                "hand_position": [round(float(v), 5) for v in hand],
                "blocks": blocks,
            }

    def _grasped_block(self) -> str | None:
        """A block is held when both fingers are in contact with it.

        Read from the contact list rather than assumed from the last command:
        commanding a close does not mean anything was caught.
        """
        fingers = set(self.scene.finger_geoms)
        touching: dict[str, set[int]] = {}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            for name, gid in self.scene.block_geoms.items():
                if gid == g1 and g2 in fingers:
                    touching.setdefault(name, set()).add(g2)
                elif gid == g2 and g1 in fingers:
                    touching.setdefault(name, set()).add(g1)
        for name, geoms in touching.items():
            bodies = {int(self.model.geom_bodyid[g]) for g in geoms}
            if len(bodies) >= 2:          # both fingers, not just a graze
                return name
        return None

    def _support_of(self, name: str) -> str:
        """What the block is resting on: the ground, or another block."""
        z = float(self.data.xpos[self.scene.block_bodies[name]][2])
        half = float(self.block_size[2] / 2.0)
        if z < half * 1.5:
            return "ground"
        xy = np.asarray(self.data.xpos[self.scene.block_bodies[name]][:2])
        best, best_z = "air", -np.inf
        for other, body in self.scene.block_bodies.items():
            if other == name:
                continue
            p = np.asarray(self.data.xpos[body])
            if p[2] < z - half and float(np.linalg.norm(p[:2] - xy)) < half * 2:
                if p[2] > best_z:
                    best, best_z = other, float(p[2])
        return best

    # ── acting ────────────────────────────────────────────────────────────

    def execute_path(
        self,
        waypoints: Sequence[Mapping[str, float]] | np.ndarray,
        times_s: Sequence[float] | None = None,
        default_speed_rad_s: float = 0.6,
        settle_s: float = 0.3,
    ) -> dict[str, Any]:
        """Track a joint-space path with the arm's position actuators.

        Args:
            waypoints: name-keyed configurations, or a ``(T, 7)`` array in
                ``joint_names`` order.
            times_s: per-waypoint times from ``retime``. Strongly preferred: the
                reference is interpolated onto the control grid, and a path with
                no timing is executed at a fixed joint speed that may be far
                from what the motion needs. Commanding a step change pulls a
                held block out of the gripper.
            settle_s: extra time holding the final waypoint, so the reported
                state is settled rather than mid-transient.
        """
        arr = self._as_array(waypoints)
        if arr.shape[0] < 1:
            raise ValueError("execute_path needs at least one waypoint")

        start = self.arm_config()
        gap = float(np.max(np.abs(arr[0] - start)))
        if gap > 0.05:
            raise ValueError(
                f"path starts {gap:.3f} rad from the arm's current configuration. "
                "Execution does not teleport: plan from the configuration "
                "observe() reports."
            )

        retimed = times_s is not None
        if retimed:
            t = np.asarray(times_s, dtype=np.float64).reshape(-1)
            if t.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"times_s has {t.shape[0]} entries for {arr.shape[0]} waypoints"
                )
        else:
            step = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(step / max(default_speed_rad_s, 1e-6))])
        duration = float(t[-1]) if t[-1] > 0 else 0.1

        n_ticks = max(2, int(np.ceil(duration / self.control_dt)))
        grid = np.linspace(0.0, duration, n_ticks)
        ref = np.stack(
            [np.interp(grid, t, arr[:, j]) for j in range(ARM_DOF)], axis=1
        )

        errors = []
        with self._lock:
            for k in range(n_ticks):
                self.data.ctrl[:ARM_DOF] = ref[k]
                self.data.ctrl[ARM_DOF] = self._gripper
                self._advance(self._steps_per_control)
                errors.append(np.abs(self.arm_config() - ref[k]))
                if self.realtime:
                    time.sleep(self.control_dt * 0.5)
            for _ in range(int(settle_s / self.control_dt)):
                self.data.ctrl[:ARM_DOF] = ref[-1]
                self.data.ctrl[ARM_DOF] = self._gripper
                self._advance(self._steps_per_control)

        err = np.asarray(errors)
        record = ExecutionRecord(
            kind="execute_path",
            duration_s=duration,
            max_tracking_error_rad=float(err.max()),
            mean_tracking_error_rad=float(err.mean()),
            n_waypoints=int(arr.shape[0]),
            retimed=retimed,
        )
        self._history.append(record)

        result = {
            "success": True,
            "duration_s": round(duration, 3),
            "n_waypoints": int(arr.shape[0]),
            "retimed": retimed,
            "max_tracking_error_rad": round(record.max_tracking_error_rad, 5),
            "mean_tracking_error_rad": round(record.mean_tracking_error_rad, 5),
            "observation": self.observe(),
        }
        if not retimed:
            result["warning"] = (
                "no times_s supplied, so this ran at a fixed joint speed. Use "
                "the pyroffi server's retime() and pass its times: a reference "
                "the arm cannot follow pulls a held block out of the gripper."
            )
        return result

    def set_gripper(self, action: str, settle_s: float = 0.8) -> dict[str, Any]:
        """Open or close the gripper and report what actually happened.

        Closing is a *command*, not an outcome: the response reports whether a
        block ended up between the fingers, which is the only thing worth
        knowing.
        """
        if action not in ("open", "close"):
            raise ValueError(f"action must be 'open' or 'close', got {action!r}")
        target = GRIPPER_OPEN if action == "open" else GRIPPER_CLOSE
        held_before = self._grasped_block()

        with self._lock:
            self._gripper = target
            hold = self.arm_config()
            for _ in range(int(settle_s / self.control_dt)):
                self.data.ctrl[:ARM_DOF] = hold
                self.data.ctrl[ARM_DOF] = target
                self._advance(self._steps_per_control)
                if self.realtime:
                    time.sleep(self.control_dt * 0.5)

        held = self._grasped_block()
        obs = self.observe()
        result: dict[str, Any] = {
            "success": True,
            "action": action,
            "held_block": held,
            "gripper_opening_m": obs["gripper_opening_m"],
            "observation": obs,
        }
        if action == "close" and held is None:
            result["success"] = False
            result["note"] = (
                "the gripper closed on nothing. Check the hand pose against the "
                "block centre with forward_kinematics: a top-down grasp wants the "
                "hand frame grasp_standoff_m above the block, +z pointing down."
            )
        if action == "open" and held_before is not None:
            result["released"] = held_before
        return result

    def _as_array(self, waypoints: Any) -> np.ndarray:
        if isinstance(waypoints, np.ndarray):
            return np.asarray(waypoints, dtype=np.float64).reshape(-1, ARM_DOF)
        rows = []
        for wp in waypoints:
            if isinstance(wp, Mapping):
                missing = [n for n in self.joint_names if n not in wp]
                if missing:
                    raise ValueError(
                        f"waypoint is missing joint values for {missing}; "
                        f"waypoints are name-keyed over {list(self.joint_names)}"
                    )
                rows.append([float(wp[n]) for n in self.joint_names])
            else:
                rows.append([float(v) for v in wp])
        return np.asarray(rows, dtype=np.float64).reshape(-1, ARM_DOF)

    # ── scoring ───────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        """Performance report: did the tower get built, and at what cost."""
        goal = self.task["goal"]
        tol = goal["tolerances"]
        size = float(self.block_size[2])
        x, y = goal["base_position_xy"]

        per_block = {}
        ok = True
        for level, name in enumerate(goal["order"]):
            target = np.array([x, y, size * (level + 0.5)])
            actual = np.asarray(self.data.xpos[self.scene.block_bodies[name]])
            xy_err = float(np.linalg.norm(actual[:2] - target[:2]))
            z_err = float(abs(actual[2] - target[2]))
            block_ok = xy_err <= float(tol["stack_xy_m"]) and z_err <= float(
                tol["stack_xy_m"]
            )
            ok = ok and block_ok
            per_block[name] = {
                "level": level,
                "target": [round(float(v), 4) for v in target],
                "actual": [round(float(v), 4) for v in actual],
                "xy_error_m": round(xy_err, 4),
                "z_error_m": round(z_err, 4),
                "resting_on": self._support_of(name),
                "ok": block_ok,
            }

        moves = [r for r in self._history if r.kind == "execute_path"]
        return {
            "task": self.task["task_id"],
            "variant": self.variant,
            "success": ok,
            "blocks": per_block,
            "n_motions": len(moves),
            "n_unretimed_motions": sum(1 for r in moves if not r.retimed),
            "commanded_duration_s": round(sum(r.duration_s for r in moves), 2),
            "sim_time_s": round(float(self.data.time), 2),
            "wall_time_s": round(time.time() - self._t_start, 1),
            "max_tracking_error_rad": (
                round(max(r.max_tracking_error_rad for r in moves), 5)
                if moves else None
            ),
            "still_held": self._grasped_block(),
        }

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.stop()
