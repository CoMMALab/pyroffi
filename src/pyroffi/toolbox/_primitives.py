"""The motion-primitive operations, transport-agnostic.

Every operation here is a *primitive*, not a policy. They expose their knobs
(``num_seeds``, ``solver``, tolerances) and report what actually happened —
including partial success, which for batched IK is real information ("47/64
restarts converged, best residual 1.2e-4"). None of them retries, escalates to
a different solver, or applies a cost-acceptance heuristic: those are the
orchestrator's decisions, and a primitive that hides a retry makes the agent's
model of the world wrong.

Responses come in two registers. The default is agent-facing: a handle plus the
decision summary, and nothing else — a 64x7 float64 path is ~4k tokens of noise
to a VLM. Raw joint arrays cross the boundary only through the explicit
``export_path`` / ``import_path`` pair.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger

from . import _exchange as ex
from ._retiming import default_acceleration_limits, retime_path
from ._session import Session, bucket_length, pad_path

_DEFAULT_MARGIN = 0.0
"""Clearance below which a pair counts as colliding, in metres. Zero means true
geometric contact; a caller wanting a safety buffer passes its own margin."""

_IK_POS_TOL = 1e-3
_IK_ROT_TOL = 1e-2


class Toolbox:
    """Primitives bound to one warm :class:`Session`.

    Deliberately holds no planning state: no notion of "the current plan", no
    memory of what the agent was trying to do. Handles and the scene are the
    only state, and both are explicit.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._request_counter = itertools.count(1)
        self._failures: dict[str, dict[str, Any]] = {}

    # ── response plumbing ─────────────────────────────────────────────────

    def _new_request(self) -> tuple[str, float]:
        return f"req_{next(self._request_counter):04d}", time.perf_counter()

    def _envelope(
        self,
        request_id: str,
        t0: float,
        compiled: bool,
        success: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        out = {
            "request_id": request_id,
            "success": bool(success),
            "solve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "compiled": bool(compiled),
            "scene_version": self.session.scene.version,
        }
        out.update(fields)
        return out

    def _record_failure(
        self, request_id: str, cause: str, detail: dict[str, Any]
    ) -> None:
        """Stash a structured cause so ``explain_failure`` has something real to say."""
        self._failures[request_id] = {"cause": cause, **detail}

    def _inherit_failure(self, outer_id: str, *inner_ids: str) -> None:
        """Republish an inner call's failure under the id the caller was handed.

        Wrappers (``check_reachable``, ``optimize_between``) delegate to other
        primitives, which record failures under *their* request id — an id the
        caller never sees. Without this, ``explain_failure`` on the only id the
        agent has says "no failure recorded", which reads as "it succeeded".
        """
        for inner in inner_ids:
            record = self._failures.get(inner)
            if record is not None:
                self._failures[outer_id] = {**record, "recorded_by": inner}
                return

    # ── scene ─────────────────────────────────────────────────────────────

    def create_scene_info(self) -> dict[str, Any]:
        """Session capabilities. The first call an orchestrator should make."""
        request_id, t0 = self._new_request()
        return self._envelope(
            request_id, t0, compiled=False, capabilities=self.session.capabilities()
        )

    def reset_scene(self, keep_ground_plane: bool = True) -> dict[str, Any]:
        """Wipe the problem, keep the session: empty scene, no attachments, no
        handles, robot back at its default configuration.

        This is the between-problems operation for a long-lived server, and it
        is deliberately not ``create_scene``. Recreating the session reparses
        the URDF and throws away every compiled function, so the next call pays
        the tens of seconds of XLA compilation again; a reset only mutates
        state the problem owns, so the warm session survives it. Detaching does
        change collision array shapes and so recompiles, but only if something
        was actually attached.

        Args:
            keep_ground_plane: Keep the ``ground`` halfspace if the session was
                built with one. It belongs to the world rather than to any one
                problem, and dropping it silently lets the next problem plan
                paths through the floor.
        """
        request_id, t0 = self._new_request()
        s = self.session
        detached = list(s.attachments.names())
        for name in detached:
            s.detach_object(name)
        removed = [
            name
            for name in s.scene.names()
            if not (keep_ground_plane and name == "ground")
        ]
        for name in removed:
            s.scene.remove_object(name)
        s.handles.clear()
        self._failures.clear()
        s.robot_state = np.asarray(s.robot.default_cfg, dtype=np.float64)
        return self._envelope(
            request_id,
            t0,
            compiled=bool(detached),
            detached=detached,
            removed_objects=removed,
            handles_invalidated=True,
            n_objects=len(s.scene.names()),
            remaining_objects=s.scene.names(),
        )

    def add_object(
        self,
        name: str,
        shape: str,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add or move a named obstacle. Adding never changes array shapes."""
        request_id, t0 = self._new_request()
        s = self.session
        obj = s.scene.add_object(
            name, shape, position=position, wxyz=wxyz, params=dict(params or {})
        )
        # World collisions against bolted-down links are excluded from every
        # later query (no motion can clear them), so an object that intersects
        # one has to be surfaced here or not at all.
        intersects_static = self._static_link_overlap(name)
        return self._envelope(
            request_id,
            t0,
            compiled=False,
            object=obj.to_dict(),
            n_objects=len(s.scene.names()),
            free_slots=len(s.scene._free[obj.shape]),
            intersects_static_links=intersects_static or None,
            **(
                {
                    "warning": f"{name!r} overlaps the robot's fixed base geometry; "
                    "collisions there are excluded from validation because no motion "
                    "can clear them. Check the object's placement."
                }
                if intersects_static
                else {}
            ),
        )

    def _static_link_overlap(self, object_name: str) -> list[str]:
        """Static links this object intersects at the current robot state."""
        s = self.session
        if not s.static_link_indices:
            return []
        try:
            import jax

            geoms = s.world_geoms()
            cfg = s.as_array(s.robot_state)
            hits: list[str] = []
            for pool, (shape, slot_names) in zip(geoms, s.scene.geom_names()):
                if object_name not in slot_names:
                    continue
                slot = slot_names.index(object_name)
                # Via the session ledger, not a fresh `jax.jit(lambda ...)`: a
                # new lambda every call is a new cache key, so add_object paid a
                # full trace+compile per object while advertising itself as free.
                fn, _ = s.jitted(
                    f"static_overlap:{shape}",  # pools are fixed-capacity per shape
                    lambda: jax.jit(
                        lambda c, g: s.robot_coll.compute_world_collision_distance(
                            s.robot, c, g
                        )
                    ),
                )
                d = np.asarray(fn(cfg, pool))
                for i in s.static_link_indices:
                    if d[i, slot] < 0.0:
                        hits.append(s.link_names[i])
            return sorted(set(hits))
        except Exception as exc:  # pragma: no cover - advisory only
            logger.debug(f"static-link overlap check skipped: {exc}")
            return []

    def remove_object(self, name: str) -> dict[str, Any]:
        request_id, t0 = self._new_request()
        self.session.scene.remove_object(name)
        return self._envelope(
            request_id, t0, compiled=False, removed=name,
            n_objects=len(self.session.scene.names()),
        )

    def attach_object(
        self,
        name: str,
        link: str | None = None,
        ignore_links: Sequence[str] = (),
        ignore_objects: Sequence[str] = (),
        mass: float | None = None,
    ) -> dict[str, Any]:
        """Grasp a scene object: it stops being an obstacle and becomes part of
        the robot's body, moving with ``link``.

        Not free, unlike ``add_object``: obstacle *count* is padded, but a
        carried object genuinely lengthens the robot's collision array, so this
        changes array shapes and forces a recompile of anything that reduces
        over them. That is the deliberate trade -- a handful of recompiles per
        plan (pick, place, handoff) in exchange for attach state that costs
        nothing per query afterwards.

        ``ignore_links`` are links the object is *allowed* to touch -- the
        gripper fingers holding it. Leave it empty and the fingers will report a
        collision with the thing they are gripping.

        ``ignore_objects`` is the same idea for world obstacles, and the surface
        the object was picked up from almost always belongs in it: the bounding
        sphere of a 5 cm cube has a 0.0433 m radius against a 0.025 m half
        height, so the block overlaps the table it is resting on the instant it
        is attached, and the lift-off that fixes it validates as invalid.

        ``mass`` is optional because scene objects are pure geometry: without it
        the attachment is collision-only, which is all a kinematic planner needs.
        Supply it to also load the robot's *dynamics*, so torque limits and
        inverse dynamics account for what is being carried.
        """
        request_id, t0 = self._new_request()
        report = self.session.attach_object(
            name,
            link=link,
            ignore_links=tuple(ignore_links),
            ignore_objects=tuple(ignore_objects),
            mass=mass,
        )
        return self._envelope(
            request_id, t0, compiled=True,
            n_objects=len(self.session.scene.names()),
            attachments=self.session.attached(),
            **report,
        )

    def detach_object(self, name: str) -> dict[str, Any]:
        """Release an attached object back into the world where the robot is
        currently holding it. The exact inverse of ``attach_object``; also
        changes array shapes, so it also recompiles."""
        request_id, t0 = self._new_request()
        report = self.session.detach_object(name)
        return self._envelope(
            request_id, t0, compiled=True,
            n_objects=len(self.session.scene.names()),
            attachments=self.session.attached(),
            **report,
        )

    def list_attachments(self) -> dict[str, Any]:
        """List objects currently carried by the robot. FREE."""
        request_id, t0 = self._new_request()
        return self._envelope(
            request_id, t0, compiled=False,
            attachments=self.session.attached(),
            n_attached=len(self.session.attachments),
        )

    def list_objects(self) -> dict[str, Any]:
        """Cheap. Call it whenever unsure — stale scene state is the most likely
        practical failure in a long agent session."""
        request_id, t0 = self._new_request()
        return self._envelope(
            request_id, t0, compiled=False, **self.session.scene.to_dict()
        )

    def export_scene(self, format: str = "primitives") -> dict[str, Any]:
        """Hand the same world to another server.

        A path validated against pyroffi's scene means nothing if the foreign
        planner saw different obstacles, so the payload carries
        ``scene_version``; compare it against the one on any validation result.
        """
        request_id, t0 = self._new_request()
        if format == "primitives":
            payload: Any = ex.export_scene_primitives(
                self.session.scene, self.session.robot_spec
            )
        elif format == "urdf":
            payload = {
                "format": "urdf",
                "scene_version": self.session.scene.version,
                "urdf": ex.export_scene_urdf(
                    self.session.scene, self.session.robot_spec
                ),
                "note": "half-spaces are approximated as 40 m thin slabs (URDF has "
                        "no half-space primitive)",
            }
        else:
            raise ValueError(f"unknown format {format!r}; expected 'primitives' or 'urdf'")
        return self._envelope(request_id, t0, compiled=False, **payload)

    def set_robot_state(
        self, config: Mapping[str, float] | Sequence[float] | str
    ) -> dict[str, Any]:
        """Set the session's current configuration (used to warm-start IK)."""
        request_id, t0 = self._new_request()
        s = self.session
        if isinstance(config, str):
            if config == "default":
                cfg = np.asarray(s.robot.default_cfg, dtype=np.float64)
            else:
                cfg = s.handles.get(config, "config").values
        else:
            cfg = ex.config_from_payload(config, s.joint_names, defaults=s.robot_state)
        violations = s.limit_violations(cfg)
        s.robot_state = cfg
        return self._envelope(
            request_id,
            t0,
            compiled=False,
            success=not violations,
            joint_limit_violations=violations,
        )

    # ── collision core ────────────────────────────────────────────────────

    def _collision_fn(self, n_configs: int):
        """Jitted ``(K, dof) -> (self distances, per-pool world distances)``.

        Batched with ``vmap`` rather than by adding leading dimensions:
        ``RobotCollision.at_config`` transforms a fixed ``(n_links,)`` geometry
        pytree in place, so a leading batch dim trips its shape assertion.

        World geometry is a *runtime argument*, not a closure: the scene's pools
        have fixed shapes, so moving or adding an obstacle changes array values
        only and never retraces.
        """
        s = self.session
        key = f"coll:{n_configs}:{s.scene.max_objects}"

        def build():
            import jax
            from jax import numpy as jnp

            robot, robot_coll = s.robot, s.robot_coll
            # Per-(row, slot), not per-row: an attachment may be permitted to
            # touch the surface it was picked up from, which a row-wide mask
            # cannot express.
            pair_masks = s.world_pair_masks()
            far = jnp.asarray(1.0e4)

            def single(cfg, geoms):
                self_d = robot_coll.compute_self_collision_distance(robot, cfg)
                # Statically-posed links are pushed out of range rather than
                # dropped, so the array keeps the (n_links, M) shape that the
                # host-side slot-name tables index into.
                world_d = tuple(
                    jnp.where(
                        mask,
                        robot_coll.compute_world_collision_distance(robot, cfg, g),
                        far,
                    )
                    for mask, g in zip(pair_masks, geoms)
                )
                return self_d, world_d

            return jax.jit(jax.vmap(single, in_axes=(0, None)))

        return s.jitted(key, build)

    def _distances(self, cfg: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], bool]:
        """Self and world signed distances for one config or a batch of them.

        A single ``(dof,)`` config returns unbatched arrays — ``(P,)`` and
        ``(N, M)`` per pool — so callers can index the two cases uniformly.
        """
        s = self.session
        arr = np.asarray(cfg, dtype=np.float64)
        single = arr.ndim == 1
        flat = arr.reshape(-1, s.dof)
        fn, compiled = self._collision_fn(flat.shape[0])
        self_d, world_d = fn(s.as_array(flat), s.world_geoms())

        self_np = np.asarray(self_d, dtype=np.float64)
        world_np = [np.asarray(d, dtype=np.float64) for d in world_d]
        if single:
            self_np = self_np[0]
            world_np = [d[0] for d in world_np]
        return self_np, world_np, compiled

    def _named_world_pairs(
        self, world_d: list[np.ndarray], margin: float, index: tuple = ()
    ) -> list[dict[str, Any]]:
        """Below-margin (link, object) pairs, named.

        Parked pool slots sit 10 km away, so anything under the margin is
        necessarily a real object — no mask arithmetic needed here.
        """
        s = self.session
        pairs: list[dict[str, Any]] = []
        pool_names = s.scene.geom_names()
        # Rows, not links: an attached object adds a row, and it is exactly the
        # row a pick-and-place agent most needs to see named.
        row_names = s.collision_row_names()
        for pool_idx, dists in enumerate(world_d):
            _shape, slot_names = pool_names[pool_idx]
            mat = dists[index] if index else dists          # (N_links, M_slots)
            hits = np.argwhere(mat < margin)
            for link_i, slot in hits:
                name = slot_names[int(slot)]
                if name is None:  # defensive: a parked slot can't be this close
                    continue
                pairs.append(
                    {
                        "link": row_names[int(link_i)],
                        "object": name,
                        "distance_m": round(float(mat[link_i, slot]), 6),
                    }
                )
        return sorted(pairs, key=lambda p: p["distance_m"])

    def _named_self_pairs(
        self, self_d: np.ndarray, margin: float, index: tuple = ()
    ) -> list[dict[str, Any]]:
        s = self.session
        vec = self_d[index] if index else self_d
        names = s.self_pair_names()
        out = []
        for k in np.argwhere(vec < margin).reshape(-1):
            link_a, link_b = names[int(k)]
            out.append(
                {
                    "link_a": link_a,
                    "link_b": link_b,
                    "distance_m": round(float(vec[int(k)]), 6),
                }
            )
        return sorted(out, key=lambda p: p["distance_m"])

    @staticmethod
    def _min_clearance(self_d: np.ndarray, world_d: list[np.ndarray]) -> float:
        vals = [self_d.min()] + [d.min() for d in world_d]
        return float(min(vals))

    # ── kinematics ────────────────────────────────────────────────────────

    def forward_kinematics(
        self,
        config: str | Mapping[str, float] | Sequence[float],
        links: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Where a configuration actually puts the links. ~1 ms warm."""
        request_id, t0 = self._new_request()
        s = self.session
        cfg = self._resolve_config(config)
        link_names = tuple(links) if links else (s.ee_link,)
        indices = [s.link_index(n) for n in link_names]

        # forward_kinematics is already jdc.jit'd; the ledger only records whether
        # this shape signature has been paid for before.
        key = f"fk:{cfg.shape}"
        compiled = key not in s.ledger
        s.ledger.visit(key)
        poses = np.asarray(
            s.robot.forward_kinematics(s.as_array(cfg)), dtype=np.float64
        )
        return self._envelope(
            request_id,
            t0,
            compiled,
            poses={
                name: ex.pose_payload(poses[i, :4], poses[i, 4:7])
                for name, i in zip(link_names, indices)
            },
        )

    def _pose_error(
        self, cfg: np.ndarray, link: str, target
    ) -> tuple[float, float]:
        """(position error in m, rotation error in rad) of *cfg* against *target*."""
        import jaxlie
        from jax import numpy as jnp

        s = self.session
        poses = s.robot.forward_kinematics(s.as_array(cfg))
        actual = jaxlie.SE3(poses[..., s.link_index(link), :])
        pos_err = float(
            jnp.linalg.norm(actual.translation() - target.translation())
        )
        rot_err = float(
            jnp.linalg.norm((actual.rotation().inverse() @ target.rotation()).log())
        )
        return pos_err, rot_err

    def _collision_constraint(self):
        """A stable softplus collision penalty for constrained IK.

        Built once per session and cached: the constraint callable must be the
        same Python object across calls or every IK solve retraces. The scene's
        padded pools are passed as ``constraint_args``, so moving an obstacle
        only updates array values.
        """
        s = self.session
        cached = getattr(self, "_coll_constraint_fn", None)
        if cached is not None:
            return cached

        import jax
        from jax import numpy as jnp

        eps = 0.005  # softplus smoothing radius [m]; C-inf, cleaner LM gradients
        robot_coll = s.robot_coll

        def penalty(cfg, robot, *pools):
            total = 0.0
            for pool in pools:
                d = robot_coll.compute_world_collision_distance(robot, cfg, pool)
                total = total + jnp.sum(jax.nn.softplus(-d / eps) * eps)
            return total

        self._coll_constraint_fn = penalty
        return penalty

    def solve_ik(
        self,
        link: str | None = None,
        pose: Mapping[str, Any] | None = None,
        num_seeds: int = 32,
        solver: str = "hjcd",
        seed_config: str | Mapping[str, float] | Sequence[float] | None = None,
        num_restarts: int = 1,
        collision_free: bool = False,
        collision_weight: float = 1e8,
        fixed_joints: Sequence[str] = (),
        pos_tolerance: float = _IK_POS_TOL,
        rot_tolerance: float = _IK_ROT_TOL,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Inverse kinematics for one target pose. ~ms warm.

        Batched by construction over ``num_seeds``, so the interesting output is
        a distribution, not just a winner. ``num_restarts > 1`` repeats the whole
        multi-seed solve with fresh randomness and reports how many restarts
        actually converged — genuine information about how hard this target is,
        which a single "success: true" would throw away.

        ``solver`` is the caller's choice and is never overridden.
        """
        request_id, t0 = self._new_request()
        s = self.session
        if pose is None:
            raise ValueError("solve_ik requires a target 'pose'")
        link = link or s.ee_link
        s.link_index(link)  # validate early
        target = ex.se3_from_payload(pose)

        if solver not in ("hjcd", "ls"):
            raise ValueError(f"unknown solver {solver!r}; expected 'hjcd' or 'ls'")

        seed_cfg = (
            self._resolve_config(seed_config)
            if seed_config is not None
            else s.robot_state
        )
        mask = self._fixed_joint_mask(fixed_joints)

        kwargs: dict[str, Any] = {}
        if collision_free:
            kwargs.update(
                constraints=(self._collision_constraint(),),
                constraint_args=tuple(s.world_geoms()),
                constraint_weights=s.as_array([collision_weight]),
            )

        key = f"ik:{solver}:{num_seeds}:{link}:{collision_free}:{bool(len(fixed_joints))}"
        compiled = key not in s.ledger
        s.ledger.visit(key)

        import jax

        results = []
        for r in range(max(1, int(num_restarts))):
            rng = jax.random.PRNGKey(int(seed) + r)
            cfg = np.asarray(
                s.robot.inverse_kinematics(
                    link,
                    target,
                    rng_key=rng,
                    previous_cfg=s.as_array(seed_cfg),
                    solver=solver,
                    num_seeds=num_seeds,
                    fixed_joint_mask=mask,
                    **kwargs,
                ),
                dtype=np.float64,
            )
            pos_err, rot_err = self._pose_error(cfg, link, target)
            results.append((cfg, pos_err, rot_err))

        pos_errs = np.array([r[1] for r in results])
        rot_errs = np.array([r[2] for r in results])
        converged = (pos_errs <= pos_tolerance) & (rot_errs <= rot_tolerance)
        best = int(np.argmin(pos_errs + rot_errs))
        best_cfg, best_pos, best_rot = results[best]

        violations = s.limit_violations(best_cfg)
        clearance = None
        in_collision = None
        if collision_free:
            self_d, world_d, _ = self._distances(best_cfg)
            clearance = self._min_clearance(self_d, world_d)
            in_collision = clearance < _DEFAULT_MARGIN

        entry = s.handles.insert(
            "config",
            best_cfg,
            s.joint_names,
            s.scene.version,
            meta={
                "source": "solve_ik",
                "request_id": request_id,
                "link": link,
                "solver": solver,
                "pos_error_m": best_pos,
                "rot_error_rad": best_rot,
            },
        )

        success = bool(converged[best]) and not violations and not (in_collision or False)
        if not success:
            self._record_failure(
                request_id,
                "ik_did_not_converge" if not converged[best]
                else ("ik_in_collision" if in_collision else "ik_joint_limits"),
                {
                    "link": link,
                    "solver": solver,
                    "pos_error_m": best_pos,
                    "rot_error_rad": best_rot,
                    "pos_tolerance": pos_tolerance,
                    "rot_tolerance": rot_tolerance,
                    "joint_limit_violations": violations,
                    "min_clearance_m": clearance,
                    "hint": "target may be outside the reachable workspace; try "
                            "check_reachable, more num_seeds, or more num_restarts",
                },
            )

        out: dict[str, Any] = {
            "config_id": entry.handle,
            "solver": solver,
            "num_seeds": num_seeds,
            "num_restarts": int(max(1, num_restarts)),
            "restarts_converged": int(converged.sum()),
            "pos_error_m": round(best_pos, 8),
            "rot_error_rad": round(best_rot, 8),
            "pos_error_median_m": round(float(np.median(pos_errs)), 8),
            "pos_error_worst_m": round(float(pos_errs.max()), 8),
            "joint_limit_violations": violations,
        }
        if collision_free:
            out["min_clearance_m"] = round(float(clearance), 6)
            out["in_collision"] = bool(in_collision)
        return self._envelope(request_id, t0, compiled, success=success, **out)

    def solve_ik_batch(
        self,
        targets: Sequence[Mapping[str, Any]],
        link: str | None = None,
        num_seeds: int = 32,
        max_iter: int = 60,
        seed_config: str | Mapping[str, float] | Sequence[float] | None = None,
        collision_free: bool = False,
        collision_weight: float = 1e8,
        pos_tolerance: float = _IK_POS_TOL,
        rot_tolerance: float = _IK_ROT_TOL,
        seed: int = 0,
    ) -> dict[str, Any]:
        """N target poses in **one** GPU dispatch. ~ms warm for tens of targets.

        Prefer this over a loop of ``solve_ik`` whenever more than one candidate
        is in play — enumerating grasps or placements is exactly the case where
        the batched kernel is nearly free per extra target. Uses the pure-JAX
        multi-seed LM solver (``ls``), vmapped over targets.

        Target count is a static shape, so it is bucketed like path length: 5
        targets and 7 targets share the 8-wide program.

        ``collision_free=True`` adds the same softplus obstacle penalty
        ``solve_ik`` uses, and every result then carries ``min_clearance_m`` and
        ``in_collision`` — enumerating grasps is precisely where a candidate
        that reaches the pose *through* an obstacle needs filtering out.
        """
        request_id, t0 = self._new_request()
        s = self.session
        if not targets:
            raise ValueError("solve_ik_batch requires at least one target")
        link = link or s.ee_link
        link_idx = s.link_index(link)

        poses = [ex.se3_from_payload(t) for t in targets]
        n = len(poses)
        # Coarse buckets on purpose: each distinct width is a separate ~3 s
        # compile, and padding a target costs almost nothing on GPU. A single
        # target keeps its own width because it is the common case.
        n_padded = bucket_length(n, (1, 8, 32, 128))
        stacked = np.stack(
            [np.asarray(p.wxyz_xyz, dtype=np.float64) for p in poses]
            + [np.asarray(poses[-1].wxyz_xyz, dtype=np.float64)] * (n_padded - n),
            axis=0,
        )

        seed_cfg = (
            self._resolve_config(seed_config)
            if seed_config is not None
            else s.robot_state
        )

        import jax
        import jaxlie
        from pyroffi.optimization_engines._ls_ik import ls_ik_solve

        pools = tuple(s.world_geoms()) if collision_free else ()
        key = (
            f"ik_batch:{n_padded}:{num_seeds}:{max_iter}:{link}:{collision_free}"
        )

        def build():
            constraint_fns = (self._collision_constraint(),) if collision_free else ()
            weights = (
                s.as_array([collision_weight]) if collision_free else None
            )

            def one(pose_vec, prev_cfg, rng, *pool_args):
                return ls_ik_solve(
                    s.robot,
                    (link_idx,),
                    (jaxlie.SE3(pose_vec),),
                    rng,
                    prev_cfg,
                    num_seeds=num_seeds,
                    max_iter=max_iter,
                    constraint_fns=constraint_fns,
                    constraint_args=pool_args,
                    constraint_weights=weights,
                )

            # The pools are arguments with ``in_axes=None``, not a closure: the
            # scene is shared by every target, and closing over it would bake
            # obstacle positions into the compiled program, so moving a block
            # would silently plan against where it used to be.
            return jax.jit(
                jax.vmap(one, in_axes=(0, 0, 0) + (None,) * len(pools))
            )

        fn, compiled = s.jitted(key, build)

        keys = jax.random.split(jax.random.PRNGKey(int(seed)), n_padded)
        prev = np.tile(np.asarray(seed_cfg, dtype=np.float64), (n_padded, 1))
        cfgs = np.asarray(
            fn(s.as_array(stacked), s.as_array(prev), keys, *pools), dtype=np.float64
        )[:n]

        clearances: list[float] | None = None
        if collision_free:
            self_d, world_d, _ = self._distances(cfgs)
            clearances = [
                self._min_clearance(self_d[i], [d[i] for d in world_d])
                for i in range(n)
            ]

        results = []
        n_converged = 0
        for i in range(n):
            pos_err, rot_err = self._pose_error(cfgs[i], link, poses[i])
            ok = pos_err <= pos_tolerance and rot_err <= rot_tolerance
            n_converged += int(ok)
            entry = s.handles.insert(
                "config",
                cfgs[i],
                s.joint_names,
                s.scene.version,
                meta={
                    "source": "solve_ik_batch",
                    "request_id": request_id,
                    "target_index": i,
                    "link": link,
                    "pos_error_m": pos_err,
                },
            )
            result = {
                "target_index": i,
                "config_id": entry.handle,
                "converged": bool(ok),
                "pos_error_m": round(pos_err, 8),
                "rot_error_rad": round(rot_err, 8),
            }
            if clearances is not None:
                result["min_clearance_m"] = round(clearances[i], 6)
                result["in_collision"] = bool(clearances[i] < 0.0)
            results.append(result)

        if n_converged == 0:
            self._record_failure(
                request_id,
                "ik_batch_all_failed",
                {
                    "n_targets": n,
                    "link": link,
                    "hint": "every target missed tolerance — check the frame and "
                            "quaternion convention of the incoming poses",
                },
            )

        return self._envelope(
            request_id,
            t0,
            compiled,
            success=n_converged > 0,
            solver="ls",
            link=link,
            collision_free=collision_free,
            n_targets=n,
            n_padded=n_padded,
            n_converged=n_converged,
            results=results,
        )

    def check_reachable(
        self,
        link: str | None = None,
        pose: Mapping[str, Any] | None = None,
        num_seeds: int = 32,
        pos_tolerance: float = _IK_POS_TOL,
        rot_tolerance: float = _IK_ROT_TOL,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Can the arm get there at all? Thin wrapper over batched IK.

        Answers the pruning question without leaving a config handle behind for
        an agent that was only filtering candidates.
        """
        request_id, t0 = self._new_request()
        res = self.solve_ik(
            link=link,
            pose=pose,
            num_seeds=num_seeds,
            seed=seed,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance,
        )
        self.session.handles.drop(res["config_id"])
        self._inherit_failure(request_id, res["request_id"])
        return self._envelope(
            request_id,
            t0,
            res["compiled"],
            success=True,
            reachable=bool(res["success"]),
            pos_error_m=res["pos_error_m"],
            rot_error_rad=res["rot_error_rad"],
        )

    # ── validation ────────────────────────────────────────────────────────

    def check_collision(
        self,
        config: str | Mapping[str, float] | Sequence[float],
        margin: float = _DEFAULT_MARGIN,
    ) -> dict[str, Any]:
        """Collision state of one configuration, with **named** pairs. ~ms."""
        request_id, t0 = self._new_request()
        s = self.session
        cfg = self._resolve_config(config)
        self_d, world_d, compiled = self._distances(cfg)

        world_pairs = self._named_world_pairs(world_d, margin)
        self_pairs = self._named_self_pairs(self_d, margin)
        clearance = self._min_clearance(self_d, world_d)
        violations = s.limit_violations(cfg)
        free = not world_pairs and not self_pairs and not violations

        if not free:
            self._record_failure(
                request_id,
                "configuration_in_collision" if (world_pairs or self_pairs)
                else "joint_limits",
                {
                    "world_collisions": world_pairs,
                    "self_collisions": self_pairs,
                    "joint_limit_violations": violations,
                    "min_clearance_m": clearance,
                },
            )

        return self._envelope(
            request_id,
            t0,
            compiled,
            success=True,
            collision_free=free,
            min_clearance_m=round(clearance, 6),
            margin_m=margin,
            world_collisions=world_pairs,
            self_collisions=self_pairs,
            joint_limit_violations=violations,
        )

    def check_edge(
        self,
        config_a: str | Mapping[str, float] | Sequence[float],
        config_b: str | Mapping[str, float] | Sequence[float],
        resolution: int = 32,
        margin: float = _DEFAULT_MARGIN,
    ) -> dict[str, Any]:
        """Is the straight-line joint-space motion valid, and where does it first
        fail? Cheap — the natural pre-filter before anything expensive."""
        request_id, t0 = self._new_request()
        s = self.session
        qa = self._resolve_config(config_a)
        qb = self._resolve_config(config_b)

        n = bucket_length(max(2, int(resolution)), (8, 16, 32, 64, 128, 256))
        alpha = np.linspace(0.0, 1.0, n)[:, None]
        samples = qa[None, :] * (1.0 - alpha) + qb[None, :] * alpha

        self_d, world_d, compiled = self._distances(samples)
        per_sample = np.minimum(
            self_d.min(axis=-1),
            np.min([d.reshape(n, -1).min(axis=-1) for d in world_d], axis=0),
        )
        bad = np.argwhere(per_sample < margin).reshape(-1)
        valid = bad.size == 0

        first_fail = None
        if not valid:
            k = int(bad[0])
            first_fail = {
                "fraction": round(float(alpha[k, 0]), 4),
                "sample_index": k,
                "clearance_m": round(float(per_sample[k]), 6),
                "world_collisions": self._named_world_pairs(world_d, margin, (k,)),
                "self_collisions": self._named_self_pairs(self_d, margin, (k,)),
            }
            self._record_failure(
                request_id, "edge_in_collision", {"first_failure": first_fail}
            )

        return self._envelope(
            request_id,
            t0,
            compiled,
            success=True,
            valid=valid,
            n_samples=n,
            min_clearance_m=round(float(per_sample.min()), 6),
            joint_distance_rad=round(float(np.linalg.norm(qb - qa)), 6),
            first_failure=first_fail,
        )

    def validate_path(
        self,
        path: str | Sequence[Any],
        edge_substeps: int = 4,
        margin: float = _DEFAULT_MARGIN,
    ) -> dict[str, Any]:
        """Validate a path — the primary consumer of ``import_path``.

        This is what an orchestrator calls on whatever a foreign RRT produced.
        Checks every waypoint, subdivides every edge, and reports joint-limit
        violations, min clearance, and the first failing edge by name.

        The response carries ``scene_version``: if the scene has moved since the
        external planner saw it, a stale validation is detectable rather than
        merely wrong.
        """
        request_id, t0 = self._new_request()
        s = self.session
        wp, source_handle = self._resolve_path(path)
        n_true = wp.shape[0]
        if n_true < 1:
            raise ValueError("path has no waypoints")

        n_bucket = bucket_length(n_true, s.path_buckets)
        padded = pad_path(wp, n_bucket)

        # Waypoints.
        self_d, world_d, compiled_wp = self._distances(padded)
        wp_clear = np.minimum(
            self_d.min(axis=-1),
            np.min([d.reshape(n_bucket, -1).min(axis=-1) for d in world_d], axis=0),
        )[:n_true]

        invalid_waypoints = []
        for k in np.argwhere(wp_clear < margin).reshape(-1):
            invalid_waypoints.append(
                {
                    "index": int(k),
                    "clearance_m": round(float(wp_clear[int(k)]), 6),
                    "world_collisions": self._named_world_pairs(
                        world_d, margin, (int(k),)
                    ),
                    "self_collisions": self._named_self_pairs(self_d, margin, (int(k),)),
                }
            )

        # Edges: subdivide interior of each edge (endpoints already covered).
        n_sub = max(1, int(edge_substeps))
        n_edges = n_bucket - 1
        edge_clear = np.full(max(n_true - 1, 0), np.inf)
        compiled_edge = False
        if n_edges > 0 and n_sub > 0:
            frac = (np.arange(1, n_sub + 1) / (n_sub + 1.0))[None, :, None]
            a = padded[:-1][:, None, :]
            b = padded[1:][:, None, :]
            samples = (a * (1.0 - frac) + b * frac).reshape(n_edges * n_sub, -1)
            e_self, e_world, compiled_edge = self._distances(samples)
            per = np.minimum(
                e_self.min(axis=-1),
                np.min(
                    [d.reshape(n_edges * n_sub, -1).min(axis=-1) for d in e_world],
                    axis=0,
                ),
            ).reshape(n_edges, n_sub)
            edge_clear = per.min(axis=1)[: max(n_true - 1, 0)]

        invalid_edges = [
            {"edge": int(k), "from_index": int(k), "to_index": int(k) + 1,
             "clearance_m": round(float(edge_clear[int(k)]), 6)}
            for k in np.argwhere(edge_clear < margin).reshape(-1)
        ]

        # Joint limits, per waypoint.
        limit_hits = []
        for k in range(n_true):
            v = s.limit_violations(wp[k])
            if v:
                limit_hits.append({"index": k, "violations": v})

        clearances = [float(wp_clear.min())]
        if edge_clear.size:
            clearances.append(float(edge_clear.min()))
        min_clearance = min(clearances)

        valid = not invalid_waypoints and not invalid_edges and not limit_hits
        if not valid:
            self._record_failure(
                request_id,
                "path_invalid",
                {
                    "first_invalid_waypoint": invalid_waypoints[0]
                    if invalid_waypoints else None,
                    "first_invalid_edge": invalid_edges[0] if invalid_edges else None,
                    "joint_limit_violations": limit_hits[:4],
                    "hint": "optimize_path can often repair a locally-invalid seed; "
                            "a globally trapped path needs a planner, not a smoother",
                },
            )

        stale = None
        if source_handle is not None:
            entry = s.handles.get(source_handle)
            if entry.scene_version != s.scene.version:
                stale = {
                    "path_scene_version": entry.scene_version,
                    "current_scene_version": s.scene.version,
                    "note": "the scene changed after this path was created",
                }

        return self._envelope(
            request_id,
            t0,
            compiled_wp or compiled_edge,
            success=True,
            valid=valid,
            n_waypoints=n_true,
            n_padded=n_bucket,
            edge_substeps=n_sub,
            min_clearance_m=round(min_clearance, 6),
            path_length_rad=round(self._path_length(wp), 6),
            n_invalid_waypoints=len(invalid_waypoints),
            n_invalid_edges=len(invalid_edges),
            invalid_waypoints=invalid_waypoints[:8],
            invalid_edges=invalid_edges[:8],
            joint_limit_violations=limit_hits[:8],
            stale_scene=stale,
        )

    # ── exchange ──────────────────────────────────────────────────────────

    def import_path(
        self,
        waypoints: Sequence[Any],
        source: str = "external",
    ) -> dict[str, Any]:
        """Take a real joint array from a foreign planner and hand back a handle.

        Waypoints may be name-keyed objects (preferred) or full-length positional
        arrays in ``joint_names`` order. Passive and mimic joints are not part of
        that ordering and must not be sent.
        """
        request_id, t0 = self._new_request()
        s = self.session
        arr = ex.path_from_payload(waypoints, s.joint_names)
        entry = s.handles.insert(
            "path", arr, s.joint_names, s.scene.version,
            meta={"source": source, "request_id": request_id},
        )
        return self._envelope(
            request_id,
            t0,
            compiled=False,
            path_id=entry.handle,
            n_waypoints=entry.n_waypoints,
            path_length_rad=round(self._path_length(arr), 6),
            joint_names=list(s.joint_names),
        )

    def export_path(self, path_id: str, include_times: bool = True) -> dict[str, Any]:
        """Machine-facing: the actual numbers, for handoff to another server.

        Explicit because it is expensive in an agent's context — a 64x7 path is
        thousands of tokens. Waypoints are name-keyed so the receiver cannot
        mis-order them.
        """
        request_id, t0 = self._new_request()
        s = self.session
        entry = s.handles.get(path_id)
        values = entry.values if entry.values.ndim == 2 else entry.values[None, :]
        payload: dict[str, Any] = {
            "path_id": path_id,
            "joint_names": list(entry.joint_names),
            "waypoints": [ex.joint_dict(row, entry.joint_names) for row in values],
            "n_waypoints": int(values.shape[0]),
            "units": ex.UNITS,
            "quaternion_convention": ex.QUATERNION_CONVENTION,
            "path_scene_version": entry.scene_version,
        }
        if include_times and entry.times is not None:
            payload["times_s"] = [float(t) for t in entry.times]
            payload["duration_s"] = float(entry.times[-1])
        return self._envelope(request_id, t0, compiled=False, **payload)

    def export_config(self, config_id: str) -> dict[str, Any]:
        """Machine-facing: one configuration as a name-keyed object."""
        request_id, t0 = self._new_request()
        entry = self.session.handles.get(config_id, "config")
        return self._envelope(
            request_id,
            t0,
            compiled=False,
            config_id=config_id,
            joint_values=ex.joint_dict(entry.values, entry.joint_names),
            units={"angle": "rad"},
        )

    # ── optimization ──────────────────────────────────────────────────────

    def optimize_path(
        self,
        path: str | Sequence[Any],
        n_batch: int = 25,
        noise_scale: float = 0.05,
        n_outer_iters: int = 20,
        n_inner_iters: int = 50,
        w_smooth: float = 10.0,
        w_collision: float = 5.0,
        collision_margin: float = 0.02,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Refine a caller-supplied path with SCO trajopt.

        **Seeded from the path you pass in** — this is the primary trajopt entry
        point, because in the intended pipeline the seed comes from an external
        planner.

        Cost is dominated by ``n_outer_iters * n_inner_iters``. Measured on a
        32-waypoint Panda path with ``n_batch=16`` (A5000, float64): the defaults
        take ~3.6 s warm and ~8 s on the first call for a given shape (watch
        ``compiled``). The iteration defaults are deliberately not the maximum —
        on the obstacle cases tested, 50x100 (~11.7 s) reached exactly the same
        final clearance as 5x15 (~0.7 s), so the extra iterations bought nothing.
        Harder problems may need more, which is why the knobs are exposed.

        The seed is resampled by arc length to the nearest compiled bucket
        length, so a 47-waypoint input comes back as 64 waypoints rather than
        triggering its own compile.
        """
        request_id, t0 = self._new_request()
        s = self.session
        wp, _ = self._resolve_path(path)
        if wp.shape[0] < 2:
            raise ValueError("optimize_path needs at least 2 waypoints")

        n_t = bucket_length(wp.shape[0], s.path_buckets)
        seed_path = self._resample(wp, n_t)

        before_self, before_world, _ = self._distances(seed_path)
        clearance_before = self._min_clearance(before_self, before_world)

        import jax
        from jax import numpy as jnp
        from pyroffi.optimization_engines import ScoTrajOptConfig
        from pyroffi.optimization_engines._sco_optimization import sco_trajopt

        opt_cfg = ScoTrajOptConfig(
            n_outer_iters=n_outer_iters,
            n_inner_iters=n_inner_iters,
            w_smooth=w_smooth,
            w_collision=w_collision,
            collision_margin=collision_margin,
        )
        key = (
            f"trajopt:{n_t}:{n_batch}:{n_outer_iters}:{n_inner_iters}:"
            f"{w_smooth}:{w_collision}:{collision_margin}:{s.scene.max_objects}"
        )

        def build():
            def fn(init_trajs, start, goal, geoms):
                return sco_trajopt(
                    init_trajs, start, goal, s.robot, s.robot_coll, geoms, opt_cfg
                )

            return jax.jit(fn)

        fn, compiled = s.jitted(key, build)

        base = jnp.asarray(s.as_array(seed_path))
        noise = jax.random.normal(
            jax.random.PRNGKey(int(seed)), (n_batch, n_t, s.dof)
        ) * noise_scale
        # Endpoints stay put: trajopt pins them, and noising them only wastes
        # the first outer iteration pulling them back.
        noise = noise.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        init_trajs = base[None] + noise.astype(base.dtype)

        best, costs, _ = fn(
            init_trajs, base[0], base[-1], s.world_geoms()
        )
        best = np.asarray(best, dtype=np.float64)
        costs = np.asarray(costs, dtype=np.float64)

        after_self, after_world, _ = self._distances(best)
        clearance_after = self._min_clearance(after_self, after_world)
        finite = np.isfinite(costs)

        entry = s.handles.insert(
            "path", best, s.joint_names, s.scene.version,
            meta={
                "source": "optimize_path",
                "request_id": request_id,
                "cost": float(costs.min()),
            },
        )

        # Honest success: the optimizer ran and produced a *valid* path, not
        # merely a lower-cost one. Cost acceptance is the caller's policy.
        valid = clearance_after >= 0.0 and not any(
            s.limit_violations(row) for row in best
        )
        if not valid:
            self._record_failure(
                request_id,
                "optimize_path_still_invalid",
                {
                    "min_clearance_m": clearance_after,
                    "cost": float(costs.min()),
                    "hint": "trajopt is a local optimizer; a seed on the wrong side "
                            "of an obstacle stays there. Get a better seed from a "
                            "sampling planner.",
                },
            )

        return self._envelope(
            request_id,
            t0,
            compiled,
            success=bool(valid),
            path_id=entry.handle,
            n_waypoints=n_t,
            n_waypoints_in=int(wp.shape[0]),
            n_batch=n_batch,
            cost_after=round(float(costs.min()), 6),
            cost_median=round(float(np.median(costs[finite])), 6) if finite.any() else None,
            n_diverged=int((~finite).sum()),
            path_length_rad_before=round(self._path_length(seed_path), 6),
            path_length_rad=round(self._path_length(best), 6),
            min_clearance_m_before=round(clearance_before, 6),
            min_clearance_m=round(clearance_after, 6),
        )

    def optimize_between(
        self,
        config_a: str | Mapping[str, float] | Sequence[float] | None = None,
        config_b: str | Mapping[str, float] | Sequence[float] | None = None,
        pose_a: Mapping[str, Any] | None = None,
        pose_b: Mapping[str, Any] | None = None,
        link: str | None = None,
        n_timesteps: int | None = None,
        **optimize_kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience: seed trajopt itself by joint-space interpolation, then optimize.

        **A local optimizer, not a planner.** It will happily fail in a maze, and
        it fails by returning a path that still intersects an obstacle rather
        than by reporting "no path exists". If the endpoints are separated by
        anything topologically interesting, get a seed from a sampling planner
        (a foreign RRT server) and use ``optimize_path`` instead.

        Endpoints may be configs or SE(3) poses (IK'd here).
        """
        request_id, t0 = self._new_request()
        s = self.session
        link = link or s.ee_link

        ik_ids = []
        if config_a is None:
            if pose_a is None:
                raise ValueError("optimize_between needs config_a or pose_a")
            r = self.solve_ik(link=link, pose=pose_a, collision_free=True)
            ik_ids.append(r)
            config_a = r["config_id"]
        if config_b is None:
            if pose_b is None:
                raise ValueError("optimize_between needs config_b or pose_b")
            r = self.solve_ik(link=link, pose=pose_b, collision_free=True, seed=1)
            ik_ids.append(r)
            config_b = r["config_id"]

        qa = self._resolve_config(config_a)
        qb = self._resolve_config(config_b)
        n_t = bucket_length(int(n_timesteps or s.n_timesteps), s.path_buckets)
        alpha = np.linspace(0.0, 1.0, n_t)[:, None]
        seed_path = qa[None, :] * (1.0 - alpha) + qb[None, :] * alpha

        res = self.optimize_path(
            [ex.joint_dict(row, s.joint_names) for row in seed_path],
            **optimize_kwargs,
        )
        # The envelope's request_id is rewritten to this call's, so the inner
        # ids have to be forwarded first or the failure record is orphaned
        # behind an id the caller never receives. Trajopt's own failure wins
        # over an endpoint IK failure: it is the later, more specific one.
        self._inherit_failure(
            request_id, res["request_id"], *[r["request_id"] for r in ik_ids]
        )
        res["request_id"] = request_id
        res["solve_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        res["seeded_by"] = "joint-space linear interpolation"
        res["endpoint_ik"] = [
            {"config_id": r["config_id"], "converged": r["success"]} for r in ik_ids
        ]
        res["caveat"] = (
            "local optimizer seeded by a straight line; a failure here does not "
            "mean no path exists"
        )
        return res

    def concat_paths(self, path_ids: Sequence[str], tolerance: float = 1e-4) -> dict[str, Any]:
        """Join path segments, checking that segment k ends where k+1 begins.

        Without this an agent hand-tracks continuity across segments and
        eventually gets it wrong; the discontinuity then surfaces as a mystery
        collision or a torque spike much later.
        """
        request_id, t0 = self._new_request()
        s = self.session
        if len(path_ids) < 1:
            raise ValueError("concat_paths needs at least one path")

        entries = [s.handles.get(pid) for pid in path_ids]
        gaps = []
        for i in range(len(entries) - 1):
            end = entries[i].values[-1]
            start = entries[i + 1].values[0]
            gap = float(np.max(np.abs(end - start)))
            if gap > tolerance:
                worst = int(np.argmax(np.abs(end - start)))
                gaps.append(
                    {
                        "between": [path_ids[i], path_ids[i + 1]],
                        "max_gap_rad": round(gap, 6),
                        "worst_joint": s.joint_names[worst],
                    }
                )
        if gaps:
            self._record_failure(
                request_id, "discontinuous_segments", {"gaps": gaps}
            )
            return self._envelope(
                request_id, t0, compiled=False, success=False,
                discontinuities=gaps, tolerance_rad=tolerance,
            )

        chunks = [entries[0].values]
        for e in entries[1:]:
            chunks.append(e.values[1:])  # drop the duplicated junction waypoint
        joined = np.concatenate(chunks, axis=0)
        entry = s.handles.insert(
            "path", joined, s.joint_names, s.scene.version,
            meta={"source": "concat_paths", "request_id": request_id,
                  "segments": list(path_ids)},
        )
        return self._envelope(
            request_id, t0, compiled=False,
            path_id=entry.handle,
            n_waypoints=entry.n_waypoints,
            n_segments=len(entries),
            path_length_rad=round(self._path_length(joined), 6),
        )

    def retime(
        self,
        path: str | Sequence[Any],
        velocity_scale: float = 1.0,
        acceleration_scale: float = 1.0,
        time_to_peak: float | None = None,
    ) -> dict[str, Any]:
        """Turn a geometric path into a timed trajectory. ~ms, CPU.

        Velocity limits come from the URDF. URDFs carry no acceleration limits,
        so they are inferred from the velocity limits unless the caller supplies
        a ``time_to_peak``; the scales let a caller back off from the ceiling.

        Nothing downstream that mentions duration means anything before this is
        called.
        """
        request_id, t0 = self._new_request()
        s = self.session
        wp, source = self._resolve_path(path)

        vmax = s.velocity_limits * float(velocity_scale)
        amax = (
            default_acceleration_limits(s.velocity_limits, time_to_peak)
            if time_to_peak is not None
            else s.acceleration_limits
        ) * float(acceleration_scale)

        result = retime_path(wp, vmax, amax, joint_names=s.joint_names)
        entry = s.handles.insert(
            "trajectory",
            wp,
            s.joint_names,
            s.scene.version,
            meta={
                "source": "retime",
                "request_id": request_id,
                "from": source,
                # simulate's feedforward term wants the profiles retiming already
                # computed; finite-differencing them again would be lossier.
                "velocities": result.velocities,
                "accelerations": result.accelerations,
            },
            times=result.times,
        )
        return self._envelope(
            request_id,
            t0,
            compiled=False,
            success=result.feasible,
            trajectory_id=entry.handle,
            **result.to_dict(),
        )

    # ── dynamics ──────────────────────────────────────────────────────────

    def simulate(
        self,
        trajectory: str,
        kp: float = 100.0,
        kd: float = 10.0,
        substeps: int | None = None,
        feedforward: bool = True,
        use_cuda: bool = False,
    ) -> dict[str, Any]:
        """Roll the timed trajectory forward under computed-torque control. ~100 ms.

        Verifies before committing: reports tracking error, peak torque, and
        divergence. Requires a retimed trajectory — a rollout needs ``dt``, and
        inventing one would make every number here meaningless.

        Two details are load-bearing, and both were arrived at the hard way:

        * **The controller runs at ``dt / substeps``, not at the waypoint rate.**
          Torque passed to ``Robot.step`` is held constant across its substeps,
          so closing the loop per waypoint samples the velocity feedback at the
          waypoint period. With a panda's smallest mass-matrix eigenvalue
          (~0.1 kg m^2), ``kd * dt / lambda_min`` exceeds 2 at any realistic
          waypoint spacing and the rollout diverges — a sampled-data
          instability in the *controller*, not the trajectory or the integrator.
        * **Inverse-dynamics feedforward** (``tau = ID(q_r, qd_r, qdd_r) + PD``)
          leaves the PD term correcting only small residuals, which is what a
          real manipulator controller does and what makes ``peak_torque`` mean
          the torque the motion actually demands.

        A divergence reported here is therefore about the trajectory, which is
        the only thing worth reporting.
        """
        request_id, t0 = self._new_request()
        s = self.session
        entry = s.handles.get(trajectory)
        if entry.times is None:
            raise ValueError(
                f"{trajectory} has no times; call retime(path_id) first — a rollout "
                "needs dt, and a made-up dt makes every reported number meaningless"
            )
        if s.robot.dynamics is None:
            raise RuntimeError(
                "this robot has no usable dynamics (the URDF lacks inertials or uses "
                "mimic joints), so simulate is unavailable"
            )

        import jax
        from jax import numpy as jnp

        q_ref = entry.values
        times = entry.times
        n = q_ref.shape[0]
        dt_wp = float(np.median(np.diff(times))) if n > 1 else 0.01
        if substeps is None:
            n_sub, stability = self._auto_substeps(q_ref[0], dt_wp, kp, kd)
        else:
            n_sub = max(1, int(substeps))
            stability = {"auto": False}

        # Reference resampled onto the control grid, with the velocity and
        # acceleration profiles retiming already computed (finite-differenced if
        # this trajectory came from somewhere else).
        qd_ref_wp = entry.meta.get("velocities")
        qdd_ref_wp = entry.meta.get("accelerations")
        ctrl_t, q_r, qd_r, qdd_r = self._control_grid(
            q_ref, times, n_sub, qd_ref_wp, qdd_ref_wp
        )
        dt_ctrl = float(np.median(np.diff(ctrl_t))) if ctrl_t.size > 1 else 0.01
        dt_waypoint = float(np.median(np.diff(times))) if n > 1 else dt_ctrl

        key = f"sim:{q_r.shape[0]}:{use_cuda}:{kp}:{kd}:{feedforward}"

        def build():
            def rollout(refs, q0):
                def body(carry, ref):
                    q, qd = carry
                    q_t, qd_t, qdd_t = ref
                    tau = kp * (q_t - q) + kd * (qd_t - qd)
                    if feedforward:
                        tau = tau + s.robot.inverse_dynamics(q_t, qd_t, qdd_t)
                    q_next, qd_next = s.robot.step(
                        q, qd, tau, dt=dt_ctrl, use_cuda=use_cuda, substeps=1
                    )
                    return (q_next, qd_next), (q_next, qd_next, tau)

                zeros = jnp.zeros_like(q0)
                _, out = jax.lax.scan(body, (q0, zeros), refs)
                return out

            return jax.jit(rollout)

        fn, compiled = s.jitted(key, build)
        q_out, qd_out, tau_out = fn(
            (s.as_array(q_r), s.as_array(qd_r), s.as_array(qdd_r)),
            s.as_array(q_ref[0]),
        )
        q_out = np.asarray(q_out, dtype=np.float64)
        qd_out = np.asarray(qd_out, dtype=np.float64)
        tau_out = np.asarray(tau_out, dtype=np.float64)

        diverged = not (
            np.all(np.isfinite(q_out))
            and np.all(np.isfinite(qd_out))
            and np.all(np.abs(q_out) < 1e3)
        )
        tracking = np.abs(q_out - q_r)
        peak_tau_idx = int(np.argmax(np.abs(tau_out)) % s.dof)

        final_pose = None
        if not diverged:
            fk = np.asarray(
                s.robot.forward_kinematics(s.as_array(q_out[-1])), dtype=np.float64
            )
            i = s.link_index(s.ee_link)
            final_pose = ex.pose_payload(fk[i, :4], fk[i, 4:7])

        if diverged:
            self._record_failure(
                request_id,
                "simulation_diverged",
                {
                    "dt_control_s": dt_ctrl,
                    "substeps": n_sub,
                    "hint": "raise substeps (the controller runs at dt/substeps), lower "
                            "kp/kd, or retime with a lower acceleration_scale",
                },
            )

        return self._envelope(
            request_id,
            t0,
            compiled,
            success=not diverged,
            diverged=bool(diverged),
            dt_control_s=round(dt_ctrl, 6),
            dt_waypoint_s=round(dt_waypoint, 6),
            control_substeps=n_sub,
            control_rate=stability,
            duration_s=round(float(times[-1]), 6),
            n_steps=int(q_r.shape[0]),
            max_tracking_error_rad=round(float(tracking.max()), 6) if not diverged else None,
            mean_tracking_error_rad=round(float(tracking.mean()), 6) if not diverged else None,
            peak_torque_nm=round(float(np.abs(tau_out).max()), 4) if not diverged else None,
            peak_torque_joint=s.joint_names[peak_tau_idx] if not diverged else None,
            peak_velocity_rad_s=round(float(np.abs(qd_out).max()), 4) if not diverged else None,
            final_ee_pose=final_pose,
        )

    def optimize_transport(
        self,
        object_name: str,
        goal_position: Sequence[float],
        goal_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        grip_link: str | None = None,
        pinch_offset: float = 0.095,
        object_mass: float = 0.2,
        n_timesteps: int = 32,
        n_stages: int = 5,
        n_inner_iters: int = 50,
        dt: float = 0.1,
        tau_max: float = 87.0,
    ) -> dict[str, Any]:
        """Contact-aware transport of a grasped scene object via ``flat_contact_trajopt``.

        Differential-flatness formulation: the object's SE(3) pose is the flat
        output and the arm's configuration is slaved to it, so grasp closure and
        object dynamics hold by construction rather than by penalty.

        Requires the GRiD CUDA dynamics backend (built out-of-tree). The object
        must already exist in the scene as a ``box``; ``pinch_offset`` is the
        grasp standoff along the grip link's local +z, and the caller owns it —
        a wrong value puts the contact point inside the palm.
        """
        request_id, t0 = self._new_request()
        s = self.session
        if object_name not in s.scene:
            raise ValueError(
                f"no object named {object_name!r}; add it with add_object first"
            )
        obj = next(o for o in s.scene.objects() if o.name == object_name)
        if obj.shape != "box":
            raise ValueError(
                f"optimize_transport currently supports box objects; {object_name!r} "
                f"is a {obj.shape}"
            )

        grip_link = grip_link or s.ee_link
        try:
            from pyroffi.dynamics import GRiDDynamics
            from pyroffi.dynamics._contact import (
                ContactSystem,
                GraspedObject,
                ManipulatorSpec,
            )
            from pyroffi.optimization_engines import (
                FlatContactTrajOptConfig,
                flat_contact_trajopt,
            )

            grid = GRiDDynamics.from_robot(s.robot)
        except Exception as exc:  # pragma: no cover - depends on out-of-tree build
            raise RuntimeError(
                "optimize_transport needs the GRiD CUDA dynamics backend, which is "
                f"built out-of-tree and is unavailable here: {exc}"
            ) from exc

        import numpy as _np
        from pyroffi.collision import Box

        half = _np.array(
            [
                float(obj.params["length"]) / 2.0,
                float(obj.params["width"]) / 2.0,
                float(obj.params["height"]) / 2.0,
            ]
        )
        box_geom = Box.from_center_and_half_lengths(
            obj.position, half, wxyz=obj.wxyz
        ).with_physical_properties(mass=object_mass, friction=1.0)

        # Endpoint arm configs: grasp pose above the object now, and at the goal.
        def _grasp_pose(position, wxyz):
            return {"position": list(np.asarray(position, dtype=float)), "wxyz": list(wxyz)}

        start_ik = self.solve_ik(
            link=grip_link,
            pose=_grasp_pose(obj.position + np.array([0.0, 0.0, pinch_offset]), (0.0, 0.0, 1.0, 0.0)),
            num_seeds=64,
        )
        goal_ik = self.solve_ik(
            link=grip_link,
            pose=_grasp_pose(
                np.asarray(goal_position, dtype=float) + np.array([0.0, 0.0, pinch_offset]),
                goal_wxyz,
            ),
            num_seeds=64,
            seed_config=start_ik["config_id"],
        )
        q0 = s.handles.get(start_ik["config_id"]).values
        q1 = s.handles.get(goal_ik["config_id"]).values

        from jax import numpy as jnp

        arm = ManipulatorSpec(
            s.robot, grid, grip_link, base_xy_yaw=(0.0, 0.0, 0.0),
            p_local=(0.0, 0.0, pinch_offset),
        )
        system = ContactSystem(
            manipulators=(arm,), body=GraspedObject(geom=box_geom), grasp_offsets=()
        )
        cfg = FlatContactTrajOptConfig(
            n_stages=n_stages, n_inner_iters=n_inner_iters, dt=dt, tau_max=tau_max
        )

        alpha = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]
        init = s.as_array(q0)[None] * (1 - alpha) + s.as_array(q1)[None] * alpha

        key = f"transport:{n_timesteps}:{n_stages}:{n_inner_iters}"
        compiled = key not in s.ledger
        s.ledger.visit(key)

        traj, forces, residuals, _centers, dt_out = flat_contact_trajopt(
            init, s.as_array(q0), s.as_array(q1), system, cfg
        )
        traj = np.asarray(traj, dtype=np.float64)
        dt_final = float(np.asarray(dt_out))

        entry = s.handles.insert(
            "trajectory",
            traj,
            s.joint_names,
            s.scene.version,
            meta={"source": "optimize_transport", "request_id": request_id,
                  "object": object_name},
            times=np.arange(traj.shape[0]) * dt_final,
        )
        res = np.asarray(residuals, dtype=np.float64)
        return self._envelope(
            request_id,
            t0,
            compiled,
            success=bool(np.all(np.isfinite(traj))),
            trajectory_id=entry.handle,
            object=object_name,
            n_waypoints=int(traj.shape[0]),
            dt_s=round(dt_final, 6),
            duration_s=round(dt_final * (traj.shape[0] - 1), 6),
            peak_contact_force_n=round(float(np.abs(np.asarray(forces)).max()), 4),
            final_residual=round(float(res.reshape(-1)[-1]), 8) if res.size else None,
            endpoint_ik=[start_ik["success"], goal_ik["success"]],
        )

    # ── inspection ────────────────────────────────────────────────────────

    def explain_failure(self, request_id: str) -> dict[str, Any]:
        """Structured cause for an earlier failed request.

        Turns a one-shot tool into something an agent can iterate against: which
        joint hit its limit, which waypoint collided with which object, whether
        the optimizer plateaued.
        """
        rid, t0 = self._new_request()
        record = self._failures.get(request_id)
        if record is None:
            return self._envelope(
                rid,
                t0,
                compiled=False,
                success=False,
                explained_request=request_id,
                note="no failure recorded for that request id — it either succeeded "
                     "or predates this session",
            )
        return self._envelope(
            rid, t0, compiled=False, explained_request=request_id, **record
        )

    def render_scene(
        self,
        config: str | Mapping[str, float] | Sequence[float] | None = None,
        resolution: tuple[int, int] = (640, 480),
    ) -> dict[str, Any]:
        """Offscreen still frame of the scene, base64 PNG.

        Uses trimesh's offscreen rasteriser, which needs a working GL stack;
        headless boxes usually need ``MUJOCO_GL=egl``-style setup or an
        X server. Reported as unavailable rather than crashing when GL is
        missing — an unrenderable scene is not a planning failure.
        """
        request_id, t0 = self._new_request()
        s = self.session
        cfg = self._resolve_config(config) if config is not None else s.robot_state

        try:
            import base64

            import trimesh

            scene = trimesh.Scene()
            robot_geom = s.robot_coll.at_config(s.robot, s.as_array(cfg))
            scene.add_geometry(robot_geom.to_trimesh(), node_name="robot")
            for pool, (shape, names) in zip(s.world_geoms(), s.scene.geom_names()):
                for slot, name in enumerate(names):
                    if name is None or shape == "halfspace":
                        continue
                    scene.add_geometry(pool[slot].to_trimesh(), node_name=name)
            png = scene.save_image(resolution=resolution, visible=False)
            if not png:
                raise RuntimeError("renderer returned no image data")
            return self._envelope(
                request_id,
                t0,
                compiled=False,
                image_base64=base64.b64encode(png).decode("ascii"),
                mime_type="image/png",
                resolution=list(resolution),
            )
        except Exception as exc:
            logger.warning(f"render_scene unavailable: {exc}")
            return self._envelope(
                request_id,
                t0,
                compiled=False,
                success=False,
                error="renderer_unavailable",
                detail=str(exc),
                note="no offscreen GL context available on this host",
            )

    def warmup(
        self,
        include_trajopt: bool = True,
        path_lengths: Sequence[int] | None = None,
        n_batch: int = 25,
    ) -> dict[str, Any]:
        """Pay the compile cost up front, deliberately.

        Compilation is explicit in this design rather than hidden: an agent that
        never calls ``warmup`` will see a 40-second ``optimize_path`` with
        ``compiled: true``, which is information, not a stall. Warming up with
        the *exact* configuration you will later time matters — a mismatched
        warmup silently recompiles.
        """
        request_id, t0 = self._new_request()
        s = self.session
        lengths = tuple(path_lengths) if path_lengths else (s.n_timesteps,)
        stages: list[dict[str, Any]] = []

        def stage(name: str, fn):
            t = time.perf_counter()
            try:
                fn()
                ok, err = True, None
            except Exception as exc:  # keep warming the rest
                ok, err = False, str(exc)
                logger.warning(f"warmup stage {name} failed: {exc}")
            stages.append(
                {
                    "stage": name,
                    "ok": ok,
                    "seconds": round(time.perf_counter() - t, 3),
                    **({"error": err} if err else {}),
                }
            )

        mid = (s.lower_limits + s.upper_limits) / 2.0
        pose = self.forward_kinematics(mid, links=[s.ee_link])["poses"][s.ee_link]

        stage("forward_kinematics", lambda: self.forward_kinematics(mid))
        stage("check_collision", lambda: self.check_collision(mid))
        stage("check_edge", lambda: self.check_edge(mid, s.clip_to_limits(mid + 0.1)))
        stage("solve_ik", lambda: self.solve_ik(pose=pose))
        stage("solve_ik_batch", lambda: self.solve_ik_batch([pose]))
        for n in lengths:
            stage(
                f"validate_path[{n}]",
                lambda n=n: self.validate_path(
                    [ex.joint_dict(r, s.joint_names) for r in self._resample(
                        np.stack([mid, s.clip_to_limits(mid + 0.1)]), n)]
                ),
            )
        if include_trajopt:
            for n in lengths:
                stage(
                    f"optimize_path[{n}]",
                    lambda n=n: self.optimize_path(
                        [ex.joint_dict(r, s.joint_names) for r in self._resample(
                            np.stack([mid, s.clip_to_limits(mid + 0.1)]), n)],
                        n_batch=n_batch,
                    ),
                )

        return self._envelope(
            request_id,
            t0,
            compiled=True,
            success=all(st["ok"] for st in stages),
            stages=stages,
            total_seconds=round(time.perf_counter() - t0, 3),
            note="subsequent calls with these exact shapes/configs report compiled=false",
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _resolve_config(
        self, config: str | Mapping[str, float] | Sequence[float]
    ) -> np.ndarray:
        """Handle, name-keyed object, or positional array → ``(dof,)``."""
        s = self.session
        if isinstance(config, str):
            entry = s.handles.get(config)
            if entry.values.ndim == 2:
                raise ValueError(
                    f"{config} is a path ({entry.n_waypoints} waypoints), not a config"
                )
            return entry.values
        return ex.config_from_payload(config, s.joint_names, defaults=s.robot_state)

    def _resolve_path(
        self, path: str | Sequence[Any]
    ) -> tuple[np.ndarray, str | None]:
        """Handle or inline waypoint list → ``((T, DOF), source_handle_or_None)``."""
        s = self.session
        if isinstance(path, str):
            entry = s.handles.get(path)
            values = entry.values
            return (values if values.ndim == 2 else values[None, :]), path
        return ex.path_from_payload(path, s.joint_names), None

    def _fixed_joint_mask(self, fixed_joints: Sequence[str]):
        """Boolean mask over actuated joints, or None when nothing is fixed."""
        s = self.session
        if not fixed_joints:
            return None
        unknown = sorted(set(fixed_joints) - set(s.joint_names))
        if unknown:
            raise ValueError(f"unknown joints {unknown} in fixed_joints")
        from jax import numpy as jnp

        return jnp.asarray(
            [name in set(fixed_joints) for name in s.joint_names], dtype=bool
        )

    def _auto_substeps(
        self,
        q0: np.ndarray,
        dt_waypoint: float,
        kp: float,
        kd: float,
        safety: float = 0.25,
        cap: int = 128,
    ) -> tuple[int, dict[str, Any]]:
        """Control substeps that keep the sampled feedback loop stable.

        The binding constraint is the *stiffest* inertial direction, i.e. the
        smallest eigenvalue of the mass matrix — not its diagonal. Explicit
        velocity feedback needs ``kd * dt / lambda_min < 2`` and the position
        loop needs ``dt * sqrt(kp / lambda_min) < 2``; a quarter of that bound
        leaves room for the trajectory's own dynamics.

        Deriving this rather than defaulting to a magic number is what keeps
        ``simulate`` from silently diverging on a robot whose inertia differs
        from the one the default was tuned on.
        """
        s = self.session
        try:
            import numpy.linalg as _la

            M = np.asarray(s.robot.mass_matrix(s.as_array(q0)), dtype=np.float64)
            lam_min = float(_la.eigvalsh(M).min())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"auto-substeps fell back to a fixed rate: {exc}")
            return 16, {"auto": True, "fallback": True}

        lam_min = max(lam_min, 1e-6)
        dt_vel = 2.0 * lam_min / max(kd, 1e-9)
        dt_pos = 2.0 / max(np.sqrt(kp / lam_min), 1e-9)
        dt_target = safety * min(dt_vel, dt_pos)
        n_sub = int(np.clip(np.ceil(dt_waypoint / dt_target), 1, cap))
        return n_sub, {
            "auto": True,
            "lambda_min_mass_matrix": round(lam_min, 6),
            "dt_stability_limit_s": round(min(dt_vel, dt_pos), 6),
            "dt_target_s": round(dt_target, 6),
        }

    @staticmethod
    def _control_grid(
        q_wp: np.ndarray,
        times: np.ndarray,
        substeps: int,
        qd_wp: np.ndarray | None,
        qdd_wp: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reference (t, q, qd, qdd) resampled onto the control grid.

        Linear interpolation in time between waypoints. ``qd``/``qdd`` come from
        retiming when available; otherwise they are finite-differenced, which is
        adequate for the feedforward term and is what a caller who skipped
        ``retime`` implicitly asked for.
        """
        n = q_wp.shape[0]
        if n < 2:
            z = np.zeros_like(q_wp)
            return times.copy(), q_wp.copy(), z, z

        if qd_wp is None or qdd_wp is None:
            dt = np.diff(times)[:, None]
            seg_v = np.diff(q_wp, axis=0) / np.maximum(dt, 1e-9)
            qd_wp = np.concatenate([seg_v[:1] * 0.0, seg_v], axis=0)
            qdd_wp = np.concatenate(
                [np.zeros((1, q_wp.shape[1])),
                 np.diff(qd_wp, axis=0) / np.maximum(dt, 1e-9)],
                axis=0,
            )

        n_ticks = (n - 1) * substeps + 1
        t_ctrl = np.linspace(times[0], times[-1], n_ticks)
        interp = lambda arr: np.stack(
            [np.interp(t_ctrl, times, arr[:, j]) for j in range(arr.shape[1])], axis=1
        )
        return t_ctrl, interp(q_wp), interp(np.asarray(qd_wp)), interp(np.asarray(qdd_wp))

    @staticmethod
    def _path_length(path: np.ndarray) -> float:
        """Total joint-space arc length (rad)."""
        path = np.asarray(path, dtype=np.float64)
        if path.ndim == 1 or path.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    @staticmethod
    def _resample(path: np.ndarray, n: int) -> np.ndarray:
        """Resample a path to exactly *n* waypoints, parameterised by arc length.

        Preferred over zero-padding whenever the result is fed to an optimizer:
        padding leaves duplicate waypoints the smoother has to work out of the
        trajectory, while resampling preserves the seed's geometry exactly.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.shape[0] == n:
            return path.copy()
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 0.0:  # a stationary path: just repeat it
            return np.repeat(path[:1], n, axis=0)
        u = cum / cum[-1]
        target = np.linspace(0.0, 1.0, n)
        return np.stack(
            [np.interp(target, u, path[:, j]) for j in range(path.shape[1])], axis=1
        )
