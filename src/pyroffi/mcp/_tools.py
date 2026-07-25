"""Tool schemas and the mapping from MCP calls to toolbox primitives.

This module is the thin part. It owns three things and nothing else: the JSON
schema for each tool, which toolbox method it dispatches to, and the wording of
the description the model actually reads.

The descriptions carry **relative cost** on purpose, and the numbers in them are
measured rather than assumed (Panda, RTX A5000, float64 — see
``docs/mcp_server.md``). A model that knows ``check_edge`` is ~3 ms while
``optimize_path`` is ~3.6 s warm and ~8 s cold will self-order its calls — prune
with validation, batch the kinematics, optimize once — without any policy logic
on the server side. The spread that matters here is three orders of magnitude,
so these figures need to be roughly right, not precise.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

# ── schema helpers ───────────────────────────────────────────────────────────

_POSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "World-frame SE(3) pose. Position in metres; quaternion scalar-first "
        "(wxyz). An xyzw quaternion is rejected, not reinterpreted."
    ),
    "properties": {
        "position": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": "[x, y, z] in metres, world frame.",
        },
        "wxyz": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "Quaternion [w, x, y, z], scalar first. Default identity.",
        },
        "quaternion_convention": {
            "type": "string",
            "enum": ["wxyz"],
            "description": "Must be 'wxyz' if given.",
        },
    },
    "required": ["position"],
}

_CONFIG_SCHEMA: dict[str, Any] = {
    "description": (
        "A configuration: either a config handle ('cfg_0001'), or a name-keyed "
        "object {joint_name: radians} covering the actuated joints. Passive and "
        "mimic joints are not part of the interface."
    ),
    "anyOf": [
        {"type": "string"},
        {"type": "object", "additionalProperties": {"type": "number"}},
        {"type": "array", "items": {"type": "number"}},
    ],
}

_PATH_SCHEMA: dict[str, Any] = {
    "description": (
        "A path: either a path handle ('path_0001') or an inline list of "
        "name-keyed waypoints."
    ),
    "anyOf": [
        {"type": "string"},
        {
            "type": "array",
            "items": {"type": "object", "additionalProperties": {"type": "number"}},
        },
    ],
}


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: its schema, its description, and how it dispatches."""

    name: str
    description: str
    input_schema: dict[str, Any]
    method: str
    """Name of the :class:`~pyroffi.toolbox.Toolbox` method to call."""

    def to_mcp(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ── the tool table ───────────────────────────────────────────────────────────

TOOLS: tuple[ToolSpec, ...] = (
    # ── scene ────────────────────────────────────────────────────────────
    ToolSpec(
        name="get_capabilities",
        description=(
            "Session capabilities: DOF, joint names and limits, end-effector link, "
            "collision model quality, available backends (CUDA/GRiD/VAMP), whether "
            "float64 is on, and the compiled path-length buckets. FREE. Call this "
            "first — everything else is expressed in these joint names and units."
        ),
        input_schema=_obj({}),
        method="create_scene_info",
    ),
    ToolSpec(
        name="create_scene",
        description=(
            "Rebuild the session with a different robot or capacity, returning the "
            "new capabilities. Costs a few seconds (URDF parse + collision "
            "calibration) and INVALIDATES every existing handle. GPU selection and "
            "float64 are process-level and fixed when the server launched, so they "
            "cannot be changed here."
        ),
        input_schema=_obj(
            {
                "robot": {
                    "type": "string",
                    "description": "URDF path, robot_descriptions name, or alias "
                                   "('panda', 'panda_spherized', 'ur5', 'iiwa').",
                },
                "max_objects": {
                    "type": "integer",
                    "description": "Per-shape obstacle capacity. Fixed for the scene's "
                                   "life because array shapes must stay static; size it "
                                   "generously (padding is nearly free).",
                    "minimum": 1,
                },
                "n_timesteps": {"type": "integer", "minimum": 2},
                "collision_model": {
                    "type": "string",
                    "enum": ["auto", "capsule", "spherized"],
                    "description": "'spherized' is much more faithful for self-collision "
                                   "but needs primitive (non-mesh) collision geometry.",
                },
            }
        ),
        method="recreate_session",
    ),
    ToolSpec(
        name="add_object",
        description=(
            "Add or move a named obstacle (box / sphere / capsule / halfspace). FREE — "
            "obstacle count is padded, so this never changes array shapes or triggers "
            "a recompile. Re-adding an existing name moves it in place."
        ),
        input_schema=_obj(
            {
                "name": {"type": "string"},
                "shape": {
                    "type": "string",
                    "enum": ["box", "sphere", "capsule", "halfspace"],
                },
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Centre in metres, world frame. For a halfspace, a "
                                   "point on the plane.",
                },
                "wxyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Shape parameters, all in metres: box needs length/width/height "
                        "(full extents), sphere needs radius, capsule needs "
                        "radius/height, halfspace needs normal ([x,y,z], outward)."
                    ),
                    "additionalProperties": True,
                },
            },
            required=["name", "shape"],
        ),
        method="add_object",
    ),
    ToolSpec(
        name="remove_object",
        description=(
            "Remove a named obstacle from the scene. FREE. Bumps scene_version, so "
            "any path validated before this call should be re-validated."
        ),
        input_schema=_obj({"name": {"type": "string"}}, required=["name"]),
        method="remove_object",
    ),
    ToolSpec(
        name="list_objects",
        description=(
            "List scene objects with poses and the current scene_version. FREE — call "
            "it freely. Stale scene state is the most common practical failure in a "
            "long session, and this is how you rule it out."
        ),
        input_schema=_obj({}),
        method="list_objects",
    ),
    ToolSpec(
        name="export_scene",
        description=(
            "Export the obstacle set so another planner can load the same world. FREE. "
            "'primitives' is a plain typed list; 'urdf' emits fixed-jointed links "
            "(half-spaces become thin slabs). Compare the returned scene_version "
            "against any later validation to detect a stale handoff."
        ),
        input_schema=_obj(
            {"format": {"type": "string", "enum": ["primitives", "urdf"]}}
        ),
        method="export_scene",
    ),
    ToolSpec(
        name="set_robot_state",
        description=(
            "Set the current configuration, used to warm-start IK and as the default "
            "seed. FREE. Accepts a config handle, a name-keyed object, or 'default'."
        ),
        input_schema=_obj({"config": _CONFIG_SCHEMA}, required=["config"]),
        method="set_robot_state",
    ),
    # ── kinematics ───────────────────────────────────────────────────────
    ToolSpec(
        name="forward_kinematics",
        description=(
            "World poses of the named links in a configuration. ~1 ms warm. Use it to "
            "confirm where a config actually puts the hand."
        ),
        input_schema=_obj(
            {
                "config": _CONFIG_SCHEMA,
                "links": {"type": "array", "items": {"type": "string"}},
            },
            required=["config"],
        ),
        method="forward_kinematics",
    ),
    ToolSpec(
        name="solve_ik",
        description=(
            "Inverse kinematics for ONE target pose. ~170 ms warm (64 seeds), ~7 s first call. Returns a config "
            "handle plus the residual and how many restarts converged — partial "
            "success is real information, so read restarts_converged rather than "
            "just success. Set collision_free=true to fold the scene into the solve. "
            "'solver' is your choice and is never overridden. For more than one "
            "target, use solve_ik_batch instead: it is barely more expensive than one."
        ),
        input_schema=_obj(
            {
                "pose": _POSE_SCHEMA,
                "link": {
                    "type": "string",
                    "description": "Target link. Defaults to the session's ee_link.",
                },
                "num_seeds": {
                    "type": "integer",
                    "description": "Seeds refined in parallel. 32 is a good default; "
                                   "raise for hard targets.",
                },
                "solver": {"type": "string", "enum": ["hjcd", "ls"]},
                "seed_config": _CONFIG_SCHEMA,
                "num_restarts": {
                    "type": "integer",
                    "description": "Independent multi-seed solves. >1 reports a "
                                   "convergence distribution instead of one answer.",
                },
                "collision_free": {"type": "boolean"},
                "fixed_joints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Joints to hold fixed, e.g. gripper fingers.",
                },
                "pos_tolerance": {"type": "number"},
                "rot_tolerance": {"type": "number"},
                "seed": {"type": "integer"},
            },
            required=["pose"],
        ),
        method="solve_ik",
    ),
    ToolSpec(
        name="solve_ik_batch",
        description=(
            "N target poses in ONE GPU dispatch. Measured warm (Panda, 64 seeds): "
            "1 target ~160 ms, 8 targets ~410 ms, 32 targets ~1.2 s — so the "
            "per-target cost falls from ~160 ms to ~40 ms. PREFER THIS over repeated "
            "solve_ik whenever you are enumerating candidate grasps, placements or "
            "approach poses. Returns one config handle per target, with per-target "
            "convergence so you can see which candidates are actually viable."
        ),
        input_schema=_obj(
            {
                "targets": {"type": "array", "items": _POSE_SCHEMA, "minItems": 1},
                "link": {"type": "string"},
                "num_seeds": {"type": "integer"},
                "seed_config": _CONFIG_SCHEMA,
                "pos_tolerance": {"type": "number"},
                "rot_tolerance": {"type": "number"},
                "seed": {"type": "integer"},
            },
            required=["targets"],
        ),
        method="solve_ik_batch",
    ),
    ToolSpec(
        name="check_reachable",
        description=(
            "Can the end-effector reach this pose at all? ~170 ms warm. Thin wrapper over "
            "batched IK that leaves no handle behind — for pruning candidates you do "
            "not intend to use."
        ),
        input_schema=_obj(
            {
                "pose": _POSE_SCHEMA,
                "link": {"type": "string"},
                "num_seeds": {"type": "integer"},
            },
            required=["pose"],
        ),
        method="check_reachable",
    ),
    # ── validation ───────────────────────────────────────────────────────
    ToolSpec(
        name="check_collision",
        description=(
            "Collision state of one configuration: named colliding pairs, min "
            "clearance, joint-limit violations. ~2 ms warm. Check get_capabilities' "
            "self_collision_calibration.reliable before trusting self-collision "
            "results on a coarse collision model."
        ),
        input_schema=_obj(
            {
                "config": _CONFIG_SCHEMA,
                "margin": {
                    "type": "number",
                    "description": "Clearance in metres below which a pair counts as "
                                   "colliding. 0 = true contact.",
                },
            },
            required=["config"],
        ),
        method="check_collision",
    ),
    ToolSpec(
        name="check_edge",
        description=(
            "Is the straight-line joint-space motion between two configs valid, and "
            "where does it first fail? ~3 ms warm. The cheapest useful filter — run it "
            "before asking for any optimization."
        ),
        input_schema=_obj(
            {
                "config_a": _CONFIG_SCHEMA,
                "config_b": _CONFIG_SCHEMA,
                "resolution": {"type": "integer"},
                "margin": {"type": "number"},
            },
            required=["config_a", "config_b"],
        ),
        method="check_edge",
    ),
    ToolSpec(
        name="validate_path",
        description=(
            "Validate a whole path: every waypoint, every subdivided edge, joint "
            "limits, min clearance. ~8 ms warm. THIS IS WHAT YOU CALL ON WHATEVER AN "
            "EXTERNAL PLANNER RETURNED. The response carries scene_version and flags "
            "a stale scene, so a validation against moved obstacles is detectable."
        ),
        input_schema=_obj(
            {
                "path": _PATH_SCHEMA,
                "edge_substeps": {
                    "type": "integer",
                    "description": "Interior samples per edge. 4 is usually enough; "
                                   "raise it for thin obstacles.",
                },
                "margin": {"type": "number"},
            },
            required=["path"],
        ),
        method="validate_path",
    ),
    # ── exchange ─────────────────────────────────────────────────────────
    ToolSpec(
        name="import_path",
        description=(
            "Bring a joint-space path in from a foreign planner and get a handle. "
            "FREE. Waypoints must be name-keyed, radians, in the actuated joint names "
            "from get_capabilities — do not send passive or mimic joints."
        ),
        input_schema=_obj(
            {
                "waypoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "minItems": 1,
                },
                "source": {"type": "string"},
            },
            required=["waypoints"],
        ),
        method="import_path",
    ),
    ToolSpec(
        name="export_path",
        description=(
            "Get the ACTUAL joint numbers for a path, name-keyed, for handoff to "
            "another server or to a controller. FREE to compute but EXPENSIVE in "
            "context — a 64-waypoint 7-DOF path is thousands of tokens. Only call it "
            "when the numbers must leave this server; otherwise pass handles around."
        ),
        input_schema=_obj(
            {
                "path_id": {"type": "string"},
                "include_times": {"type": "boolean"},
            },
            required=["path_id"],
        ),
        method="export_path",
    ),
    ToolSpec(
        name="export_config",
        description=(
            "One configuration as name-keyed joint values in radians. FREE and small "
            "(one object, not a path) — safe to call when you need the actual numbers "
            "for a single pose, e.g. to hand a grasp config to another server."
        ),
        input_schema=_obj({"config_id": {"type": "string"}}, required=["config_id"]),
        method="export_config",
    ),
    # ── optimization ─────────────────────────────────────────────────────
    ToolSpec(
        name="optimize_path",
        description=(
            "Smooth and repair a path you supply, with SCO trajectory optimization. "
            "THE EXPENSIVE TOOL: ~3.6 s warm at the default iteration counts, ~8 s on "
            "the FIRST call for a given path length (check the 'compiled' field before "
            "concluding the server is slow; call warmup to pay it up front). Lower "
            "n_outer_iters/n_inner_iters to trade quality for speed — 10/25 runs in "
            "~1.5 s. Prune with check_edge and validate_path before spending this. "
            "This is the main trajopt entry point: it is seeded from "
            "your path, so it composes with an external sampling planner. Local "
            "optimizer — it repairs a locally-bad seed well and cannot escape a seed "
            "trapped on the wrong side of an obstacle."
        ),
        input_schema=_obj(
            {
                "path": _PATH_SCHEMA,
                "n_batch": {
                    "type": "integer",
                    "description": "Parallel perturbed candidates. More is nearly free "
                                   "on GPU and improves the result.",
                },
                "collision_margin": {
                    "type": "number",
                    "description": "Desired clearance in metres the cost drives toward.",
                },
                "w_smooth": {"type": "number"},
                "w_collision": {"type": "number"},
                "n_outer_iters": {"type": "integer"},
                "n_inner_iters": {"type": "integer"},
                "seed": {"type": "integer"},
            },
            required=["path"],
        ),
        method="optimize_path",
    ),
    ToolSpec(
        name="optimize_between",
        description=(
            "Convenience: interpolate between two endpoints and optimize. Same cost as "
            "optimize_path. HONEST LIMIT: this is a local optimizer seeded by a "
            "straight line, not a planner. In anything maze-like it returns a path "
            "that still hits an obstacle rather than reporting 'no path' — for that, "
            "get a seed from a sampling planner and use optimize_path. Endpoints may "
            "be configs or poses (IK'd here)."
        ),
        input_schema=_obj(
            {
                "config_a": _CONFIG_SCHEMA,
                "config_b": _CONFIG_SCHEMA,
                "pose_a": _POSE_SCHEMA,
                "pose_b": _POSE_SCHEMA,
                "link": {"type": "string"},
                "n_timesteps": {"type": "integer"},
            }
        ),
        method="optimize_between",
    ),
    ToolSpec(
        name="concat_paths",
        description=(
            "Join path segments into one handle, verifying that each segment ends "
            "where the next begins. FREE. Use it instead of tracking continuity "
            "yourself — a silent discontinuity surfaces much later as a mystery "
            "collision or torque spike."
        ),
        input_schema=_obj(
            {
                "path_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "tolerance": {"type": "number"},
            },
            required=["path_ids"],
        ),
        method="concat_paths",
    ),
    ToolSpec(
        name="retime",
        description=(
            "Turn a geometric path into a timed trajectory under velocity and "
            "acceleration limits. <1 ms, CPU, closed form. REQUIRED before simulate, and before "
            "any statement about how long a motion takes: an un-retimed path has no "
            "duration. Feasible by construction, not time-optimal."
        ),
        input_schema=_obj(
            {
                "path": _PATH_SCHEMA,
                "velocity_scale": {
                    "type": "number",
                    "description": "Fraction of the URDF velocity limits to use (0-1].",
                },
                "acceleration_scale": {"type": "number"},
                "time_to_peak": {
                    "type": "number",
                    "description": "Seconds to reach full joint speed, used to infer "
                                   "acceleration limits (URDFs do not carry them).",
                },
            },
            required=["path"],
        ),
        method="retime",
    ),
    # ── dynamics and inspection ──────────────────────────────────────────
    ToolSpec(
        name="simulate",
        description=(
            "Roll a RETIMED trajectory forward under computed-torque control. ~300 ms warm. "
            "Reports tracking error, peak torque and joint, peak velocity, final "
            "end-effector pose, and divergence. Verify here before committing a "
            "motion. The control rate is chosen from the mass matrix for stability, so "
            "a divergence reported here is about the trajectory."
        ),
        input_schema=_obj(
            {
                "trajectory": {
                    "type": "string",
                    "description": "A trajectory handle from retime (not a bare path).",
                },
                "kp": {"type": "number"},
                "kd": {"type": "number"},
                "substeps": {
                    "type": "integer",
                    "description": "Control ticks per waypoint. Omit to derive a stable "
                                   "rate automatically (recommended).",
                },
                "feedforward": {"type": "boolean"},
            },
            required=["trajectory"],
        ),
        method="simulate",
    ),
    ToolSpec(
        name="optimize_transport",
        description=(
            "Contact-aware transport of a grasped box via differential-flatness "
            "trajectory optimization: grasp closure and object dynamics hold by "
            "construction. SLOW (tens of seconds) and requires the GRiD CUDA dynamics "
            "backend, which is built out-of-tree and may be unavailable. The object "
            "must already be in the scene as a box."
        ),
        input_schema=_obj(
            {
                "object_name": {"type": "string"},
                "goal_position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "goal_wxyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "grip_link": {"type": "string"},
                "pinch_offset": {
                    "type": "number",
                    "description": "Grasp standoff along the grip link's local +z, in "
                                   "metres. Robot-specific; a wrong value puts the "
                                   "contact point inside the palm.",
                },
                "object_mass": {"type": "number"},
                "n_timesteps": {"type": "integer"},
            },
            required=["object_name", "goal_position"],
        ),
        method="optimize_transport",
    ),
    ToolSpec(
        name="explain_failure",
        description=(
            "Structured cause for an earlier failed request, by its request_id: which "
            "joint hit a limit, which waypoint collided with which named object, "
            "whether the optimizer plateaued, plus a hint. FREE. Call this instead of "
            "guessing why something failed."
        ),
        input_schema=_obj(
            {"request_id": {"type": "string"}}, required=["request_id"]
        ),
        method="explain_failure",
    ),
    ToolSpec(
        name="render_scene",
        description=(
            "Offscreen PNG of the robot and obstacles, to look at. ~200 ms. Needs a GL "
            "context on the host; returns success=false with 'renderer_unavailable' "
            "when there is none, which is not a planning failure."
        ),
        input_schema=_obj({"config": _CONFIG_SCHEMA}),
        method="render_scene",
    ),
    ToolSpec(
        name="warmup",
        description=(
            "Compile everything up front. SLOW ON PURPOSE (~40 s with trajopt). "
            "Compilation in this server is explicit rather than hidden: warm up once "
            "and every later call reports compiled=false and runs in milliseconds. "
            "Warm up with the SAME path lengths you will actually use — a mismatched "
            "warmup silently recompiles later."
        ),
        input_schema=_obj(
            {
                "include_trajopt": {"type": "boolean"},
                "path_lengths": {"type": "array", "items": {"type": "integer"}},
                "n_batch": {"type": "integer"},
            }
        ),
        method="warmup",
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}


def list_tool_payloads() -> list[dict[str, Any]]:
    """The ``tools/list`` payload."""
    return [t.to_mcp() for t in TOOLS]


def dispatch(toolbox: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route one tool call to its toolbox method.

    Unknown tools and bad arguments raise; the server turns those into MCP
    errors. Nothing here reinterprets or "fixes up" arguments — a silently
    corrected argument is how a wrong answer becomes plausible.
    """
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown tool {name!r}; known tools: {sorted(TOOLS_BY_NAME)}")

    method: Callable = getattr(toolbox, spec.method, None)  # type: ignore[assignment]
    if method is None:
        raise RuntimeError(f"tool {name!r} maps to missing method {spec.method!r}")

    allowed = set(spec.input_schema.get("properties", {}))
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        raise ValueError(
            f"{name}: unexpected argument(s) {unexpected}; accepted: {sorted(allowed)}"
        )
    missing = [k for k in spec.input_schema.get("required", []) if k not in arguments]
    if missing:
        raise ValueError(f"{name}: missing required argument(s) {missing}")

    return method(**arguments)
