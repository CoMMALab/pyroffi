"""Tool schemas for the sandbox MCP server.

Same shape as :mod:`pyroffi.mcp._tools` — a table of specs, each naming the
method it dispatches to — and deliberately a *separate* table, because these
tools command something and the planning tools do not.

The descriptions carry the two facts an agent gets wrong here if nobody tells
it: motions execute from wherever the arm actually is (there is no teleport),
and an unretimed reference pulls a held block out of the gripper.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

_WAYPOINTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "description": (
        "Joint-space waypoints, name-keyed over the 7 arm joints "
        "(panda_joint1..7), radians. This is exactly what the pyroffi planning "
        "server's export_path returns."
    ),
    "items": {"type": "object", "additionalProperties": {"type": "number"}},
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
    name: str
    description: str
    input_schema: dict[str, Any]
    method: str

    def to_mcp(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_task",
        description=(
            "The task: blocks and their starting poses, obstacles, the goal "
            "stack, the grasp convention, and the tolerances you will be scored "
            "against. FREE. Call this first — it also tells you the robot and "
            "joint names to configure the pyroffi planning server with."
        ),
        input_schema=_obj({}),
        method="get_task",
    ),
    ToolSpec(
        name="observe",
        description=(
            "The simulator's ACTUAL state: arm configuration, gripper opening, "
            "which block (if any) is between the fingers, and every block's "
            "measured pose and what it is resting on. FREE. This is ground "
            "truth, not your plan — if a place went 2 cm wrong, this is where "
            "you find out. Call it after every motion."
        ),
        input_schema=_obj({}),
        method="observe",
    ),
    ToolSpec(
        name="render",
        description=(
            "A PNG of the scene through the viser viewer, so you can look at it. "
            "~1 s. Viewpoints: 'iso', 'front', 'side', 'top', or omit for the "
            "camera a human is currently looking through. REQUIRES a browser "
            "connected to the viewer URL — with none, this fails saying so "
            "rather than substituting a different renderer."
        ),
        input_schema=_obj(
            {
                "viewpoint": {
                    "type": "string",
                    "enum": ["iso", "front", "side", "top"],
                },
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            }
        ),
        method="render",
    ),
    ToolSpec(
        name="execute_path",
        description=(
            "Run a joint-space path on the arm and report what it actually did. "
            "Takes roughly the trajectory's own duration in wall time. "
            "TWO THINGS THAT WILL BITE YOU: (1) the path must START at the "
            "configuration observe() reports — execution does not teleport, and "
            "a mismatch is rejected; (2) pass times_s from the planning server's "
            "retime(). Without timing this runs at a fixed joint speed, and a "
            "reference the arm cannot follow pulls a held block straight out of "
            "the gripper. The gripper holds its current state throughout."
        ),
        input_schema=_obj(
            {
                "waypoints": _WAYPOINTS_SCHEMA,
                "times_s": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Per-waypoint times in seconds from retime(), "
                                   "same length as waypoints.",
                },
            },
            required=["waypoints"],
        ),
        method="execute_path",
    ),
    ToolSpec(
        name="set_gripper",
        description=(
            "Open or close the gripper. ~1 s. Closing is a COMMAND, not an "
            "outcome: the response tells you whether a block actually ended up "
            "between the fingers, and closing on nothing reports success=false. "
            "The fingers are real geometry closing on a real block — approach "
            "with the hand frame the task's grasp_standoff_m above the block "
            "centre, +z pointing down."
        ),
        input_schema=_obj(
            {"action": {"type": "string", "enum": ["open", "close"]}},
            required=["action"],
        ),
        method="set_gripper",
    ),
    ToolSpec(
        name="reset",
        description=(
            "Put the world back to its starting state and clear the performance "
            "history. ~1 s. Use it after wedging the scene; note that the report "
            "counts motions since the last reset."
        ),
        input_schema=_obj({}),
        method="reset",
    ),
    ToolSpec(
        name="report",
        description=(
            "The performance report: whether the goal stack is built, per-block "
            "position error against target, how many motions it took, how many "
            "of them were run unretimed, commanded duration, and worst tracking "
            "error. FREE. Call it when you believe you are done — it reads the "
            "simulator, so it cannot be satisfied by a plan that was never run."
        ),
        input_schema=_obj({}),
        method="report",
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}


def list_tool_payloads() -> list[dict[str, Any]]:
    return [t.to_mcp() for t in TOOLS]


def dispatch(target: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route one tool call. Nothing here reinterprets arguments."""
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown tool {name!r}; known tools: {sorted(TOOLS_BY_NAME)}")
    method: Callable = getattr(target, spec.method, None)  # type: ignore[assignment]
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
