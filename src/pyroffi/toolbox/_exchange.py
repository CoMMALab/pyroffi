"""The interop contract: what a joint array, a pose, and a scene mean when they
cross the server boundary.

Every silent failure in a multi-server TAMP stack lives here, so each
convention is pinned in code rather than assumed:

* **Joint ordering** — name-keyed, always. Positional arrays are accepted only
  when they are exactly ``dof`` long, and the response echoes the names so the
  caller can check its own ordering assumption.
* **Quaternion convention** — ``wxyz`` (pyroffi/jaxlie), scalar first. Payloads
  are labelled; a ``quaternion_convention`` other than ``wxyz`` is rejected
  rather than reinterpreted.
* **Units and frames** — radians, metres, and the scene's world frame.
* **Passive/mimic joints** — never crossed. Only ``robot.joints.actuated_names``
  appear, so a URDF's mimic finger joint cannot arrive as a free variable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jaxlie
import numpy as np
from jax import numpy as jnp

QUATERNION_CONVENTION = "wxyz"
UNITS = {"length": "m", "angle": "rad", "time": "s"}


def joint_dict(
    values: np.ndarray, joint_names: Sequence[str]
) -> dict[str, float]:
    """Name-keyed view of a single configuration."""
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != len(joint_names):
        raise ValueError(
            f"got {values.shape[0]} joint values for {len(joint_names)} actuated joints"
        )
    return {name: float(v) for name, v in zip(joint_names, values)}


def config_from_payload(
    payload: Mapping[str, float] | Sequence[float],
    joint_names: Sequence[str],
    defaults: np.ndarray | None = None,
) -> np.ndarray:
    """Turn an inbound configuration into a positional array.

    A mapping is resolved by name (unknown names are an error; missing names
    fall back to *defaults* if given, else are an error).  A bare sequence is
    accepted only at full length, and is interpreted in ``joint_names`` order.
    """
    names = tuple(joint_names)
    if isinstance(payload, Mapping):
        unknown = sorted(set(payload) - set(names))
        if unknown:
            raise ValueError(
                f"unknown joint names {unknown}; this robot's actuated joints are "
                f"{list(names)}"
            )
        if defaults is None:
            missing = [n for n in names if n not in payload]
            if missing:
                raise ValueError(
                    f"missing joint values for {missing}; supply all {len(names)} "
                    "actuated joints or pass a seed/default configuration"
                )
            base = np.zeros(len(names), dtype=np.float64)
        else:
            base = np.asarray(defaults, dtype=np.float64).reshape(len(names)).copy()
        for i, name in enumerate(names):
            if name in payload:
                base[i] = float(payload[name])
        return base

    arr = np.asarray(payload, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(names):
        raise ValueError(
            f"positional joint arrays must have exactly {len(names)} entries (got "
            f"{arr.shape[0]}), ordered as {list(names)}; prefer a name-keyed object"
        )
    return arr


def path_from_payload(
    waypoints: Sequence[Any],
    joint_names: Sequence[str],
) -> np.ndarray:
    """``(n_waypoints, dof)`` from a list of name-keyed or positional waypoints."""
    if len(waypoints) == 0:
        raise ValueError("path has no waypoints")
    rows = [config_from_payload(wp, joint_names) for wp in waypoints]
    return np.stack(rows, axis=0)


def pose_payload(wxyz: Any, position: Any) -> dict[str, Any]:
    """Outbound pose, explicitly labelled."""
    return {
        "position": [float(v) for v in np.asarray(position).reshape(3)],
        "wxyz": [float(v) for v in np.asarray(wxyz).reshape(4)],
        "frame": "world",
        "quaternion_convention": QUATERNION_CONVENTION,
        "units": {"length": UNITS["length"]},
    }


def se3_from_payload(pose: Mapping[str, Any]) -> jaxlie.SE3:
    """Inbound pose → ``jaxlie.SE3``, rejecting a mislabelled quaternion.

    Accepts ``{"position": [x,y,z], "wxyz": [w,x,y,z]}``.  ``xyzw`` input is
    refused rather than silently reinterpreted — that mistake shows up as a
    plausible-looking rotation error, not as a crash.
    """
    convention = str(pose.get("quaternion_convention", QUATERNION_CONVENTION)).lower()
    if convention != QUATERNION_CONVENTION:
        raise ValueError(
            f"quaternion_convention={convention!r} is not supported; pyroffi is "
            f"{QUATERNION_CONVENTION!r} (scalar first). Reorder on your side and "
            "relabel, so the conversion is explicit."
        )
    if "position" not in pose:
        raise ValueError("pose requires a 'position' [x, y, z] in metres, world frame")
    position = np.asarray(pose["position"], dtype=np.float64).reshape(3)
    wxyz = np.asarray(
        pose.get("wxyz", (1.0, 0.0, 0.0, 0.0)), dtype=np.float64
    ).reshape(4)
    norm = float(np.linalg.norm(wxyz))
    if norm < 1e-8:
        raise ValueError("quaternion has ~zero norm")
    wxyz = wxyz / norm
    # No explicit dtype: JAX canonicalises float64 -> float32 itself when x64 is
    # off, and keeps the full precision when it is on. Hard-coding float32 here
    # threw away ~1e-7 m of the target under x64, which is the same order as the
    # IK tolerances the caller is being measured against.
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.asarray(wxyz)), jnp.asarray(position)
    )


def export_scene_primitives(scene, robot_name: str) -> dict[str, Any]:
    """Plain primitive list — the lowest-friction handoff to another server."""
    payload = scene.to_dict()
    payload["robot"] = robot_name
    payload["format"] = "primitives"
    return payload


def export_scene_urdf(scene, robot_name: str) -> str:
    """The obstacle set as a standalone URDF of fixed-jointed links.

    Emitted for planners that only ingest URDF. Half-spaces become large thin
    boxes, since URDF has no half-space primitive — stated here because the
    approximation is the sort of thing that otherwise bites silently.
    """
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{robot_name}_world">',
        '  <link name="world"/>',
    ]
    for obj in scene.objects():
        if obj.shape == "sphere":
            geom = f'<sphere radius="{float(obj.params["radius"]):.6f}"/>'
        elif obj.shape == "box":
            geom = (
                f'<box size="{float(obj.params["length"]):.6f} '
                f'{float(obj.params["width"]):.6f} '
                f'{float(obj.params["height"]):.6f}"/>'
            )
        elif obj.shape == "capsule":
            geom = (
                f'<cylinder radius="{float(obj.params["radius"]):.6f}" '
                f'length="{float(obj.params["height"]):.6f}"/>'
            )
        else:  # half-space → thin slab, 40 m square
            geom = '<box size="40 40 0.01"/>'
        rpy = _rpy_from_wxyz(obj.wxyz)
        origin = (
            f'<origin xyz="{obj.position[0]:.6f} {obj.position[1]:.6f} '
            f'{obj.position[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>'
        )
        lines += [
            f'  <link name="{obj.name}">',
            f"    <visual>{origin}<geometry>{geom}</geometry></visual>",
            f"    <collision>{origin}<geometry>{geom}</geometry></collision>",
            "  </link>",
            f'  <joint name="{obj.name}_fixed" type="fixed">',
            '    <parent link="world"/>',
            f'    <child link="{obj.name}"/>',
            "  </joint>",
        ]
    lines.append("</robot>")
    return "\n".join(lines)


def _rpy_from_wxyz(wxyz: np.ndarray) -> tuple[float, float, float]:
    """wxyz quaternion → URDF fixed-axis roll-pitch-yaw."""
    w, x, y, z = (float(v) for v in np.asarray(wxyz).reshape(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)
