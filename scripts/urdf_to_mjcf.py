"""Convert a spherized URDF (real inertial data + sphere collision geometry) to MJCF.

Spherized URDFs in resources/ replace mesh collision geometry with fitted collision
spheres. This script walks the URDF kinematic tree with yourdfpy and emits an
equivalent MJCF XML: one <body> per link (pos/quat = parent joint origin), one
<joint> per non-fixed URDF joint (hinge/slide), an <inertial> from the URDF's
<mass>/<inertia>, and one <geom type="sphere"> per URDF collision sphere.

Every link must carry a complete URDF <inertial> element (mass + full 3x3 inertia).
This is a hard requirement, not a fallback: unlike MuJoCo's own URDF importer, this
script never invents mass/inertia for a link, since the whole point is to source
real system-identified dynamics parameters. A link with a missing <inertial>, or a
<collision> geometry that isn't a <sphere>, raises ValueError.

Usage:
    python scripts/urdf_to_mjcf.py path/to/robot_spherized.urdf [out.xml]
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yourdfpy


def _quat_wxyz_from_matrix(mat: np.ndarray) -> np.ndarray:
    R = mat[:3, :3]
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _fmt(vals) -> str:
    return " ".join(f"{v:.10g}" for v in np.asarray(vals).reshape(-1))


def _check_inertial(urdf: yourdfpy.URDF) -> None:
    missing = []
    bad_collisions = []
    for name, link in urdf.link_map.items():
        if link.inertial is None or link.inertial.mass is None or link.inertial.inertia is None:
            missing.append(name)
        for coll in link.collisions:
            if coll.geometry.sphere is None:
                bad_collisions.append((name, coll.name))
    if missing:
        raise ValueError(
            f"URDF '{urdf.robot.name}' is missing <inertial> mass/inertia data for link(s): "
            f"{missing}. This converter requires real per-link mass and inertia "
            "(system-identified or CAD-derived), not an inferred/default value — "
            "add <inertial> to these links in the URDF before converting."
        )
    if bad_collisions:
        raise ValueError(
            f"URDF '{urdf.robot.name}' has non-sphere <collision> geometry on link(s)/"
            f"collision(s): {bad_collisions}. This converter only supports spherized "
            "URDFs (sphere-only collision geometry)."
        )


def _build_body(
    parent_elem: ET.Element,
    link_name: str,
    urdf: yourdfpy.URDF,
    children_of: dict,
    origin: np.ndarray,
) -> None:
    link = urdf.link_map[link_name]
    body = ET.SubElement(parent_elem, "body", name=link_name)
    body.set("pos", _fmt(origin[:3, 3]))
    body.set("quat", _fmt(_quat_wxyz_from_matrix(origin)))

    # MuJoCo's <inertial> forbids specifying both `quat` and `fullinertia` (the
    # inertia matrix must be expressed directly in the body frame). Rotate the
    # URDF inertia tensor from the <inertial> origin's frame into the body frame
    # instead of passing that rotation along as a MJCF quat.
    inertial = link.inertial
    inert_elem = ET.SubElement(body, "inertial")
    inert_elem.set("pos", _fmt(inertial.origin[:3, 3]))
    inert_elem.set("mass", f"{inertial.mass:.10g}")
    R = inertial.origin[:3, :3]
    I = R @ inertial.inertia @ R.T
    inert_elem.set(
        "fullinertia", _fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    )

    for i, coll in enumerate(link.collisions):
        sphere = coll.geometry.sphere
        geom = ET.SubElement(body, "geom", name=f"{link_name}_col{i}", type="sphere")
        geom.set("size", f"{sphere.radius:.10g}")
        geom.set("pos", _fmt(coll.origin[:3, 3]))
        geom.set("quat", _fmt(_quat_wxyz_from_matrix(coll.origin)))
        geom.set("group", "3")
        geom.set("rgba", "0.8 0.3 0.3 0.5")

    for joint in children_of.get(link_name, []):
        if joint.type != "fixed":
            j = ET.SubElement(body, "joint", name=joint.name)
            j.set("type", {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}[joint.type])
            j.set("axis", _fmt(joint.axis if joint.axis is not None else [0, 0, 1]))
            if joint.type == "continuous":
                j.set("limited", "false")
            elif joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
                j.set("limited", "true")
                j.set("range", f"{joint.limit.lower:.10g} {joint.limit.upper:.10g}")
        _build_body(body, joint.child, urdf, children_of, joint.origin)


def urdf_to_mjcf(urdf_path: Path) -> ET.ElementTree:
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=False)
    _check_inertial(urdf)

    children_of: dict = {}
    for joint in urdf.joint_map.values():
        children_of.setdefault(joint.parent, []).append(joint)

    child_links = {j.child for j in urdf.joint_map.values()}
    roots = [name for name in urdf.link_map if name not in child_links]
    if len(roots) != 1:
        raise ValueError(
            f"URDF '{urdf.robot.name}' has {len(roots)} root link(s) {roots}; expected exactly one."
        )
    root_name = roots[0]

    mujoco_elem = ET.Element("mujoco", model=urdf.robot.name or urdf_path.stem)
    ET.SubElement(mujoco_elem, "compiler", angle="radian", autolimits="true")
    worldbody = ET.SubElement(mujoco_elem, "worldbody")
    _build_body(worldbody, root_name, urdf, children_of, np.eye(4))

    tree = ET.ElementTree(mujoco_elem)
    ET.indent(tree, space="  ")
    return tree


def convert(urdf_path: Path, out_path: Path | None = None) -> Path:
    tree = urdf_to_mjcf(urdf_path)
    if out_path is None:
        out_path = urdf_path.with_suffix(".xml")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("out", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    out_path = convert(args.urdf, args.out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
