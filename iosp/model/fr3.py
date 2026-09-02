"""The FR3 as iosp's forward-model robot: a 7-DOF variant of the spherized URDF.

The teleop demonstrations in `sim_teleop` were recorded on an FR3 (remu's
`default_fr3_mjcf`, Franka Hand attached), while every synthetic iosp result to
date used `resources/panda/panda_spherized.urdf`.  Those are different robots --
the FR3's link lengths and joint limits differ from the Panda's -- so fitting an
FR3 demonstration through a Panda forward model would charge the kinematic
mismatch to the cost weights, which is exactly the confound this experiment
exists to avoid.

`resources/fr3/fr3_spherized.urdf` is the right geometry but the wrong DOF
count: it exposes `fr3_finger_joint1` as an eighth actuated joint, and the
demonstrations are 7-vectors.  Rather than keep a hand-edited second copy of a
700-line URDF in sync with the original, this module DERIVES the 7-DOF model:
the two finger joints become `fixed` at their recorded pose and everything else
-- link geometry, inertias, collision spheres, joint limits -- is untouched.
The result is cached next to `iosp/data/` and regenerated whenever the source
URDF is newer, so editing the source cannot leave a stale derivative behind.

The fingers are welded, not deleted: they still carry collision spheres the
SRDF names, and a hand that plans as if it had no fingers would sweep them
through the table.  Welded at the OPEN width, because that is the wider swept
volume and therefore the conservative choice for a clearance cost.
"""

import pathlib
import xml.etree.ElementTree as ET

RESOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "resources"
FR3_DIR = RESOURCE_ROOT / "fr3"
SRC_URDF = FR3_DIR / "fr3_spherized.urdf"
SRDF_PATH = FR3_DIR / "fr3_spherized.srdf"
MESH_DIR = FR3_DIR / "meshes"
CACHE_URDF = pathlib.Path(__file__).resolve().parent.parent / "data" / "fr3_7dof.urdf"

EE_LINK = "fr3_hand"
FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
FINGER_OPEN_M = 0.04  # per-finger travel at the Franka Hand's 0.08 m opening


def build_7dof_urdf(path=CACHE_URDF, finger_open=FINGER_OPEN_M):
    """Write (or refresh) the 7-DOF FR3 and return its path."""
    path = pathlib.Path(path)
    if path.exists() and path.stat().st_mtime >= SRC_URDF.stat().st_mtime:
        return path

    tree = ET.parse(SRC_URDF)
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") not in FINGER_JOINTS:
            continue
        joint.set("type", "fixed")
        # A fixed joint has no axis/limit/mimic, and yourdfpy rejects a mimic
        # that points at a joint no longer actuated.  The finger's opening is
        # folded into the origin instead, along the axis it used to travel.
        axis = joint.find("axis")
        offset = [float(v) for v in (axis.get("xyz").split() if axis is not None
                                     else (0, 0, 0))]
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", rpy="0 0 0", xyz="0 0 0")
        base = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        origin.set("xyz", " ".join(f"{b + finger_open * a:.6g}"
                                   for b, a in zip(base, offset)))
        for tag in ("axis", "limit", "mimic", "dynamics", "safety_controller"):
            for child in joint.findall(tag):
                joint.remove(child)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def paths():
    """`(urdf, srdf, mesh_dir, ee_link)` for `PickPlaceProblem.load`."""
    return str(build_7dof_urdf()), str(SRDF_PATH), str(MESH_DIR), EE_LINK
