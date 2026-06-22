"""Process-local shim: make ``yourdfpy==0.0.58`` work under numpy >= 2.

yourdfpy 0.0.58 computes joint FK with ``float(q)`` where ``q`` can be a size-1
array.  numpy >= 2 makes ``float(<size-1 array>)`` an error, so loading any URDF
with revolute/continuous joints raises ``TypeError: only 0-dimensional arrays
can be converted to Python scalars``.

This re-binds ``URDF._forward_kinematics_joint`` with an identical implementation
that coerces ``q`` to a scalar safely.  It does NOT modify any installed files
and is a no-op once yourdfpy/numpy are mutually compatible.

Call :func:`apply` once before loading URDFs.
"""

from __future__ import annotations


def apply() -> bool:
    """Install the shim if needed.  Returns True if it was applied."""
    import numpy as np
    import trimesh.transformations as tra
    from yourdfpy.urdf import URDF

    # Already compatible? (numpy < 2 accepts float() on size-1 arrays.)
    try:
        float(np.zeros(1))
        return False
    except TypeError:
        pass

    def _forward_kinematics_joint(self, joint, q=None):
        origin = np.eye(4) if joint.origin is None else joint.origin

        if joint.mimic is not None:
            if joint.mimic.joint in self.actuated_joint_names:
                mimic_joint_index = self.actuated_joint_names.index(joint.mimic.joint)
                q = (
                    self._cfg[mimic_joint_index] * joint.mimic.multiplier
                    + joint.mimic.offset
                )
            else:
                q = 0.0 + joint.mimic.offset

        if joint.type in ["revolute", "prismatic", "continuous"]:
            if q is None:
                q = self.cfg[
                    self.actuated_dof_indices[
                        self.actuated_joint_names.index(joint.name)
                    ]
                ]
            if joint.type == "prismatic":
                matrix = origin @ tra.translation_matrix(q * joint.axis)
            else:
                angle = float(np.asarray(q).reshape(-1)[0])
                matrix = origin @ tra.rotation_matrix(angle, joint.axis)
        else:
            matrix = origin

        return matrix, q

    URDF._forward_kinematics_joint = _forward_kinematics_joint
    return True
