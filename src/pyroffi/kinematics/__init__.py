"""Robot kinematics: forward/inverse kinematics as free functions.

The ``Robot`` class methods (``Robot.forward_kinematics`` etc.) delegate to
this module; both call styles are supported.
"""

from ._fk import (
    forward_kinematics as forward_kinematics,
    forward_kinematics_joints_jax as forward_kinematics_joints_jax,
    link_poses_from_joint_poses as link_poses_from_joint_poses,
)
from ._ik import inverse_kinematics as inverse_kinematics
