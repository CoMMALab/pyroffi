"""Synthetic low-DOF URDFs so bench2d's toy benchmarks can be driven through GRiD.

GRiD has no analytic low-DOF shortcut -- ``GRiDDynamics`` compiles CUDA kernels
from a real URDF's kinematic/dynamic tree.  "Reduced dimensionality" therefore
means a synthetic URDF whose joint DOF matches a benchmark's state dimension
exactly, so the state *is* the joint vector and no separate FK is needed to
relate them:

- d=2 (``racing``, ``field``, ``segments``): a 2-DOF Cartesian point mass --
  prismatic-x + prismatic-y, nominal point mass/inertia.
- d=3 (``unicycle``): a 3-DOF planar mobile base -- prismatic-x + prismatic-y +
  revolute-yaw, nominal mass + yaw inertia (mirrors the ``base_xy_yaw``
  convention ``ManipulatorSpec`` already uses for planar bases).
"""

import dataclasses

import pyroffi as pk

# Nominal inertial parameters: a small point mass, negligible link inertia,
# and just enough yaw inertia for the 3-DOF base's revolute joint to be
# numerically well-posed.
_MASS = 1.0
_EPS_I = 1e-4
_YAW_I = 0.05

_URDF_2D = """<?xml version="1.0"?>
<robot name="bench2d_point_mass">
  <link name="world"/>
  <link name="link_x">
    <inertial>
      <mass value="{eps}"/>
      <inertia ixx="{eps}" ixy="0" ixz="0" iyy="{eps}" iyz="0" izz="{eps}"/>
    </inertial>
  </link>
  <link name="link_y">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{eps}" ixy="0" ixz="0" iyy="{eps}" iyz="0" izz="{eps}"/>
    </inertial>
  </link>
  <joint name="joint_x" type="prismatic">
    <parent link="world"/>
    <child link="link_x"/>
    <axis xyz="1 0 0"/>
    <limit lower="-100" upper="100" effort="1000" velocity="100"/>
  </joint>
  <joint name="joint_y" type="prismatic">
    <parent link="link_x"/>
    <child link="link_y"/>
    <axis xyz="0 1 0"/>
    <limit lower="-100" upper="100" effort="1000" velocity="100"/>
  </joint>
</robot>
""".format(mass=_MASS, eps=_EPS_I)

_URDF_3D = """<?xml version="1.0"?>
<robot name="bench2d_planar_base">
  <link name="world"/>
  <link name="link_x">
    <inertial>
      <mass value="{eps}"/>
      <inertia ixx="{eps}" ixy="0" ixz="0" iyy="{eps}" iyz="0" izz="{eps}"/>
    </inertial>
  </link>
  <link name="link_y">
    <inertial>
      <mass value="{eps}"/>
      <inertia ixx="{eps}" ixy="0" ixz="0" iyy="{eps}" iyz="0" izz="{eps}"/>
    </inertial>
  </link>
  <link name="link_yaw">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{eps}" ixy="0" ixz="0" iyy="{eps}" iyz="0" izz="{yaw_i}"/>
    </inertial>
  </link>
  <joint name="joint_x" type="prismatic">
    <parent link="world"/>
    <child link="link_x"/>
    <axis xyz="1 0 0"/>
    <limit lower="-100" upper="100" effort="1000" velocity="100"/>
  </joint>
  <joint name="joint_y" type="prismatic">
    <parent link="link_x"/>
    <child link="link_y"/>
    <axis xyz="0 1 0"/>
    <limit lower="-100" upper="100" effort="1000" velocity="100"/>
  </joint>
  <joint name="joint_yaw" type="revolute">
    <parent link="link_y"/>
    <child link="link_yaw"/>
    <axis xyz="0 0 1"/>
    <limit lower="-100" upper="100" effort="1000" velocity="100"/>
  </joint>
</robot>
""".format(mass=_MASS, eps=_EPS_I, yaw_i=_YAW_I)


@dataclasses.dataclass(frozen=True)
class Robot2DProblem:
    """Same surface `ioc.robot.problem.RobotProblem` gives the robot pipeline.

    State-space and joint-space coincide by construction here, so `unpack` is
    literally `ioc.bench2d.problems.unpack`, imported unchanged by callers.
    """

    robot: object
    grid: object
    dof: int

    @staticmethod
    def load(d):
        """Build the d=2 point-mass or d=3 planar-base synthetic robot."""
        import io

        import yourdfpy

        if d == 2:
            xml = _URDF_2D
        elif d == 3:
            xml = _URDF_3D
        else:
            raise ValueError(f"no synthetic robot for d={d}")
        urdf = yourdfpy.URDF.load(io.BytesIO(xml.encode()), mesh_dir="")
        robot = pk.Robot.from_urdf(urdf)
        grid = pk.dynamics.GRiDDynamics(urdf)
        return Robot2DProblem(robot=robot, grid=grid, dof=d)
