"""Execution sandbox: a stepped MuJoCo world an agent can act on.

Separate from :mod:`pyroffi.mcp` on purpose. That server computes and never
commands; this one commands a simulation. Keeping them in different packages
with different entry points is what makes that boundary structural rather than
a promise in a docstring.

    from pyroffi.sandbox import Sandbox

    sandbox = Sandbox(task, variant="wall")      # viser viewer starts with it
    sandbox.set_gripper("close")
    sandbox.execute_path(waypoints, times_s=times)
    sandbox.report()

Or over MCP::

    pyroffi-sandbox --task examples/tasks/block_stacking_panda.json --variant wall
"""

from ._sandbox import ExecutionRecord as ExecutionRecord
from ._sandbox import Sandbox as Sandbox
from ._scene import GRIPPER_CLOSE as GRIPPER_CLOSE
from ._scene import GRIPPER_OPEN as GRIPPER_OPEN
from ._scene import MENAGERIE_ARM_JOINTS as MENAGERIE_ARM_JOINTS
from ._scene import SceneHandles as SceneHandles
from ._scene import build_scene as build_scene
