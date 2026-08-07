"""Session/toolbox layer: pyroffi as a composable toolbox of motion primitives.

This is the substance underneath the MCP adapter, and it is transport-agnostic
by construction — nothing here imports ``mcp``. It owns exactly what a stateless
library call cannot: the persistent scene, the warm jitted caches, the handle
table, shape bucketing, time parameterisation, and honest reporting of what
happened.

It deliberately does **not** own planning policy. No automatic retry escalation,
no silent solver substitution, no cost-acceptance heuristic — the orchestrator
makes those calls, and the primitives just report, including partial success.

Typical use::

    from pyroffi.toolbox import Session, Toolbox, configure_process

    configure_process(gpu=1, x64=True)      # before JAX initialises
    tb = Toolbox(Session(robot="panda", max_objects=8, n_timesteps=64))
    tb.add_object("shelf", "box", position=(0.5, 0.0, 0.3),
                  params={"length": 0.4, "width": 0.1, "height": 0.6})
    tb.warmup()                              # pay the compile cost on purpose
    tb.solve_ik(pose={"position": [0.4, 0.2, 0.4], "wxyz": [0, 0, 1, 0]})
"""

from ._exchange import QUATERNION_CONVENTION as QUATERNION_CONVENTION
from ._exchange import config_from_payload as config_from_payload
from ._exchange import export_scene_urdf as export_scene_urdf
from ._exchange import joint_dict as joint_dict
from ._exchange import path_from_payload as path_from_payload
from ._exchange import pose_payload as pose_payload
from ._exchange import se3_from_payload as se3_from_payload
from ._handles import Entry as Entry
from ._handles import HandleTable as HandleTable
from ._primitives import Toolbox as Toolbox
from ._retiming import RetimingResult as RetimingResult
from ._retiming import default_acceleration_limits as default_acceleration_limits
from ._retiming import retime_path as retime_path
from ._scene import SHAPES as SHAPES
from ._scene import Scene as Scene
from ._scene import SceneObject as SceneObject
from ._session import DEFAULT_PATH_BUCKETS as DEFAULT_PATH_BUCKETS
from ._session import CompileLedger as CompileLedger
from ._session import Session as Session
from ._session import bucket_length as bucket_length
from ._session import configure_process as configure_process
from ._session import load_urdf as load_urdf
from ._session import pad_path as pad_path
