"""Visualization for pyroffi, built on viser.

The render layer is deliberately split into a **source** (where the world state
comes from) and a **view** (how it is drawn). A MuJoCo simulation, a planned
pyroffi configuration, and a perception stack watching a physical cell all
reduce to a :class:`WorldState`, so moving from simulation to hardware changes
the source and nothing else — and two renders from different sources are
directly comparable because the same code drew them.

    from pyroffi.viewer import RenderViewer, MuJoCoSource

    viewer = RenderViewer(MuJoCoSource(model, data, joint_names, bodies, geom))
    viewer.start()
    viewer.wait_for_client()          # capture renders through the browser
    png = viewer.capture_png("top")

Capture is viser or nothing: :meth:`RenderViewer.capture` raises
:class:`NoViewerClient` when no browser is connected rather than quietly
substituting a different renderer.
"""

# from ._batched_urdf import BatchedURDF as BatchedURDF
from ._manipulability_ellipse import ManipulabilityEllipse as ManipulabilityEllipse
from ._render_viewer import DEFAULT_VIEWPOINTS as DEFAULT_VIEWPOINTS
from ._render_viewer import NoViewerClient as NoViewerClient
from ._render_viewer import RenderViewer as RenderViewer
from ._render_viewer import Viewpoint as Viewpoint
from ._render_viewer import ee_path_positions as ee_path_positions
from ._scene_view import SceneView as SceneView
from ._weight_tuner import WeightTuner as WeightTuner
from ._world import CallableSource as CallableSource
from ._world import MuJoCoSource as MuJoCoSource
from ._world import ObjectGeometry as ObjectGeometry
from ._world import Pose as Pose
from ._world import ToolboxSource as ToolboxSource
from ._world import WorldDescription as WorldDescription
from ._world import WorldSource as WorldSource
from ._world import WorldState as WorldState
