import os as _os
import sys as _sys
from pathlib import Path as _Path

# cudals is the external/cudals submodule; PYROFFI_CUDALS_PATH overrides the checkout.
_cudals_src = str(_Path(_os.environ.get("PYROFFI_CUDALS_PATH", _Path(__file__).resolve().parents[2] / "external" / "cudals")) / "src")
if _cudals_src not in _sys.path:
    _sys.path.insert(0, _cudals_src)

from . import attachments as attachments
from . import bindings as bindings
from . import collision as collision
from . import costs as costs
from . import kinematics as kinematics
from . import motion_generators as motion_generators
from . import topp as topp
from . import viewer as viewer
from ._robot import Robot as Robot
from cudals import AnalyzedLeastSquaresProblem as AnalyzedLeastSquaresProblem
from cudals import AugmentedLagrangianConfig as AugmentedLagrangianConfig
from cudals import ConjugateGradientConfig as ConjugateGradientConfig
from cudals import Cost as Cost
from cudals import LeastSquaresProblem as LeastSquaresProblem
from cudals import SE2Var as SE2Var
from cudals import SE3Var as SE3Var
from cudals import SO2Var as SO2Var
from cudals import SO3Var as SO3Var
from cudals import SolveSummary as SolveSummary
from cudals import TerminationConfig as TerminationConfig
from cudals import TrustRegionConfig as TrustRegionConfig
from cudals import Var as Var
from cudals import VarValues as VarValues
from ._splines import linear_interpolate as linear_interpolate
from ._splines import cubic_spline_interpolate as cubic_spline_interpolate
from ._splines import bspline_interpolate as bspline_interpolate
from ._splines import make_spline_init_trajs as make_spline_init_trajs

__version__ = "0.2.0"
