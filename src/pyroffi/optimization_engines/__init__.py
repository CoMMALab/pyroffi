"""IK and trajectory optimisation engines.

Collision guarantees differ by CUDA IK solver
---------------------------------------------

The four CUDA IK solvers do NOT offer the same collision guarantee, and picking
one for speed can silently give up the constraint. Both self-collision and world
obstacles behave as described below.

===========  ==========================  ==================================
solver       guarantee                   depends on ``collision_weight``?
===========  ==========================  ==================================
``sqp_ik``   hard constraint             no
``ls_ik``    soft penalty, correct grad  yes
``hjcd_ik``  soft penalty, merit only    yes, strongly
``mppi_ik``  soft penalty, merit only    yes, strongly
===========  ==========================  ==================================

**sqp_ik** solves a genuinely constrained QP: linearised rows
``grad(d)^T p >= margin - d`` enter the subproblem (ADMM, OSQP's indirect form),
and acceptance is lexicographic on feasibility before pose error. It returns
collision-free configurations even at a negligible ``collision_weight``.

**ls_ik** has correct collision gradients -- the term reaches the normal
equations, so the solver steps *away* from a collision rather than merely
declining to step further in -- but Levenberg-Marquardt has no constraint
mechanism. Feasibility is bought with weight. The shipped default (1e4) clears
self-collision on the Panda; a much smaller weight will not.

**hjcd_ik** and **mppi_ik** carry the penalty in the merit function only: it
ranks candidates and rejects steps but never enters a step direction. They need
the largest weights and offer the weakest guarantee. ``hjcd_ik`` does well in
practice because its penalty ranks coarse-phase seeds, which is a strong filter
across many seeds -- but that is selection, not constraint satisfaction.

Self-collision activates automatically when ``collision_checker`` is a
``RobotCollisionSpherized``, which MUST have been built with an SRDF; without
one the model treats adjacent links as permanently overlapping and every
configuration is rejected.

See ``tests/test_collision_constraints.py``, whose low-``collision_weight`` case
is what distinguishes the hard row of this table from the soft ones.
"""

from ._hjcd_ik import hjcd_solve as hjcd_solve
from ._quik_ik import (
    QuIKSolver as QuIKSolver,
    quik_ik_solve as quik_ik_solve,
)
from ._halley_ik import (
    HalleyJAXSolver as HalleyJAXSolver,
    halley_ik_solve as halley_ik_solve,
)
from ._ls_ik import ls_ik_solve as ls_ik_solve
from ._ls_ik import ls_ik_solve_cuda as ls_ik_solve_cuda
from ._sqp_ik import sqp_ik_solve as sqp_ik_solve
from ._sqp_ik import sqp_ik_solve_cuda as sqp_ik_solve_cuda
from ._mppi_ik import mppi_ik_solve as mppi_ik_solve
from ._mppi_ik import mppi_ik_solve_cuda as mppi_ik_solve_cuda
from ._region_ik import brownian_motion_sample_box_region_cuda as brownian_motion_sample_box_region_cuda
from ._region_ik import svgd_sample_box_region_cuda as svgd_sample_box_region_cuda
from ._region_ik import hit_and_run_sample_box_region_cuda as hit_and_run_sample_box_region_cuda
from ._region_ik import direct_sample_box_region_cuda as direct_sample_box_region_cuda
from ._region_ik_jax import direct_sample_box_region_jax as direct_sample_box_region_jax
from ._region_ik_jax import svgd_sample_box_region_jax as svgd_sample_box_region_jax
from ._learned_ik import (
    IKFlowNet as IKFlowNet,
    encode_pose as encode_pose,
    make_learned_ik_solve as make_learned_ik_solve,
    save_learned_ik as save_learned_ik,
    load_learned_ik as load_learned_ik,
    get_default_model_path as get_default_model_path,
)
from ._sco_optimization import ScoTrajOptConfig as ScoTrajOptConfig
from ._sco_optimization import TrajOptConfig as TrajOptConfig
from ._sco_optimization import sco_trajopt as sco_trajopt
from ._sco_optimization import make_init_trajs as make_init_trajs
from ._contact_trajopt import ContactTrajOptConfig as ContactTrajOptConfig
from ._contact_trajopt import contact_sco_trajopt as contact_sco_trajopt
from ._flat_contact_trajopt import (
    FlatContactTrajOptConfig as FlatContactTrajOptConfig,
)
from ._flat_contact_trajopt import flat_contact_trajopt as flat_contact_trajopt
from ._contact_rich_trajopt import (
    ContactRichTrajOptConfig as ContactRichTrajOptConfig,
)
from ._contact_rich_trajopt import contact_rich_trajopt as contact_rich_trajopt
from ._chomp_optimization import ChompTrajOptConfig as ChompTrajOptConfig
from ._chomp_optimization import chomp_trajopt as chomp_trajopt
from ._stomp_optimization import StompTrajOptConfig as StompTrajOptConfig
from ._stomp_optimization import stomp_trajopt as stomp_trajopt
from ._ls_trajopt_optimization import LsTrajOptConfig as LsTrajOptConfig
from ._ls_trajopt_optimization import ls_trajopt as ls_trajopt
from ._lbfgs_trajopt_optimization import LbfgsTrajOptConfig as LbfgsTrajOptConfig
from ._lbfgs_trajopt_optimization import lbfgs_trajopt as lbfgs_trajopt
from ..kinematics._analytic_ik import (
    analytic_ik_solve as analytic_ik_solve,
    analytic_ik_solve_batched as analytic_ik_solve_batched,
    build_geometry as build_analytic_ik_geometry,
)
