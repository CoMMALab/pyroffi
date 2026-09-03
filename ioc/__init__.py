"""Inverse optimal control through a differentiable trajectory optimizer.

Every experiment in this package instantiates the same bilevel problem.  A
demonstration set {x~_i, c_i} is assumed to be (noisy) optimal under an unknown
cost that is linear in a known feature basis,

    inner:  x*(theta, c) = argmin_x  J_theta(x, c),  J_theta = sum_k theta_k phi_k(x, c)
    outer:  min_theta    L(theta) = (1/M) sum_i  l( x*(theta, c_i),  x~_i )

with theta constrained to the simplex via theta = softmax(z).  The features are
written as residual vectors r_k with phi_k = ||r_k||^2, which makes the inner
problem a weighted nonlinear least-squares problem: its Gauss-Newton Hessian
sum_k theta_k J_k^T J_k is PSD by construction, so x*(theta) is a well-defined
and locally smooth function of theta.  That smoothness is the precondition for
everything else here -- with an indefinite Hessian the solver stalls in
negative-curvature regions and its iterate jumps discontinuously under
infinitesimal changes of theta, which invalidates both the implicit function
theorem and finite differences.

Modules
-------
`inner`     the inner solver and its two differentiation paths (implicit
            adjoint, truncated unrolling)
`outer`     optimizers over z that only need L and dL/dz (Adam, finite
            differences, CMA-ES)
`analytic`  baselines that never solve the inner problem at all (Inverse KKT,
            CIOC, EIV-TLS)
`metrics`   scoring of a recovered theta against ground truth
`robot`     7-DoF Panda experiments (E1 identifiability, E2 cost dimension,
            E3 dynamics)
`bench2d`   cheap, drawable 2D benchmarks (racing, reward field, unicycle,
            time-segmented quadratics)
`analysis`  aggregation and quick-look plots per experiment

`collect.py` reproduces `data/`, `plots.py` renders `figures/`; see README.md.
"""

from ioc.analytic import cioc_fit, eiv_fit, kkt_fit
from ioc.inner import InnerSolver, make_inner_solver
from ioc.metrics import cosine, simplex_metrics
from ioc.outer import adam, adam_scan, cma_es, fd_grad_fn

__all__ = [
    "InnerSolver",
    "adam",
    "adam_scan",
    "cioc_fit",
    "cma_es",
    "eiv_fit",
    "cosine",
    "fd_grad_fn",
    "kkt_fit",
    "make_inner_solver",
    "simplex_metrics",
]
