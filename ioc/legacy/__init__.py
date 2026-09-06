"""Legacy / reference baselines kept out of the main IOC path.

`pdp_oc` is a standalone, faithful port of Pontryagin Differentiable Programming
(Jin et al. 2020; Safe-PDP). It is retained as a reference baseline, not a
component of the IOC pipeline.

Conclusion it documents: the IOC implicit-diff gradient functionally rederives
PMP/PDP when the inner solver is SCO + augmented Lagrangian -- same optimality
condition, machine-identical gradients (cos = 1.0) on a shared OCP. PDP's
Riccati recursion is O(T) where the dense implicit KKT solve is O((T*m)^3); the
higher time complexity is the price paid so the inner cost can be arbitrary and
generalize to IOSP, where no control model / PMP structure exists.
"""

from ioc.legacy import pdp_oc

__all__ = ["pdp_oc"]
