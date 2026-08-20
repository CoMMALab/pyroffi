"""IOSP -- Inverse (Optimal) Sequential Planning.

Generalizes `ioc` (IOC on one trajopt segment) to a fixed, hardcoded task
skeleton of chained segments (approach -> grasp -> transport -> place), so a
single human teleop rollout through a whole pick-and-place task can be
inverted for per-segment cost weights.

See `iosp.pickplace` for the composed planner and its docstring for the
segment-composition design decision.  All differentiation (implicit adjoint,
FD, CMA-ES) is reused unmodified from `ioc.inner` / `ioc.outer` -- see
`iosp.pickplace`'s module docstring for why that reuse works cleanly.
"""
