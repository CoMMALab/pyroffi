"""IOSP -- inverse optimal control on a composed pick-and-place task.

Generalizes `ioc` (IOC on ONE trajopt segment) to a task skeleton of chained
segments (IK -> approach -> grasp -> transport -> place), so a single rollout of
a whole pick-and-place can be inverted for per-segment cost weights.  All
differentiation -- implicit adjoint, FD, CMA-ES, subspace refit -- is reused
unmodified from `ioc.inner` / `ioc.outer` / `ioc.identifiability`.

    config      every path, task constant, ground-truth weight, solver default
    model       the composed forward model (`pickplace`) and its `scenes`
    fit         the bilevel machinery: `parametric`, `procedure`, `params`,
                `multistart`
    experiments one module per experiment, runnable with `python -m`
    checks      diagnostics that gate the experiments
    record      run a fit, save every outer step to .npz
    viz         renderers reading those .npz files
    analysis    aggregate multi-run sweeps into tables
    shelved     deliberately out of scope, kept for the negative results

See README.md for what each experiment asks and how to reproduce it, and
`iosp.model.pickplace`'s docstring for the segment-composition design decision.
"""
