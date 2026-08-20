"""Cheap, interpretable, drawable 2D inverse-optimal-control benchmarks.

The robot experiments cost seconds per forward solve, which caps how far the
baselines can be pushed.  These 2D systems cost milliseconds, so the same
comparison runs to convergence, at matched solve budgets, and at cost dimensions
large enough to actually stress derivative-free search.  They are also drawable,
so a recovered cost can be inspected rather than only scored.

`problems`  the four benchmarks, their contexts and feature whitening
`run`       the driver that fits every method under a matched solve budget
"""

from ioc.bench2d.problems import BENCHMARKS, Ctx

__all__ = ["BENCHMARKS", "Ctx"]
