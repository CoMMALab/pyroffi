"""Fitting machinery, shared by every experiment.

`parametric`  build the path-A bilevel forward map (known cost library)
`procedure`   the five-stage fit-wide/Gram/select/refit/report driver
`params`      the `u -> theta` parameterization and its softmax gauge
`multistart`  many candidates (IK branch x cost start) as one batched program
"""
