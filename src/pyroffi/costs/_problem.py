"""The jaxls-shaped problem API: variables, values, costs and least-squares problems.

Names, signatures and semantics follow jaxls, so code written against jaxls ports by
replacing ``jaxls.X`` with ``pk.X``. The solver does not: ``analyze()`` compiles the
problem to one CUDA kernel (see :mod:`._compile`), and ``solve()`` is a single launch.
There is no JAX fallback; a problem that cannot be compiled, or a machine without a
CUDA device and nvcc, raises.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, ClassVar, Generic, Iterable, Literal, TypeVar

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

T = TypeVar("T")

CostKind = Literal["l2_squared", "constraint_eq_zero", "constraint_leq_zero",
                   "constraint_geq_zero"]


class Var(Generic[T]):
    """A symbolic optimization variable, identified by its type and integer ``id``.

    Subclass with a default value, as in jaxls::

        class JointVar(pk.Var[jax.Array], default_factory=lambda: jnp.zeros(7)): ...

    ``id`` may be an int or an integer array (a batch of variables of this type).
    """

    _default_factory: ClassVar[Callable[[], Any]]
    tangent_dim: ClassVar[int]
    retract_fn: ClassVar[Callable[[Any, jax.Array], Any] | None] = None
    """``None`` for Euclidean variables; a manifold retraction otherwise."""

    def __init_subclass__(cls, default_factory: Callable[[], Any] | None = None,
                          tangent_dim: int | None = None,
                          retract_fn: Callable[[Any, jax.Array], Any] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if retract_fn is None and getattr(cls, "retract_fn", None) is not None:
            # A subclass of a Lie-group variable (e.g. a new default) keeps its manifold.
            retract_fn, tangent_dim = cls.retract_fn, cls.tangent_dim
        if default_factory is not None:
            cls._default_factory = staticmethod(default_factory)
            if retract_fn is None:
                cls.retract_fn = None
                cls.tangent_dim = sum(int(np.prod(s.shape)) for s in
                                      jax.tree.leaves(jax.eval_shape(default_factory)))
            else:
                cls.retract_fn = staticmethod(retract_fn)
                cls.tangent_dim = int(tangent_dim)
        jax.tree_util.register_pytree_node(
            cls, lambda v: ((v.id,), None), lambda _, ch: cls(ch[0]))

    def __init__(self, id: int | jax.Array):
        self.id = id

    @classmethod
    def default_factory(cls) -> T:
        return cls._default_factory()

    def with_value(self, value: T) -> "VarWithValue[T]":
        """Pair with a value, for :meth:`VarValues.make`."""
        return VarWithValue(self, value)

    def __getitem__(self, index_or_slice):
        return type(self)(self.id[index_or_slice])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"


@dataclasses.dataclass(frozen=True)
class VarWithValue(Generic[T]):
    variable: Var[T]
    value: T


def _concrete(x) -> np.ndarray:
    """The value of ``x``, including a tracer computed only from constants inside jax.jit.

    Variable ids decide the kernel's structure, so they must be known when compiling, but
    jaxls-style code builds them inside jit (``JointVar(jnp.arange(T))``). Such a tracer's
    value is recoverable: replay the equations of the enclosing trace that produce it. One
    that depends on the jitted function's inputs raises.
    """
    if not isinstance(x, jax.core.Tracer):
        return np.asarray(x)
    from jax._src.core import Literal, Var as JVar

    frame = getattr(getattr(x, "_trace", None), "frame", None)
    var = getattr(x, "val", None)
    if frame is None or var is None:
        raise ValueError("variable ids must be concrete (or computed from constants inside jit)")
    if isinstance(var, Literal):
        return np.asarray(var.val)
    needed, chain = {var}, []
    for eqn in reversed(frame.get_eqns()):
        if needed.intersection(eqn.outvars):
            chain.append(eqn)
            needed.difference_update(eqn.outvars)
            needed.update(v for v in eqn.invars if type(v) is JVar)
    consts = frame.constvar_to_val
    if any(v not in consts for v in needed):
        raise ValueError("variable ids depend on the inputs of a jitted function; ids decide "
                         "the compiled problem's structure and must be known at trace time")
    env = {v: consts[v] for v in needed}
    with jax.ensure_compile_time_eval():
        for eqn in reversed(chain):
            args = [v.val if isinstance(v, Literal) else env[v] for v in eqn.invars]
            outs = eqn.primitive.bind(*args, **eqn.params)
            for v, o in zip(eqn.outvars, outs if eqn.primitive.multiple_results else [outs]):
                env[v] = o
    return np.asarray(env[var])


def _ids(var: Var) -> np.ndarray:
    return np.atleast_1d(_concrete(var.id)).ravel()


class VarValues:
    """A mapping from variables to values: ``vals[var]`` or ``vals[VarType]`` (stacked by id)."""

    def __init__(self, values: dict[tuple[type, int], Any]):
        self._values = values

    @staticmethod
    def make(variables: Iterable[Var | VarWithValue]) -> "VarValues":
        """Values for ``variables``: the given ones, or each variable type's default."""
        values: dict[tuple[type, int], Any] = {}
        for item in variables:
            var, value = (item.variable, item.value) if isinstance(item, VarWithValue) else (item, None)
            ids = _ids(var)
            batched = np.ndim(var.id) > 0  # works on tracers (static shape)
            for k, vid in enumerate(ids):
                if value is None:
                    v = type(var).default_factory()
                else:
                    v = jax.tree.map(lambda a: a[k], value) if batched else value
                values[(type(var), int(vid))] = v
        return VarValues(values)

    def get_value(self, var: Var[T]) -> T:
        if np.ndim(var.id) > 0:
            vals = [self._values[(type(var), int(i))] for i in _ids(var)]
            stacked = jax.tree.map(lambda *a: jnp.stack(a), *vals)
            return jax.tree.map(lambda a: a.reshape(np.shape(var.id) + a.shape[1:]), stacked)
        return self._values[(type(var), int(_concrete(var.id)))]

    def get_stacked_value(self, var_type: type[Var[T]]) -> T:
        keys = sorted(k for k in self._values if k[0] is var_type)
        return jax.tree.map(lambda *a: jnp.stack(a), *[self._values[k] for k in keys])

    def __getitem__(self, var_or_type):
        if isinstance(var_or_type, type):
            return self.get_stacked_value(var_or_type)
        return self.get_value(var_or_type)

    def __repr__(self) -> str:
        return "VarValues(" + ", ".join(f"{t.__name__}({i})" for t, i in self._values) + ")"


jax.tree_util.register_pytree_node(
    VarValues,
    lambda v: (list(v._values.values()), list(v._values.keys())),
    lambda keys, vals: VarValues(dict(zip(keys, vals))))


@dataclasses.dataclass(frozen=True)
class Cost:
    """A cost or constraint term; build with :meth:`Cost.factory`, as in jaxls.

    ``kind``: ``"l2_squared"`` (minimise ``||r||^2``), or a constraint ``r == 0``,
    ``r <= 0`` or ``r >= 0``. A leading batch axis on the arguments (and on the
    variables' ids) makes a batch of costs.
    """

    compute_residual: Callable
    args: tuple
    kind: CostKind = "l2_squared"
    jac_mode: Literal["auto", "forward", "reverse"] = "auto"
    jac_batch_size: int | None = None
    jac_custom_fn: Callable | None = None
    jac_custom_with_cache_fn: Callable | None = None
    name: str | None = None
    # A NativeResidual (costs/_native.py): rows and Jacobian from device code; the traced
    # residual only reserves the rows.
    native: Any = None

    @staticmethod
    def factory(compute_residual: Callable | None = None, *, kind: CostKind = "l2_squared",
                jac_mode: Literal["auto", "forward", "reverse"] = "auto",
                jac_batch_size: int | None = None, jac_custom_fn: Callable | None = None,
                jac_custom_with_cache_fn: Callable | None = None, name: str | None = None):
        """Decorate ``residual(vals, *args, **kwargs) -> Array`` into a cost factory.

        Jacobian options are accepted for jaxls compatibility; the compiled solver always
        differentiates the residual exactly.
        """
        def decorator(fn: Callable) -> Callable[..., Cost]:
            def make(*args, **kwargs) -> Cost:
                return Cost(
                    compute_residual=lambda vals, a, kw: fn(vals, *a, **kw),
                    args=(args, kwargs), kind=kind, jac_mode=jac_mode,
                    jac_batch_size=jac_batch_size, jac_custom_fn=jac_custom_fn,
                    jac_custom_with_cache_fn=jac_custom_with_cache_fn,
                    name=name if name is not None else fn.__name__)
            make.__name__ = getattr(fn, "__name__", "cost")
            make.__doc__ = getattr(fn, "__doc__", None)
            return make

        return decorator if compute_residual is None else decorator(compute_residual)

    create_factory = factory  # older jaxls name

    def _get_name(self) -> str:
        return self.name or "cost"


# ── solver configuration (jaxls names; the compiled solver reads the fields noted) ──

@dataclasses.dataclass(frozen=True)
class TrustRegionConfig:
    lambda_initial: float = 5e-4  # used
    lambda_factor: float = 2.0
    lambda_min: float = 1e-5
    lambda_max: float = 1e6
    step_quality_min: float = 1e-3


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    max_iterations: int = 100  # used
    early_termination: bool = True
    cost_tolerance: float = 1e-5  # used
    gradient_tolerance: float = 1e-4
    gradient_tolerance_start_step: int = 10
    parameter_tolerance: float = 1e-6


@dataclasses.dataclass(frozen=True)
class AugmentedLagrangianConfig:
    penalty_factor: float = 4.0
    penalty_max: float = 1e7
    penalty_min: float = 1e-6
    penalty_initial: float | None = None
    tolerance_absolute: float = 1e-5  # used
    tolerance_relative: float = 1e-4
    violation_reduction_threshold: float = 0.5
    lambda_min: float = -1e7
    lambda_max: float = 1e7
    inner_solve_tolerance: float = 1e-2
    max_iterations: int = 20  # used: outer iterations


@dataclasses.dataclass(frozen=True)
class ConjugateGradientConfig:
    """Accepted for jaxls compatibility; the compiled solver factors densely."""

    tolerance_min: float = 1e-7
    tolerance_max: float = 1e-2
    eisenstat_walker_gamma: float = 0.9
    eisenstat_walker_alpha: float = 2.0
    preconditioner: str | None = "block_jacobi"


@dataclasses.dataclass(frozen=True)
class SolveSummary:
    cost: jax.Array
    """Sum of squared cost residuals at the solution."""
    constraint_violation: jax.Array
    """Largest constraint violation at the solution."""


class LeastSquaresProblem:
    """Costs and the variables they are minimised over, as in jaxls."""

    def __init__(self, costs: Iterable[Cost], variables: Iterable[Var], fixed: Iterable[Var] = ()):
        """``fixed``: variables held at their initial value (e.g. a trajectory's endpoints);
        a pyroffi extension, solved by the banded (trajectory) solver."""
        self.costs = list(costs)
        self.variables = list(variables)
        self.fixed = list(fixed)

    def analyze(self, use_onp: bool = False, structure: str = "auto") -> "AnalyzedLeastSquaresProblem":
        """Compile to a CUDA kernel (built once per problem structure, then cached).

        ``structure``: ``"dense"`` (up to 128 tangent dimensions), ``"banded"`` (trajectories:
        costs batched over knots, each touching a few neighbouring variables), or ``"auto"``
        (banded above 128 variables or with fixed variables). Raises if there is no CUDA
        device or nvcc, or if the problem uses something the compiler does not support.
        """
        del use_onp
        from ._compile import analyze

        return analyze(self, structure)


# Lie-group variables, as in jaxls: stored as the group's parameters, solved in its tangent
# space, stepped by right-multiplying exp(delta).
def _lie_var(name: str, group):
    return type(name, (Var,), {}, default_factory=group.identity,
                tangent_dim=group.tangent_dim,
                retract_fn=lambda x, delta: x @ group.exp(delta))


SO2Var = _lie_var("SO2Var", jaxlie.SO2)
SE2Var = _lie_var("SE2Var", jaxlie.SE2)
SO3Var = _lie_var("SO3Var", jaxlie.SO3)
SE3Var = _lie_var("SE3Var", jaxlie.SE3)
