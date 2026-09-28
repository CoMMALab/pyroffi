"""PyRoKi-named cost factories (jaxls) and their CUDA compilation (pyroffi.costs.compile).

The emitter tests run on CPU: they evaluate the generated scalar graph numerically and
compare it to JAX, which checks every primitive handler and fold rule without nvcc. The
compiled-solve tests need a CUDA device and nvcc, and are skipped otherwise.
"""

import shutil

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description

jaxls = pytest.importorskip("jaxls")

import pyroffi as pk
from pyroffi.costs import _emit as E

# The factories in PyRoKi's costs.py (upstream 388e43e).
PYROKI_COSTS = [
    "pose_cost", "pose_cost_analytic_jac", "pose_cost_numerical_jac", "pose_cost_with_base",
    "limit_cost", "limit_velocity_cost", "rest_cost", "rest_with_base_cost", "smoothness_cost",
    "manipulability_cost", "self_collision_cost", "world_collision_cost",
    "five_point_velocity_cost", "five_point_acceleration_cost", "five_point_jerk_cost",
    "limit_constraint", "limit_velocity_constraint", "world_collision_constraint",
]


def _has_cuda() -> bool:
    try:
        return bool(jax.devices("cuda")) and shutil.which("nvcc") is not None
    except RuntimeError:
        return False


cuda = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA device and nvcc")


@pytest.fixture(scope="module")
def panda():
    robot = pk.Robot.from_urdf(load_robot_description("panda_description"))
    return robot, robot.links.names.index("panda_hand")


def _ik_problem(robot, link, pos, wxyz):
    jv = robot.joint_var_cls(0)
    target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)
    costs = [
        pk.costs.pose_cost_analytic_jac(robot, jv, target, jnp.array(link, jnp.int32),
                                        pos_weight=50.0, ori_weight=10.0),
        pk.costs.limit_constraint(robot, jv),
        pk.costs.rest_cost(jv, jnp.array(jv.default_factory()), 0.01),
    ]
    return jaxls.LeastSquaresProblem(costs=costs, variables=[jv]), jv


def _pose_error(robot, link, q, pos, wxyz):
    T = jaxlie.SE3(robot.forward_kinematics(q)[link])
    d = (T.inverse() @ jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)).log()
    return float(jnp.linalg.norm(d[:3])), float(jnp.linalg.norm(d[3:]))


# ── drop-in names ───────────────────────────────────────────────────────────

def test_pyroki_cost_names_exist():
    missing = [n for n in PYROKI_COSTS if not callable(getattr(pk.costs, n, None))]
    assert not missing


def test_joint_var_cls_defaults_to_default_cfg(panda):
    robot, _ = panda
    jv = robot.joint_var_cls(0)
    assert isinstance(jv, jaxls.Var)
    np.testing.assert_allclose(jv.default_factory(), robot.default_cfg)


def test_jaxls_solve_unchanged(panda):
    """PyRoKi's basic-IK snippet, with only the import swapped."""
    robot, link = panda
    pos, wxyz = jnp.array([0.5, 0.1, 0.4]), jnp.array([0.0, 1.0, 0.0, 0.0])
    problem, jv = _ik_problem(robot, link, pos, wxyz)
    sol = problem.analyze().solve(verbose=False, linear_solver="dense_cholesky",
                                  trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))
    pos_err, ori_err = _pose_error(robot, link, sol[jv], pos, wxyz)
    assert pos_err < 1e-4 and ori_err < 1e-3


# ── emitter (CPU) ───────────────────────────────────────────────────────────

def _eval_graph(g: E.Graph, outs: np.ndarray, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the symbolic graph numerically: an independent check of what emit() prints."""
    val = {}

    def get(a):
        return val[a.idx] if isinstance(a, E.Sym) else a

    for n in g.nodes:
        if n.op in inputs:
            val[n.idx] = np.float32(inputs[n.op][n.args[0]])
        elif n.op == "cast":
            val[n.idx] = E._NP_DTYPE[n.kind](get(n.args[0]))
        else:
            with np.errstate(all="ignore"):
                val[n.idx] = E._FOLD[n.op](*[get(a) for a in n.args])
    return np.array([float(get(v)) for v in outs.flat]).reshape(outs.shape)


def _check_emitted(fn, *args, rtol=1e-4, atol=1e-5):
    """Trace ``fn`` symbolically, evaluate the graph, compare with JAX."""
    args = [np.asarray(a, np.float32) for a in args]
    closed = jax.make_jaxpr(fn)(*args)
    g = E.Graph()
    names = ["x", "p", "w", "s"][: len(args)]
    outs = E.Evaluator(g).run(closed, [E.symbolic(g, n, a.shape) for n, a in zip(names, args)])
    ref = jax.tree.leaves(fn(*args))
    for o, r in zip(outs, ref):
        got = _eval_graph(g, o, dict(zip(names, args)))
        np.testing.assert_allclose(got, np.asarray(r, np.float64), rtol=rtol, atol=atol)
    g.emit([(f"out[{i}]", v) for i, v in enumerate(outs[0].flat)])  # every node has a C++ form
    return g, outs


def test_emitter_pose_residual_and_jacobian(panda):
    robot, link = panda
    robot = jax.tree.map(np.asarray, robot)  # baked, as compile() does
    target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0.1, 0.2, 0.3), jnp.array([0.4, 0.1, 0.5]))

    def f(q, p):
        T = jaxlie.SE3(robot.forward_kinematics(q)[link])
        return jnp.concatenate([(T.inverse() @ jaxlie.SE3(p)).log() * 50.0,
                                jnp.maximum(0.0, q - robot.joints.upper_limits)])

    q = np.asarray(robot.default_cfg) + 0.3
    _, (J, r) = _check_emitted(
        lambda q, p: jax.jacfwd(lambda q: (f(q, p),) * 2, has_aux=True)(q), q, target.wxyz_xyz)
    # The fingers do not move the hand: those Jacobian columns fold to constant zero.
    assert sum(isinstance(v, E.Sym) for v in J.flat) < J.size


@pytest.mark.parametrize("name,fn,args", [
    ("scan", lambda x: jax.lax.scan(lambda c, a: (c * a + 1.0, jnp.sin(c)), x[0], x)[0], [np.arange(1, 5) * 0.3]),
    ("cond", lambda x: jax.lax.cond(x[0] > 0.5, jnp.sin, jnp.cos, x), [np.array([0.7, 0.2])]),
    ("dynamic_index", lambda x, p: x[jnp.int32(p[0])] * 2.0, [np.arange(4.0), np.array([2.0])]),
    ("dot_general", lambda x: jnp.reshape(x, (2, 3)) @ jnp.reshape(x, (3, 2)), [np.arange(6.0) / 7]),
    ("reductions", lambda x: jnp.stack([x.sum(), x.max(), jnp.cumsum(x)[-1], jnp.argmax(x) * 1.0]),
     [np.array([0.3, -1.0, 2.0, 0.5])]),
    ("fori_loop", lambda x: jax.lax.fori_loop(0, 3, lambda i, c: c * x + i, x), [np.array([0.5, 1.5])]),
    ("where_pow", lambda x: jnp.where(x > 0, x**3, jnp.sqrt(jnp.abs(x))), [np.array([-0.5, 2.0])]),
])
def test_emitter_primitives(name, fn, args):
    _check_emitted(fn, *args)


def test_emitter_rejects_data_dependent_loops():
    fn = lambda x: jax.lax.while_loop(lambda c: c[0] < 10.0, lambda c: c * 2.0, x)
    with pytest.raises(E.EmitError):
        _check_emitted(fn, np.array([1.0]))


# ── compiled solves (CUDA) ──────────────────────────────────────────────────

@cuda
def test_compiled_ik_matches_jaxls(panda):
    robot, link = panda
    pos, wxyz = jnp.array([0.5, 0.1, 0.4]), jnp.array([0.0, 1.0, 0.0, 0.0])
    problem, jv = _ik_problem(robot, link, pos, wxyz)
    solver = pk.costs.compile(problem)
    sol = solver.solve(trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))
    q = sol[jv]
    pos_err, ori_err = _pose_error(robot, link, q, pos, wxyz)
    assert pos_err < 1e-4 and ori_err < 1e-3
    assert jnp.all(q <= robot.joints.upper_limits + 1e-5)
    assert jnp.all(q >= robot.joints.lower_limits - 1e-5)

    # A new target is a runtime parameter: same build, no recompilation.
    pos2 = jnp.array([0.4, -0.2, 0.5])
    problem2, _ = _ik_problem(robot, link, pos2, wxyz)
    q2 = solver.solve(problem2, trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))[jv]
    pos_err, ori_err = _pose_error(robot, link, q2, pos2, wxyz)
    assert pos_err < 1e-4 and ori_err < 1e-3


def test_structure_check_rejects_other_robot_and_baked_values(panda, monkeypatch):
    """A problem with another robot, variable class or baked weight must not reuse a build."""
    from pyroffi.costs import _compile as C

    monkeypatch.setattr(C, "_build", lambda header: "unused")  # no nvcc needed
    robot, link = panda
    wxyz = jnp.array([0.0, 1.0, 0.0, 0.0])
    solver = pk.costs.compile(_ik_problem(robot, link, jnp.zeros(3), wxyz)[0])
    solver.params(_ik_problem(robot, link, jnp.ones(3), wxyz)[0])  # new target: fine

    other = pk.Robot.from_urdf(load_robot_description("panda_description"))
    with pytest.raises(ValueError):
        solver.params(_ik_problem(other, link, jnp.zeros(3), wxyz)[0])
    jv = robot.joint_var_cls(0)
    target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), jnp.zeros(3))
    reweighted = jaxls.LeastSquaresProblem([
        pk.costs.pose_cost_analytic_jac(robot, jv, target, jnp.array(link, jnp.int32),
                                        pos_weight=10.0, ori_weight=10.0),
        pk.costs.limit_constraint(robot, jv),
        pk.costs.rest_cost(jv, jnp.array(jv.default_factory()), 0.01)], [jv])
    with pytest.raises(ValueError):
        solver.params(reweighted)


@cuda
def test_compiled_batch_under_jit(panda):
    """Problems built inside jax.jit from traced targets reuse the compiled kernel."""
    robot, link = panda
    wxyz = jnp.array([0.0, 1.0, 0.0, 0.0])
    solver = pk.costs.compile(_ik_problem(robot, link, jnp.zeros(3), wxyz)[0], lambda_initial=1.0)
    targets = jnp.array([[0.5, 0.1, 0.4], [0.4, -0.2, 0.5], [0.3, 0.3, 0.6]])

    @jax.jit
    def solve(targets):
        params = jax.vmap(lambda t: solver.params(_ik_problem(robot, link, t, wxyz)[0]))(targets)
        x0 = jnp.broadcast_to(solver.initial_x(), (targets.shape[0], solver.n_x))
        return solver.solve_batch(params, x0)

    q, cost, viol = solve(targets)
    assert q.shape == (3, solver.n_x) and jnp.all(viol <= 1e-5)
    for qi, t in zip(q, targets):
        pos_err, ori_err = _pose_error(robot, link, qi, t, wxyz)
        assert pos_err < 1e-4 and ori_err < 1e-3


@cuda
def test_compiled_equality_and_geq_constraints():
    """Constraint kinds on a toy problem: min |x|^2 s.t. x0 + x1 == 1, x1 >= 0.8."""

    class V(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)): ...

    v = V(0)
    small = jaxls.Cost.factory(lambda vals, v: vals[v])
    eq = jaxls.Cost.factory(kind="constraint_eq_zero")(lambda vals, v: vals[v][:1] + vals[v][1:] - 1.0)
    geq = jaxls.Cost.factory(kind="constraint_geq_zero")(lambda vals, v: vals[v][1:] - 0.8)
    problem = jaxls.LeastSquaresProblem([small(v), eq(v), geq(v)], [v])
    x = pk.costs.compile(problem).solve()[v]
    np.testing.assert_allclose(x, [0.2, 0.8], atol=1e-3)
