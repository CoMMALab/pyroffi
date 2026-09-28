"""pyroffi's jaxls-shaped API (pk.Var, pk.Cost, pk.LeastSquaresProblem, ...) and PyRoKi's
cost factories, compiled to CUDA by ``analyze()``.

The emitter tests run on CPU: they evaluate the generated scalar graph numerically and
compare it to JAX, which checks every primitive handler and fold rule without nvcc. The
solve tests need a CUDA device and nvcc, and are skipped otherwise.
"""

import shutil

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as np
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroffi as pk
from pyroffi.costs import _compile as C
from pyroffi.costs import _emit as E

# The factories in PyRoKi's costs.py (upstream 388e43e).
PYROKI_COSTS = [
    "pose_cost", "pose_cost_analytic_jac", "pose_cost_numerical_jac", "pose_cost_with_base",
    "limit_cost", "limit_velocity_cost", "rest_cost", "rest_with_base_cost", "smoothness_cost",
    "manipulability_cost", "self_collision_cost", "world_collision_cost",
    "five_point_velocity_cost", "five_point_acceleration_cost", "five_point_jerk_cost",
    "limit_constraint", "limit_velocity_constraint", "world_collision_constraint",
]
# jaxls's public names that pyroffi mirrors at top level.
JAXLS_NAMES = ["Var", "VarValues", "Cost", "LeastSquaresProblem", "AnalyzedLeastSquaresProblem",
               "TrustRegionConfig", "TerminationConfig", "AugmentedLagrangianConfig",
               "ConjugateGradientConfig", "SolveSummary", "SE2Var", "SE3Var", "SO2Var", "SO3Var"]


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


def _ik_problem(robot, link, pos, wxyz, pos_weight=50.0):
    jv = robot.joint_var_cls(0)
    target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)
    costs = [
        pk.costs.pose_cost_analytic_jac(robot, jv, target, jnp.array(link, jnp.int32),
                                        pos_weight=pos_weight, ori_weight=10.0),
        pk.costs.limit_constraint(robot, jv),
        pk.costs.rest_cost(jv, jnp.array(jv.default_factory()), 0.01),
    ]
    return pk.LeastSquaresProblem(costs=costs, variables=[jv]), jv


def _pose_error(robot, link, q, pos, wxyz):
    T = jaxlie.SE3(robot.forward_kinematics(q)[link])
    d = (T.inverse() @ jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)).log()
    return float(jnp.linalg.norm(d[:3])), float(jnp.linalg.norm(d[3:]))


# ── API surface (CPU) ───────────────────────────────────────────────────────

def test_names_exist():
    assert not [n for n in PYROKI_COSTS if not callable(getattr(pk.costs, n, None))]
    assert not [n for n in JAXLS_NAMES if not hasattr(pk, n)]


def test_joint_var_cls_and_var_values(panda):
    robot, _ = panda
    JointVar = robot.joint_var_cls
    np.testing.assert_allclose(JointVar(0).default_factory(), robot.default_cfg)
    vals = pk.VarValues.make([JointVar(jnp.arange(3)), JointVar(5).with_value(jnp.ones(robot.nq))])
    assert vals[JointVar(jnp.arange(3))].shape == (3, robot.nq)
    np.testing.assert_allclose(vals[JointVar(5)], 1.0)
    assert vals[JointVar].shape == (4, robot.nq)  # stacked, sorted by id


def test_traced_variable_ids_inside_jit(panda):
    """jaxls-style code makes ids inside jit (``JointVar(jnp.arange(T))``); they are
    recovered from the trace, but ids that depend on a jitted input are refused."""
    robot, _ = panda
    JointVar = robot.joint_var_cls

    @jax.jit
    def f(y):
        ids = jnp.arange(1, 4) - 1
        vals = pk.VarValues.make([JointVar(ids).with_value(jnp.ones((3, robot.nq)) * y)])
        return vals[JointVar(ids)], vals[JointVar(jnp.full((), 2))]

    all_, two = f(2.0)
    assert all_.shape == (3, robot.nq)
    np.testing.assert_allclose(two, 2.0)

    with pytest.raises(ValueError, match="depend on the inputs"):
        jax.jit(lambda i: pk.VarValues.make([JointVar(i)])[JointVar(i)])(jnp.int32(0))


def test_analyze_fails_without_gpu(panda, monkeypatch):
    def no_cuda(*a, **k):
        raise RuntimeError("no cuda")

    monkeypatch.setattr(C.jax, "devices", no_cuda)
    robot, link = panda
    problem, _ = _ik_problem(robot, link, jnp.zeros(3), jnp.array([1.0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="CUDA device"):
        problem.analyze()


def test_unsupported_problems_raise(monkeypatch):
    monkeypatch.setattr(C, "_require_cuda", lambda: None)
    monkeypatch.setattr(C, "_build", lambda header: "unused")
    monkeypatch.setattr(C, "_KERNELS", {})  # keep mocked builds out of the real cache
    se3 = pk.SE3Var(0)
    lie = pk.Cost.factory(lambda vals, v: vals[v].translation())
    with pytest.raises(NotImplementedError, match="Euclidean"):
        pk.LeastSquaresProblem([lie(se3)], [se3]).analyze()

    class Big(pk.Var[jax.Array], default_factory=lambda: jnp.zeros(40)): ...

    small = pk.Cost.factory(lambda vals, v: vals[v])
    with pytest.raises(NotImplementedError, match="up to 32"):
        pk.LeastSquaresProblem([small(Big(0))], [Big(0)]).analyze()


def test_builds_are_cached_by_baked_values(panda, monkeypatch):
    """A new target reuses the build; another robot or baked weight gets its own."""
    monkeypatch.setattr(C, "_require_cuda", lambda: None)
    monkeypatch.setattr(C, "_build", lambda header: "unused")
    monkeypatch.setattr(C, "_KERNELS", {})  # keep mocked builds out of the real cache
    robot, link = panda
    wxyz = jnp.array([0.0, 1.0, 0.0, 0.0])
    a = _ik_problem(robot, link, jnp.zeros(3), wxyz)[0].analyze()
    assert _ik_problem(robot, link, jnp.ones(3), wxyz)[0].analyze()._kernel is a._kernel
    assert _ik_problem(robot, link, jnp.zeros(3), wxyz, pos_weight=10.0)[0].analyze()._kernel is not a._kernel
    other = pk.Robot.from_urdf(load_robot_description("panda_description"))
    assert _ik_problem(other, link, jnp.zeros(3), wxyz)[0].analyze()._kernel is not a._kernel


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
    g.emit_function([("out", outs[0])])  # every node has a C++ form
    return g, outs


def test_emitter_pose_residual_and_jacobian(panda):
    robot, link = panda
    robot = jax.tree.map(np.asarray, robot)  # baked, as analyze() does
    target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0.1, 0.2, 0.3), jnp.array([0.4, 0.1, 0.5]))

    def f(q, p):
        T = jaxlie.SE3(robot.forward_kinematics(q)[link])
        return jnp.concatenate([(T.inverse() @ jaxlie.SE3(p)).log() * 50.0,
                                jnp.maximum(0.0, q - robot.joints.upper_limits)])

    q = np.asarray(robot.default_cfg) + 0.3
    jf = lambda q, p: jax.jacfwd(lambda q: (f(q, p),) * 2, has_aux=True)(q)
    # Fully scalarized (the thread tier's variant) and at the cooperative tiers' threshold.
    J, r = _check_loops(jf, q, target.wxyz_xyz, loop_min=float("inf"))
    _check_loops(jf, q, target.wxyz_xyz, loop_min=C.COOP_LOOP_MIN)
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


# ── loop IR (CPU): a schedule interpreter mirrors the emitted C++ ─────────────

def _run_schedule(g: E.Graph, outputs: list, inputs: dict) -> dict:
    """Execute ``g.schedule(outputs)`` in Python with one lane: the same statements, loop
    bodies and scratch offsets the printed kernel uses."""
    import re

    sch = g.schedule(outputs)
    out = {name: np.zeros(int(np.prod(v.shape, dtype=int)), np.float32) for name, v in outputs}
    scratch = np.zeros(max(sch.scratch, 1), np.float32)
    val = {n.idx: np.float32(inputs[n.op][n.args[0]]) for n in g.nodes if n.op in inputs}
    get = lambda a: val[a.idx] if isinstance(a, E.Sym) else a
    cast = lambda n, v: E._NP_DTYPE[n.kind](v)

    def ev(n):
        if n.op == "tload":
            return cast(n, g.tables[n.args[0].idx][int(get(n.args[1]))])
        if n.op == "bload":
            return cast(n, scratch[sch.offsets[n.args[0].idx] + int(get(n.args[1]))])
        if n.op == "cast":
            return cast(n, get(n.args[0]))
        with np.errstate(all="ignore"):
            return cast(n, E._FOLD[n.op](*[get(a) for a in n.args]))

    for n in sch.top:
        if n.op == "loop":
            idx, value, red = n.args[0], n.args[1], n.extra["reduce"]
            members = sch.body.get(n.idx, [])
            rid = red[0].idx if red else None
            outer = [m for m in members if rid not in sch.deps[m.idx]]
            inner = [m for m in members if rid in sch.deps[m.idx]]
            dst, off = ((out[n.extra["target"]], 0) if n.extra["target"]
                        else (scratch, sch.offsets[n.idx]))
            for k in range(n.extra["extent"]):
                val[idx.idx] = np.int32(k)
                for m in outer:
                    val[m.idx] = ev(m)
                if red:
                    acc = cast(n, E._reduce_init(red[2], n.kind))
                    for r in range(red[1]):
                        val[red[0].idx] = np.int32(r)
                        for m in inner:
                            val[m.idx] = ev(m)
                        acc = cast(n, E._FOLD[red[2]](acc, get(value)))
                    dst[off + k] = acc
                else:
                    dst[off + k] = get(value)
        elif n.op == "pack":
            for i, v in enumerate(n.args):
                scratch[sch.offsets[n.idx] + i] = get(v)
        else:
            val[n.idx] = ev(n)
    for lhs, v in sch.writes:
        name, i = re.match(r"(\w+)\[(\d+)\]", lhs).groups()
        out[name][int(i)] = get(v)
    return out


def _check_loops(fn, *args, loop_min=64, rtol=1e-4, atol=1e-5):
    """Trace ``fn`` with a low loop threshold (so small arrays take the loop paths), run the
    schedule, compare with JAX. Returns the emitted values."""
    args = [np.asarray(a, np.float32) for a in args]
    g = E.Graph()
    names = ["x", "p"][: len(args)]
    outs = E.Evaluator(g, loop_min).run(jax.make_jaxpr(lambda *a: tuple(jax.tree.leaves(fn(*a))))(*args),
                                        [E.symbolic(g, n, a.shape) for n, a in zip(names, args)])
    named = [(f"o{k}", o) for k, o in enumerate(outs)]
    got = _run_schedule(g, named, dict(zip(names, args)))
    for k, ref in enumerate(jax.tree.leaves(fn(*args))):
        np.testing.assert_allclose(got[f"o{k}"], np.asarray(ref, np.float64).ravel(), rtol=rtol, atol=atol)
    g.emit_function(named)  # every node prints
    return outs


_X = np.random.default_rng(0).standard_normal(600).astype(np.float32)
_IDX = np.random.default_rng(1).integers(0, 30, size=40)


@pytest.mark.parametrize("name,fn", [
    ("ew_chain", lambda x: jnp.sin(x) * x + jnp.sqrt(jnp.abs(x))),
    ("broadcast_ew", lambda x: x.reshape(20, 30, 1) * x[:30].reshape(1, 30, 1) + x[:20].reshape(20, 1, 1)),
    ("transpose", lambda x: x.reshape(20, 30).T * 2.0),
    ("slice", lambda x: x.reshape(20, 30)[3:17, 5:25] + 1.0),
    ("concat3", lambda x: jnp.concatenate([x[:200], x[200:400] * 3.0, jnp.cos(x[400:])])),
    ("stack", lambda x: jnp.stack([x[:300], x[300:] + 1.0], axis=1)),
    ("pad", lambda x: jnp.pad(x.reshape(20, 30), ((1, 2), (3, 0)), constant_values=7.0)),
    ("gather_fancy", lambda x: x.reshape(20, 30)[:, _IDX]),
    ("scatter_set", lambda x: jnp.asarray(x).at[_IDX * 10].set(99.0) * 2.0),
    ("dynamic_index", lambda x: x.reshape(20, 30)[jnp.int32(jnp.abs(x[0]) * 5) % 20] * 2.0),
    ("dot_small", lambda x: jnp.einsum("nij,nj->ni", x[:540].reshape(60, 3, 3), x[:180].reshape(60, 3))),
    ("dot_self", lambda x: jnp.einsum("nij,njk->nik", x[:576].reshape(36, 4, 4), x[:576].reshape(36, 4, 4))),
    ("dot_large", lambda x: x.reshape(20, 30) @ x.reshape(30, 20)),
    ("reduce_small", lambda x: x.reshape(200, 3).sum(1) + x.reshape(200, 3).max(1)),
    ("reduce_large", lambda x: jnp.concatenate([x.reshape(20, 30).sum(1), x.reshape(20, 30).min(0)])),
    ("argmin", lambda x: jnp.argmin(x.reshape(20, 30), axis=1).astype(jnp.float32) + x[:20]),
    ("select_pow", lambda x: jnp.clip(x, -0.5, 0.5) + jnp.where(x > 0, x ** 3, 1.0 / (1.0 + x * x))),
    ("scan_xs", lambda x: jax.lax.scan(lambda c, a: (c + a, a * c), jnp.zeros(30), x.reshape(20, 30))[1]),
    ("cond", lambda x: jax.lax.cond(x[0] > 0, lambda y: y * 2.0, lambda y: y - 1.0, x)),
    ("pairwise_min", lambda x: jnp.min(jnp.linalg.norm(x[:180].reshape(60, 1, 3) - x[180:360].reshape(1, 60, 3), axis=-1), axis=1)),
])
def test_loop_ir_primitives(name, fn):
    _check_loops(fn, _X)


def test_loop_ir_collision_jacobian(panda):
    """The self-collision distance and its Jacobian through the loop IR, capsule model."""
    robot, _ = panda
    urdf = load_robot_description("panda_description")
    coll = jax.tree.map(np.asarray, pk.collision.RobotCollision.from_urdf(urdf))
    rb = jax.tree.map(np.asarray, robot)
    f = lambda q: jax.jacfwd(lambda q: (coll.compute_self_collision_distance(rb, q),) * 2, has_aux=True)(q)
    outs = _check_loops(f, np.asarray(robot.default_cfg) + 0.2, loop_min=256)
    assert any(isinstance(o, E.Lazy) for o in outs)


# ── compiled solves (CUDA) ──────────────────────────────────────────────────

@cuda
@pytest.mark.parametrize("tier", ["auto", "thread", "warp", "block"])
def test_ik_solve(panda, tier):
    robot, link = panda
    pos, wxyz = jnp.array([0.5, 0.1, 0.4]), jnp.array([0.0, 1.0, 0.0, 0.0])
    problem, jv = _ik_problem(robot, link, pos, wxyz)
    sol, summary = problem.analyze().solve(
        verbose=False, linear_solver="dense_cholesky",
        trust_region=pk.TrustRegionConfig(lambda_initial=1.0), return_summary=True, tier=tier)
    q = sol[jv]
    pos_err, ori_err = _pose_error(robot, link, q, pos, wxyz)
    assert pos_err < 1e-4 and ori_err < 1e-3
    assert float(summary.constraint_violation) <= 1e-5
    assert jnp.all(q <= robot.joints.upper_limits + 1e-5)
    assert jnp.all(q >= robot.joints.lower_limits - 1e-5)


@cuda
def test_pyroki_style_solve_inside_jit(panda):
    """PyRoKi's pattern: robot and targets are jit arguments, the problem is built and
    solved inside the trace. The traced robot resolves to its concrete original."""
    robot, link = panda

    @jdc.jit
    def solve(robot: pk.Robot, pos: jax.Array, wxyz: jax.Array, idx: jax.Array) -> jax.Array:
        jv = robot.joint_var_cls(0)
        target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)
        costs = [pk.costs.pose_cost_analytic_jac(robot, jv, target, idx, pos_weight=50.0, ori_weight=10.0),
                 pk.costs.limit_constraint(robot, jv)]
        sol = pk.LeastSquaresProblem(costs=costs, variables=[jv]).analyze().solve(
            verbose=False, trust_region=pk.TrustRegionConfig(lambda_initial=1.0))
        return sol[jv]

    wxyz = jnp.array([0.0, 1.0, 0.0, 0.0])
    for pos in (jnp.array([0.5, 0.1, 0.4]), jnp.array([0.4, -0.2, 0.5])):
        q = solve(robot, pos, wxyz, jnp.array(link, jnp.int32))
        pos_err, ori_err = _pose_error(robot, link, q, pos, wxyz)
        assert pos_err < 1e-4 and ori_err < 1e-3


@cuda
def test_batch_of_problems(panda):
    robot, link = panda
    wxyz = jnp.array([0.0, 1.0, 0.0, 0.0])
    targets = jnp.array([[0.5, 0.1, 0.4], [0.4, -0.2, 0.5], [0.3, 0.3, 0.6]])

    @jax.jit
    def solve(targets):
        params = jax.vmap(lambda t: _ik_problem(robot, link, t, wxyz)[0].analyze().params())(targets)
        analyzed = _ik_problem(robot, link, targets[0], wxyz)[0].analyze()
        x0 = jnp.broadcast_to(analyzed.initial_x(), (targets.shape[0], analyzed.n_x))
        return analyzed.solve_batch(params, x0, lambda_initial=1.0)

    q, cost, viol = solve(targets)
    assert q.shape[0] == 3 and jnp.all(viol <= 1e-5)
    for qi, t in zip(q, targets):
        pos_err, ori_err = _pose_error(robot, link, qi, t, wxyz)
        assert pos_err < 1e-4 and ori_err < 1e-3


@cuda
@pytest.mark.parametrize("tier", ["thread", "warp", "block"])
def test_equality_and_geq_constraints(tier):
    """min |x|^2 s.t. x0 + x1 == 1, x1 >= 0.8."""

    class V(pk.Var[jax.Array], default_factory=lambda: jnp.zeros(2)): ...

    v = V(0)
    small = pk.Cost.factory(lambda vals, v: vals[v])
    eq = pk.Cost.factory(kind="constraint_eq_zero")(lambda vals, v: vals[v][:1] + vals[v][1:] - 1.0)
    geq = pk.Cost.factory(kind="constraint_geq_zero")(lambda vals, v: vals[v][1:] - 0.8)
    x = pk.LeastSquaresProblem([small(v), eq(v), geq(v)], [v]).analyze().solve(tier=tier)[v]
    np.testing.assert_allclose(x, [0.2, 0.8], atol=1e-3)
