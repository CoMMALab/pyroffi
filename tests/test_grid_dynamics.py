"""CUDA (GRiD-generated kernels) vs pure-JAX dynamics agreement tests.

Requires a CUDA GPU and nvcc; the first run per robot JIT-compiles the
kernels (cached under ~/.cache/pyroffi/grid).
"""

import jax
import jax.numpy as jnp
import pytest
import yourdfpy

import pyroffi

ATOL = 2e-4
RTOL = 1e-4


def _close(a, b, atol=ATOL, rtol=None):
    import jax.numpy as _jnp

    tol = atol + (rtol if rtol is not None else RTOL) * _jnp.abs(b)
    return bool((_jnp.abs(a - b) <= tol).all())
PANDA_URDF = "resources/panda/panda_spherized.urdf"
FETCH_URDF = "resources/fetch/fetch_grid.urdf"

pytest.importorskip("jax")
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(scope="module", params=[PANDA_URDF, FETCH_URDF])
def setup(request):
    from pyroffi.dynamics import GRiDDynamics

    urdf = yourdfpy.URDF.load(request.param, load_meshes=False)
    robot = pyroffi.Robot.from_urdf(urdf)
    gd = GRiDDynamics(urdf)
    return robot, gd


def _state(gd, key, batch=(16,)):
    return jax.random.normal(key, (3, *batch, gd.num_dof), dtype=jnp.float32)


def test_inverse_dynamics_agreement(setup):
    robot, gd = setup
    q, qd, qdd = _state(gd, jax.random.PRNGKey(0))
    assert _close(gd.inverse_dynamics(q, qd, qdd), robot.inverse_dynamics(q, qd, qdd))


def test_forward_dynamics_agreement(setup):
    robot, gd = setup
    q, qd, u = _state(gd, jax.random.PRNGKey(1))
    assert _close(gd.forward_dynamics(q, qd, u), robot.forward_dynamics(q, qd, u))


def test_minv_agreement(setup):
    robot, gd = setup
    q = _state(gd, jax.random.PRNGKey(2))[0]
    Minv_ref = jnp.linalg.inv(robot.mass_matrix(q))
    assert _close(gd.mass_matrix_inv(q), Minv_ref)


def test_gradient_kernels_vs_autodiff(setup):
    robot, gd = setup
    n = gd.num_dof
    q, qd, x = _state(gd, jax.random.PRNGKey(3), batch=())
    G_id = gd.inverse_dynamics_gradient(q, qd, x)
    J_ref = jax.jacobian(lambda q_: robot.inverse_dynamics(q_, qd, x))(q)
    assert _close(G_id[..., :n], J_ref, atol=1e-3, rtol=1e-3)
    G_fd = gd.forward_dynamics_gradient(q, qd, x)
    J_fd = jax.jacobian(lambda qd_: robot.forward_dynamics(q, qd_, x))(qd)
    assert _close(G_fd[..., n:], J_fd, atol=1e-3, rtol=1e-3)


def test_custom_vjp_matches_jax(setup):
    robot, gd = setup
    q, qd, x = _state(gd, jax.random.PRNGKey(4))

    for arg in range(3):  # d/dq, d/dqd, d/d{qdd,u}
        args = [q, qd, x]

        def wrt(v, fn, args=args, arg=arg):
            a = list(args)
            a[arg] = v
            return fn(*a).sum()

        g_id = jax.grad(lambda v: wrt(v, gd.inverse_dynamics))(args[arg])
        g_id_ref = jax.grad(lambda v: wrt(v, robot.inverse_dynamics))(args[arg])
        assert _close(g_id, g_id_ref, atol=1e-3, rtol=1e-3), f"ID grad arg {arg}"

        g_fd = jax.grad(lambda v: wrt(v, gd.forward_dynamics))(args[arg])
        g_fd_ref = jax.grad(lambda v: wrt(v, robot.forward_dynamics))(args[arg])
        assert _close(g_fd, g_fd_ref, atol=1e-3, rtol=1e-3), f"FD grad arg {arg}"


def test_batched_shapes_and_jit(setup):
    _, gd = setup
    n = gd.num_dof
    q, qd, x = _state(gd, jax.random.PRNGKey(5), batch=(2, 3))
    assert gd.inverse_dynamics(q, qd, x).shape == (2, 3, n)
    assert gd.mass_matrix_inv(q).shape == (2, 3, n, n)
    f = jax.jit(gd.forward_dynamics)
    assert f(q, qd, x).shape == (2, 3, n)


def test_mass_matrix_agreement(setup):
    """GRiD's generated crba_kernel M(q) vs the pure-JAX CRBA."""
    robot, gd = setup
    q = _state(gd, jax.random.PRNGKey(6))[0]
    M = gd.mass_matrix(q)
    M_ref = robot.mass_matrix(q)
    assert _close(M, M_ref)
    # And it must actually invert the direct-Minv kernel's output.
    assert _close(
        jnp.einsum("...ij,...jk->...ik", M, gd.mass_matrix_inv(q)),
        jnp.broadcast_to(jnp.eye(gd.num_dof), M.shape),
        atol=5e-3,
        rtol=5e-3,
    )


def test_fext_agreement(setup):
    """f_ext composited around the GRiD kernels vs pure-JAX f_ext plumbing."""
    robot, gd = setup
    q, qd, x = _state(gd, jax.random.PRNGKey(7))
    f_ext = jax.random.normal(
        jax.random.PRNGKey(8), (*q.shape, 6), dtype=jnp.float32
    )
    assert _close(
        gd.inverse_dynamics(q, qd, x, f_ext),
        robot.inverse_dynamics(q, qd, x, f_ext=f_ext),
    )
    assert _close(
        gd.forward_dynamics(q, qd, x, f_ext),
        robot.forward_dynamics(q, qd, x, f_ext=f_ext),
    )


def test_step_agreement(setup):
    robot, gd = setup
    q, qd, u = _state(gd, jax.random.PRNGKey(9))
    dt = 1e-3
    for method in ("semi_implicit", "euler", "rk4"):
        q1, qd1 = gd.step(q, qd, u, dt, method=method)
        q1_ref, qd1_ref = robot.step(q, qd, u, dt, method=method)
        assert _close(q1, q1_ref) and _close(qd1, qd1_ref), method


def test_cache_hit(setup):
    """Reconstruction must load the cached .so without recompiling."""
    import time

    from pyroffi.dynamics import GRiDDynamics

    _, gd = setup
    # Codegen (sympy) still runs, but nvcc must be skipped; generously bounded.
    t0 = time.monotonic()
    gd2 = GRiDDynamics.__new__(GRiDDynamics)  # noqa: F841 - just checking cache path
    from pyroffi.dynamics._grid_codegen import compile_grid_library

    so = compile_grid_library(gd._grid_model)
    assert so.is_file()
    assert time.monotonic() - t0 < 120.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
