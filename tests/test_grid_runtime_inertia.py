"""GRiD ``runtime_inertia``: the mutable device inertia table, and payloads on it.

Ordered as the plan requires, because each test is the gate for the next:

  1. bit-identity of ``runtime_inertia=True`` against the baked build with
     unmodified parameters -- everything downstream rests on this;
  2. the 10-parameter round trip, including the parallel-axis term, against the
     pure-JAX ``I_body`` (the step most likely to be silently wrong);
  3. GRiD-with-payload vs. JAX-with-payload;
  4. the purity guard actually fires on a tracer instead of uploading.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
import yourdfpy

import pyroffi
from pyroffi.attachments import Attachment, AttachmentSet

PANDA_URDF = "resources/panda/panda_spherized.urdf"

pytest.importorskip("jax")
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(scope="module")
def urdf():
    return yourdfpy.URDF.load(PANDA_URDF, load_meshes=False)


@pytest.fixture(scope="module")
def robot(urdf):
    return pyroffi.Robot.from_urdf(urdf)


@pytest.fixture(scope="module")
def baked(urdf):
    from pyroffi.dynamics import GRiDDynamics

    return GRiDDynamics(urdf)


@pytest.fixture(scope="module")
def mutable(urdf):
    from pyroffi.dynamics import GRiDDynamics

    return GRiDDynamics(urdf, runtime_inertia=True)


def _state(n, batch=8, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), (3, batch, n), dtype=jnp.float32)


# --- 1. the gate -----------------------------------------------------------


def test_runtime_inertia_is_bit_identical_to_baked(baked, mutable):
    """With the URDF's own parameters uploaded, the runtime path must reproduce
    the baked path exactly -- upstream claims this by construction; verify it,
    because every later test assumes it."""
    q, qd, qdd = _state(baked.num_dof)
    onp.testing.assert_array_equal(
        onp.asarray(mutable.inverse_dynamics(q, qd, qdd)),
        onp.asarray(baked.inverse_dynamics(q, qd, qdd)),
    )
    onp.testing.assert_array_equal(
        onp.asarray(mutable.forward_dynamics(q, qd, qdd)),
        onp.asarray(baked.forward_dynamics(q, qd, qdd)),
    )
    onp.testing.assert_array_equal(
        onp.asarray(mutable.mass_matrix(q)), onp.asarray(baked.mass_matrix(q))
    )


def test_baked_build_is_unaffected_by_the_flag(baked):
    """The flag must be free when unused: a baked library exposes no mutator."""
    assert baked.runtime_inertia is False
    with pytest.raises(AttributeError, match="without runtime_inertia"):
        baked.model_state


# --- 2. the parameter basis ------------------------------------------------


def test_param_extraction_roundtrips_a_spatial_inertia():
    from pyroffi.attachments import spatial_inertia
    from pyroffi.dynamics._grid_runtime_inertia import inertia_params_from_spatial

    m, c = 2.5, onp.array([0.07, -0.12, 0.2])
    Ic = onp.diag([0.01, 0.02, 0.03])
    I = onp.asarray(
        spatial_inertia(jnp.asarray(m), jnp.asarray(c), jnp.asarray(Ic))
    )
    pi = inertia_params_from_spatial(I)
    onp.testing.assert_allclose(pi[0], m, rtol=1e-6)
    onp.testing.assert_allclose(pi[1:4], m * c, rtol=1e-6)
    # I_O must carry the parallel-axis term, not the about-COM inertia.
    I_O_expected = Ic + m * (c @ c * onp.eye(3) - onp.outer(c, c))
    onp.testing.assert_allclose(
        pi[4:],
        [
            I_O_expected[0, 0],
            I_O_expected[0, 1],
            I_O_expected[0, 2],
            I_O_expected[1, 1],
            I_O_expected[1, 2],
            I_O_expected[2, 2],
        ],
        rtol=1e-6,
        atol=1e-9,
    )


def test_baseline_table_matches_the_jax_body_inertias(mutable, robot):
    """The table GRiD ships and pyroffi's own ``I_body`` must describe the same
    bodies -- this is where a missing parallel-axis term or a frame mismatch
    would show up as plausible-but-wrong torques."""
    from pyroffi.dynamics._grid_runtime_inertia import inertia_params_from_spatial

    perm = onp.asarray(mutable._grid_model.joint_perm)
    jax_params = inertia_params_from_spatial(
        onp.asarray(robot.dynamics.I_body, dtype=onp.float64)
    )
    onp.testing.assert_allclose(
        mutable.model_state.baseline,
        jax_params[perm],
        rtol=1e-5,
        atol=1e-8,
    )


# --- 3. payloads -----------------------------------------------------------


def _payload(robot, mass=3.0, offset=(0.0, 0.0, 0.15)):
    link = robot.links.num_links - 1
    T = jnp.array([1.0, 0.0, 0.0, 0.0, *offset])
    return AttachmentSet.empty().attach(
        Attachment.from_mass(jnp.asarray(mass), link, T, name="payload")
    )


def test_payload_changes_the_torques(mutable, robot):
    q, qd, qdd = _state(mutable.num_dof)
    before = onp.asarray(mutable.inverse_dynamics(q, qd, qdd))
    try:
        mutable.set_attachments(robot, _payload(robot))
        after = onp.asarray(mutable.inverse_dynamics(q, qd, qdd))
    finally:
        mutable.reset_inertia()
    assert not onp.allclose(before, after)
    # and the reset really restores the unloaded model
    onp.testing.assert_array_equal(
        onp.asarray(mutable.inverse_dynamics(q, qd, qdd)), before
    )


def test_grid_payload_agrees_with_the_jax_payload(mutable, robot):
    """The headline check: the same attachment, through the GPU table and
    through the pure-JAX inertia composition, must agree."""
    aset = _payload(robot)
    q, qd, qdd = _state(mutable.num_dof)
    loaded = robot.with_attachments(aset)
    ref = onp.asarray(loaded.inverse_dynamics(q, qd, qdd))
    try:
        mutable.set_attachments(robot, aset)
        got = onp.asarray(mutable.inverse_dynamics(q, qd, qdd))
    finally:
        mutable.reset_inertia()
    # float32 kernels against a float32 JAX reference; the JAX RNEA is the
    # noisier of the two here (see test_grid_dynamics), hence the loose bound.
    onp.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


def test_payload_mass_scales_the_extra_torque_linearly(mutable, robot):
    """pi is linear in mass, so doubling the payload must exactly double the
    torque *difference* it induces."""
    q, qd, qdd = _state(mutable.num_dof, batch=4, seed=3)
    base = onp.asarray(mutable.inverse_dynamics(q, qd, qdd))
    try:
        mutable.set_attachments(robot, _payload(robot, mass=1.0))
        d1 = onp.asarray(mutable.inverse_dynamics(q, qd, qdd)) - base
        mutable.set_attachments(robot, _payload(robot, mass=2.0))
        d2 = onp.asarray(mutable.inverse_dynamics(q, qd, qdd)) - base
    finally:
        mutable.reset_inertia()
    onp.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-3, atol=1e-4)


# --- 4. the purity guard ---------------------------------------------------


def test_guard_raises_on_a_traced_payload(mutable):
    """A silently-stale table is an invisible wrong-dynamics bug, so the guard
    must key on tracer-ness and fail loudly rather than skip the upload."""
    state = mutable.model_state

    @jax.jit
    def upload(params):
        state.upload(params)
        return params

    with pytest.raises(TypeError, match="cannot be set from inside"):
        upload(jnp.asarray(state.baseline))


def test_set_inertia_params_is_never_reached_from_inside_jit(mutable, robot):
    aset = _payload(robot)

    @jax.jit
    def f(_):
        mutable.set_attachments(robot, aset)
        return 0.0

    with pytest.raises(TypeError, match="cannot be set from inside"):
        f(jnp.asarray(0.0))
    mutable.reset_inertia()
