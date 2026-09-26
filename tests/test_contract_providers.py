"""Every provider behind the traced-robot contract computes the same forward dynamics.

cricket (thread tier), GRiD (block tier) and PyRoFFI's JAX dynamics, all float32, against
pinocchio in float64, on the rewritten benchmark URDFs (see tests/bench_dynamics.py).
"""

import shutil

import jax
import numpy as np
import pytest
import yourdfpy

pytest.importorskip("cricket")
pin = pytest.importorskip("pinocchio")
if shutil.which("nvcc") is None or jax.default_backend() != "gpu":
    pytest.skip("contract kernels need nvcc and a GPU", allow_module_level=True)

from pyroffi import Robot
from pyroffi.dynamics._contract_dynamics import ContractDynamics

URDFS = {
    "panda": "resources/panda/panda_spherized__dynbench.urdf",
    "fetch": "resources/fetch/fetch_grid__dynbench.urdf",
    "g1": "resources/g1_description/g1_29dof__dynbench.urdf",
}


@pytest.mark.parametrize("robot", list(URDFS))
def test_forward_dynamics_providers_agree(robot):
    path = URDFS[robot]
    urdf = yourdfpy.URDF.load(path, load_meshes=False)
    contract = ContractDynamics(urdf)
    pyroffi_robot = Robot.from_urdf(urdf)
    actuated = list(pyroffi_robot.joints.actuated_names)

    model = pin.buildModelFromUrdf(path, mimic=True)
    data = model.createData()
    to_pin = [actuated.index(model.names[j]) for j in range(1, model.njoints) if model.joints[j].nq]

    rng = np.random.default_rng(0)
    lo, hi = np.asarray(pyroffi_robot.joints.lower_limits), np.asarray(pyroffi_robot.joints.upper_limits)
    q = rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi), (64, len(actuated))).astype(np.float32)
    qd, tau = (rng.uniform(-1, 1, q.shape).astype(np.float32) for _ in range(2))

    ref = np.empty_like(q, dtype=np.float64)
    for i in range(len(q)):
        out = pin.aba(model, data, *(x[i, to_pin].astype(np.float64) for x in (q, qd, tau)))
        ref[i, to_pin] = out

    results = {tier: np.asarray(contract.forward_dynamics(q, qd, tau, tier)) for tier in contract.tiers}
    results["jax"] = np.asarray(jax.vmap(pyroffi_robot.forward_dynamics)(q, qd, tau))
    assert set(results) == {"thread", "block", "jax"}
    # Measured 2026-09-26 (worst of panda/fetch/g1): thread 1.0e-6, block 2.3e-6, jax 1.2e-4.
    # The JAX dynamics is ~100x looser than either traced provider on fetch and g1.
    bound = {"thread": 1e-5, "block": 1e-5, "jax": 1e-3}
    for name, out in results.items():
        err = np.linalg.norm(out - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1.0)
        assert err.max() < bound[name], (name, np.median(err), err.max())
