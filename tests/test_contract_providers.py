"""Every provider behind the traced-robot contract computes the same dynamics.

cricket at thread and warp-split block tier, GRiD, and (for forward dynamics) the
mass-matrix + GLASS Cholesky route, all float32, against pinocchio float64 on the rewritten
benchmark URDFs (see tests/bench_dynamics.py). The JAX dynamics is checked too.
"""

import json
import shutil
from pathlib import Path

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


@pytest.fixture(scope="module", params=list(URDFS))
def robot(request):
    path = URDFS[request.param]
    urdf = yourdfpy.URDF.load(path, load_meshes=False)
    contract = ContractDynamics(urdf)
    pyroffi_robot = Robot.from_urdf(urdf)
    model = pin.buildModelFromUrdf(path, mimic=True)
    actuated = contract.actuated
    to_pin = [actuated.index(model.names[j]) for j in range(1, model.njoints) if model.joints[j].nq]
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(pyroffi_robot.joints.lower_limits), np.asarray(pyroffi_robot.joints.upper_limits)
    q = rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi), (48, len(actuated))).astype(np.float32)
    qd, u = (rng.uniform(-1, 1, q.shape).astype(np.float32) for _ in range(2))
    return request.param, contract, pyroffi_robot, model, np.array(to_pin), (q, qd, u)


def _reference(op, model, to_pin, q, qd, u):
    data = model.createData()
    n = q.shape[1]
    out = []
    for i in range(len(q)):
        a, b, c = (x[i, to_pin].astype(np.float64) for x in (q, qd, u))
        if op == "id":
            r = np.empty(n); r[to_pin] = pin.rnea(model, data, a, b, c)
        elif op == "fd":
            r = np.empty(n); r[to_pin] = pin.aba(model, data, a, b, c)
        elif op == "crba":
            M = pin.crba(model, data, a); M = np.triu(M) + np.triu(M, 1).T
            r = np.empty((n, n)); r[np.ix_(to_pin, to_pin)] = M
        else:
            dq, dv, _ = pin.computeRNEADerivatives(model, data, a, b, c)
            r = np.empty((n, 2 * n)); r[np.ix_(to_pin, to_pin)] = dq; r[np.ix_(to_pin, n + to_pin)] = dv
        out.append(r)
    return np.stack(out)


# Every provider, the JAX forward dynamics included (ABA since 2026-09-26; the earlier
# CRBA + Cholesky route was ~1e-4 on fetch and g1), stays within 1e-4 of pinocchio.
BOUND: dict = {}


@pytest.mark.parametrize("op", ["id", "crba", "fd", "id_du"])
def test_providers_agree_with_pinocchio(robot, op):
    name, contract, pyroffi_robot, model, to_pin, (q, qd, u) = robot
    ref = _reference(op, model, to_pin, q, qd, u).reshape(len(q), -1)
    call = {"id": contract.inverse_dynamics, "crba": lambda a, *_, tier: contract.mass_matrix(a, tier=tier),
            "fd": contract.forward_dynamics, "id_du": contract.inverse_dynamics_gradient}[op]
    results = {tier: np.asarray(call(q, qd, u, tier=tier)) for tier in contract.available(op)}
    if op == "fd":
        results["jax"] = np.asarray(jax.vmap(pyroffi_robot.forward_dynamics)(q, qd, u))
    assert {"thread", "block", "grid"} <= set(results)
    for tier, out in results.items():
        err = np.linalg.norm(out.reshape(len(q), -1) - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1.0)
        assert err.max() < BOUND.get(tier, 1e-4), (name, op, tier, float(np.median(err)), float(err.max()))


def test_auto_tier_follows_table(monkeypatch):
    """G1: GRiD wins the ID gradient at B=16 and thread wins every op from 4k (default table)."""
    contract = ContractDynamics(yourdfpy.URDF.load(URDFS["g1"], load_meshes=False))
    # The committed default table, not whatever this machine's per-robot calibration cached.
    monkeypatch.setattr(contract, "_table", json.loads(Path(
        "resources/tier_tables/contract_dynamics_sm86.json").read_text()))
    assert contract.select("id_du", 16) == "grid"
    assert contract.select("crba", 256) == "block"
    assert all(contract.select(op, 65536) == "thread" for op in ("id", "crba", "fd", "id_du"))
    monkeypatch.setenv("PYROFFI_TIER", "block")
    assert contract.select("id", 16) == "block"
    monkeypatch.delenv("PYROFFI_TIER")
    q = np.random.default_rng(1).uniform(-1, 1, (16, contract.n_q)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(contract.inverse_dynamics_gradient(q, q, q)),
                               np.asarray(contract.inverse_dynamics_gradient(q, q, q, tier="grid")), rtol=1e-6)
