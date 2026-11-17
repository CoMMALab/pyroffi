"""Tiered rigid-body dynamics from GRiM's mantis-traced kernels.

GRiM's :class:`grim.motion.dynamics.MantisDynamics` compiles these ops of one robot from
mantis traces at three tiers (one configuration per thread, warp or block):

======== ======================================= ==========================================
op       computes                                 inputs -> outputs
======== ======================================= ==========================================
id       inverse dynamics (RNEA)                  q, qd, qdd -> tau
crba     mass matrix (CRBA)                       q -> M
fd       forward dynamics (ABA)                   q, qd, tau -> qdd
fd_crba  forward dynamics as M qdd = tau - b      q, qd, tau -> qdd (Cholesky)
id_du    inverse-dynamics derivatives             q, qd, qdd -> [dtau/dq | dtau/dqd]
======== ======================================= ==========================================

``tier="auto"`` picks the tier per call at trace time from the batch size, using this robot's
calibration (:meth:`ContractDynamics.calibrate`) when one exists, else the measured default
table. ``fd`` also offers ``<tier>_crba`` (the ``fd_crba`` route). Arguments and results are
float32 in PyRoFFI's actuated joint order; gravity is Pinocchio's -9.81. mantis models no
joint damping, so the URDF should carry none.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_TABLE = _REPO / "resources" / "tier_tables" / "contract_dynamics_sm86.json"
# A tier replaces thread only when calibration measures it this much faster: below it the
# difference is launch/dispatch noise (~50 us per call through JAX) and flips run to run.
_MARGIN = 0.85
TIERS = ("thread", "warp", "block")
OPS = ("id", "crba", "fd", "fd_crba", "id_du")
CANDIDATES = {
    "id": TIERS,
    "crba": TIERS,
    "fd": TIERS + tuple(f"{t}_crba" for t in TIERS),
    "id_du": TIERS,
}


class ContractDynamics:
    """Dynamics of one robot at every tier mantis provides.

    Methods take ``(B, n)`` float32 arrays and a ``tier``: ``"auto"``, or one of
    :data:`CANDIDATES` for that op.
    """

    def __init__(self, urdf, ops: tuple[str, ...] | None = None):
        """``ops`` defaults to every op mantis supports for the robot (on mimic robots,
        forward dynamics only through CRBA)."""
        from .. import _grim
        from .._cuda_backends import _urdf_xml
        from .._robot import Robot

        _grim.ensure_importable()
        from grim.motion import MotionRobot
        from grim.motion.dynamics import CK_MIMIC_OPS, MantisDynamics

        self.actuated = list(Robot.from_urdf(urdf).joints.actuated_names)
        self.n_q = len(self.actuated)
        mrobot = MotionRobot.from_urdf(_urdf_xml(urdf))
        if list(mrobot.actuated_names) != self.actuated:
            raise ValueError(f"GRiM's actuated joints {mrobot.actuated_names} differ from "
                             f"pyroffi's {self.actuated}")
        if ops is None:
            mimic = bool((mrobot.mimic_act_idx != -1).any())
            ops = tuple(o for o in OPS if not mimic or o in CK_MIMIC_OPS)
        self._ck = MantisDynamics(mrobot, "mantis-f32", ops)
        self.ops = self._ck.ops
        # The build's content key (GRiM names targets grim_<symbol>_<key>): keys calibrations.
        self.key = next(iter(self._ck._targets.values())).rsplit("_", 1)[-1]
        self._table = self._load_table()

    # ── public ops ───────────────────────────────────────────────────────────────────────────
    def inverse_dynamics(self, q, qd, qdd, tier: str = "auto") -> Array:
        return self._run("id", tier, q, qd, qdd)

    def mass_matrix(self, q, tier: str = "auto") -> Array:
        return self._run("crba", tier, q)

    def forward_dynamics(self, q, qd, tau, tier: str = "auto") -> Array:
        return self._run("fd", tier, q, qd, tau)

    def inverse_dynamics_gradient(self, q, qd, qdd, tier: str = "auto") -> Array:
        """``[dtau/dq | dtau/dqd]``, shape ``(B, n, 2n)``, rows the output joint."""
        return self._run("id_du", tier, q, qd, qdd)

    # ── tier selection ───────────────────────────────────────────────────────────────────────
    def available(self, op: str) -> tuple[str, ...]:
        have = set(TIERS) if op in self.ops else set()
        if op == "fd" and "fd_crba" in self.ops:
            have |= {f"{t}_crba" for t in TIERS}
        return tuple(t for t in CANDIDATES[op] if t in have)

    def select(self, op: str, batch: int) -> str:
        """The tier ``tier="auto"`` uses for ``op`` at ``batch`` configurations.

        Precedence: ``$PYROFFI_TIER``, then the nearest calibrated (n, batch) point in log space
        (this robot's own calibration if it has one, else the default table), then ``thread``.
        """
        env = os.environ.get("PYROFFI_TIER")
        if env:
            return env
        rows = self._table.get(op)
        if not rows:
            return "thread"
        best = min(rows, key=lambda r: np.log(r["n"] / self.n_q) ** 2 + np.log(r["batch"] / batch) ** 2)
        return best["winner"] if best["winner"] in self.available(op) else "thread"

    def calibrate(self, batches=(16, 64, 256, 1024, 4096, 65536), repeats: int = 20) -> dict:
        """Time every tier of every op at ``batches`` on this GPU; cache and use the winners.

        A tier that fails at a size (one whose shared memory does not fit) is skipped
        there. The winner is the fastest tier if it beats ``thread`` by ``_MARGIN``, else
        ``thread``.
        """
        rows = {}
        for op in (o for o in ("id", "crba", "fd", "id_du") if o in self.ops):
            rows[op] = []
            for batch in batches:
                args = [jax.random.uniform(jax.random.PRNGKey(k), (batch, self.n_q), minval=-1, maxval=1)
                        for k in range(1 if op == "crba" else 3)]
                times = {}
                for tier in self.available(op):
                    f = jax.jit(lambda *a, tier=tier, op=op: self._run(op, tier, *a))
                    try:
                        for _ in range(3):
                            jax.block_until_ready(f(*args))
                        t0 = time.perf_counter()
                        for _ in range(repeats):
                            res = f(*args)
                        jax.block_until_ready(res)
                    except jax.errors.JaxRuntimeError:
                        continue
                    times[tier] = (time.perf_counter() - t0) / repeats * 1e3
                fastest = min(times, key=times.get)
                winner = fastest if times[fastest] < _MARGIN * times.get("thread", np.inf) else "thread"
                rows[op].append({"n": self.n_q, "batch": batch, "winner": winner,
                                 "ms": {k: round(v, 4) for k, v in times.items()}})
        self._table = rows
        path = self._calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        return rows

    # ── internals ────────────────────────────────────────────────────────────────────────────
    def _calibration_path(self) -> Path:
        from grim.motion._build import cache_root

        gpu = jax.devices()[0].device_kind.replace(" ", "_")
        return cache_root() / "tier_tables" / f"{self.key}_{gpu}.json"

    def _load_table(self) -> dict:
        for path in (self._calibration_path(), _DEFAULT_TABLE):
            if path.is_file():
                return json.loads(path.read_text())
        return {}

    def _run(self, op: str, tier: str, *args) -> Array:
        n = self.n_q
        args = [jnp.asarray(a, jnp.float32).reshape(-1, n) for a in args]
        batch = args[0].shape[0]
        if tier == "auto":
            tier = self.select(op, batch)
        if tier not in self.available(op):
            raise ValueError(f"{op}: tier {tier!r} not available ({self.available(op)})")
        if tier.endswith("_crba"):
            op, tier = "fd_crba", tier.removesuffix("_crba")
        return self._ck(op, *args, tier=tier)
