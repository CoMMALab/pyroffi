"""Dynamics benchmark for pyroffi (bench_ik.py-style).

One file, two roles:

Dispatcher (no --solver):
    Runs in the pyroffi env on CPU with float64 (JAX_PLATFORMS=cpu and
    JAX_ENABLE_X64=1 are set before jax is imported, cf. bench_ik). For each
    robot it generates deterministic state targets (PRNGKey(0)), computes the
    float64 references with pyroffi-jax on CPU, writes
    resources/bench_dynamics_targets_<robot>.npz, and launches one isolated
    subprocess per (robot, solver) so each baseline runs in its own conda
    env with the GPU to itself.

Child (--robot R --solver S):
    Loads the sidecar, JITs the solver's ops, times them, and appends CSV
    rows to resources/bench_dynamics_results.csv. The dispatcher strips
    XLA_PYTHON_CLIENT_PREALLOCATE / JAX_ENABLE_X64 / JAX_PLATFORMS from the
    child environment, so GPU children get the default preallocating
    allocator, the branch-default float32, and the GPU exclusively.

Solvers and envs:
    pyroffi_jax     pyroffi (this env)        ID, FD, CRBA
    pyroffi_grid    pyroffi (this env)        ID, FD, CRBA, MINV, IDdu, FDdu
    grid_rbd        dynbench (grid-rbd 0.5.0) ID, FD, CRBA, MINV, IDdu, FDdu
    frax            dynbench (frax 0.0.5)     ID, FD, CRBA
    mjx             dynbench (mujoco-mjx 3.11) ID, FD, CRBA
    brax            dynbench (brax 0.14.2)    ID, FD
    pinocchio       dynbench-cpu (pin 4.1.0)  ID, FD, CRBA
    rbdl            dynbench-cpu (source)     ID, FD, CRBA

Ops:
    ID    inverse dynamics        (q, qd, qdd) -> tau
    FD    forward dynamics        (q, qd, tau) -> qdd
    CRBA  composite rigid-body M  (q) -> M
    MINV  inverse mass matrix     (q) -> M^-1
    IDdu  d tau / d(q, qd)        layout [d/dq | d/dqd]
    FDdu  d qdd / d(q, qd)        layout [d/dq | d/dqd]

Conventions:
    - All solvers run on a rewritten URDF (see _rewritten_urdf): zero-mass
      links get 1e-9 (MuJoCo drops zero-mass "phantom" bodies and their
      placeholder inertias), a floating root joint becomes fixed (pyroffi's
      dynamics expose only actuated DOFs, so its g1 is a fixed-base 29-DOF
      model), joint <dynamics> elements are stripped (frax/GRiD fold joint
      damping into the bias term; pyroffi's dynamics API does not model it),
      and degenerate inertia tensors (any non-positive diagonal) become
      spheres (MuJoCo rejects non-positive-definite inertias), MJCF-only
      <compiler> directives are dropped (MuJoCo would double-prefix mesh
      paths that already carry the meshdir), and <collision>/<visual> geoms
      are stripped (no solver uses geometry, and mjx cannot compile
      baxter's cylinder x mesh collision pair), and inertial-origin
      rotations are folded into the tensor (I' = R I R^T, origin rpy -> 0;
      RBDL's loader rejects rotated inertial origins, baxter's six gripper
      links). The first three rewrites are dynamics-neutral for the pyroffi
      reference (verified 0.0 rel diff); the last four only touch degenerate
      links, loader-only directives, geometry, and inertial frames, and
      every solver loads the identical file. Both the reference and every
      baseline use the same rewritten file, so all solvers compute the same
      model.
    - Joint vectors are in pyroffi order (URDF document order of actuated
      joints, robot.dynamics.dof_names). Baselines with a different internal
      order are permuted internally; a misordering shows up as a large
      rel_err_max vs the float64 reference.
    - Gravity -9.81 m/s^2 along -z (Featherstone spatial accel
      [0,0,9.81,0,0,0], frax's own default).
    - GPU children run float32; CPU children float64 (native); references
      are always float64.
    - brax ID is M(q)@qdd + bias: brax has no qdd-aware RNEA (its
      dynamics.inverse computes the bias force only).
    - rbdl assumes URDF document joint order (its wrapper has no name API).

Timing:
    - GPU/JAX children: a jit-compiled timer repeats the op
      N_DEVICE_REPEATS times in a lax.scan; median/p95 over N_TIMED outer
      runs (cf. bench_ik _time_scan).
    - CPU children (pinocchio, rbdl): wall-clock median/p95 over
      N_CPU_ITERS per-call loop iterations.
    - GPU children are sampled by an NVML monitor (20 ms) for peak/avg GPU
      utilization and peak VRAM.

rel_err_max = max|out - ref| / max(max|ref|, 1e-6), normalized by the
batch-max reference magnitude (robust to near-zero reference components;
empty for gradient ops, which are correctness-covered by
tests/test_grid_dynamics.py).

Usage:
    python tests/bench_dynamics.py                      # full dispatcher run
    python tests/bench_dynamics.py --disable-robot g1   # subset
    python tests/bench_dynamics.py --robot panda --solver frax   # one child
"""

import argparse
import csv
import datetime
import os
import pathlib
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np

# --- pre-import env ---------------------------------------------------------
# Children (launched with --solver) must NOT inherit the dispatcher's
# float64/CPU/no-prealloc settings; the dispatcher pops them from the child
# environment (see _run_solver_subprocess). The dispatcher itself needs
# float64 + CPU to compute references without touching the GPU.
_IS_CHILD = "--solver" in sys.argv[1:]
if not _IS_CHILD:
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_FILE = REPO_ROOT / "resources" / "bench_dynamics_results.csv"

ROBOTS = {
    "panda": "resources/panda/panda_spherized.urdf",
    "fetch": "resources/fetch/fetch_grid.urdf",
    "baxter": "resources/baxter/baxter.urdf",
    "g1": "resources/g1_description/g1_29dof.urdf",
}

BATCH_SIZES = [1, 16, 64, 256, 1024, 4096]
N_TIMED = 5
N_DEVICE_REPEATS = 5
N_CPU_ITERS = 50

SOLVERS = [
    "pyroffi_jax", "pyroffi_grid", "grid_rbd", "frax",
    "mjx", "brax", "pinocchio", "rbdl",
]
SOLVER_BACKEND = {
    "pyroffi_jax": "gpu", "pyroffi_grid": "gpu", "grid_rbd": "gpu",
    "frax": "gpu", "mjx": "gpu", "brax": "gpu",
    "pinocchio": "cpu", "rbdl": "cpu",
}
SOLVER_OPS = {
    "pyroffi_jax": ["ID", "FD", "CRBA"],
    "pyroffi_grid": ["ID", "FD", "CRBA", "MINV", "IDdu", "FDdu"],
    "grid_rbd": ["ID", "FD", "CRBA", "MINV", "IDdu", "FDdu"],
    "frax": ["ID", "FD", "CRBA"],
    "mjx": ["ID", "FD", "CRBA"],
    "brax": ["ID", "FD"],
    "pinocchio": ["ID", "FD", "CRBA"],
    "rbdl": ["ID", "FD", "CRBA"],
}
SOLVER_ENV_VAR = {
    "pyroffi_jax": None, "pyroffi_grid": None,
    "grid_rbd": "DYNBENCH_PYTHON", "frax": "DYNBENCH_PYTHON",
    "mjx": "DYNBENCH_PYTHON", "brax": "DYNBENCH_PYTHON",
    "pinocchio": "DYNBENCH_CPU_PYTHON", "rbdl": "DYNBENCH_CPU_PYTHON",
}
SOLVER_ENV_NAME = {
    "grid_rbd": "dynbench", "frax": "dynbench",
    "mjx": "dynbench", "brax": "dynbench",
    "pinocchio": "dynbench-cpu", "rbdl": "dynbench-cpu",
}
SOLVER_NOTE = {
    "pyroffi_jax": "",
    "pyroffi_grid": "pyroffi CUDA path (GRiD codegen via JAX FFI)",
    "grid_rbd": "stock GRiD 0.5.0 jax backend; joint order from handle _meta",
    "frax": "",
    "mjx": "URDF via built-in compiler; ID = mul_m + qfrc_bias after explicit crb (mjx.inverse reads a stale M); CRBA via crb + full_m",
    "brax": "ID = M(q)@qdd + bias (no qdd-aware RNEA in brax)",
    "pinocchio": "python-loop batching",
    "rbdl": "python-loop batching; assumes URDF document joint order",
}

_CSV_FIELDS = [
    "timestamp", "robot", "solver", "op", "batch", "n_dof", "backend",
    "t_med_ms", "t_p95_ms", "rel_err_max",
    "peak_gpu_pct", "avg_gpu_pct", "peak_vram_mb", "note",
]


# --- sidecar (dispatcher only; lazy imports) ---------------------------------

def _make_sidecar(robot, outdir):
    import jax
    import yourdfpy
    import pyroffi
    from pyroffi.dynamics import forward_dynamics, inverse_dynamics, mass_matrix

    urdf = yourdfpy.URDF.load(
        _rewritten_urdf(str(REPO_ROOT / ROBOTS[robot])), load_meshes=False
    )
    pb = pyroffi.Robot.from_urdf(urdf)
    dof_names = [str(s) for s in pb.dynamics.dof_names]
    n = len(dof_names)

    data = {
        "dof_names": np.array(dof_names),
        "n_dof": n,
        "urdf": np.array(ROBOTS[robot]),
    }
    keys = jax.random.split(jax.random.PRNGKey(0), len(BATCH_SIZES))
    for B, key in zip(BATCH_SIZES, keys):
        kq, kqd, kqdd, ktau = jax.random.split(key, 4)
        q = jax.random.normal(kq, (B, n))
        qd = jax.random.normal(kqd, (B, n))
        qdd = 5.0 * jax.random.normal(kqdd, (B, n))
        tau = 10.0 * jax.random.normal(ktau, (B, n))
        data[f"q_{B}"] = np.asarray(q)
        data[f"qd_{B}"] = np.asarray(qd)
        data[f"qdd_{B}"] = np.asarray(qdd)
        data[f"tau_{B}"] = np.asarray(tau)
        data[f"tau_ref_{B}"] = np.asarray(inverse_dynamics(pb, q, qd, qdd))
        data[f"qdd_ref_{B}"] = np.asarray(forward_dynamics(pb, q, qd, tau))
        data[f"M_ref_{B}"] = np.asarray(mass_matrix(pb, q))

    sidecar = outdir / f"bench_dynamics_targets_{robot}.npz"
    np.savez_compressed(sidecar, **data)
    print(f"[{robot}] sidecar: {sidecar}  n_dof={n}  dofs={dof_names}")


def _dof_names(z):
    return [str(s) for s in z["dof_names"]]


def _urdf_path(z):
    return str(REPO_ROOT / str(z["urdf"]))


_BASELINE_URDF_CACHE = {}


def _rewritten_urdf(src):
    """URDF rewritten so every solver sees exactly the model pyroffi sees.

    Seven rewrites. The first three are verified dynamics-neutral for
    pyroffi (identical M and tau to 0.0 rel diff); the last four only touch
    degenerate links, loader-only directives, geometry no solver uses, or
    inertial frames:

    1. Zero-mass links get mass 1e-9. MuJoCo's importer drops "phantom"
       bodies (zero mass, no geom) and silently discards their placeholder
       inertia. pyroffi's URDFs use zero-mass links as virtual attach frames
       with 0.1 kg-m^2 placeholder inertias (panda_link8, panda_grasptarget);
       un-rewritten, MuJoCo's M is missing exactly those inertias (~5-10%).
       The 1e-9 mass keeps the bodies non-phantom; MuJoCo then merges them
       through their fixed joints exactly as pyroffi composes them.
    2. A floating root joint becomes fixed. pyroffi's dynamics expose only
       actuated DOFs, so a free base is a frozen 0-DOF root -- a fixed-base
       model. The GRiD adapter (and frax) reject `floating` joints outright,
       and MuJoCo/Pinocchio would add a free joint; fixing the root makes all
       solvers compute the same fixed-base dynamics.
    3. Joint `<dynamics damping="...">` elements are dropped. pyroffi's
       dynamics API does not model joint damping, but frax and stock GRiD
       honor the URDF value and fold -d*qd into the bias term; on fetch
       (damping 1-5 per joint) that alone makes their ID/FD deviate ~20-50%
       from the reference while CRBA stays exact. Stripping the elements
       makes every solver compute the undamped model pyroffi computes.
    4. Degenerate inertia tensors become spheres. MuJoCo's compiler requires
       positive-definite inertias (positive eigenvalues AND the triangle
       inequalities A+B>=C); fetch's gripper fingers (ixx>0, iyy=izz=0) and
       baxter's degenerate fingertip links (all-zero tensor) fail that check
       outright. A link with any non-positive diagonal term gets all three
       diagonal terms set to the largest diagonal (or 1e-9 if the tensor is
       entirely zero) -- a physically plausible sphere at the same scale,
       loaded identically by every solver.
    5. MJCF-only `<compiler .../>` directives are dropped. Every URDF loader
       except MuJoCo's ignores them, but MuJoCo honors `meshdir`; g1's mesh
       filenames already carry the `meshes/` prefix, so keeping meshdir
       double-prefixes every path (`meshes/meshes/...`).
    6. `<collision>` and `<visual>` geoms are stripped. No solver in this
       bench uses geometry (dynamics read only joints and inertials), and
       MuJoCo/MJX would otherwise compile a collision kernel for every geom
       pair -- which fails outright on baxter (mjx does not implement
       cylinder x mesh). Bodies survive the strip because rule 1 guarantees
       every body has mass >= 1e-9, so none is "phantom" (zero mass, no geom).
    7. Inertial-origin rotations are folded into the tensor: `<inertial>`
       origins keep their CoM translation but get rpy="0 0 0", with the
       inertia tensor re-expressed in the link frame (I' = R I R^T). RBDL's
       URDF loader accepts a CoM offset but throws
       "rotation of body frames not yet supported" on a rotated inertial
       origin (baxter's six gripper links). Pure frame change: every other
       solver loads the same physics.
    """
    import math
    import re

    if src in _BASELINE_URDF_CACHE:
        return _BASELINE_URDF_CACHE[src]
    text = open(src).read()
    fixed = re.sub(r'(<mass\s+value=")0(\.0+)?(")', r"\g<1>1e-9\3", text)
    fixed = re.sub(r'(<joint name="[^"]*" type=)"floating"', r'\1"fixed"', fixed)
    fixed = re.sub(r'\s*<dynamics\b[^>]*/>', '', fixed)
    fixed = re.sub(r'\s*<dynamics\b[^>]*>.*?</dynamics>', '', fixed, flags=re.S)
    fixed = re.sub(r'\s*<compiler\b[^>]*/>', '', fixed)
    fixed = re.sub(r'\s*<collision\b[^>]*/>', '', fixed)
    fixed = re.sub(r'\s*<collision\b[^>]*>.*?</collision>', '', fixed, flags=re.S)
    fixed = re.sub(r'\s*<visual\b[^>]*/>', '', fixed)
    fixed = re.sub(r'\s*<visual\b[^>]*>.*?</visual>', '', fixed, flags=re.S)

    def _sphericalize_inertia(m):
        tag = m.group(0)
        vals = {}
        for k in ("ixx", "iyy", "izz"):
            mm = re.search(rf'{k}="([^"]+)"', tag)
            vals[k] = float(mm.group(1)) if mm else 0.0
        if min(vals.values()) > 0:
            return tag
        s = max(vals.values()) or 1e-9
        for k in ("ixx", "iyy", "izz"):
            tag = re.sub(rf'{k}="[^"]*"', f'{k}="{s:g}"', tag)
        return tag

    def _unrotate_inertial(m):
        # Rule 7: I' = R I R^T into the link frame, origin rpy -> "0 0 0"
        # (xyz translation kept). Runs before _sphericalize_inertia so a
        # rotated-but-degenerate tensor is still caught by rule 4.
        tag = m.group(0)
        om = re.search(r'<origin\b[^>]*/>', tag)
        if not om:
            return tag
        rpy = re.search(r'rpy="([^"]+)"', om.group(0))
        if not rpy:
            return tag
        r, p, y = (float(x) for x in rpy.group(1).split())
        if abs(r) < 1e-12 and abs(p) < 1e-12 and abs(y) < 1e-12:
            return tag
        im = re.search(r'<inertia\b[^>]*/>', tag)
        if not im:
            return tag
        c = {}
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            mm = re.search(rf'{k}="([^"]+)"', im.group(0))
            c[k] = float(mm.group(1)) if mm else 0.0
        I = [[c["ixx"], c["ixy"], c["ixz"]],
             [c["ixy"], c["iyy"], c["iyz"]],
             [c["ixz"], c["iyz"], c["izz"]]]
        cr_, sr_ = math.cos(r), math.sin(r)
        cp_, sp_ = math.cos(p), math.sin(p)
        cy_, sy_ = math.cos(y), math.sin(y)
        R = [[cy_ * cp_, cy_ * sp_ * sr_ - sy_ * cr_, cy_ * sp_ * cr_ + sy_ * sr_],
             [sy_ * cp_, sy_ * sp_ * sr_ + cy_ * cr_, sy_ * sp_ * cr_ - cy_ * sr_],
             [-sp_, cp_ * sr_, cp_ * cr_]]
        # I' = R I R^T
        Ip = [[sum(R[i][k] * I[k][j] for k in range(3)) for j in range(3)]
              for i in range(3)]
        Ip = [[sum(Ip[i][k] * R[j][k] for k in range(3)) for j in range(3)]
              for i in range(3)]
        inertia = im.group(0)
        keys = (("ixx", "ixy", "ixz"),
                ("ixy", "iyy", "iyz"),
                ("ixz", "iyz", "izz"))
        for i in range(3):
            for j in range(3):
                key = keys[i][j]
                inertia = re.sub(rf'({key}=")[^"]*(")',
                                 rf'\g<1>{Ip[i][j]:.17g}\2', inertia)
        new_tag = tag.replace(im.group(0), inertia)
        return new_tag.replace(f'rpy="{rpy.group(1)}"', 'rpy="0 0 0"')

    fixed = re.sub(r'<inertial\b.*?</inertial>', _unrotate_inertial, fixed, flags=re.S)
    fixed = re.sub(r'<inertia\b[^>]*?/>', _sphericalize_inertia, fixed)
    if fixed == text:
        _BASELINE_URDF_CACHE[src] = src
        return src
    # Write next to the source (deterministic name, gitignored): relative
    # paths like `meshes/foo.STL` inside the URDF resolve against the file's
    # own directory, so a /tmp copy breaks every baseline that loads meshes.
    out = os.path.join(os.path.dirname(src) or ".",
                       os.path.splitext(os.path.basename(src))[0]
                       + "__dynbench.urdf")
    with open(out, "w") as f:
        f.write(fixed)
    _BASELINE_URDF_CACHE[src] = out
    return out


def _urdf_path_for_solver(z):
    """Rewritten URDF path for this robot (see _rewritten_urdf)."""
    return _rewritten_urdf(_urdf_path(z))


def _permutations(z, base_names):
    """Index arrays between pyroffi joint order and `base_names` order.

    P_in[i]  = pyroffi index of base_names[i]  (index inputs:  pyroffi -> base)
    P_out[i] = base index of dof_names[i]      (index outputs: base -> pyroffi)
    """
    dof_names = _dof_names(z)
    P_in = np.asarray([dof_names.index(nm) for nm in base_names], dtype=np.int32)
    P_out = np.asarray([base_names.index(dn) for dn in dof_names], dtype=np.int32)
    return P_in, P_out


# --- solver builders (each child imports only its own stack) ------------------

def _build_pyroffi_jax(robot, z):
    import jax
    import yourdfpy
    import pyroffi
    from pyroffi.dynamics import forward_dynamics, inverse_dynamics, mass_matrix

    urdf = yourdfpy.URDF.load(_urdf_path_for_solver(z), load_meshes=False)
    pb = pyroffi.Robot.from_urdf(urdf)
    return {
        "ID": jax.jit(lambda q, qd, qdd, tau: inverse_dynamics(pb, q, qd, qdd)),
        "FD": jax.jit(lambda q, qd, qdd, tau: forward_dynamics(pb, q, qd, tau)),
        "CRBA": jax.jit(lambda q, qd, qdd, tau: mass_matrix(pb, q)),
    }


def _build_pyroffi_grid(robot, z):
    import jax
    import yourdfpy
    from pyroffi.dynamics import GRiDDynamics

    urdf = yourdfpy.URDF.load(_urdf_path_for_solver(z), load_meshes=False)
    gd = GRiDDynamics(urdf)
    return {
        "ID": jax.jit(lambda q, qd, qdd, tau: gd.inverse_dynamics(q, qd, qdd)),
        "FD": jax.jit(lambda q, qd, qdd, tau: gd.forward_dynamics(q, qd, tau)),
        "CRBA": jax.jit(lambda q, qd, qdd, tau: gd.mass_matrix(q)),
        "MINV": jax.jit(lambda q, qd, qdd, tau: gd.mass_matrix_inv(q)),
        "IDdu": jax.jit(lambda q, qd, qdd, tau: gd.inverse_dynamics_gradient(q, qd, qdd)),
        "FDdu": jax.jit(lambda q, qd, qdd, tau: gd.forward_dynamics_gradient(q, qd, tau)),
    }


def _build_grid_rbd(robot, z):
    import jax
    import jax.numpy as jnp
    import grid_rbd.jax as grid_jax

    h = grid_jax.register_robot(
        name=f"bench_dyn_{robot}",
        urdf_path=_urdf_path_for_solver(z),
        max_batch_size=max(BATCH_SIZES),
    )
    # GRiD v-slot order; the jax handle keeps it in a private _meta dict.
    base_names = [str(s) for s in h._base._meta["joint_names"]]
    P_in, P_out = _permutations(z, base_names)
    P_in, P_out = jnp.asarray(P_in), jnp.asarray(P_out)
    n = len(_dof_names(z))

    def _pin(x):
        return x[:, P_in]

    def _pout(x):
        return x[:, P_out]

    def _pmat(Mg):
        return Mg[:, P_out][:, :, P_out]

    def _grad2(out):
        # out: (*batch, n, 2n) [d/dq | d/dqd] in base order -> pyroffi order
        return jnp.stack((_pout(out[..., :n]), _pout(out[..., n:])), axis=-1)

    return {
        "ID": jax.jit(lambda q, qd, qdd, tau: _pout(h.inverse_dynamics(_pin(q), _pin(qd), _pin(qdd)))),
        "FD": jax.jit(lambda q, qd, qdd, tau: _pout(h.forward_dynamics(_pin(q), _pin(qd), _pin(tau)))),
        "CRBA": jax.jit(lambda q, qd, qdd, tau: _pmat(h.crba(_pin(q)))),
        "MINV": jax.jit(lambda q, qd, qdd, tau: _pmat(h.minv(_pin(q)))),
        "IDdu": jax.jit(lambda q, qd, qdd, tau: _grad2(h.inverse_dynamics_gradient(_pin(q), _pin(qd), _pin(qdd)))),
        "FDdu": jax.jit(lambda q, qd, qdd, tau: _grad2(h.forward_dynamics_gradient(_pin(q), _pin(qd), _pin(tau)))),
    }


def _build_frax(robot, z):
    import jax
    import jax.numpy as jnp
    from frax import Robot

    dof_names = _dof_names(z)
    r = Robot(_urdf_path_for_solver(z), joint_ordering=dof_names)
    if [str(s) for s in r.joint_names] != dof_names:
        raise ValueError(f"frax joint order mismatch: {r.joint_names} != {dof_names}")
    grav = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
    # Uniform 4-arg (q, qd, qdd, tau) signature: the dispatch below maps all
    # four inputs through vmap; each op uses the subset it needs.
    return {
        "ID": jax.jit(jax.vmap(lambda q1, qd1, qdd1, t1: r.rnea(q1, qd1, qdd1, grav, None))),
        "FD": jax.jit(jax.vmap(lambda q1, qd1, qdd1, t1: r.forward_dynamics(q1, qd1, t1, None))),
        "CRBA": jax.jit(jax.vmap(lambda q1, qd1, qdd1, t1: r.crba(q1))),
    }


def _build_mjx(robot, z):
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    m = mujoco.MjModel.from_xml_path(_urdf_path_for_solver(z))
    mj = mjx.put_model(m)  # C++ model -> JAX model (rne/crb require it)
    base_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in range(m.njnt)]
    P_in, P_out = _permutations(z, base_names)
    P_in, P_out = jnp.asarray(P_in), jnp.asarray(P_out)
    d0 = mjx.make_data(mj)

    # Uniform 4-arg (q, qd, qdd, tau) signature: the dispatch below maps all
    # four inputs through vmap; each op uses the subset it needs.
    # NOTE: make_data() leaves most derived fields (cinert, cvel, qfrc_bias,
    # qfrc_inverse, qacc_smooth) uninitialized (0 or nan), so each op must run
    # the pipeline steps that produce the field it reads -- mirroring the C
    # API where mj_inverse / mj_fwdPosition+mj_fwdVelocity / mj_crb do the
    # same.
    def _id1(q1, qd1, qdd1, t1):
        d = d0.replace(qpos=q1[P_in], qvel=qd1[P_in], qacc=qdd1[P_in])
        d = mjx.fwd_position(mj, d)   # cinert
        d = mjx.fwd_velocity(mj, d)   # qfrc_bias
        d = mjx.crb(mj, d)            # _impl.M -- mjx.inverse reads _impl.M but
        # never calls crb, so its tau is computed against a stale matrix
        return (mjx.mul_m(mj, d, d.qacc) + d.qfrc_bias)[P_out]

    def _fd1(q1, qd1, qdd1, t1):
        d = d0.replace(qpos=q1[P_in], qvel=qd1[P_in])
        d = mjx.fwd_position(mj, d)   # xpos, cinert, cdof_dot
        d = mjx.fwd_velocity(mj, d)   # cvel, qfrc_bias, qfrc_passive
        d = d.replace(qfrc_applied=t1[P_in])
        d = mjx.fwd_acceleration(mj, d)  # qacc_smooth = M^-1 (tau - bias + passive)
        return d.qacc_smooth[P_out]

    def _m1(q1, qd1, qdd1, t1):
        d = mjx.fwd_position(mj, d0.replace(qpos=q1[P_in]))
        d = mjx.crb(mj, d)
        return mjx.full_m(mj, d)[P_out][:, P_out]

    return {
        "ID": jax.jit(jax.vmap(_id1)),
        "FD": jax.jit(jax.vmap(_fd1)),
        "CRBA": jax.jit(jax.vmap(_m1)),
    }


def _build_brax(robot, z):
    import jax
    import jax.numpy as jnp
    import mujoco
    import brax
    from brax import kinematics as brax_kin
    from brax.generalized import dynamics, mass
    from brax.generalized.base import State
    from brax.io import mjcf as brax_mjcf

    mj = mujoco.MjModel.from_xml_path(_urdf_path_for_solver(z))
    sys = brax_mjcf.load_model(mj)
    base_names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in range(mj.njnt)]
    P_in, P_out = _permutations(z, base_names)
    P_in, P_out = jnp.asarray(P_in), jnp.asarray(P_out)

    def _init1(q1, qd1):
        x, xd = brax_kin.forward(sys, q1[P_in], qd1[P_in])
        st = State.init(q1[P_in], qd1[P_in], x, xd)
        st = dynamics.transform_com(sys, st)
        st = mass.matrix_inv(sys, st, 0)
        return st

    def _id1(q1, qd1, qdd1, t1):
        st = _init1(q1, qd1)
        bias = dynamics.inverse(sys, st)  # bias force only (no qdd term)
        return (mass.matrix(sys, st) @ qdd1[P_in] + bias)[P_out]

    def _fd1(q1, qd1, qdd1, t1):
        st = _init1(q1, qd1)
        qfrc = dynamics.forward(sys, st, t1[P_in])  # net joint force, not qdd
        return (st.mass_mx_inv @ qfrc)[P_out]

    return {
        "ID": jax.jit(jax.vmap(_id1)),
        "FD": jax.jit(jax.vmap(_fd1)),
    }


def _build_pinocchio(robot, z):
    import pinocchio as pin

    dof_names = _dof_names(z)
    model = pin.buildModelFromUrdf(_urdf_path_for_solver(z))
    data = model.createData()
    base_names = [str(model.names[j]) for j in range(1, model.njoints)]
    P_in, P_out = _permutations(z, base_names)
    P_in, P_out = list(P_in), list(P_out)
    n = len(dof_names)

    def _id(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n))
        for i in range(q.shape[0]):
            pin.rnea(model, data, q[i, P_in], qd[i, P_in], qdd[i, P_in])
            out[i] = data.tau[P_out]
        return out

    def _fd(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n))
        for i in range(q.shape[0]):
            a = pin.aba(model, data, q[i, P_in], qd[i, P_in], tau[i, P_in])
            out[i] = a[P_out]
        return out

    def _crba(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n, n))
        for i in range(q.shape[0]):
            pin.crba(model, data, q[i, P_in])
            out[i] = data.M[np.ix_(P_out, P_out)]
        return out

    return {"ID": _id, "FD": _fd, "CRBA": _crba}


def _build_rbdl(robot, z):
    import rbdl

    n = len(_dof_names(z))
    model = rbdl.loadModel(_urdf_path_for_solver(z), floating_base=False)
    if model.dof_count != n:
        raise ValueError(f"rbdl dof_count {model.dof_count} != {n}")

    def _id(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n))
        for i in range(q.shape[0]):
            t = np.zeros(n)
            rbdl.InverseDynamics(model, q[i], qd[i], qdd[i], t)
            out[i] = t
        return out

    def _fd(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n))
        for i in range(q.shape[0]):
            a = np.zeros(n)
            rbdl.ForwardDynamics(model, q[i], qd[i], tau[i], a)
            out[i] = a
        return out

    def _crba(q, qd, qdd, tau):
        out = np.empty((q.shape[0], n, n))
        for i in range(q.shape[0]):
            H = np.zeros((n, n))
            rbdl.CompositeRigidBodyAlgorithm(model, q[i], H)
            out[i] = H
        return out

    return {"ID": _id, "FD": _fd, "CRBA": _crba}


_BUILDERS = {
    "pyroffi_jax": _build_pyroffi_jax,
    "pyroffi_grid": _build_pyroffi_grid,
    "grid_rbd": _build_grid_rbd,
    "frax": _build_frax,
    "mjx": _build_mjx,
    "brax": _build_brax,
    "pinocchio": _build_pinocchio,
    "rbdl": _build_rbdl,
}


# --- timing / correctness -----------------------------------------------------

def _reference_for(op, B, z):
    if op == "ID":
        return z[f"tau_ref_{B}"]
    if op == "FD":
        return z[f"qdd_ref_{B}"]
    if op == "CRBA":
        return z[f"M_ref_{B}"]
    if op == "MINV":
        return np.linalg.inv(z[f"M_ref_{B}"])
    return None  # IDdu / FDdu: correctness covered by tests/test_grid_dynamics.py


def _rel_err(out, ref):
    """max|out-ref| normalized by max|ref| (floor 1e-6).

    Elementwise |out-ref|/|ref| blows up on near-zero reference components
    (a float32 abs error of 1e-3 vs a ref component of 1e-4 reads as rel
    error 10). Normalizing by the batch-max keeps the metric bounded at the
    float32 level (~1e-5..1e-4) and comparable across solvers and batches.
    """
    if ref is None:
        return float("nan")
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        raise ValueError(f"output shape {out.shape} != reference shape {ref.shape}")
    scale = max(float(np.max(np.abs(ref))), 1e-6)
    return float(np.max(np.abs(out - ref)) / scale)


def _make_gpu_timer(op_fn):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def timer(q, qd, qdd, tau):
        def body(carry, _):
            out = op_fn(q, qd, qdd, tau)
            return carry + jnp.sum(out).astype(jnp.float32), None

        carry, _ = jax.lax.scan(body, jnp.float32(0.0),
                                jnp.arange(N_DEVICE_REPEATS))
        return carry

    return timer


def _time_gpu(timer_fn, *args):
    import jax

    jax.block_until_ready(timer_fn(*args))  # compile + upload, untimed
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        out = timer_fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / N_DEVICE_REPEATS)
    return float(np.median(times)) * 1e3, float(np.percentile(times, 95)) * 1e3


def _time_cpu(fn, *args):
    for _ in range(2):
        fn(*args)  # warm (first calls allocate)
    times = []
    for _ in range(N_CPU_ITERS):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, float(np.percentile(times, 95)) * 1e3


# --- GPU monitoring (NVML, cf. bench_ik) ---------------------------------------

_nvml_handle = None


def _get_nvml_handle():
    global _nvml_handle
    if _nvml_handle is not None:
        return _nvml_handle or None  # cached False (no NVML) -> None
    try:
        import pynvml
    except ImportError:
        _nvml_handle = False
        return None
    pynvml.nvmlInit()
    idx = 0
    cudas = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cudas:
        first = cudas.split(",")[0].strip()
        if first.isdigit():
            idx = int(first)
    try:
        _nvml_handle = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx))
    except Exception:
        _nvml_handle = False
        return None
    return _nvml_handle


@contextmanager
def _gpu_monitor(interval_s=0.02):
    samples = {"gpu_util": [], "vram_mb": []}
    handle = _get_nvml_handle()
    if handle is None:
        yield samples
        return
    pynvml, dev = handle
    stop_evt = threading.Event()

    def _loop():
        while not stop_evt.wait(interval_s):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(dev)
                mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
                samples["gpu_util"].append(float(util.gpu))
                samples["vram_mb"].append(float(mem.used) / 1024**2)
            except Exception:
                pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_evt.set()
        t.join(timeout=1.0)


# --- CSV ------------------------------------------------------------------------

def _write_csv_rows(csv_file, rows):
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file.exists() or csv_file.stat().st_size == 0
    with open(csv_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# --- child ----------------------------------------------------------------------

def _run_child(robot, solver, csv_file):
    z = np.load(csv_file.parent / f"bench_dynamics_targets_{robot}.npz",
                allow_pickle=True)
    n = int(z["n_dof"])
    backend = SOLVER_BACKEND[solver]
    is_gpu = backend == "gpu"
    print(f"=== bench_dynamics child: {robot} / {solver} ({backend}, n_dof={n}) ===")

    ops = _BUILDERS[solver](robot, z)
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for B in BATCH_SIZES:
        if is_gpu:
            q = z[f"q_{B}"].astype(np.float32)
            qd = z[f"qd_{B}"].astype(np.float32)
            qdd = z[f"qdd_{B}"].astype(np.float32)
            tau = z[f"tau_{B}"].astype(np.float32)
        else:
            q = z[f"q_{B}"]
            qd = z[f"qd_{B}"]
            qdd = z[f"qdd_{B}"]
            tau = z[f"tau_{B}"]

        rows = []
        for op in SOLVER_OPS[solver]:
            fn = ops[op]
            out = np.asarray(fn(q, qd, qdd, tau))  # also transfers GPU arrays
            ref = _reference_for(op, B, z)
            rel = _rel_err(out, ref)

            if is_gpu:
                timer_fn = _make_gpu_timer(fn)
                with _gpu_monitor() as samples:
                    t_med, t_p95 = _time_gpu(timer_fn, q, qd, qdd, tau)
                if samples["gpu_util"]:
                    gpu_fields = (
                        f"{max(samples['gpu_util']):.1f}",
                        f"{float(np.mean(samples['gpu_util'])):.1f}",
                        f"{max(samples['vram_mb']):.0f}",
                    )
                else:
                    gpu_fields = ("", "", "")
            else:
                t_med, t_p95 = _time_cpu(fn, q, qd, qdd, tau)
                gpu_fields = ("", "", "")

            rows.append({
                "timestamp": ts, "robot": robot, "solver": solver, "op": op,
                "batch": B, "n_dof": n, "backend": backend,
                "t_med_ms": f"{t_med:.4f}", "t_p95_ms": f"{t_p95:.4f}",
                "rel_err_max": f"{rel:.3e}" if not np.isnan(rel) else "",
                "peak_gpu_pct": gpu_fields[0], "avg_gpu_pct": gpu_fields[1],
                "peak_vram_mb": gpu_fields[2], "note": SOLVER_NOTE[solver],
            })
            rel_s = "nan" if np.isnan(rel) else f"{rel:.2e}"
            print(f"  {op:5s} B={B:5d}: {t_med:8.3f} ms   rel_err={rel_s}")
        _write_csv_rows(csv_file, rows)
    print(f"=== done {robot} / {solver} ===")


# --- subprocess (cf. bench_ik _curobo_python_cmd / _run_solver_subprocess) -------

def _env_python_cmd(env_var, env_name):
    override = os.environ.get(env_var)
    if override:
        if os.path.isfile(override):
            return [override]
        print(f"[warn] {env_var}={override} is not a file; ignoring")
    for parent in pathlib.Path(sys.executable).resolve().parents:
        cand = parent / "envs" / env_name / "bin" / "python"
        if cand.is_file():
            return [str(cand)]
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        cand = pathlib.Path(prefix).parent / env_name / "bin" / "python"
        if cand.is_file():
            return [str(cand)]
    return ["conda", "run", "--no-capture-output", "-n", env_name]


def _run_solver_subprocess(robot, solver, csv_file, args):
    env_var = SOLVER_ENV_VAR[solver]
    prefix = [sys.executable] if env_var is None else _env_python_cmd(
        env_var, SOLVER_ENV_NAME[solver])
    cmd = prefix + [str(__file__), "--robot", robot, "--solver", solver]
    if args.outdir is not None:
        cmd += ["--outdir", str(args.outdir)]
    env = os.environ.copy()
    # Children get the default preallocating allocator, the branch-default
    # float32, and the GPU to themselves; references are already float64.
    for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_ENABLE_X64", "JAX_PLATFORMS"):
        env.pop(k, None)
    print(f"=== Running {robot} / {solver} in subprocess ===")
    subprocess.run(cmd, env=env, check=True)


# --- main ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disable-robot", action="append", choices=list(ROBOTS),
                   default=[], help="skip a robot (dispatcher only)")
    p.add_argument("--outdir", type=pathlib.Path, default=None,
                   help="write CSV + sidecars here instead of resources/")
    p.add_argument("--robot", choices=list(ROBOTS), help="child: which robot")
    p.add_argument("--solver", choices=SOLVERS, help="child: which solver")
    args = p.parse_args()

    if args.solver is not None:
        if args.robot is None:
            p.error("--solver requires --robot")
        csv_file = (args.outdir / CSV_FILE.name) if args.outdir is not None else CSV_FILE
        _run_child(args.robot, args.solver, csv_file)
        return

    csv_file = (args.outdir / CSV_FILE.name) if args.outdir is not None else CSV_FILE
    for robot in [r for r in ROBOTS if r not in args.disable_robot]:
        print(f"=== {robot}: generating targets + references ===")
        _make_sidecar(robot, csv_file.parent)
        for solver in SOLVERS:
            _run_solver_subprocess(robot, solver, csv_file, args)


if __name__ == "__main__":
    main()
