"""Screen: with Path A's setup held FIXED and only the cost basis swapped, does
the RKHS inner solve actually converge -- and at what iteration budget?

`study3_identifiable_refit` gives Path A and Path B the same `n_iters=60`
forward solver.  Equal iteration COUNT is not equal convergence: the budget was
calibrated on the named cost, and the last run measured transport stationarity
at 1.483e-01 on the RKHS surface (`scratch/logs/study3_v3_both.log`), which
`ioc/inner.py` says invalidates the adjoint the study eigendecomposes.

This sweeps n_iters x kernel config and reports ||grad_x C|| per phase, for the
named cost (control) and for each kernel.  Everything else -- problem, scenes,
seeds, per-phase calibration, IK weights, the other three phases' weights, the
transport mass -- is byte-identical to `build_parametric` / `build_rkhs`.
"""
import sys, pathlib, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from iosp import pickplace as pp
from iosp.config import THETA_IK_STAR, Z_TRAJOPT_STAR
from iosp.model.pickplace import split_trajopt as _split_trajopt
from iosp.study0_segment_ablation import MESH_DIR, SRDF_PATH, URDF_PATH
from iosp.study3_identifiable_refit import _build_inner, _scenes_ab, scene_a

TOL = 1e-3


def rff_residual_fn(problem, M, key, ls, mode="base", form="sq"):
    """`make_rff_residual_fn` generalised over descriptor and feature form.
    `mode="base"`, `form="sq"`, M=16, ls=1.0 reproduces it exactly."""
    k1, k2 = jax.random.split(key)
    dof = problem.dof
    dim = {"base": 2 * dof, "geom": 2 * dof + 13}[mode]
    Omega = jax.random.normal(k1, (M, dim), dtype=jnp.float32) / ls
    b = jax.random.uniform(k2, (M,), dtype=jnp.float32) * 2.0 * jnp.pi
    amp = jnp.sqrt(2.0 / M)

    def residual_fn(x_flat, scene):
        q = problem.unpack(x_flat, scene)
        dq = q[1:] - q[:-1]
        u = jnp.concatenate([q[:-1], dq], axis=-1)
        if mode == "geom":
            p_l = problem.robot.forward_kinematics(q)[..., 4:7]
            d = jnp.linalg.norm(p_l - scene.obs_center, axis=-1)
            u = jnp.concatenate([u, d[:-1]], axis=-1)
        z = u @ Omega.T + b
        # form "sq":  residual  amp*cos(z)        -> feature amp^2 cos^2(z)
        # form "lin": residual  sqrt2*sin(z/2)    -> feature 1 - cos(z), exactly,
        #   via 1-cos(z) = 2 sin^2(z/2).  Writing it this way keeps the residual
        #   smooth: sqrt(max(1-cos,0)) has an infinite derivative at every zero
        #   and NaNs the Gauss-Newton solve.
        ph = amp * jnp.cos(z) if form == "sq" else jnp.sqrt(2.0) * jnp.sin(0.5 * z)
        return tuple(ph[:, j] for j in range(M))

    return residual_fn


KERNELS = [
    ("current  base M=16  ls=1.0  cos^2", dict(M=16,  ls=1.0,  mode="base", form="sq")),
    ("tuned    base M=64  ls=3.0  cos^2", dict(M=64,  ls=3.0,  mode="base", form="sq")),
    ("tuned    base M=256 ls=10   cos^2", dict(M=256, ls=10.0, mode="base", form="sq")),
    ("tuned    GEOM M=256 ls=10   1-cos", dict(M=256, ls=10.0, mode="geom", form="lin")),
]
N_ITERS = (60, 200)


def main():
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    scenes = _scenes_ab()
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)
    star_by_phase = _split_trajopt(theta_trajopt_star)
    transport_mass = float(jnp.sum(star_by_phase["transport"]))
    print(f"transport mass (pinned, same as build_rkhs) = {transport_mass:.4f}")

    def screen(inner, by_phase, tag):
        x0, phase_scenes, _, _ = prob.seeds(scenes, THETA_IK_STAR)
        worst, cells = 0.0, []
        for p in pp.PHASES:
            s = np.asarray(jax.vmap(inner[p].stationarity, in_axes=(0, None, 0))(
                x0[p], by_phase[p], phase_scenes[p]))
            m = float(s.max())
            if not np.isfinite(m):
                worst = float("nan")          # `max(x, nan)` silently returns x
            elif np.isfinite(worst):
                worst = max(worst, m)
            cells.append(f"{p[:5]} A={s[0]:.1e} B={s[1]:.1e}")
        ok = ("NaN" if not np.isfinite(worst)
              else "OK " if worst <= TOL else "BAD")
        print(f"    {tag:36s} worst={worst:.3e} {ok}  | " + " | ".join(cells), flush=True)
        return worst

    for n_iters in N_ITERS:
        print(f"\n=== forward solver n_iters = {n_iters} ===", flush=True)
        fs = pp.make_composed_forward_solver(n_iters=n_iters)

        # control: Path A's named library, identical call path
        t0 = time.perf_counter()
        inner_a, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, fs, 0)
        screen(inner_a, star_by_phase, "PATH A control (named library)")
        print(f"      ({time.perf_counter() - t0:.0f}s)", flush=True)

        for name, cfg in KERNELS:
            t0 = time.perf_counter()
            rff = rff_residual_fn(prob.seg["transport"], cfg["M"],
                                  jax.random.PRNGKey(7), cfg["ls"],
                                  cfg["mode"], cfg["form"])
            inner_k, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, fs, 0,
                                      {"transport": rff})
            by_phase = dict(star_by_phase)
            by_phase["transport"] = transport_mass * jnp.full(
                (cfg["M"],), 1.0 / cfg["M"], dtype=jnp.float32)
            screen(inner_k, by_phase, name)
            print(f"      ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
