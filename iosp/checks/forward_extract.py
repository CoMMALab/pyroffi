"""Extract a domain's FORWARD-pass joint trajectory + scene targets at Z_STAR.

The demonstrations the inverse pass tries to recover are rollouts of the composed
trajopt at the ground-truth cost `Z_STAR`.  This dumps, for one scene, the joint
path `q` (N_FULL, dof) plus the pick/place targets and block/obstacle geometry, so
a MuJoCo pass can check the trajectory is actually feasible (reachable, collision-
free, and the block ends where the skeleton says).

    python -m iosp.checks.forward_extract tower --out scratch/feas/tower.npz
    python -m iosp.checks.forward_extract tetris --out scratch/feas/tetris.npz
"""
import argparse
import pathlib

from iosp import config
config.setup()

import jax
import jax.numpy as jnp
import numpy as np


def _zstar_override(default, names):
    """Env `ZSTAR_SKEL` overrides the skeleton logit for the feasibility test."""
    import os
    boost = os.environ.get("ZSTAR_SKEL")
    if boost is None:
        return default
    z = np.asarray(default, np.float32).copy()
    z[list(names).index("skeleton")] = float(boost)
    print(f"  [Z_STAR override] skeleton logit -> {boost}", flush=True)
    return jnp.asarray(z)


def _tower():
    from iosp.experiments import e9_tower as E
    from iosp.model import tower as tw
    built = E.build(seed=0, n_iters=60, n_scenes=6, stack_level=0)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, tw.FEATURE_NAMES)

    def q_of(z):
        theta = jax.nn.softmax(z)
        xs, seg_scenes, full_sc, _, _ = prob.solve(
            scenes, built["inner_by_phase"], theta[:tw.K_SEG], theta, refine)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc)

    q = np.asarray(jax.jit(q_of)(zstar))            # (S, N_FULL, dof)
    meta = dict(
        domain="tower", idx_pick=tw.IDX_PICK, idx_place=tw.IDX_PLACE,
        n_full=tw.N_FULL, block_half=tw.BLOCK_HALF,
        pick_pos=np.asarray(scenes.pick_pos), place_pos=np.asarray(scenes.place_pos),
        target_z=np.asarray(scenes.target_z), q_start=np.asarray(scenes.q_start),
        obs_center=np.asarray(scenes.obs_center), obs_radius=np.asarray(scenes.obs_radius),
    )
    return q, meta


def _tetris():
    from iosp.experiments import e8_tetris as E
    from iosp.model import tetris as tt
    built = E.build(seed=0, n_iters=60, n_scenes=6, num_blocks=1)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, tt.FEATURE_NAMES)

    def q_of(z):
        theta = jax.nn.softmax(z)
        xs, seg_scenes, full_sc, _, _ = prob.solve(
            scenes, built["inner_by_phase"], theta[:tt.K_SEG], theta, refine)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc)

    q = np.asarray(jax.jit(q_of)(zstar))
    meta = dict(
        domain="tetris", idx_pick=tt.IDX_PICK, idx_place=tt.IDX_PLACE,
        n_full=tt.N_FULL,
        pick_pos=np.asarray(scenes.pick_pos), place_pos=np.asarray(scenes.place_pos),
        q_start=np.asarray(scenes.q_start),
    )
    return q, meta


def _pickplace():
    from iosp.experiments import e4_three_stage as E
    from iosp.model import pickplace as pp
    built = E.build(seed=0, n_iters=60, n_scenes=6)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, list(pp.THETA_SHARED_NAMES))

    def q_of(z):
        theta = jax.nn.softmax(z)
        theta_seg, theta_full = prob.split_shared(theta)
        x0, phase_scenes, q_pick, q_place = prob.seeds(scenes, E.THETA_IK)
        _, _, xs, ps = prob.solve(
            E.THETA_IK, {p: theta_seg for p in pp.PHASES}, scenes,
            built["inner_by_phase"], x0, refine=refine, theta_full=theta_full)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], ps["full"])

    q = np.asarray(jax.jit(q_of)(zstar))
    meta = dict(
        domain="pickplace",
        idx_pick=pp.SKELETON_PICK[1], idx_place=pp.SKELETON_PLACE[0],
        n_full=pp.N_FULL,
        pick_pos=np.asarray(scenes.pick_pos), place_pos=np.asarray(scenes.place_pos),
    )
    return q, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["tower", "tetris", "pickplace"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q, meta = {"tower": _tower, "tetris": _tetris,
               "pickplace": _pickplace}[args.domain]()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, q=q, **{k: np.asarray(v) for k, v in meta.items()
                          if not isinstance(v, str)},
             domain=meta["domain"])
    print(f"wrote {out}: q{q.shape}  pick={meta['pick_pos'][0]}  "
          f"place={meta['place_pos'][0]}", flush=True)
    # cheap sanity: EE at the pick/place skeleton rows vs the targets
    prob_ee = None
    # joint-limit check
    print(f"q range: [{q.min():.2f}, {q.max():.2f}] rad  "
          f"(dof={q.shape[-1]}, N_FULL={q.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
