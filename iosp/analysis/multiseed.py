"""Aggregate the multi-seed sweep into the numbers a paper can quote.

Stage A is reported as a PAIRED comparison: joint and EE at the same seed share
the feature-scale calibration and the cost starts, so the per-seed difference is
the statistic.  Reporting two independent means instead would throw away the
pairing and inflate the spread with variance that cancels.

Every stage-A row is read at the FINAL step, not at each run's own best.  Taking
each run's minimum would be selecting on the held-out criterion -- the exact
leakage `study6.select_winner` asserts against -- and would flatter both arms.
The best-held-out column is printed alongside only to show how much is being
left on the table by fixing the budget, never as the headline.
"""

import glob
import os
import re

import numpy as np


def _seed(path):
    """Seed from the filename; the pre-sweep seed-0 recordings carry no `seedN`
    tag in their name, so an untagged file IS seed 0 rather than unknown."""
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


RESULTS_ROOT = os.environ.get("IOSP_RESULTS", "iosp/data/viz")


def stage_a(root=None, seed0=None):
    root = root or os.path.join(RESULTS_ROOT, "multiseed")
    """joint vs EE, paired by seed, at the final step."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(root, "A_*_seed*.npz"))):
        d = np.load(f, allow_pickle=True)
        space = "joint" if "_joint_" in os.path.basename(f) else "ee"
        rows.setdefault(_seed(f), {})[space] = dict(
            ee_fit=float(d["ee_fit"][-1]), ee_held=float(d["ee_held"][-1]),
            best_held=float(np.min(d["ee_held"])),
            best_step=int(np.argmin(d["ee_held"])))
    if seed0:
        for space, v in seed0.items():
            rows.setdefault(0, {})[space] = v

    paired = sorted(s for s, v in rows.items() if {"joint", "ee"} <= set(v))
    print(f"\n=== Stage A: joint vs EE loss, {len(paired)} paired seeds "
          f"(final step, no held-out selection) ===")
    print(f"{'seed':>5} {'ee_held(ee)':>12} {'ee_held(joint)':>15} {'delta':>9} "
          f"{'rel':>8}   {'ee_fit(ee)':>11} {'ee_fit(joint)':>14}")
    dl = []
    for s in paired:
        e, j = rows[s]["ee"], rows[s]["joint"]
        d = j["ee_held"] - e["ee_held"]
        dl.append(d / e["ee_held"])
        print(f"{s:5d} {e['ee_held']:12.5f} {j['ee_held']:15.5f} {d:+9.5f} "
              f"{100*d/e['ee_held']:+7.1f}%   {e['ee_fit']:11.5f} {j['ee_fit']:14.5f}")
    if not paired:
        print("  (no paired seeds yet)")
        return rows
    dl = np.asarray(dl)
    he = np.array([rows[s]["ee"]["ee_held"] for s in paired])
    hj = np.array([rows[s]["joint"]["ee_held"] for s in paired])
    print(f"\n  ee    ee_held: mean {he.mean():.5f}  sd {he.std(ddof=1):.5f}  "
          f"min {he.min():.5f}  max {he.max():.5f}")
    print(f"  joint ee_held: mean {hj.mean():.5f}  sd {hj.std(ddof=1):.5f}  "
          f"min {hj.min():.5f}  max {hj.max():.5f}")
    print(f"  paired relative change: mean {100*dl.mean():+.1f}%  "
          f"sd {100*dl.std(ddof=1):.1f}%  "
          f"joint better on {int((dl < 0).sum())}/{len(dl)} seeds")
    # A sign test, not a t-test: 4-6 seeds is far too few to lean on normality,
    # and "wins on k of n seeds" is the claim actually being made.
    print(f"  -> the honest claim is the win RATE and the paired spread, "
          f"not a p-value at n={len(dl)}")
    return rows


def stage_b(root=None, extra=None):
    root = root or os.path.join(RESULTS_ROOT, "multiseed")
    extra = extra if extra is not None else (
        os.path.join(RESULTS_ROOT, "multistart_behavior.npz"),)
    files = sorted(glob.glob(os.path.join(root, "B_seed*.npz"))) + [
        f for f in extra if os.path.exists(f)]
    if not files:
        print("\n=== Stage B: no runs yet ===")
        return
    print(f"\n=== Stage B: 9-candidate multistart, 2x-harder held-out scene "
          f"({len(files)} seeds) ===")
    print(f"{'seed':>5} {'winner':>7} {'branch':>7} {'ee_held(win)':>13} "
          f"{'spread':>9} {'sel==best?':>11} {'regret':>9}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        w, S = int(d["winner"]), int(d["S"])
        eeh, he = d["ee_held_hist"][-1], d["held_hist"][-1]
        best = int(np.argmin(eeh))
        # Regret: what training-loss selection cost against an oracle that could
        # see the held-out criterion.  0.0 means the rule picked the best one.
        regret = float(eeh[w] - eeh[best])
        print(f"{_seed(f):5d} {w:7d} {w//S:7d} {float(eeh[w]):13.5f} "
              f"{float(he.max()/he.min()):8.1f}x {str(best == w):>11} {regret:9.5f}")


if __name__ == "__main__":
    # Seed 0 was recorded before the sweep existed, under the same protocol
    # (40 steps, LR 0.05, n_iters 60) but with the spectrum enabled and a
    # different output path, so it is folded in explicitly rather than silently
    # re-run or silently dropped.
    seed0 = {}
    for space, path in (("joint", os.path.join(RESULTS_ROOT, "loss_space_joint.npz")),
                        ("ee", os.path.join(RESULTS_ROOT, "loss_space_ee.npz"))):
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            seed0[space] = dict(ee_fit=float(d["ee_fit"][-1]),
                                ee_held=float(d["ee_held"][-1]),
                                best_held=float(np.min(d["ee_held"])),
                                best_step=int(np.argmin(d["ee_held"])))
    stage_a(seed0=seed0 or None)
    stage_b()
