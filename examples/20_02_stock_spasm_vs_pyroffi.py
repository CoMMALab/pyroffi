"""Stock SPaSM vs. the pyroffi-backed port: same solver, different backend

The dynamics claim in 20_01 only means something if the pyroffi backend is not
paying for it in throughput. This example is the control: it runs the *same*
SPaSM tetris-packing solver twice — once on stock SPaSM's own hand-written
kinematics (``commalab/spasm``, vendored at ``tamp/external/spasm_stack``), and
once on the pyroffi port — and compares wall-clock and solution quality.

Everything about the solver is held fixed: same ``SpasmParams``, same batch
sizes, same optimiser steps, same random seed, same cost function. Only the
kinematics/collision layer underneath differs:

* stock   — SPaSM's bespoke JAX FK, written for this one robot
* pyroffi — ``pyroffi.Robot`` / ``RobotCollisionSpherized`` loaded from the
            *same URDF*, giving the same 59 collision spheres

The expected result is parity, not a speedup. pyroffi is a general library
being asked to do a job a hand-specialised implementation was written for; the
claim being supported is that you give up nothing in throughput by moving to it
— and 20_01 shows what you gain.

Both solvers are timed after a warm-up call, because the first call compiles.

Run::

    python examples/20_02_stock_spasm_vs_pyroffi.py --blocks 3
    python examples/20_02_stock_spasm_vs_pyroffi.py --blocks 3 5 --trials 5
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TAMP_ROOT = REPO / "tamp"
STOCK_ROOT = TAMP_ROOT / "external" / "spasm_stock"

# Stock SPaSM and the port both define a top-level `spasm` package. They cannot
# coexist in one interpreter, so each side is timed in its own subprocess with
# the appropriate root on PYTHONPATH. This is also the honest way to measure it:
# neither side can warm the other's JIT cache.
_TIMER = r'''
import sys, time, json
import jax
sys.path.insert(0, {root!r})
{imports}

params = SpasmParams()
params.sampling_batch, params.opt_batch, params.opt_steps = {sb}, {ob}, {steps}
params.cost_thresh = {thresh}

sim = Simulation(num_blocks={n})
key = jax.random.key({seed})

# Warm-up: the first call pays compilation.
out = solve(params, sim, key)
jax.block_until_ready(out)

times = []
for _ in range({trials}):
    t0 = time.perf_counter()
    out = solve(params, sim, key)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)

# `solve` returns the best particle only; score it with the solver's own
# cost function so both sides are graded by identical math.
score = float(cost(params, sim, out))
print("RESULT " + json.dumps({{"times": times, "cost": score}}))
'''

_STOCK_IMPORTS = textwrap.dedent("""
    from spasm.solve import SpasmParams, solve, cost
    from spasm.tetris_env import Simulation
""").strip()

_PORT_IMPORTS = textwrap.dedent("""
    from spasm.tetris.solve import SpasmParams, solve, cost
    from spasm.tetris.env import Simulation
""").strip()


@contextlib.contextmanager
def meshcat_server(port=6000):
    """Stock SPaSM's ``Simulation.__init__`` connects to meshcat eagerly and
    blocks forever without a server; the port made the viewer lazy. Running one
    for the stock side keeps the comparison possible. It sits outside the timed
    region — ``Simulation`` is constructed before timing starts — so it costs
    the stock side nothing.
    """
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            yield          # already running; leave it alone
            return

    proc = subprocess.Popen(["meshcat-server"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise RuntimeError("meshcat-server did not come up on port 6000")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def time_side(label, root, imports, n, trials, seed, cfg):
    import json
    import os

    src = _TIMER.format(root=str(root), imports=imports, n=n, trials=trials,
                        seed=seed, **cfg)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    proc = subprocess.run([sys.executable, "-c", src], cwd=str(root),
                          capture_output=True, text=True, env=env, timeout=1800)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")),
                None)
    if line is None:
        print(f"  {label}: FAILED")
        print(textwrap.indent((proc.stderr or proc.stdout)[-1500:], "      "))
        return None
    payload = json.loads(line[len("RESULT "):])
    med = float(np.median(payload["times"])) * 1e3
    print(f"  {label:<10} {med:8.1f} ms   (cost {payload['cost']:.4f})", flush=True)
    payload["median_ms"] = med
    return payload


# Per-problem-size solver settings, copied from stock SPaSM's own __main__ so
# both sides run the configuration the original author tuned.
CONFIGS = {
    3: dict(sb=512, ob=64, steps=25, thresh=0.44),
    5: dict(sb=4096, ob=256, steps=25, thresh=0.42),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, nargs="+", default=[3])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not STOCK_ROOT.exists():
        sys.exit(f"missing {STOCK_ROOT} — run tamp/setup_externals.sh")

    rows = []
    for n in args.blocks:
        cfg = CONFIGS.get(n, CONFIGS[3])
        print(f"\ntetris packing, {n} blocks "
              f"(sampling_batch={cfg['sb']}, opt_steps={cfg['steps']}):")
        with meshcat_server():
            stock = time_side("stock", STOCK_ROOT, _STOCK_IMPORTS, n,
                              args.trials, args.seed, cfg)
        port = time_side("pyroffi", TAMP_ROOT, _PORT_IMPORTS, n,
                         args.trials, args.seed, cfg)
        rows.append((n, stock, port))

    print("\n=== Stock SPaSM vs pyroffi backend ===")
    print(f"{'blocks':>7} {'stock (ms)':>12} {'pyroffi (ms)':>14} {'ratio':>8}")
    for n, stock, port in rows:
        if stock is None or port is None:
            print(f"{n:>7} {'—':>12} {'—':>14} {'—':>8}")
            continue
        ratio = port["median_ms"] / stock["median_ms"]
        print(f"{n:>7} {stock['median_ms']:>12.1f} {port['median_ms']:>14.1f} "
              f"{ratio:>7.2f}x")

    print("\nParity here is the point: the pyroffi backend is a general library "
          "loaded\nfrom a URDF, matching a hand-specialised implementation on "
          "its own task —\nwhile additionally providing the differentiable "
          "dynamics that 20_01 uses.")


if __name__ == "__main__":
    main()
