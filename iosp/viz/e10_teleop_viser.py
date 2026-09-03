"""Viser viewer for E10 method comparison: reconstructed EE paths vs demos.

Reads from saved data (iosp/data/results/e10_methods/paths.npz), does NOT
re-solve.  Shows each method's reconstruction overlaid on the demo, with
per-episode and per-method toggles.

Usage:
    python -m iosp.viz.e10_teleop_viser [--data-dir iosp/data/results/e10_methods]
"""
import argparse
import json
import pathlib

import numpy as np
import viser

METHOD_COLORS = {
    "implicit": (0x3b, 0x7d, 0xd8),   # blue
    "fd":       (0xd9, 0x53, 0x4f),    # red
    "cmaes":    (0x2e, 0x8b, 0x57),    # green
    "unrolled": (0xe6, 0x9f, 0x00),    # amber
}
DEMO_COLOR = (0x22, 0x22, 0x22)


def _polyline(server, name, pts, color, width):
    pts = np.asarray(pts, np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return server.scene.add_line_segments(
        name, points=segs,
        colors=np.tile(np.asarray(color, np.uint8), (len(segs), 2, 1)),
        line_width=width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default=str(pathlib.Path(__file__).resolve().parents[1]
                                    / "data" / "results" / "e10_methods"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    paths = np.load(data_dir / "paths.npz")
    demo = paths["demo"]  # (B, T, 3)
    B, T, _ = demo.shape

    summary = {}
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    episodes = summary.get("episodes", [f"ep_{i}" for i in range(B)])
    n_fit = summary.get("n_fit", B)

    methods = [m for m in ["implicit", "fd", "cmaes", "unrolled"] if m in paths]

    server = viser.ViserServer(port=args.port)
    print(f"Viser server at http://localhost:{args.port}", flush=True)

    # Controls
    with server.gui.add_folder("Episodes"):
        ep_toggles = []
        for i, name in enumerate(episodes):
            tag = "fit" if i < n_fit else "held"
            cb = server.gui.add_checkbox(f"{name} ({tag})", initial_value=True)
            ep_toggles.append(cb)

    with server.gui.add_folder("Methods"):
        method_toggles = {}
        for m in methods:
            method_toggles[m] = server.gui.add_checkbox(m, initial_value=True)
        demo_toggle = server.gui.add_checkbox("demo", initial_value=True)

    line_width_slider = server.gui.add_slider("Line width", min=1.0, max=10.0,
                                               step=0.5, initial_value=3.0)

    # Draw everything
    handles = {}

    def redraw(_=None):
        for key, h in handles.items():
            h.remove()
        handles.clear()

        lw = line_width_slider.value
        for i in range(B):
            if not ep_toggles[i].value:
                continue
            if demo_toggle.value:
                h = _polyline(server, f"demo/ep{i}", demo[i], DEMO_COLOR, lw + 1)
                handles[f"demo_{i}"] = h
                # Mark start/end
                server.scene.add_icosphere(f"demo/ep{i}/start", radius=0.008,
                                           color=DEMO_COLOR, position=demo[i, 0].tolist())
                handles[f"demo_{i}_start"] = server.scene._children[f"demo/ep{i}/start"]

            for m in methods:
                if not method_toggles[m].value:
                    continue
                p = paths[m]  # (B, T, 3)
                color = METHOD_COLORS.get(m, (0x80, 0x80, 0x80))
                h = _polyline(server, f"{m}/ep{i}", p[i], color, lw)
                handles[f"{m}_{i}"] = h

    redraw()

    # Register callbacks
    for cb in ep_toggles:
        cb.on_update(redraw)
    for cb in method_toggles.values():
        cb.on_update(redraw)
    demo_toggle.on_update(redraw)
    line_width_slider.on_update(redraw)

    # Add metrics panel if summary exists
    if "methods" in summary:
        with server.gui.add_folder("Metrics"):
            for m in methods:
                if m in summary["methods"]:
                    ms = summary["methods"][m]["metrics"]
                    server.gui.add_markdown(
                        f"**{m}**: EE fit={ms.get('ee_rmse_fit', '?'):.4f}m, "
                        f"gen={ms.get('ee_rmse_gen', '?'):.4f}m, "
                        f"compile={summary['methods'][m].get('wall_compile_s', '?'):.0f}s"
                    )

    # Keep alive
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
