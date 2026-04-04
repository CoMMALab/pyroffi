"""Visualize kinematic chain of a URDF as a BFS tree with link names.

Usage:
    python scripts/visualize_kinematic_chain.py resources/g1_description/g1_29dof_with_hand_rev_1_0_spherized.urdf
    python scripts/visualize_kinematic_chain.py resources/g1_description/g1_29dof_with_hand_rev_1_0_spherized.urdf --out g1_chain.png
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict, deque

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import yourdfpy


def build_link_graph(urdf: yourdfpy.URDF) -> tuple[nx.DiGraph, str]:
    """Build a directed graph of links connected by joints, return graph and root link name."""
    G = nx.DiGraph()

    # Add all links as nodes
    for link in urdf.robot.links:
        G.add_node(link.name)

    # Add edges: parent_link -> child_link, labeled by joint name and type
    for joint in urdf.robot.joints:
        parent = joint.parent
        child = joint.child
        G.add_edge(parent, child, joint_name=joint.name, joint_type=joint.type)

    # Find root: node with no incoming edges
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    assert len(roots) == 1, f"Expected exactly one root, got: {roots}"
    return G, roots[0]


def bfs_layout(G: nx.DiGraph, root: str) -> dict[str, tuple[float, float]]:
    """Compute node positions using BFS level-by-level, spreading children evenly."""
    # BFS to get levels
    levels: dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in G.successors(node):
            if child not in levels:
                levels[child] = levels[node] + 1
                queue.append(child)

    # Group nodes by level
    level_nodes: dict[int, list[str]] = defaultdict(list)
    for node, lvl in levels.items():
        level_nodes[lvl].append(node)

    # Assign x positions within each level, y = -level (so root is at top)
    pos: dict[str, tuple[float, float]] = {}
    for lvl, nodes in level_nodes.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * 2.0
            y = -lvl * 3.0
            pos[node] = (x, y)

    return pos


JOINT_TYPE_COLORS = {
    "revolute": "#4C9BE8",
    "continuous": "#7EC8A0",
    "prismatic": "#F4A261",
    "fixed": "#AAAAAA",
    "floating": "#C77DFF",
    "planar": "#FF6B6B",
}


def draw_tree(
    G: nx.DiGraph,
    root: str,
    title: str,
    out_path: pathlib.Path | None,
) -> None:
    pos = bfs_layout(G, root)

    # Determine figure size based on graph size
    n_nodes = G.number_of_nodes()
    max_level = max(p[1] for p in pos.values()) if pos else 0
    max_width = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()) if pos else 1
    fig_w = max(24, max_width * 0.8)
    fig_h = max(16, abs(max_level) * 0.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.axis("off")

    # Node color by joint type of incoming edge
    node_colors = []
    for node in G.nodes:
        in_edges = list(G.in_edges(node, data=True))
        if not in_edges:
            node_colors.append("#E63946")  # root: red
        else:
            jtype = in_edges[0][2].get("joint_type", "fixed")
            node_colors.append(JOINT_TYPE_COLORS.get(jtype, "#CCCCCC"))

    font_size = max(4, min(8, 200 // n_nodes))
    node_size = max(200, min(1200, 12000 // n_nodes))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        edge_color="#555555",
        width=0.8,
        connectionstyle="arc3,rad=0.0",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.92,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=font_size,
        font_color="black",
        font_weight="normal",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#E63946", label="root"),
    ]
    for jtype, color in JOINT_TYPE_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=jtype))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize URDF kinematic chain via BFS.")
    parser.add_argument("urdf", type=pathlib.Path, help="Path to URDF file.")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Output image path (PNG/PDF/SVG). If omitted, shows interactively.")
    args = parser.parse_args()

    urdf_path: pathlib.Path = args.urdf
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print(f"Loading URDF: {urdf_path}")
    urdf = yourdfpy.URDF.load(str(urdf_path))

    G, root = build_link_graph(urdf)
    n_links = G.number_of_nodes()
    n_joints = G.number_of_edges()
    print(f"  Links: {n_links},  Joints: {n_joints},  Root: {root}")

    title = f"Kinematic Chain — {urdf_path.stem}\n{n_links} links, {n_joints} joints  (root: {root})"

    out_path = args.out
    if out_path is None:
        out_path = urdf_path.with_suffix(".png").with_name(urdf_path.stem + "_chain.png")
        print(f"No --out specified, saving to {out_path}")

    draw_tree(G, root, title, out_path)


if __name__ == "__main__":
    main()
