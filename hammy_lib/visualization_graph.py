"""Matplotlib visualization of graph topology and distributions.

Functions:
    visualize_graph(graph, ax=None, title=None, max_edges=2000)
    visualize_distribution(graph, distribution, ax=None, title=None, max_edges=2000)
    visualize_tier(tier, output_path=None, max_edges=1500)
    visualize_collection(output_dir="results/graphs")
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from pathlib import Path

from .graph import Graph, CubicLattice, Torus2D, Torus3D, PathGraph, CycleGraph, IcosahedralSphere


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tm_edges(graph: Graph, max_edges: int) -> list[tuple[int, int]]:
    """Upper-triangle edges from the transition matrix (undirected)."""
    tm = graph._results["transition_matrix"].values
    r, c = np.where(tm > 1e-12)
    mask = r < c
    r, c = r[mask], c[mask]
    edges = list(zip(r.tolist(), c.tolist()))
    if len(edges) > max_edges:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(edges), max_edges, replace=False)
        edges = [edges[k] for k in sorted(idx)]
    return edges


def _node_size(n: int) -> float:
    return max(1.5, min(25.0, 6000.0 / n))


def _node_values_default(graph: Graph) -> np.ndarray:
    """Degree (number of distinct neighbours) as a proxy for colouring."""
    tm = graph._results["transition_matrix"].values
    n = tm.shape[0]
    diag = tm[np.arange(n), np.arange(n)]
    return (1.0 - diag) / (1.0 / (2 * graph.size) + 1e-15)   # rough off-diag mass


# ---------------------------------------------------------------------------
# 2-D drawing
# ---------------------------------------------------------------------------

def _draw_2d(ax: plt.Axes, positions: np.ndarray, edges: list[tuple[int, int]],
             node_values: np.ndarray | None, cmap: str, title: str | None,
             n: int) -> None:
    """Draw nodes (col=x, row=y) and edges aligned to that convention."""
    # Edges: each endpoint (col_i, row_i) to match scatter below
    if edges:
        segs = [((positions[i, 1], positions[i, 0]),
                 (positions[j, 1], positions[j, 0])) for i, j in edges]
        lc = mc.LineCollection(segs, linewidths=0.6, colors="dimgray", alpha=0.5, zorder=1)
        ax.add_collection(lc)

    x, y = positions[:, 1], positions[:, 0]
    s = _node_size(n)
    if node_values is not None:
        sc = ax.scatter(x, y, c=node_values, cmap=cmap, s=s, zorder=2, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    else:
        ax.scatter(x, y, c=node_values or _node_values_default_from_pos(positions, n),
                   cmap="viridis", s=s, zorder=2, linewidths=0)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def _node_values_default_from_pos(positions, n):
    return None   # caller handles


def _draw_2d_full(ax, graph, positions, node_values, edges, cmap, title):
    """2-D draw with auto-colouring."""
    if edges:
        segs = [((positions[i, 1], positions[i, 0]),
                 (positions[j, 1], positions[j, 0])) for i, j in edges]
        lc = mc.LineCollection(segs, linewidths=0.6, colors="dimgray", alpha=0.45, zorder=1)
        ax.add_collection(lc)

    s = _node_size(graph.size)
    if node_values is not None:
        vals = node_values
    else:
        # colour by number of off-diagonal neighbours
        tm = graph._results["transition_matrix"].values
        n = graph.size
        vals = (tm > 1e-12).sum(axis=1) - 1   # subtract self-loop

    sc = ax.scatter(positions[:, 1], positions[:, 0],
                    c=vals, cmap=cmap, s=s, zorder=2, linewidths=0)
    if node_values is not None:
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# 1-D drawing  (PathGraph)
# ---------------------------------------------------------------------------

def _draw_1d(ax: plt.Axes, graph: Graph, positions: np.ndarray,
             node_values: np.ndarray | None, edges: list[tuple[int, int]],
             cmap: str, title: str | None) -> None:
    if edges:
        segs = [((positions[i, 0], 0.0), (positions[j, 0], 0.0)) for i, j in edges]
        lc = mc.LineCollection(segs, linewidths=1.0, colors="steelblue", alpha=0.7, zorder=1)
        ax.add_collection(lc)

    s = _node_size(graph.size)
    if node_values is not None:
        sc = ax.scatter(positions[:, 0], np.zeros(graph.size),
                        c=node_values, cmap=cmap, s=s, zorder=2, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    else:
        ax.scatter(positions[:, 0], np.zeros(graph.size),
                   color="steelblue", s=s, zorder=2, linewidths=0)

    ax.autoscale_view()
    ax.set_aspect("auto")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Torus 2-D: proper donut embedding in 3-D
# ---------------------------------------------------------------------------

def _torus2d_positions_3d(graph: Torus2D) -> np.ndarray:
    rows, cols = graph.rows, graph.cols
    R, r = 1.0, 0.32
    out = np.zeros((graph.size, 3))
    for i in range(graph.size):
        row, col = i // cols, i % cols
        theta = 2 * np.pi * row / rows
        phi = 2 * np.pi * col / cols
        out[i, 0] = (R + r * np.cos(theta)) * np.cos(phi)
        out[i, 1] = (R + r * np.cos(theta)) * np.sin(phi)
        out[i, 2] = r * np.sin(theta)
    return out


def _draw_torus2d(ax, graph: Torus2D, node_values: np.ndarray | None,
                  edges: list[tuple[int, int]], cmap: str, title: str | None) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    p = _torus2d_positions_3d(graph)

    if edges:
        segs = [[(p[i, 0], p[i, 1], p[i, 2]), (p[j, 0], p[j, 1], p[j, 2])]
                for i, j in edges]
        lc = Line3DCollection(segs, linewidths=0.4, colors="dimgray", alpha=0.35, zorder=1)
        ax.add_collection(lc)

    s = _node_size(graph.size)
    if node_values is not None:
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                        c=node_values, cmap=cmap, s=s, depthshade=True, linewidths=0)
    else:
        tm = graph._results["transition_matrix"].values
        vals = (tm > 1e-12).sum(axis=1) - 1
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=vals, cmap="viridis", s=s, depthshade=True, linewidths=0)

    ax.set_box_aspect([1, 1, 0.35])
    ax.set_axis_off()
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_zlim(-0.35, 0.35)
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# 3-D drawing: CubicLattice / Torus3D
# ---------------------------------------------------------------------------

def _draw_3d(ax, graph: Graph, positions: np.ndarray,
             node_values: np.ndarray | None, edges: list[tuple[int, int]],
             cmap: str, title: str | None) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if edges:
        segs = [[(positions[i, 0], positions[i, 1], positions[i, 2]),
                 (positions[j, 0], positions[j, 1], positions[j, 2])]
                for i, j in edges]
        lc = Line3DCollection(segs, linewidths=0.35, colors="dimgray", alpha=0.3, zorder=1)
        ax.add_collection(lc)

    s = _node_size(graph.size)
    if node_values is not None:
        sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=node_values, cmap=cmap, s=s, depthshade=True, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.04)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=positions[:, 2], cmap="viridis", s=s, depthshade=True, linewidths=0)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_torus2d(g: Graph) -> bool:
    return isinstance(g, Torus2D) and not isinstance(g, Torus3D)


def _is_3d(g: Graph) -> bool:
    return isinstance(g, (CubicLattice, Torus3D, IcosahedralSphere)) or _is_torus2d(g)


def _is_1d(g: Graph) -> bool:
    pos = g._results["node_positions"].values
    return isinstance(g, PathGraph) and pos.shape[1] == 2 and np.allclose(pos[:, 1], 0)


def _draw_graph_on_ax(ax, graph: Graph, node_values: np.ndarray | None,
                      max_edges: int, cmap: str, title: str | None) -> None:
    """Dispatch to the right draw function; ax must already be 3D if needed."""
    positions = graph._results["node_positions"].values
    edges = _get_tm_edges(graph, max_edges)

    if _is_torus2d(graph):
        _draw_torus2d(ax, graph, node_values, edges, cmap, title)
    elif graph.dim == 3:
        _draw_3d(ax, graph, positions, node_values, edges, cmap, title)
    elif _is_1d(graph):
        _draw_1d(ax, graph, positions, node_values, edges, cmap, title)
    else:
        _draw_2d_full(ax, graph, positions, node_values, edges, cmap, title)


def visualize_graph(graph: Graph, ax=None, title: str | None = None,
                    max_edges: int = 2000, cmap: str = "viridis") -> plt.Figure:
    """Draw graph topology (nodes coloured by degree, edges sampled).

    For Torus2D and 3D graphs, creates a 3D axes automatically when ax is None.
    """
    standalone = ax is None
    if standalone:
        if _is_3d(graph):
            fig = plt.figure(figsize=(6, 5))
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if title is None:
        title = f"{graph.__class__.__name__} (N={graph.size})"

    _draw_graph_on_ax(ax, graph, None, max_edges, cmap, title)

    if standalone:
        fig.tight_layout()
    return fig


def visualize_distribution(graph: Graph, distribution: np.ndarray,
                            ax=None, title: str | None = None,
                            max_edges: int = 2000, cmap: str = "hot_r") -> plt.Figure:
    """Draw a probability distribution on graph nodes."""
    standalone = ax is None
    if standalone:
        if _is_3d(graph):
            fig = plt.figure(figsize=(6, 5))
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if title is None:
        title = f"{graph.__class__.__name__} distribution"

    _draw_graph_on_ax(ax, graph, distribution, max_edges, cmap, title)

    if standalone:
        fig.tight_layout()
    return fig


def visualize_tier(tier: str, output_path=None, max_edges: int = 1500) -> plt.Figure:
    """Render all graph types for a given tier in a single figure."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from .collection import make_tier, GRAPH_TYPES

    graphs = make_tier(tier)
    n_graphs = len(GRAPH_TYPES)
    ncols = 4
    nrows = (n_graphs + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for i, name in enumerate(GRAPH_TYPES):
        g = graphs[name]
        title = f"{name}\nN={g.size}"
        if _is_3d(g):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        else:
            ax = fig.add_subplot(nrows, ncols, i + 1)
        _draw_graph_on_ax(ax, g, None, max_edges, "viridis", title)

    fig.suptitle(f"Graph Collection — Tier {tier}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def visualize_collection(output_dir="results/graphs") -> dict[str, Path]:
    """Generate and save tier visualizations for all tiers (XS, S, M, L)."""
    from .collection import TIERS

    output_dir = Path(output_dir)
    saved = {}
    for tier in TIERS:
        path = output_dir / f"tier_{tier}.png"
        visualize_tier(tier, output_path=path)
        plt.close("all")
        saved[tier] = path
    return saved
