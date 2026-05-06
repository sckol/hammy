"""Spatial visualization of distributions on graph nodes using Polyscope."""
import numpy as np

from .graph import Graph


def show_distribution(graph: Graph, distribution: np.ndarray,
                      quantity_name: str = "p") -> None:
    """Render a probability distribution on graph nodes via Polyscope.

    Args:
        graph: the graph (provides node_positions and faces).
        distribution: shape (N,) probability or density values per node.
        quantity_name: label shown in the Polyscope UI.
    """
    try:
        import polyscope as ps
    except ImportError as e:
        raise ImportError("polyscope is required for spatial visualization. "
                          "Install with: pip install polyscope") from e

    positions = graph._results["node_positions"].values
    faces = graph.faces

    ps.init()

    if positions.shape[1] == 2:
        positions_3d = np.column_stack([positions, np.zeros(len(positions))])
    else:
        positions_3d = positions

    mesh = ps.register_surface_mesh("graph", positions_3d, faces,
                                     smooth_shade=True)
    mesh.add_scalar_quantity(quantity_name, distribution, defined_on="vertices",
                             enabled=True)
    ps.show()
