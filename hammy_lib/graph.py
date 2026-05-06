from functools import cached_property
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay
from .hammy_object import ArrayHammyObject


class Graph(ArrayHammyObject):
    """Base class for random walk graphs.

    Stores transition_matrix (row-stochastic), node_positions, and spectral
    decomposition. Subclasses build these in __init__.

    Convention: T[i,j] = P(step to j | currently at i). Rows sum to 1.
    Distribution propagation: v_{t+1} = T.T @ v_t.
    """

    def __init__(self, transition_matrix: np.ndarray, node_positions: np.ndarray,
                 faces: list[list[int]], id: str = None):
        super().__init__(id)
        self._faces = faces
        self._results = xr.Dataset(
            {
                "transition_matrix": (
                    ["position_index_from", "position_index_to"],
                    transition_matrix,
                ),
                "node_positions": (
                    ["position_index", "spatial_dim"],
                    node_positions,
                ),
            }
        )

    @property
    def size(self) -> int:
        return self._results["transition_matrix"].shape[0]

    @property
    def dim(self) -> int:
        return self._results["node_positions"].shape[1]

    @property
    def faces(self) -> list[list[int]]:
        return self._faces

    def calculate(self) -> None:
        import scipy.linalg as la
        tm = self._results["transition_matrix"].values
        eigvals, eigvecs = np.linalg.eig(tm)
        self._results["eigenvalues"] = (["eigen_index"], eigvals)
        self._results["eigenvectors"] = (["position_index", "eigen_index"], eigvecs)
        self._results["eigenvectors_inv"] = (["eigen_index", "position_index"], la.inv(eigvecs))

    @cached_property
    def triangulation(self) -> Delaunay:
        positions = self._results["node_positions"].values
        return Delaunay(positions)

    def euclidean_to_barycentric(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        tri = self.triangulation
        idx = tri.find_simplex(point)
        T = tri.transform[idx]
        bary = T[:-1] @ (point - T[-1])
        return idx, np.append(bary, 1 - bary.sum())

    def barycentric_to_euclidean(self, simplex_idx: int, weights: np.ndarray) -> np.ndarray:
        nodes = self.triangulation.simplices[simplex_idx]
        positions = self._results["node_positions"].values
        return weights @ positions[nodes]

    def get_cells(self) -> list[tuple[int, ...]]:
        return [tuple(f) for f in self._faces]

    def get_node_to_cells(self) -> dict[int, list[int]]:
        if not hasattr(self, '_node_to_cells_cache'):
            mapping: dict[int, list[int]] = {}
            for ci, cell in enumerate(self._faces):
                for node in cell:
                    mapping.setdefault(node, []).append(ci)
            self._node_to_cells_cache = mapping
        return self._node_to_cells_cache

    def node_to_coords(self, node_index: int) -> tuple:
        positions = self._results["node_positions"].values
        return tuple(float(x) for x in positions[node_index])

    def generate_id(self) -> str:
        return f"graph_{self.__class__.__name__}_{self.generate_digest(self._results['transition_matrix'].values.tobytes().hex()[:64])}"

    @property
    def simple_name(self) -> str:
        return self.__class__.__name__


def _lazy_walk_tm(n: int, neighbors_fn, n_directions: int) -> np.ndarray:
    """Build a lazy random walk transition matrix with fixed step probabilities.

    P(attempt each of n_directions steps) = 0.5/n_directions.
    Blocked steps (boundary) are absorbed into the self-loop.
    This ensures T[i,j] = T[j,i] whenever the adjacency is undirected symmetric,
    because the off-diagonal probability is the same regardless of node degree.

    Args:
        n: number of nodes.
        neighbors_fn: callable(node_idx) -> list of neighbor indices.
        n_directions: total number of step directions in the lattice type.
    """
    tm = np.zeros((n, n))
    step_prob = 0.5 / n_directions
    for i in range(n):
        nb = neighbors_fn(i)
        tm[i, i] = 0.5 + (n_directions - len(nb)) * step_prob
        for j in nb:
            tm[i, j] = step_prob
    return tm


class LinearGraph(Graph):
    """Linear chain with lazy walk: P(stay)=0.5, P(±1)=0.25 each.

    Boundary nodes absorb the missing-neighbor probability into the self-loop.
    node_positions: (N, 2) with y=0, x = node index.
    faces: edges (pairs of adjacent nodes).
    """
    def __init__(self, length: int, id: str = None):
        node_positions = np.column_stack([np.arange(length, dtype=float),
                                          np.zeros(length)])

        def neighbors(i):
            nb = []
            if i > 0:
                nb.append(i - 1)
            if i < length - 1:
                nb.append(i + 1)
            return nb

        tm = _lazy_walk_tm(length, neighbors, n_directions=2)
        faces = [[i, i + 1] for i in range(length - 1)]
        super().__init__(tm, node_positions, faces, id)


class LatticeGraph2D(Graph):
    """2D square lattice with lazy walk.

    P(stay)=0.5, P(each of 4 steps)=0.125. Blocked steps → extra self-loop.
    Nodes indexed row-major: node(r, c) = r * cols + c.
    node_positions: (r, c) Euclidean coordinates.
    faces: unit squares (n00, n10, n01, n11).
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        node_positions = np.array(
            [(r, c) for r in range(rows) for c in range(cols)],
            dtype=float,
        )

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            nb = []
            if r > 0:
                nb.append((r - 1) * cols + c)
            if r < rows - 1:
                nb.append((r + 1) * cols + c)
            if c > 0:
                nb.append(r * cols + (c - 1))
            if c < cols - 1:
                nb.append(r * cols + (c + 1))
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=4)

        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                n00 = r * cols + c
                n10 = r * cols + (c + 1)
                n01 = (r + 1) * cols + c
                n11 = (r + 1) * cols + (c + 1)
                faces.append([n00, n10, n01, n11])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        return (node_index // self.cols, node_index % self.cols)


class TriangularGraph2D(Graph):
    """2D triangular lattice with lazy 6-direction walk.

    Six directions (in row-col index space): (0,±1), (±1,0), (+1,-1), (-1,+1).
    P(each step) = 0.5/6. Blocked steps → extra self-loop.
    node_positions: axial embedding — row r offset by 0.5*r in column direction.
      position(r,c) = (r*√3/2, c + r*0.5)
    faces: triangles.
    """
    STEPS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]

    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        sqrt3_2 = np.sqrt(3) / 2
        node_positions = np.array(
            [(r * sqrt3_2, c + r * 0.5) for r in range(rows) for c in range(cols)],
            dtype=float,
        )

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            nb = []
            for dr, dc in self.STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb.append(nr * cols + nc)
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=6)

        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                n_rc = r * cols + c
                n_rc1 = r * cols + (c + 1)
                n_r1c = (r + 1) * cols + c
                n_r1c1 = (r + 1) * cols + (c + 1)
                faces.append([n_rc, n_rc1, n_r1c])
                faces.append([n_rc1, n_r1c, n_r1c1])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        return (node_index // self.cols, node_index % self.cols)


class HexagonalGraph2D(Graph):
    """2D honeycomb lattice with lazy 3-direction walk.

    Sublattice A (r+c even): steps (+1,0), (-1,0), (0,+1).
    Sublattice B (r+c odd):  steps (+1,0), (-1,0), (0,-1).
    Adjacency is symmetric: A→B and B→A always pair correctly.
    P(each step) = 0.5/3. Blocked steps → extra self-loop.
    node_positions: B-sublattice nodes offset slightly (+0.1) in col for visual separation.
    faces: 4-node square cells (approximation for position detection).
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        node_positions = np.zeros((n, 2))
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                sublattice = (r + c) % 2
                node_positions[idx] = (float(r), float(c) + 0.1 * sublattice)

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            sublattice = (r + c) % 2
            if sublattice == 0:  # A
                steps = [(1, 0), (-1, 0), (0, 1)]
            else:  # B
                steps = [(1, 0), (-1, 0), (0, -1)]
            nb = []
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb.append(nr * cols + nc)
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=3)

        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                n00 = r * cols + c
                n10 = r * cols + (c + 1)
                n01 = (r + 1) * cols + c
                n11 = (r + 1) * cols + (c + 1)
                faces.append([n00, n10, n01, n11])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        return (node_index // self.cols, node_index % self.cols)


class BrickGraph2D(Graph):
    """2D brick (offset rectangular) lattice with lazy 6-direction walk.

    Uses the same 6 triangular-lattice directions as TriangularGraph2D but with
    brick Euclidean coordinates: odd rows offset by +0.5 in the column direction.
    This gives a symmetric adjacency and consistent step probabilities.
    P(each step) = 0.5/6. Blocked steps → extra self-loop.
    node_positions: (r, c + 0.5*(r%2)) for each node.
    faces: offset rectangular cells (4 nodes).
    """
    STEPS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]

    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        node_positions = np.array(
            [(float(r), float(c) + 0.5 * (r % 2)) for r in range(rows) for c in range(cols)],
            dtype=float,
        )

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            nb = []
            for dr, dc in self.STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb.append(nr * cols + nc)
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=6)

        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                n00 = r * cols + c
                n10 = r * cols + (c + 1)
                n01 = (r + 1) * cols + c
                n11 = (r + 1) * cols + (c + 1)
                faces.append([n00, n10, n01, n11])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        return (node_index // self.cols, node_index % self.cols)
