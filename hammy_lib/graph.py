import numpy as np
import xarray as xr
from .hammy_object import ArrayHammyObject


class Graph(ArrayHammyObject):
    def __init__(self, transition_matrix: np.ndarray, id: str = None):
        super().__init__(id)
        self._results = xr.Dataset(
            {
                "transition_matrix": (
                    ["position_index_from", "position_index_to"],
                    transition_matrix,
                )
            }
        )

    def calculate(self) -> None:
        import scipy.linalg as la
        tm = self._results["transition_matrix"].values
        eigvals, eigvecs = np.linalg.eig(tm)
        self._results["eigenvalues"] = (["eigen_index"], eigvals)
        self._results["eigenvectors"] = (["position_index", "eigen_index"], eigvecs)
        self._results["eigenvectors_inv"] = (["eigen_index", "position_index"], la.inv(eigvecs))

    def get_cells(self) -> list[tuple[int, ...]]:
        """Return list of cells. Each cell is a tuple of node indices.

        Override in subclasses with cell structure (e.g. squares for 2D lattice).
        """
        return []

    def get_node_to_cells(self) -> dict[int, list[int]]:
        """Reverse index: node → list of cell indices in get_cells()."""
        if not hasattr(self, '_node_to_cells_cache'):
            mapping: dict[int, list[int]] = {}
            for ci, cell in enumerate(self.get_cells()):
                for node in cell:
                    mapping.setdefault(node, []).append(ci)
            self._node_to_cells_cache = mapping
        return self._node_to_cells_cache

    def node_to_coords(self, node_index: int) -> tuple:
        """Convert flat node index to graph-specific coordinates. Override in subclasses."""
        return (node_index,)

    def generate_id(self) -> str:
        return f"graph_{hash(self._results['transition_matrix'].values.tobytes())}"

    @property
    def simple_name(self) -> str:
        return self.__class__.__name__


class LinearGraph(Graph):
    """Linear chain with lazy walk: P(stay)=0.5, P(±1)=0.25.

    Boundary nodes have self-loop probability 0.75 (= 0.5 + 0.25 reflected inward).
    Matches the effective bin dynamics of walk.c, which steps ±1 in raw position
    and stores raw//2 as the bin.
    """
    def __init__(self, length: int, id: str = None):
        tm = np.zeros((length, length))
        for i in range(length):
            tm[i, i] = 0.5
            if i > 0:
                tm[i, i - 1] = 0.25
            if i < length - 1:
                tm[i, i + 1] = 0.25
        tm[0, 0] = 0.75
        tm[-1, -1] = 0.75
        super().__init__(tm, id)


class LatticeGraph2D(Graph):
    """2D regular lattice with lazy walk.

    Each step picks one of 4 directions (±x, ±y) uniformly (prob 1/4 each),
    then the raw position is binned via //2.  Effective transition on bins:
    P(stay)=0.5, P(each neighbor)=0.125.  Boundary nodes absorb reflected
    probability into self-loops.

    Nodes are indexed in row-major order: node(r, c) = r * cols + c,
    matching xarray's stack(position_index=("x", "y")) ordering.
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols
        tm = np.zeros((n, n))
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                tm[idx, idx] = 0.5
                neighbors = 0
                if r > 0:
                    tm[idx, (r - 1) * cols + c] = 0.125
                    neighbors += 1
                if r < rows - 1:
                    tm[idx, (r + 1) * cols + c] = 0.125
                    neighbors += 1
                if c > 0:
                    tm[idx, r * cols + (c - 1)] = 0.125
                    neighbors += 1
                if c < cols - 1:
                    tm[idx, r * cols + (c + 1)] = 0.125
                    neighbors += 1
                # Absorb missing neighbors into self-loop
                tm[idx, idx] += (4 - neighbors) * 0.125
        super().__init__(tm, id)

    def get_cells(self) -> list[tuple[int, ...]]:
        """Enumerate all square cells in the lattice.

        Each cell is (n00, n10, n01, n11) where subscript (ds, dt):
        ds = column offset (0 or 1), dt = row offset (0 or 1).
        Node n00 is the top-left corner at (r, c).
        """
        cells = []
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                n00 = r * self.cols + c
                n10 = r * self.cols + (c + 1)
                n01 = (r + 1) * self.cols + c
                n11 = (r + 1) * self.cols + (c + 1)
                cells.append((n00, n10, n01, n11))
        return cells

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        """Convert flat node index to (row, col)."""
        return (node_index // self.cols, node_index % self.cols)
