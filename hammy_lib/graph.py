from collections import defaultdict
import numpy as np
import xarray as xr
from .hammy_object import ArrayHammyObject


def _compute_binned_transition_matrix(rows, cols, bin_min_r, bin_min_c,
                                      step_fn, n_dirs, symmetrize=True):
    """Numerically compute exact bin-to-bin transition matrix.

    For each bin, enumerate all raw positions within it (using C-style //2),
    simulate all possible step directions, determine which bin each lands in,
    and average.

    Args:
        rows, cols: number of bins in each dimension.
        bin_min_r, bin_min_c: minimum bin coordinate (e.g. -7).
        step_fn: callable(raw_x, raw_y, direction) -> (new_x, new_y).
        n_dirs: number of possible step directions.
        symmetrize: if True, symmetrize via (T + T.T)/2 then re-normalize rows.
            Needed because //2 binning creates bins of unequal size near 0,
            breaking detailed balance.  The walk itself is reversible.

    Returns:
        (rows*cols, rows*cols) transition matrix.
    """
    n = rows * cols

    def c_div(a, b):
        """C-style integer division (truncate towards zero)."""
        if a >= 0:
            return a // b
        return -((-a) // b)

    def bin_to_idx(br, bc):
        return (br - bin_min_r) * cols + (bc - bin_min_c)

    def raw_positions_in_bin(br, bc):
        """Enumerate raw positions that map to bin (br, bc) via C-style //2."""
        positions = []
        # Raw x values: 2*br and 2*br+1 for br > 0; 2*br and 2*br-1 for br < 0; -1,0,1 for br=0
        for raw_r in range(2 * br - 1, 2 * br + 3):
            if c_div(raw_r, 2) != br:
                continue
            for raw_c in range(2 * bc - 1, 2 * bc + 3):
                if c_div(raw_c, 2) != bc:
                    continue
                positions.append((raw_r, raw_c))
        return positions

    tm = np.zeros((n, n))
    bin_max_r = bin_min_r + rows - 1
    bin_max_c = bin_min_c + cols - 1

    for br in range(bin_min_r, bin_min_r + rows):
        for bc in range(bin_min_c, bin_min_c + cols):
            src_idx = bin_to_idx(br, bc)
            raw_pos = raw_positions_in_bin(br, bc)
            n_raw = len(raw_pos)

            for raw_r, raw_c in raw_pos:
                for d in range(n_dirs):
                    new_r, new_c = step_fn(raw_r, raw_c, d)
                    new_br = c_div(new_r, 2)
                    new_bc = c_div(new_c, 2)
                    # Clamp to bin range (boundary absorption)
                    new_br = max(bin_min_r, min(bin_max_r, new_br))
                    new_bc = max(bin_min_c, min(bin_max_c, new_bc))
                    dst_idx = bin_to_idx(new_br, new_bc)
                    tm[src_idx, dst_idx] += 1.0 / (n_raw * n_dirs)

    if symmetrize:
        tm = (tm + tm.T) / 2
        # Re-normalize rows to sum to 1
        row_sums = tm.sum(axis=1)
        tm = tm / row_sums[:, np.newaxis]

    return tm


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


class TriangularGraph2D(Graph):
    """2D triangular lattice with 6-direction walk (//2 binned).

    Walk directions: (±1,0), (0,±1), (-1,+1), (+1,-1).
    Each direction chosen with prob 1/6.  After //2 binning:
    P(stay)≈5/12, P(axis neighbor)≈1/8, P(diagonal neighbor)≈1/24.

    Cells are triangles: each (r,c)→(r,c+1)→(r+1,c) (upward) and
    (r,c+1)→(r+1,c)→(r+1,c+1) (downward).
    """
    # Triangular walk: 6 directions
    STEP_DX = [1, -1, 0, 0, -1, 1]
    STEP_DY = [0, 0, 1, -1, 1, -1]

    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        bin_min = -(rows // 2)

        def step_fn(rx, ry, d):
            return rx + self.STEP_DX[d], ry + self.STEP_DY[d]

        tm = _compute_binned_transition_matrix(
            rows, cols, bin_min, bin_min, step_fn, 6
        )
        super().__init__(tm, id)

    def get_cells(self) -> list[tuple[int, ...]]:
        """Enumerate all triangular cells (upward + downward)."""
        cells = []
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                n_rc = r * self.cols + c
                n_rc1 = r * self.cols + (c + 1)
                n_r1c = (r + 1) * self.cols + c
                n_r1c1 = (r + 1) * self.cols + (c + 1)
                # Upward triangle: (r,c), (r,c+1), (r+1,c)
                cells.append((n_rc, n_rc1, n_r1c))
                # Downward triangle: (r,c+1), (r+1,c), (r+1,c+1)
                cells.append((n_rc1, n_r1c, n_r1c1))
        return cells

    def node_to_coords(self, node_index: int) -> tuple[int, int]:
        return (node_index // self.cols, node_index % self.cols)


class HexagonalGraph2D(Graph):
    """2D honeycomb lattice with 3-direction walk (//2 binned).

    Sublattice A ((x+y) even): steps (+1,0), (-1,0), (0,+1).
    Sublattice B ((x+y) odd):  steps (+1,0), (-1,0), (0,-1).
    Each direction chosen with prob 1/3.

    Cells: hexagonal faces of the honeycomb, each with 6 nodes.
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        bin_min = -(rows // 2)

        def step_fn(rx, ry, d):
            sublattice = (rx + ry) & 1  # 0 = A, 1 = B
            if d == 0:
                return rx + 1, ry
            elif d == 1:
                return rx - 1, ry
            else:  # d == 2
                return (rx, ry + 1) if sublattice == 0 else (rx, ry - 1)

        tm = _compute_binned_transition_matrix(
            rows, cols, bin_min, bin_min, step_fn, 3
        )
        super().__init__(tm, id)

    def get_cells(self) -> list[tuple[int, ...]]:
        """Hexagonal cells: 6 nodes forming each honeycomb face.

        Each face is bounded by 6 edges connecting alternating A/B sublattice
        nodes.  For practical implementation, we use the smallest enclosing
        structure: 2x2 blocks that tile the lattice.
        """
        # For a honeycomb on binned grid, cells are harder to define cleanly.
        # Use 4-node square cells (same as LatticeGraph2D) as an approximation.
        # The position detection will still work; the cell shape affects only
        # the GBC computation.
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
        return (node_index // self.cols, node_index % self.cols)


class BrickGraph2D(Graph):
    """2D brick (offset rectangular) lattice with 4-direction walk (//2 binned).

    Odd rows are offset by 0.5 in the column direction.  The walker steps:
    - Same row: (0, ±1)
    - Adjacent row: depends on row parity.
      Even row up: (-1, 0) or (-1, -1); down: (+1, 0) or (+1, -1)
      Odd row up: (-1, 0) or (-1, +1); down: (+1, 0) or (+1, +1)
    Each of the 4 directions chosen with prob 1/4.

    Cells: offset rectangles (4 nodes).
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        bin_min = -(rows // 2)

        def step_fn(rx, ry, d):
            parity = rx & 1  # 0 = even row, 1 = odd row
            if d == 0:  # right
                return rx, ry + 1
            elif d == 1:  # left
                return rx, ry - 1
            elif d == 2:  # up
                if parity == 0:
                    return rx - 1, ry  # even row up: no column shift
                else:
                    return rx - 1, ry + 1  # odd row up: shift right
            else:  # down
                if parity == 0:
                    return rx + 1, ry  # even row down: no column shift
                else:
                    return rx + 1, ry + 1  # odd row down: shift right

        tm = _compute_binned_transition_matrix(
            rows, cols, bin_min, bin_min, step_fn, 4
        )
        super().__init__(tm, id)

    def get_cells(self) -> list[tuple[int, ...]]:
        """Offset rectangular cells (4 nodes each)."""
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
        return (node_index // self.cols, node_index % self.cols)
