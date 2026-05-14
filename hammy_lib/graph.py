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


class PathGraph(Graph):
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


# Backward-compatible alias
LinearGraph = PathGraph


class CycleGraph(Graph):
    """Ring (cycle) with lazy walk and periodic boundary.

    P(stay)=0.5, P(±1)=0.25 each. No boundary effects — all nodes identical.
    node_positions: unit circle, node i at angle 2πi/N.
    faces: edges (pairs of adjacent nodes including N-1 → 0 wrap).
    """
    def __init__(self, length: int, id: str = None):
        angles = 2 * np.pi * np.arange(length) / length
        node_positions = np.column_stack([np.cos(angles), np.sin(angles)])

        def neighbors(i):
            return [(i - 1) % length, (i + 1) % length]

        tm = _lazy_walk_tm(length, neighbors, n_directions=2)
        faces = [[i, (i + 1) % length] for i in range(length)]
        super().__init__(tm, node_positions, faces, id)


class SquareLattice(Graph):
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


# Backward-compatible alias
LatticeGraph2D = SquareLattice


class HexLattice(Graph):
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


# Backward-compatible alias
TriangularGraph2D = HexLattice


class BrickLattice(Graph):
    """2D brick-wall lattice with lazy walk.

    Odd rows are offset by +0.5 column, row spacing is 0.5 → 2:1 brick aspect ratio.
    Each interior node connects to 2 horizontal neighbors (same row) and 4 diagonal
    neighbors (2 above, 2 below). Diagonal directions depend on row parity so that all
    off-diagonal connections have equal Euclidean length sqrt(0.5).
    P(each of 6 step directions) = 0.5/6. Blocked → extra self-loop.
    node_positions: (r*0.5, c + 0.5*(r%2)).
    faces: rectangular cells (4 nodes).
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        # Row spacing 0.5, column spacing 1 → 2:1 brick aspect ratio
        node_positions = np.array(
            [(float(r) * 0.5, float(c) + 0.5 * (r % 2)) for r in range(rows) for c in range(cols)],
            dtype=float,
        )

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            nb = []
            # Horizontal (same row)
            for dc in (1, -1):
                nc = c + dc
                if 0 <= nc < cols:
                    nb.append(r * cols + nc)
            # Diagonal: parity-dependent so all diagonals have equal Euclidean length
            # Even row nodes at (r*0.5, c); nearest odd-row neighbors at c±0.5 → index c and c-1
            # Odd row nodes at (r*0.5, c+0.5); nearest even-row neighbors at c±0.5 → index c and c+1
            inter = [(1, 0), (1, -1), (-1, 0), (-1, -1)] if r % 2 == 0 else \
                    [(1, 0), (1,  1), (-1, 0), (-1,  1)]
            for dr, dc in inter:
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


# Backward-compatible alias
BrickGraph2D = BrickLattice


class CubicLattice(Graph):
    """3D cubic lattice with lazy 6-direction walk (±x, ±y, ±z).

    P(stay)=0.5, P(each of 6 steps)=1/12. Blocked steps → extra self-loop.
    Nodes indexed: node(x, y, z) = x * cols * depth + y * depth + z.
    node_positions: (x, y, z) Euclidean coordinates.
    faces: unit cube cells (8 nodes each) for position detection.
    """
    def __init__(self, rows: int, cols: int, depth: int, id: str = None):
        self.rows = rows
        self.cols = cols
        self.depth = depth
        n = rows * cols * depth

        node_positions = np.array(
            [(x, y, z)
             for x in range(rows)
             for y in range(cols)
             for z in range(depth)],
            dtype=float,
        )

        def idx(x, y, z):
            return x * cols * depth + y * depth + z

        def neighbors(i):
            x = i // (cols * depth)
            y = (i % (cols * depth)) // depth
            z = i % depth
            nb = []
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < rows and 0 <= ny < cols and 0 <= nz < depth:
                    nb.append(idx(nx, ny, nz))
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=6)

        # Unit cube cells (for position detection)
        faces = []
        for x in range(rows - 1):
            for y in range(cols - 1):
                for z in range(depth - 1):
                    faces.append([
                        idx(x, y, z), idx(x+1, y, z), idx(x, y+1, z), idx(x+1, y+1, z),
                        idx(x, y, z+1), idx(x+1, y, z+1), idx(x, y+1, z+1), idx(x+1, y+1, z+1),
                    ])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int, int]:
        x = node_index // (self.cols * self.depth)
        y = (node_index % (self.cols * self.depth)) // self.depth
        z = node_index % self.depth
        return (x, y, z)


class Torus2D(Graph):
    """2D square lattice with periodic boundary in both directions.

    P(stay)=0.5, P(each of 4 steps)=0.125. All nodes have exactly 4 neighbors.
    Nodes indexed row-major: node(r, c) = r * cols + c.
    node_positions: (r, c) Euclidean coordinates (flat embedding of torus).
    faces: unit squares (non-wrapping only, for visualization).
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
            return [
                ((r - 1) % rows) * cols + c,
                ((r + 1) % rows) * cols + c,
                r * cols + (c - 1) % cols,
                r * cols + (c + 1) % cols,
            ]

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


class Torus3D(Graph):
    """3D cubic lattice with periodic boundary in all three directions.

    P(stay)=0.5, P(each of 6 steps)=1/12. All nodes have exactly 6 neighbors.
    Nodes indexed: node(x, y, z) = x * cols * depth + y * depth + z.
    node_positions: (x, y, z) Euclidean coordinates (flat embedding of 3-torus).
    faces: unit cube cells (non-wrapping only, for position detection).
    """
    def __init__(self, rows: int, cols: int, depth: int, id: str = None):
        self.rows = rows
        self.cols = cols
        self.depth = depth
        n = rows * cols * depth

        node_positions = np.array(
            [(x, y, z)
             for x in range(rows)
             for y in range(cols)
             for z in range(depth)],
            dtype=float,
        )

        def idx(x, y, z):
            return x * cols * depth + y * depth + z

        def neighbors(i):
            x = i // (cols * depth)
            y = (i % (cols * depth)) // depth
            z = i % depth
            return [
                idx((x+1) % rows, y, z),
                idx((x-1) % rows, y, z),
                idx(x, (y+1) % cols, z),
                idx(x, (y-1) % cols, z),
                idx(x, y, (z+1) % depth),
                idx(x, y, (z-1) % depth),
            ]

        tm = _lazy_walk_tm(n, neighbors, n_directions=6)

        faces = []
        for x in range(rows - 1):
            for y in range(cols - 1):
                for z in range(depth - 1):
                    faces.append([
                        idx(x, y, z), idx(x+1, y, z), idx(x, y+1, z), idx(x+1, y+1, z),
                        idx(x, y, z+1), idx(x+1, y, z+1), idx(x, y+1, z+1), idx(x+1, y+1, z+1),
                    ])

        super().__init__(tm, node_positions, faces, id)

    def node_to_coords(self, node_index: int) -> tuple[int, int, int]:
        x = node_index // (self.cols * self.depth)
        y = (node_index % (self.cols * self.depth)) // self.depth
        z = node_index % self.depth
        return (x, y, z)


def _icosahedron_subdivide(k: int) -> tuple[np.ndarray, list[list[int]]]:
    """Subdivide a regular icosahedron k times per edge, project to unit sphere.

    Returns:
        vertices: (N, 3) float array on unit sphere, N = 10*k²+2.
        faces: list of [a, b, c] triangle index lists.
    """
    phi = (1 + np.sqrt(5)) / 2
    bv = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    bv /= np.linalg.norm(bv[0])

    base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    if k == 1:
        return bv, [list(f) for f in base_faces]

    verts = bv.tolist()
    edge_cache: dict[tuple[int, int], list[int]] = {}

    def get_edge_pts(a: int, b: int) -> list[int]:
        lo, hi = min(a, b), max(a, b)
        if (lo, hi) not in edge_cache:
            pa, pb = np.array(verts[lo]), np.array(verts[hi])
            pts = []
            for t in range(1, k):
                p = pa + t * (pb - pa) / k
                p /= np.linalg.norm(p)
                pts.append(len(verts))
                verts.append(p.tolist())
            edge_cache[(lo, hi)] = pts
        pts = edge_cache[(lo, hi)]
        return pts if a == lo else list(reversed(pts))

    all_tris: list[list[int]] = []

    for v0, v1, v2 in base_faces:
        # grid[r][s]: vertex index; row r has r+1 entries (0 ≤ s ≤ r ≤ k)
        # Barycentric: w0=(k-r)/k, w1=(r-s)/k, w2=s/k → w0*v0 + w1*v1 + w2*v2
        grid: list[list[int | None]] = [[None] * (r + 1) for r in range(k + 1)]
        grid[0][0] = v0
        grid[k][0] = v1
        grid[k][k] = v2

        for r, idx in enumerate(get_edge_pts(v0, v1), start=1):
            grid[r][0] = idx
        for s, idx in enumerate(get_edge_pts(v0, v2), start=1):
            grid[s][s] = idx
        for t, idx in enumerate(get_edge_pts(v1, v2), start=1):
            grid[k][t] = idx

        for r in range(2, k):
            for s in range(1, r):
                w0 = (k - r) / k
                w1 = (r - s) / k
                w2 = s / k
                p = (w0 * np.array(verts[v0]) +
                     w1 * np.array(verts[v1]) +
                     w2 * np.array(verts[v2]))
                p /= np.linalg.norm(p)
                grid[r][s] = len(verts)
                verts.append(p.tolist())

        for r in range(k):
            for s in range(r + 1):  # upward triangles
                all_tris.append([grid[r][s], grid[r + 1][s], grid[r + 1][s + 1]])
            for s in range(r):       # downward triangles
                all_tris.append([grid[r][s], grid[r][s + 1], grid[r + 1][s + 1]])

    return np.array(verts), all_tris


class IcosahedralSphere(Graph):
    """Near-isotropic sphere graph via icosahedral subdivision.

    Subdivides a regular icosahedron k times per edge and projects all vertices
    to the unit sphere. N = 10*k² + 2 nodes. Interior nodes have degree 6; the
    12 original icosahedron vertices have degree 5. All edges have nearly equal
    Euclidean length — much more isotropic than lat-lon grids.

    Lazy walk: n_directions=6 (5-degree nodes accumulate extra self-loop).
    node_positions: 3D unit sphere coordinates.
    faces: triangular sub-faces of the icosahedral subdivision.
    """
    def __init__(self, k: int, id: str = None):
        self.k = k
        verts, tris = _icosahedron_subdivide(k)
        n = len(verts)

        adj: list[set[int]] = [set() for _ in range(n)]
        for a, b, c in tris:
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])

        tm = _lazy_walk_tm(n, lambda i: sorted(adj[i]), n_directions=6)
        super().__init__(tm, verts, tris, id)

    def generate_id(self) -> str:
        return f"graph_IcosahedralSphere_k{self.k}_{self.generate_digest(str(self.k))}"

    @property
    def simple_name(self) -> str:
        return f"IcosahedralSphere(k={self.k})"


class SphereApprox(Graph):
    """Spherical lattice: lat-lon grid projected onto the unit sphere.

    rows × cols = N nodes. Latitude offset by 0.5 bin to avoid degenerate poles.
    Connectivity: 4-neighbor — periodic in longitude, non-periodic in latitude.
    node_positions: 3D unit sphere coordinates (x, y, z).
    faces: quad cells (wrapping in longitude, non-wrapping in latitude).
    """
    def __init__(self, rows: int, cols: int, id: str = None):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        # Latitude: uniform bins from south to north, offset 0.5 to avoid poles
        lats = np.pi * (np.arange(rows) + 0.5) / rows - np.pi / 2
        lons = 2.0 * np.pi * np.arange(cols) / cols

        node_positions = np.zeros((n, 3), dtype=float)
        for r in range(rows):
            for c in range(cols):
                lat, lon = lats[r], lons[c]
                node_positions[r * cols + c] = [
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat),
                ]

        def neighbors(idx):
            r, c = idx // cols, idx % cols
            nb = [
                r * cols + (c - 1) % cols,   # west (periodic)
                r * cols + (c + 1) % cols,   # east (periodic)
            ]
            if r > 0:
                nb.append((r - 1) * cols + c)  # south
            if r < rows - 1:
                nb.append((r + 1) * cols + c)  # north
            return nb

        tm = _lazy_walk_tm(n, neighbors, n_directions=4)

        # Quad faces (longitude wraps, latitude does not)
        faces = []
        for r in range(rows - 1):
            for c in range(cols):
                faces.append([
                    r * cols + c,
                    r * cols + (c + 1) % cols,
                    (r + 1) * cols + c,
                    (r + 1) * cols + (c + 1) % cols,
                ])

        super().__init__(tm, node_positions, faces, id)
