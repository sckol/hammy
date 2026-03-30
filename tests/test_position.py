"""Regression tests for position detection algorithms.

Tests verify that all algorithm variants produce correct results on known inputs.
Run with: .venv/bin/pytest tests/test_position.py -v
"""
import numpy as np
import pytest
from hammy_lib.calculations.position import (
    _find_power, _precompute_power_search,
    _identify_simplex, _identify_cell,
    _compute_position, _compute_position_cell, _compute_position_cell_fast,
    _assemble_cell_result,
)
from .conftest import make_peaked_distribution, make_gaussian_2d


# ---------- Power search ----------

class TestFindPower:
    def test_basic(self, square_graph):
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        x = make_peaked_distribution(225, 112, [113, 127])
        p = _find_power(eigvals, eigvecs_inv, x)
        assert 1.0 <= p <= 10000.0

    def test_precomputed_matches(self, square_graph):
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        x = make_peaked_distribution(225, 112, [113, 127])
        p_plain = _find_power(eigvals, eigvecs_inv, x)
        precomp = _precompute_power_search(eigvals, eigvecs_inv)
        p_fast = _find_power(eigvals, eigvecs_inv, x, precomputed=precomp)
        assert abs(p_plain - p_fast) < 0.01

    def test_spread_distribution_high_power(self, square_graph):
        """A spread-out distribution should give high power (long diffusion)."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        # Nearly uniform = very diffused = high power
        x = np.ones(225) + np.random.default_rng(42).normal(0, 0.01, 225)
        x = np.abs(x)
        x_norm = x / x.sum()
        p = _find_power(eigvals, eigvecs_inv, x_norm)
        assert p > 50  # very spread → high power


# ---------- Simplex identification ----------

class TestSimplex:
    def test_edge_on_linear_graph(self, linear_graph):
        """Simplex on 1D linear graph should find an edge (2 nodes)."""
        tm = linear_graph.results["transition_matrix"].values
        adjacency = (tm > 0) & ~np.eye(101, dtype=bool)
        indices = np.array([50, 51])
        weights = np.array([0.6, 0.4])
        nodes, bary, fq = _identify_simplex(indices, weights, adjacency, max_dim=1)
        assert len(nodes) == 2
        assert abs(fq - 1.0) < 0.01

    def test_no_triangle_on_square_lattice(self, square_graph):
        """Square lattice is bipartite — max simplex is edge, not triangle."""
        tm = square_graph.results["transition_matrix"].values
        adjacency = (tm > 0) & ~np.eye(225, dtype=bool)
        # 4 nodes of a square cell
        indices = np.array([112, 113, 127, 128])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        nodes, bary, fq = _identify_simplex(indices, weights, adjacency, max_dim=2)
        # Should find edge (2 nodes), not triangle
        assert len(nodes) == 2
        assert fq < 0.7  # only captures ~50% of weight


# ---------- Cell identification ----------

class TestCell:
    def test_square_cell_perfect_fit(self, square_graph):
        """4 corners of a square cell should give fit_quality=1.0."""
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        # nodes (7,7),(7,8),(8,7),(8,8) = indices 112,113,127,128
        indices = np.array([112, 113, 127, 128])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        cell_nodes, (s, t), cell_dim, fq = _identify_cell(indices, weights, cells, n2c)
        assert abs(fq - 1.0) < 0.01
        assert cell_dim >= 2  # face (3 = all 4 corners nonzero)
        assert abs(s - 0.5) < 0.01
        assert abs(t - 0.5) < 0.01

    def test_triangle_cell(self, triangular_graph):
        """3 corners of a triangle should give fit_quality=1.0."""
        cells = triangular_graph.get_cells()
        n2c = triangular_graph.get_node_to_cells()
        # center of 21x21 grid: node (10,10)=10*21+10=220
        # upward triangle: (10,10),(10,11),(11,10) = 220,221,241
        indices = np.array([220, 221, 241])
        weights = np.array([0.5, 0.3, 0.2])
        cell_nodes, (s, t), cell_dim, fq = _identify_cell(indices, weights, cells, n2c)
        assert abs(fq - 1.0) < 0.01
        assert cell_dim == 2  # face (triangle)

    def test_single_node_vertex(self, square_graph):
        """Single NNLS component → vertex (cell_dim=0)."""
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        indices = np.array([112])
        weights = np.array([1.0])
        cell_nodes, (s, t), cell_dim, fq = _identify_cell(indices, weights, cells, n2c)
        assert cell_dim == 0

    def test_edge_two_nodes(self, square_graph):
        """Two adjacent nodes → edge (cell_dim=1)."""
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        indices = np.array([112, 113])
        weights = np.array([0.6, 0.4])
        cell_nodes, (s, t), cell_dim, fq = _identify_cell(indices, weights, cells, n2c)
        assert cell_dim == 1


# ---------- Full position computation ----------

class TestPositionComputation:
    def test_cell_position_at_center(self, square_graph):
        """Distribution peaked at center (7,7) → position ≈ (7,7)."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = make_peaked_distribution(225, 112, [113, 127])
        result = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        assert abs(result['position_row'] - 7.0) < 0.5
        assert abs(result['position_col'] - 7.0) < 0.5
        assert result['fit_quality'] > 0.9

    def test_cell_diagonal_position(self, square_graph):
        """Gaussian between (7,7) and (8,8) → position ≈ (7.5, 7.5)."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = make_gaussian_2d(15, 15, 7.5, 7.5, sigma=2.0)
        result = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        assert result['fit_quality'] > 0.8
        assert abs(result['position_row'] - 7.5) < 1.0
        assert abs(result['position_col'] - 7.5) < 1.0

    def test_triangular_center(self, triangular_graph):
        """Triangular graph position at center → correct coordinates."""
        eigvals = triangular_graph.results["eigenvalues"].values
        eigvecs = triangular_graph.results["eigenvectors"].values
        eigvecs_inv = triangular_graph.results["eigenvectors_inv"].values
        cells = triangular_graph.get_cells()
        n2c = triangular_graph.get_node_to_cells()
        x = make_gaussian_2d(21, 21, 10.0, 10.0, sigma=3.0)
        result = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, triangular_graph.node_to_coords,
        )
        assert abs(result['position_row'] - 10.0) < 0.5
        assert abs(result['position_col'] - 10.0) < 0.5
        assert result['fit_quality'] > 0.9


# ---------- Fast vs slow comparison ----------

class TestFastMatchesSlow:
    def test_same_result_on_square(self, square_graph):
        """Fast (windowed+truncated) gives same result as slow (full NNLS)."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        tm = square_graph.results["transition_matrix"].values
        adjacency = (tm > 0) & ~np.eye(225, dtype=bool)
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = make_peaked_distribution(225, 112, [113, 127])

        slow = _compute_position_cell(
            x, eigvals, eigvecs, eigvecs_inv, adjacency,
            cells, n2c, square_graph.node_to_coords,
        )
        fast = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        assert abs(slow['position_row'] - fast['position_row']) < 0.1
        assert abs(slow['position_col'] - fast['position_col']) < 0.1
        assert abs(slow['fit_quality'] - fast['fit_quality']) < 0.05


# ---------- Matching pursuit ----------

class TestMatchingPursuit:
    def test_matches_nnls_simple(self, square_graph):
        """On a simple peaked distribution, MP finds same node as NNLS."""
        from hammy_lib.calculations.position import _compute_position_cell_mp
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = make_peaked_distribution(225, 112, [113, 127])

        fast = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        mp = _compute_position_cell_mp(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        assert abs(fast['position_row'] - mp['position_row']) < 0.5
        assert abs(fast['position_col'] - mp['position_col']) < 0.5

    def test_method_registry(self, square_graph):
        """CellPositionCalculation accepts method parameter."""
        from hammy_lib.calculations.position import CellPositionCalculation, POSITION_METHODS
        assert "nnls" in POSITION_METHODS
        assert "nnls_fast" in POSITION_METHODS
        assert "matching_pursuit" in POSITION_METHODS


# ---------- Edge cases ----------

class TestEdgeCases:
    def test_zero_distribution(self, square_graph):
        """Zero histogram should not crash."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = np.zeros(225)
        result = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        assert result['n_components'] == 0
        assert result['position_row'] == -1.0

    def test_uniform_distribution(self, square_graph):
        """Uniform distribution → should find SOME position without crashing."""
        eigvals = square_graph.results["eigenvalues"].values
        eigvecs = square_graph.results["eigenvectors"].values
        eigvecs_inv = square_graph.results["eigenvectors_inv"].values
        cells = square_graph.get_cells()
        n2c = square_graph.get_node_to_cells()
        x = np.ones(225) / 225
        result = _compute_position_cell_fast(
            x, eigvals, eigvecs, eigvecs_inv,
            cells, n2c, square_graph.node_to_coords,
        )
        # Uniform → stationary distribution → high power, low information
        assert result['power'] > 100


# ---------- Graph structure ----------

class TestGraphStructure:
    def test_transition_matrix_stochastic(self, square_graph):
        tm = square_graph.results["transition_matrix"].values
        assert np.allclose(tm.sum(axis=1), 1.0)
        assert (tm >= -1e-15).all()

    def test_transition_matrix_symmetric(self, square_graph):
        tm = square_graph.results["transition_matrix"].values
        assert np.allclose(tm, tm.T, atol=1e-10)

    def test_real_eigenvalues(self, square_graph):
        eigvals = square_graph.results["eigenvalues"].values
        assert np.allclose(eigvals.imag, 0, atol=1e-10)

    def test_cells_cover_interior(self, square_graph):
        """Every interior node should belong to at least one cell."""
        n2c = square_graph.get_node_to_cells()
        for r in range(1, 14):
            for c in range(1, 14):
                idx = r * 15 + c
                assert idx in n2c, f"Interior node ({r},{c}) not in any cell"

    @pytest.mark.parametrize("graph_fixture", [
        "square_graph", "triangular_graph", "hexagonal_graph", "brick_graph"
    ])
    def test_all_graphs_valid(self, graph_fixture, request):
        g = request.getfixturevalue(graph_fixture)
        tm = g.results["transition_matrix"].values
        assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-10)
        assert (tm >= -1e-15).all()
        eigvals = g.results["eigenvalues"].values
        assert np.max(np.abs(eigvals.imag)) < 1e-8, "Complex eigenvalues detected"
