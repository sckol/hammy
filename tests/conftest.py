"""Shared fixtures for position detection tests."""
import numpy as np
import pytest
from hammy_lib.graph import (
    LinearGraph, LatticeGraph2D, TriangularGraph2D,
    HexagonalGraph2D, BrickGraph2D,
)


@pytest.fixture(scope="session")
def linear_graph():
    """101-node linear chain (matches experiment 1 bins)."""
    g = LinearGraph(101)
    g.calculate()
    return g


@pytest.fixture(scope="session")
def square_graph():
    """15x15 square lattice (matches experiment 2)."""
    g = LatticeGraph2D(15, 15)
    g.calculate()
    return g


@pytest.fixture(scope="session")
def triangular_graph():
    """21x21 triangular lattice (matches experiment 3)."""
    g = TriangularGraph2D(21, 21)
    g.calculate()
    return g


@pytest.fixture(scope="session")
def hexagonal_graph():
    """15x15 hexagonal lattice."""
    g = HexagonalGraph2D(15, 15)
    g.calculate()
    return g


@pytest.fixture(scope="session")
def brick_graph():
    """15x15 brick lattice."""
    g = BrickGraph2D(15, 15)
    g.calculate()
    return g


def make_peaked_distribution(n, center, neighbors, center_weight=100, neighbor_weight=50):
    """Create a distribution peaked at center with given neighbors."""
    x = np.zeros(n)
    x[center] = center_weight
    for nb in neighbors:
        x[nb] = neighbor_weight
    return x / x.sum()


def make_gaussian_2d(rows, cols, center_r, center_c, sigma=3.0):
    """Create a 2D Gaussian distribution on a grid."""
    x = np.zeros(rows * cols)
    for r in range(rows):
        for c in range(cols):
            d2 = (r - center_r) ** 2 + (c - center_c) ** 2
            x[r * cols + c] = np.exp(-d2 / (2 * sigma ** 2))
    return x / x.sum()
