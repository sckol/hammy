"""End-to-end test: matrix walk pipeline.

SquareLattice → WalkTask → run on CPU via hampy → verify output.
"""
import numpy as np
import pytest
import xarray as xr

from hammy_lib.graph import SquareLattice as LatticeGraph2D, PathGraph as LinearGraph, HexLattice as TriangularGraph2D
from hammy_lib.hampy import set_backend, to_numpy
from hammy_lib.task import WalkTask
from hammy_lib.dispatcher import Dispatcher


@pytest.fixture(autouse=True)
def use_numpy_backend():
    import numpy as np
    set_backend(np)


class TestWalkTask:
    def test_output_shape(self):
        g = LatticeGraph2D(5, 5)
        g.calculate()
        initial = np.zeros(25)
        initial[12] = 1.0
        task = WalkTask(graph=g, initial=initial, t_steps=[1, 2, 4])
        result = task.run()
        assert result.dims == ("t", "node")
        assert result.shape == (3, 25)

    def test_probability_conservation(self):
        g = LatticeGraph2D(5, 5)
        g.calculate()
        initial = np.zeros(25)
        initial[12] = 1.0
        task = WalkTask(graph=g, initial=initial, t_steps=[1, 10, 100])
        result = task.run()
        for t_idx in range(3):
            total = float(result.isel(t=t_idx).sum())
            assert abs(total - 1.0) < 1e-10, f"probability not conserved at step {t_idx}"

    def test_distribution_spreads(self):
        g = LinearGraph(21)
        g.calculate()
        initial = np.zeros(21)
        initial[10] = 1.0
        task = WalkTask(graph=g, initial=initial, t_steps=[1, 50])
        result = task.run()
        early = result.sel(t=1).values
        late = result.sel(t=50).values
        # After many steps the distribution should be more spread: lower max value
        assert late.max() < early.max()

    def test_t_steps_ordering(self):
        """t_steps can be given in any order; output is sorted."""
        g = LatticeGraph2D(4, 4)
        g.calculate()
        initial = np.zeros(16)
        initial[8] = 1.0
        task = WalkTask(graph=g, initial=initial, t_steps=[4, 1, 2])
        result = task.run()
        assert list(result.coords["t"].values) == [1, 2, 4]

    def test_zero_initial_raises(self):
        g = LinearGraph(5)
        g.calculate()
        task = WalkTask(graph=g, initial=np.zeros(5), t_steps=[1])
        with pytest.raises(ValueError, match="sums to zero"):
            task.run()


class TestDispatcher:
    def test_cpu_gets_smallest(self):
        g5 = LatticeGraph2D(5, 5)
        g5.calculate()
        g10 = LatticeGraph2D(10, 10)
        g10.calculate()
        initial5 = np.ones(25) / 25
        initial10 = np.ones(100) / 100
        tasks = [
            WalkTask(graph=g10, initial=initial10, t_steps=[1]),
            WalkTask(graph=g5, initial=initial5, t_steps=[1]),
        ]
        d = Dispatcher(tasks)
        cpu_task = d.next_for_cpu()
        assert cpu_task.size == 25

    def test_gpu_gets_largest(self):
        g5 = LatticeGraph2D(5, 5)
        g5.calculate()
        g10 = LatticeGraph2D(10, 10)
        g10.calculate()
        initial5 = np.ones(25) / 25
        initial10 = np.ones(100) / 100
        tasks = [
            WalkTask(graph=g10, initial=initial10, t_steps=[1]),
            WalkTask(graph=g5, initial=initial5, t_steps=[1]),
        ]
        d = Dispatcher(tasks)
        gpu_task = d.next_for_gpu()
        assert gpu_task.size == 100


class TestGraphStructure:
    def test_transition_matrix_stochastic(self):
        for GraphClass, args in [
            (LatticeGraph2D, (7, 7)),
            (LinearGraph, (15,)),
            (TriangularGraph2D, (7, 7)),
        ]:
            g = GraphClass(*args)
            tm = g._results["transition_matrix"].values
            assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-12), \
                f"{GraphClass.__name__} rows don't sum to 1"
            assert (tm >= -1e-14).all(), f"{GraphClass.__name__} has negative entries"

    def test_node_positions_shape(self):
        g = LatticeGraph2D(5, 5)
        pos = g._results["node_positions"].values
        assert pos.shape == (25, 2)

    def test_faces_correct_count(self):
        g = LatticeGraph2D(4, 4)
        assert len(g.faces) == 9  # (4-1)*(4-1) squares

    def test_triangular_faces(self):
        g = TriangularGraph2D(4, 4)
        assert len(g.faces) == 2 * 3 * 3  # 2*(rows-1)*(cols-1) triangles
