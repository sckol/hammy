"""Task hierarchy for matrix-multiplication-based random walks."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import xarray as xr

from .graph import Graph


@dataclass
class Task:
    graph: Graph

    @property
    def size(self) -> int:
        return self.graph.size

    def run(self) -> xr.DataArray:
        raise NotImplementedError


@dataclass
class WalkTask(Task):
    """Exact probability propagation via repeated T @ v.

    Computes the distribution after each number of steps in t_steps.

    Args:
        graph: the graph providing the transition matrix.
        initial: initial probability vector, shape (N,). Will be normalized.
        t_steps: list of step counts at which to record the distribution.
    """
    initial: np.ndarray
    t_steps: list[int] = field(default_factory=list)

    def run(self) -> xr.DataArray:
        import hammy_lib.hampy as hp

        # T is row-stochastic: T[i,j] = P(go to j | at i).
        # Distribution propagation uses the transpose: v_{t+1} = T.T @ v_t.
        T_raw = hp.array(self.graph._results["transition_matrix"].values)
        T = T_raw.T
        v = hp.array(self.initial.astype(float))
        total = hp.sum(v)
        if float(total) == 0:
            raise ValueError("WalkTask: initial distribution sums to zero")
        v = v / total

        sorted_steps = sorted(set(self.t_steps))
        results = []
        current_step = 0
        for t in sorted_steps:
            steps_needed = t - current_step
            for _ in range(steps_needed):
                v = hp.matmul(T, v)
            current_step = t
            results.append(hp.to_numpy(v))

        data = np.stack(results, axis=0)
        return xr.DataArray(
            data,
            dims=["t", "node"],
            coords={
                "t": xr.DataArray(sorted_steps, dims="t"),
                "node": xr.DataArray(np.arange(self.size), dims="node"),
            },
        )
