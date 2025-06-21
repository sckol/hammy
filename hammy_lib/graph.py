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
        tm = self._results["transition_matrix"].values
        eigvals, eigvecs = np.linalg.eig(tm)
        self._results["eigenvalues"] = (["eigen_index"], eigvals)
        self._results["eigenvectors"] = (["position_index", "eigen_index"], eigvecs)

    def generate_id(self) -> str:
        return f"graph_{hash(self._results['transition_matrix'].values.tobytes())}"

    @property
    def simple_name(self) -> str:
        return self.__class__.__name__


class LinearGraph(Graph):
    def __init__(self, length: int, id: str = None):
        tm = np.zeros((length, length))
        for i in range(length):
            if i > 0:
                tm[i, i - 1] = 0.5
            if i < length - 1:
                tm[i, i + 1] = 0.5
        # Fix endpoints to only move inward
        tm[0, 1] = 1.0
        tm[0, 0] = 0.0
        tm[-1, -2] = 1.0
        tm[-1, -1] = 0.0
        super().__init__(tm, id)


class CircularGraph(Graph):
    def __init__(self, length: int, id: str = None):
        tm = np.zeros((length, length))
        for i in range(length):
            tm[i, (i - 1) % length] = 0.5
            tm[i, (i + 1) % length] = 0.5
        super().__init__(tm, id)
