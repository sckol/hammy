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
