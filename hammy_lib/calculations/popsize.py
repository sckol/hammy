import numpy as np
import xarray as xr
from typing import Callable
from ..calculation import Calculation


class PopulationSizeCalculation(Calculation):
    """
    Estimates the effective population size (N) by comparing observed distribution
    with theoretical distribution using the G-test (log-likelihood ratio).

    Returns 1/phi where phi = G/df. For correct IID data, phi ≈ 1 so ratio ≈ 1.
    Values > 1 indicate underdispersion, < 1 indicate overdispersion (correlated
    samples or model mismatch).

    The base class extend_simulation_results() cumulates across levels and adds
    TOTAL platform — so per-level values test each level's IID, while higher
    cumulated levels and TOTAL reveal seed duplication or inter-level correlation.
    """

    def __init__(
        self,
        main_input,
        theoretical_dist_func: Callable,
        distribution_dim: str = "x",
        id: str | None = None,
    ):
        """
        Args:
            main_input: ArrayHammyObject containing the observed distributions
            theoretical_dist_func: Callable(positions, target, checkpoint) -> probabilities
            distribution_dim: Name of the dimension containing the distribution bins
            id: Optional identifier
        """
        super().__init__(main_input, id)
        self.theoretical_dist_func = theoretical_dist_func
        self.distribution_dim = distribution_dim

    @property
    def independent_dimensions(self) -> list[str]:
        return [str(d) for d in self.main_input.results.dims if d != self.distribution_dim]

    @property
    def simple_type_return(self):
        return True

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> float:
        """
        Returns 1/phi where phi = G/df (G-test overdispersion estimate).

        The G-test statistic G = 2 * sum(k * log(k / E)) follows chi-squared(df)
        under IID multinomial. Unlike Pearson chi-squared, it works with any
        observed count (including k=1,2,3) without a min_expected filter.

        No capping — values > 1 (underdispersion) and < 1 (overdispersion)
        are both meaningful signals.
        """
        target = coords.get("target")
        checkpoint = coords.get("checkpoint")

        k_all = np.asarray(input_array.values, dtype=float)
        x_all = np.asarray(input_array.coords[self.distribution_dim].values)

        N_true = float(np.sum(k_all))
        if N_true == 0:
            raise ValueError(
                f"Empty histogram at target={target}, checkpoint={checkpoint}"
            )

        p_all = np.asarray(
            self.theoretical_dist_func(x_all, target=target, checkpoint=checkpoint),
            dtype=float,
        )

        if not np.all(np.isfinite(p_all)):
            raise ValueError(
                f"Theoretical distribution contains non-finite values at target={target}, checkpoint={checkpoint}"
            )

        # Only use bins where theoretical probability is positive
        mask = p_all > 0
        k = k_all[mask]
        p = p_all[mask]

        p_sum = float(np.sum(p))
        if p_sum <= 0:
            raise ValueError(
                f"Theoretical distribution sums to zero at target={target}, checkpoint={checkpoint}"
            )
        p = p / p_sum

        E = N_true * p

        # G-test: only bins with k > 0 contribute (0 * log(0/E) = 0)
        nonzero = k > 0
        k_nz = k[nonzero]
        E_nz = E[nonzero]

        m = int(mask.sum())
        df = m - 1
        if df <= 0:
            return float("nan")

        G = 2.0 * float(np.sum(k_nz * np.log(k_nz / E_nz)))
        phi = G / df

        if phi <= 0:
            raise ValueError(
                f"G-test statistic is non-positive (G={G}, df={df}) at target={target}, checkpoint={checkpoint}. This should not happen with real simulation data."
            )

        return float(1.0 / phi)


def bridged_random_walk_distribution(
    positions: np.ndarray,
    target: int,
    checkpoint: int,
    T: int = 1000,
) -> np.ndarray:
    """
    Calculate theoretical probability distribution for a bridged random walk
    whose positions are stored as raw_position // 2 (integer division).

    The walk does T steps of +/-1. Both the checkpoint position and the final
    position are floor-divided by 2 before binning. A bin value x therefore
    collects raw positions 2x and 2x+1, and the target bin collects raw
    endpoints 2*target and 2*target+1.

    P(bin = x at checkpoint | bin = target at T)
        = sum_{r in {2x, 2x+1}} sum_{e in {2*target, 2*target+1}}
              P(raw = r at checkpoint) * P(raw = e at T | raw = r at checkpoint)
          / sum_{e in {2*target, 2*target+1}} P(raw = e at T)

    Args:
        positions: Array of bin values (raw_position // 2)
        target: Target bin value (raw_endpoint // 2)
        checkpoint: Time step at which we observe
        T: Total number of steps (default 1000)

    Returns:
        Array of probabilities p_x for each bin position
    """
    from scipy.special import comb

    def random_walk_prob(x_end, x_start, t):
        displacement = x_end - x_start
        if abs(displacement) > t or (t + displacement) % 2 != 0:
            return 0.0
        k = (t + displacement) // 2
        if k < 0 or k > t:
            return 0.0
        return comb(t, k, exact=False) * (0.5**t)

    # Raw endpoints that map to the target bin
    target_raws = [2 * target, 2 * target + 1]
    Z = sum(random_walk_prob(e, 0, T) for e in target_raws)

    if Z < 1e-300:
        raise ValueError(
            f"Target bin {target} is unreachable in {T} steps (Z={Z})"
        )

    probs = np.zeros_like(positions, dtype=float)
    for i, x in enumerate(positions):
        for raw_cp in [2 * x, 2 * x + 1]:
            for raw_end in target_raws:
                p_fwd = random_walk_prob(raw_cp, 0, checkpoint)
                p_bwd = random_walk_prob(raw_end, raw_cp, T - checkpoint)
                probs[i] += p_fwd * p_bwd
    probs /= Z

    prob_sum = np.sum(probs)
    if prob_sum <= 0:
        raise ValueError(
            f"Bridge distribution sums to zero at target={target}, checkpoint={checkpoint}"
        )
    probs /= prob_sum

    return probs
