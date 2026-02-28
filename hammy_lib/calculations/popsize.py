import numpy as np
import xarray as xr
import scipy.optimize
from typing import Callable
from ..calculation import Calculation


class PopulationSizeCalculation(Calculation):
    """
    Estimates the effective population size (N) by comparing observed distribution
    with theoretical distribution using Maximum Likelihood Estimation.
    
    This is useful for detecting if samples are truly independent (IID) or have
    correlations. If samples are correlated, N_estimated < N_true.
    """
    
    def __init__(
        self, 
        main_input, 
        theoretical_dist_func: Callable,
        id: str = None
    ):
        """
        Args:
            main_input: ArrayHammyObject containing the observed distributions
            theoretical_dist_func: Callable that computes theoretical probabilities
                Signature: func(positions, target, checkpoint, **params) -> np.ndarray
                where positions are the x-coordinates and returns p_x for each position
            id: Optional identifier
        """
        super().__init__(main_input, id)
        self.theoretical_dist_func = theoretical_dist_func
    
    @property
    def independent_dimensions(self):
        """Dimensions to iterate over (e.g., target, checkpoint)"""
        # Exclude 'position_index' which contains the distribution data
        return [d for d in self.main_input.results.dims 
                if d != 'position_index']
    
    @property
    def simple_type_return(self):
        """Returns a single float (estimated N)"""
        return True
    
    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> float:
        """
        Returns N_eff / N_true using a Pearson overdispersion estimate.

        Idea:
          - N_true = sum_x k_x (actual counted samples in this slice)
          - Expected counts under IID multinomial: E_x = N_true * p_x
          - Pearson chi2: chi2 = sum_x (k_x - E_x)^2 / E_x
          - Overdispersion: phi = chi2 / df  (≈1 for well-specified IID) [web:85][web:86]
          - Effective size ratio: N_eff / N_true ≈ 1 / phi

        Notes:
          - Bins with small expected counts destabilize chi2; filter by min_expected.
          - If theoretical_probs are computed on a truncated set of bins, renormalize.
        """
        import numpy as np

        # coordinates for the theoretical distribution
        target = coords.get("target")
        checkpoint = coords.get("checkpoint")

        # observed histogram for this (target, checkpoint)
        k_all = np.asarray(input_array.values, dtype=float)
        x_all = np.asarray(input_array.coords["position_index"].values)

        N_true = float(np.sum(k_all))
        if N_true <= 0:
            return 0.0

        # theoretical probabilities on the same bins
        p_all = np.asarray(
            self.theoretical_dist_func(x_all, target=target, checkpoint=checkpoint),
            dtype=float,
        )

        # filter invalid probs
        mask = np.isfinite(p_all) & (p_all > 0)
        if not np.any(mask):
            return 0.0

        k = k_all[mask]
        p = p_all[mask]

        # If you only model a subset of positions (truncated bins), renormalize on that subset
        p_sum = float(np.sum(p))
        if p_sum <= 0:
            return 0.0
        p = p / p_sum

        E = N_true * p  # expected counts under IID multinomial

        # stabilize Pearson chi2 by removing sparse expected bins
        min_expected = 5.0
        good = E >= min_expected
        k = k[good]
        E = E[good]

        m = int(k.size)
        df = m - 1
        if df <= 0:
            return float("nan")

        eps = 1e-12
        chi2 = float(np.sum((k - E) ** 2 / (E + eps)))
        phi = chi2 / df  # ≈1 if IID and model is correct [web:85][web:86]

        ratio = 1.0 / max(phi, eps)
        # cap at 1 (sometimes numerical/model quirks can make phi slightly < 1)
        return float(min(1.0, ratio))


def bridged_random_walk_distribution(
    positions: np.ndarray,
    target: int,
    checkpoint: int,
    T: int = 1000
) -> np.ndarray:
    """
    Calculate theoretical probability distribution for a bridged random walk.
    
    A bridged random walk starts at 0, must end at target after T steps,
    and we observe the position at checkpoint time.
    
    Uses the discrete random walk bridge formula:
    P(X_t = x | X_0 = 0, X_T = target) = 
        P(X_t = x | X_0 = 0) * P(X_{T-t} = target - x | X_0 = 0) / P(X_T = target | X_0 = 0)
    
    Args:
        positions: Array of position values (already divided by 2 as in your C code)
        target: Target end position (divided by 2)
        checkpoint: Time step at which we observe (even number assumed)
        T: Total number of steps (default 1000)
    
    Returns:
        Array of probabilities p_x for each position
    """
    # Simple symmetric random walk: each step ±1 with p=0.5
    # After t steps, position has binomial distribution
    
    def random_walk_prob(x_end, x_start, t):
        """
        Probability of reaching x_end from x_start in exactly t steps.
        For symmetric random walk: need (t + (x_end - x_start))/2 steps right
        """
        displacement = x_end - x_start
        
        # Check if displacement is achievable in t steps
        if abs(displacement) > t or (t + displacement) % 2 != 0:
            return 0.0
        
        # Number of steps to the right
        k = (t + displacement) // 2
        
        if k < 0 or k > t:
            return 0.0
        
        # Binomial probability: C(t, k) * 0.5^t
        from scipy.special import comb
        prob = comb(t, k, exact=False) * (0.5 ** t)
        
        return prob
    
    # Calculate probability for the bridge
    probs = np.zeros_like(positions, dtype=float)
    
    # P(end at target from 0 in T steps) - normalization constant
    p_end = random_walk_prob(target, 0, T)
    
    if p_end < 1e-100:
        # Target is unreachable, return uniform (or handle error)
        return np.ones_like(positions) / len(positions)
    
    for i, x in enumerate(positions):
        # P(at x from 0 at checkpoint) * P(at target from x in remaining steps)
        p_forward = random_walk_prob(x, 0, checkpoint)
        p_backward = random_walk_prob(target, x, T - checkpoint)
        
        # Bridge probability
        probs[i] = (p_forward * p_backward) / p_end
    
    # Normalize to handle numerical errors
    prob_sum = np.sum(probs)
    if prob_sum > 0:
        probs /= prob_sum
    
    return probs
