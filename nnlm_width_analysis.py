"""
Compute NNLM width dynamics for a Gaussian-to-Gaussian bridge walk on 1D lattice.

Theory to check:
  - At each time t, marginal distribution is f_t(x) ∝ (T^t p0)(x) * (T^(T-t) p0)(x)
  - Apply _find_power + NNLS to f_t
  - NNLM width = sqrt( sum_i w_i (i - mean_i)^2 ) over all NNLS weights (not thresholded)
  - Compare with theoretical prediction sigma^2 * [1 - 2t/T + 2t^2/T^2]
"""

import sys
sys.path.insert(0, '/home/nsushchenko/hammy')

import numpy as np
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hammy_lib.graph import PathGraph as LinearGraph
from hammy_lib.calculations.position import _find_power, _precompute_power_search


def make_gaussian(n, center, sigma):
    """Discrete Gaussian probability vector on n nodes."""
    xs = np.arange(n, dtype=float)
    p = np.exp(-0.5 * ((xs - center) / sigma) ** 2)
    return p / p.sum()


def nnlm_width(T_power, f_norm):
    """
    Solve NNLS: T^p w ≈ f_norm.
    Return (power-weighted std of source nodes, continuous_position, weights).
    Uses ALL weights above zero (no 1% threshold or MAX_COMPONENTS limit).
    """
    weights, _ = scipy.optimize.nnls(T_power, f_norm)
    total = weights.sum()
    if total == 0:
        return 0.0, 0.0, weights
    w_norm = weights / total
    n = len(w_norm)
    xs = np.arange(n, dtype=float)
    mean_pos = np.dot(w_norm, xs)
    variance = np.dot(w_norm, (xs - mean_pos) ** 2)
    return float(np.sqrt(variance)), float(mean_pos), w_norm


def run_analysis(n=201, sigma=8.0, T_steps=500, n_checkpoints=50, center=100):
    """
    n: lattice size
    sigma: Gaussian packet width (in bins)
    T_steps: total bridge duration
    n_checkpoints: number of time points
    center: center node of both Gaussians
    """
    print(f"Parameters: n={n}, sigma={sigma}, T={T_steps}, center={center}")

    # Build graph and compute spectral decomposition
    g = LinearGraph(n)
    g.calculate()
    eigvals = g.results["eigenvalues"].values
    eigvecs = g.results["eigenvectors"].values
    eigvecs_inv = g.results["eigenvectors_inv"].values

    precomputed = _precompute_power_search(eigvals, eigvecs_inv)

    # Precompute T^t for all checkpoints via spectral decomposition
    # T^p = V @ diag(lambda^p) @ V^{-1}
    p0 = make_gaussian(n, center, sigma)

    times = np.linspace(1, T_steps - 1, n_checkpoints).astype(int)
    times = np.unique(times)

    results = []

    print(f"\n{'t':>6}  {'sigma_dist':>12}  {'power_p':>10}  {'nnlm_width':>12}  {'theory_width':>14}")
    print("-" * 60)

    for t in times:
        # Forward: T^t p0
        lam_t = eigvals.astype(np.complex128) ** t
        Tt = np.real(eigvecs @ np.diag(lam_t) @ eigvecs_inv)
        fwd = Tt @ p0

        # Backward: T^(T-t) p0
        lam_Tmt = eigvals.astype(np.complex128) ** (T_steps - t)
        TT_mt = np.real(eigvecs @ np.diag(lam_Tmt) @ eigvecs_inv)
        bwd = TT_mt @ p0

        # Bridge marginal: pointwise product, normalize
        f_t = fwd * bwd
        f_t = np.clip(f_t, 0, None)
        total = f_t.sum()
        if total == 0:
            continue
        f_norm = f_t / total

        # Distribution width
        xs = np.arange(n, dtype=float)
        mean_x = np.dot(f_norm, xs)
        sigma_dist = float(np.sqrt(np.dot(f_norm, (xs - mean_x) ** 2)))

        # Find power
        power = _find_power(eigvals, eigvecs_inv, f_norm,
                            p_min=1.0, p_max=float(T_steps * 2),
                            precomputed=precomputed)

        # Compute T^power
        lam_p = eigvals.astype(np.complex128) ** power
        Tp = np.real(eigvecs @ np.diag(lam_p) @ eigvecs_inv)

        # NNLS and NNLM width
        width, pos, w_norm = nnlm_width(Tp, f_norm)

        # Theoretical prediction: sigma * sqrt(1 - 2*tau + 2*tau^2)
        tau = t / T_steps
        theory = sigma * np.sqrt(1 - 2 * tau + 2 * tau ** 2)

        results.append({
            't': t,
            'tau': tau,
            'sigma_dist': sigma_dist,
            'power': power,
            'nnlm_width': width,
            'theory_width': theory,
            'mean_pos': pos,
        })
        print(f"{t:>6}  {sigma_dist:>12.3f}  {power:>10.1f}  {width:>12.4f}  {theory:>14.4f}")

    return results


def plot_results(results, sigma, T_steps, out_path='nnlm_width_plot.png'):
    times = [r['t'] for r in results]
    sigma_dist = [r['sigma_dist'] for r in results]
    nnlm_width = [r['nnlm_width'] for r in results]
    theory_width = [r['theory_width'] for r in results]
    power = [r['power'] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(times, sigma_dist, label='particle distribution σ', color='steelblue', lw=2)
    ax.plot(times, nnlm_width, label='NNLM width (actual)', color='tomato', lw=2)
    ax.plot(times, theory_width, label=f'theory: σ√(1−2τ+2τ²)', color='darkorange',
            lw=2, linestyle='--')
    ax.axhline(sigma / np.sqrt(2), color='gray', linestyle=':', label=f'σ/√2 = {sigma/np.sqrt(2):.2f}')
    ax.set_xlabel('t')
    ax.set_ylabel('width (nodes)')
    ax.set_title(f'NNLM width dynamics: Gaussian→Gaussian bridge (σ={sigma}, T={T_steps})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, power, color='purple', lw=2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('NNLM power p')
    ax2.set_title('Power p found by spectral matching')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    results = run_analysis(n=201, sigma=8.0, T_steps=500, n_checkpoints=30, center=100)
    plot_results(results, sigma=8.0, T_steps=500, out_path='/home/nsushchenko/hammy/nnlm_width_plot.png')
    print("\nDone.")
