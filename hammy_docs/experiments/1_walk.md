# Experiment 1: One-Dimensional Walk

## Setup

Consider a random walk on a one-dimensional lattice $\mathbb{Z}$ of length $T = 1000$ steps. At each step the particle moves $\pm 1$ with equal probability. We condition on the walk ending at a particular position from the set $\text{TARGETS} = \{0, 1, 2, 5, 10\}$. For each target, we record the distribution of the particle's position at intermediate checkpoints $\text{CHECKPOINTS} = \{100, 200, \ldots, 1000\}$.

The raw data is a histogram $h_{t,c}(x)$ counting how many walks with target $t$ had position $x$ at checkpoint $c$.

## Goal

Develop a **trajectory function** $f: h_{t,c} \mapsto \text{position}$ that:
1. Maps a distribution over graph nodes to a single position on the graph,
2. Does not depend on the labelling of the graph nodes (topology-agnostic),
3. Gives a continuous-valued position even though the graph is discrete,
4. Provides quantifiable uncertainty.

The naive approach (empirical mode or weighted mean) requires node coordinates, which don't exist on a general graph. This experiment develops and validates a coordinate-free approach.

## Raw data

The simulation ran on a Yandex Cloud V100 GPU, producing $2.7 \times 10^{11}$ data points across 5 levels (16 minutes of compute time). Three platforms (PYTHON, CFFI, CUDA) ran in parallel, producing independent datasets that are compared for correctness and then aggregated.

The raw histograms at selected checkpoints (aggregated across all platforms and levels):

![[images/1_histogram.png]]

Each column is a checkpoint (100, 300, 500, 700, 900 steps); each row is a target (0, 1, 2, 5, 10). The distributions shift linearly toward the target as the checkpoint increases, consistent with the conditioned walk drifting toward its endpoint.

## Position algorithm

### Step 1: Spectral power matching

The transition matrix $T$ of the lazy walk (with $P(\text{stay}) = 1/2$, $P(\pm 1) = 1/4$) has eigendecomposition $T = V \Lambda V^{-1}$. The column $T^p e_i$ gives the distribution after $p$ steps starting at node $i$.

Given observed distribution $x$ (normalized to a probability vector), we find the *walk power* $p$ by matching its spectral spread to the mean spectral spread of $T^p$ columns:

$$\sum_{k \geq 1} c_k^2 = \frac{1}{n} \sum_{k \geq 1} \lambda_k^{2p} \|r_k\|^2$$

where $c = V^{-1} x$ are spectral coefficients, $\lambda_k$ are eigenvalues, and $r_k$ are rows of $V^{-1}$. The right-hand side is monotonically decreasing in $p$, giving a unique root via bisection.

### Step 2: NNLS decomposition

At the optimal $p$, compute $T^p$ via spectral decomposition and solve the non-negative least squares problem:

$$\min_{w \geq 0} \|T^p w - x\|_2$$

The solution $w$ decomposes $x$ as a non-negative mixture of point-source diffusions. Components with weight $< 1\%$ of the maximum are discarded.

### Step 3: Simplex identification

The surviving NNLS components (graph nodes with positive weights) are mapped to a graph element:
- **1 node**: position is on a vertex (0-simplex)
- **2 adjacent nodes**: position is on an edge (1-simplex), weights are barycentric coordinates
- **$k+1$ mutually adjacent nodes**: position is on a $k$-simplex

A greedy algorithm builds the largest valid simplex: start with the highest-weight node, add the next-highest only if it is adjacent to all nodes already in the simplex.

**Continuous position** is computed as the weighted sum of simplex node indices using barycentric coordinates: $\hat{x} = \sum_i \beta_i \cdot \text{node}_i$.

## Results: position

The continuous position (blue line) tracks the theoretical trajectory $\hat{x}(c) = t \cdot c / T$ (black line) for all targets. The light-blue band shows the bootstrap 95% CI:

![[images/2_position_ci.png]]

Each column is a platform (CFFI, CUDA, PYTHON, TOTAL); each row is a target. Key observations:
- For targets $t \geq 1$, the position follows the theoretical line with sub-node precision.
- For $t = 0$, the position stays at node 50 (the starting point). The PYTHON column shows a wide CI because the PYTHON platform produces the fewest data points (~6M vs ~270B for CUDA).
- All platforms agree on the position, confirming simulation correctness.

### Position convergence

The position at checkpoint 500 converges rapidly across simulation levels (each level doubles the data):

![[images/3_position_convergence.png]]

For CFFI, CUDA, and TOTAL, the position is stable from level 0. The PYTHON platform converges by level 2-3 due to fewer samples.

### Numerical results

At checkpoint 500, level 4, TOTAL platform:

| Target | Theoretical | Measured | Bootstrap std | Simplex dim |
|---|---|---|---|---|
| 0 | 50.000 | 50.000 | < 0.001 | 0 (vertex) |
| 1 | 50.500 | 50.500 | < 0.001 | 1 (edge) |
| 2 | 51.000 | 51.000 | < 0.001 | 0 (vertex) |
| 5 | 52.500 | 52.500 | < 0.001 | 1 (edge) |
| 10 | 55.000 | 55.000 | < 0.001 | 0 (vertex) |

Integer targets (0, 2, 10) land exactly on vertices; half-integer targets (1, 5 mapped to 50.5, 52.5) land on edges with barycentric coordinates $\approx (0.5, 0.5)$.

## Error metrics

We considered three metrics. Only two proved useful.

### Bootstrap standard deviation (useful)

Resample the histogram $h$ via multinomial$(N, h/N)$ with $B = 200$ resamples. For each resample, run the full position pipeline and collect $\hat{x}_b$. The standard deviation of $\{\hat{x}_b\}$ gives the **position uncertainty** without assuming a true position.

![[images/4_bootstrap_std.png]]

The bootstrap std decreases with simulation level (more data) as expected. The PYTHON column shows the largest uncertainty (fewest samples), decreasing from ~0.1 at level 0 to ~0.01 at level 4. CFFI and CUDA are near zero at all levels due to their much higher throughput. For target 0, the position is exactly on a vertex (node 50) across all resamples, giving zero std.

### Simplex fit quality (useful for higher dimensions)

The fraction of total NNLS weight captured by the identified simplex: $q = \sum_{\text{simplex}} w_i / \sum_{\text{all}} w_i$.

![[images/6_fit_quality.png]]

In this 1D experiment $q = 1.0$ everywhere — all weight maps to an edge or vertex. The greedy simplex identification never fails on this graph. On higher-dimensional graphs, $q < 1$ would signal that the distribution doesn't decompose cleanly into a point on a simplex.

### NNLS residual (not useful for error estimation)

The relative residual $\|T^p w - x\|_2 / \|x\|_2$:

![[images/5_nnls_residual.png]]

The residual converges to $\approx 0.1$, not to zero. This is **model error**, not sampling noise: the conditioned walk distribution is not exactly representable as a non-negative mixture of $T^p$ columns. The non-negativity constraint in NNLS prevents the exact decomposition because conditioning on the endpoint creates a distribution shape that requires "subtracting" some columns. Since the residual is constant regardless of data quality, it does not distinguish well-localized from poorly-localized positions. It can serve as a sanity check — anomalous values would signal bugs.

## Implications for future experiments

- The position algorithm generalizes to any graph with a transition matrix. No changes needed for 2D/3D lattices except adjusting `max_simplex_dim` (1 for edges, 2 for faces).
- Bootstrap is the primary error metric. Simplex fit quality becomes important on higher-dimensional graphs where simplex identification may fail.
- At level 4, position is determined to $< 0.01$ nodes for non-zero targets. For experiments where sub-node precision is not needed, level 2 (4 minutes of GPU time) is sufficient.
- The number of simulations needed depends on the target: undirected walks ($t = 0$) have maximal positional entropy and require proportionally more data. However, on this 1D graph even the PYTHON platform (~6M data points) gives exact vertex positions for symmetric targets — the main uncertainty comes from edge positions where the walker is between two nodes.
