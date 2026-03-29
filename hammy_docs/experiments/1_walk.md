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

## Error metrics

We considered three metrics. Only two proved useful.

### Bootstrap standard deviation (useful)

Resample the histogram $h$ via multinomial$(N, h/N)$ with $B = 200$ resamples. For each resample, run the full position pipeline and collect $\hat{x}_b$. The standard deviation of $\{\hat{x}_b\}$ gives the **position uncertainty** without assuming a true position.

Key property: the bootstrap std decreases with the number of simulation data points, as expected. It is large for $t = 0$ (undirected walk, hard to localize) and small for $t = 10$ (strong directional drift).

### Simplex fit quality (useful for higher dimensions)

The fraction of total NNLS weight captured by the identified simplex: $q = \sum_{\text{simplex}} w_i / \sum_{\text{all}} w_i$. In this 1D experiment $q = 1.0$ everywhere (all weight maps to an edge or vertex). On higher-dimensional graphs, $q < 1$ would signal that the distribution doesn't decompose cleanly into a point on a simplex.

### NNLS residual (not useful)

The relative residual $\|T^p w - x\|_2 / \|x\|_2$ measures how well the NNLS mixture approximates the observed distribution. It converges to $\approx 0.1$, not to zero. This is **model error**, not sampling noise: the conditioned walk distribution is not exactly representable as a non-negative mixture of $T^p$ columns, because conditioning on the endpoint creates correlations that a point-source mixture cannot capture. Since the residual is constant regardless of data quality, it does not distinguish well-localized from poorly-localized positions.

## Results

The continuous position tracks the theoretical trajectory $\hat{x}(c) = t \cdot c / T$ for all targets, with bootstrap CIs that narrow with more data. On V100 GPU with $2.7 \times 10^{11}$ total data points (level 4), the bootstrap std at checkpoint 500 for target 5 is $< 0.001$ nodes.

### Key observations

1. **The algorithm works**: NNLS decomposition reliably identifies the correct edge of the graph, with barycentric coordinates giving sub-node precision.
2. **Simplex identification is robust**: fit quality is 1.0 across all targets, checkpoints, platforms, and levels. The greedy algorithm never fails on this graph.
3. **Bootstrap gives meaningful CIs**: uncertainty scales with $1/\sqrt{N}$ as expected, varies across targets (larger for $t = 0$), and is consistent across platforms (PYTHON, CFFI, CUDA agree).
4. **NNLS residual is a model property**: ~10% irreducible error from the non-negativity constraint. Not useful for confidence intervals but can serve as a sanity check (anomalous values signal bugs).

## Implications for future experiments

- The position algorithm generalizes to any graph with a transition matrix. No changes needed for 2D/3D lattices except adjusting `max_simplex_dim` (1 for edges, 2 for faces).
- Bootstrap is the primary error metric. Simplex fit quality becomes important on higher-dimensional graphs where simplex identification may fail.
- At level 4, position is determined to $< 0.01$ nodes for non-zero targets. For experiments where sub-node precision is not needed, level 2 (4 minutes of GPU time) is sufficient.
- The number of simulations needed depends on the target: $t = 0$ requires $\sim 100\times$ more data than $t = 10$ for the same precision, because the undirected walk has maximal positional entropy.
