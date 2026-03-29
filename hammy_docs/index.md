# hammy

Experiments with random walks on graphs and the correspondence between walk statistics and physical notions.

## Motivation

Consider a graph where we perform a random walk of fixed length from node $S$ to node $E$. At each intermediate step, the distribution of the particle's position defines a "trajectory." On an infinite lattice, this trajectory corresponds to uniform motion. By adding position-dependent termination probabilities (potential energy), the trajectory shows the particle slowing in high-potential regions.

**hammy** explores this correspondence between walk statistics and physics:
- Continuous position on a discrete graph (via NNLS spectral decomposition)
- Lagrangian and momentum (future work)
- Hamiltonian function (future work)

## Experiments

- [[experiments/1_walk|Experiment 1: One-Dimensional Walk]] — develop the position algorithm, validate error metrics, establish how many simulations are needed

## Computation Framework

- [[computation_framework/index|Architecture and Pipeline]] — dual-language simulation, caching, cloud execution
- [[computation_framework/cost_analysis|Cost Analysis]] — throughput and cost per data point across GPU/CPU configurations

## Ideas

Theoretical notes and hypotheses in [[ideas/index|Ideas]].
