# hammy: Main page

# Overview

Consider a graph where we perform a random walk with a fixed number of steps from a starting node `S`. We keep only those walks that end at a particular point `E`. We can then define a “trajectory” by identifying the most frequently visited node at each intermediate step. The aim of this project is to show that such a trajectory exhibits “physical” behavior. For example, on an infinite 1D/2D/3D lattice graph, the trajectory would correspond to a discrete version of uniform motion between `S` and `E`. We can introduce “potential energy” by terminating the random walk at each step with a probability based on the potential energy of the particle's location. In this scenario, the trajectory would show the particle slowing down in areas of high potential energy and accelerating in areas of low potential energy.

The **hammy** project aims to demonstrate these concepts and explore the correspondence between physical notions and statistics of random walks, including:

- Lagrangian,
- continuous position on a discrete graph,
- momentum and energy,
- Hamiltonian function.

# Structure

**hammy** consists of a series of experiments, each running simulations of random walks, calculating their statistics, and interpreting them in physical terms. These experiments are conducted in Jupyter Notebooks. Some experiments require substantial computational resources (memory, CPUs, and GPUs), so utilities were developed to facilitate running experiments in Yandex Cloud efficiently and cost-effectively.

Code for experiments and utilities is [available on GitHub](https://github.com/sckol/hammy).

## Experiments

Random walk simulations are prone to bugs, some of which are difficult to detect. To ensure reliability, the simulations are implemented in both Python and C. During development, it's more convenient to use CPUs to run the code. However, as experiments often require many iterations to achieve more accurate results, GPUs are used for final runs. For Python code, the [CuPy](https://cupy.dev/) library allows to write code that runs that runs on both CPUs and GPUs. For the C implementation, a simple framework of macros and code-writing rules has been developed, allowing the use of a single source for both CPU and GPU execution.

We keep separate datasets of simulation results for:

- Python implementations on CPU and GPU,
- C implementation on CPU,
- C implementation on GPU,
- aggregated data across all implementations.

Hypotheses are tested on the different datasets to identify any discrepancies (which may indicate bugs) and finally on the aggregated data.

## Utilities

# Naming conventions