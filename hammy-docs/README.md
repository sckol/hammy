# hammy: Main page

# Overview

Consider a graph where we perform a random walk with a fixed number of steps from a starting node `S`. We keep only those walks that end at a particular point `E`. We can then define a “trajectory” by identifying the most frequently visited node at each intermediate step. The aim of this project is to show that such a trajectory exhibits “physical” behavior. For example, on an infinite 1D/2D/3D lattice graph, the trajectory would correspond to a discrete version of uniform motion between `S` and `E`. We can introduce “potential energy” by terminating the random walk at each step with a probability based on the potential energy of the particle's location. In this scenario, the trajectory would show the particle slowing down in areas of high potential energy and accelerating in areas of low potential energy.

The **hammy** project aims to demonstrate these concepts and explore the correspondence between physical notions and statistics of random walks, including:

- Lagrangian,
- continuous position on a discrete graph,
- momentum and energy,
- Hamiltonian function.

# Structure

**hammy** consists of a series of experiments, each running simulations of random walks, calculating their statistics, and interpreting them in physical terms. These experiments are conducted in Jupyter Notebooks. Some experiments require substantial computational resources (memory, CPUs, and GPUs), so utilities were developed to facilitate running experiments in a cloud environment efficiently and cost-effectively.

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
The `docker` folder contains files for running experiments in Yandex Cloud using [Container Solution](https://yandex.cloud/en/docs/cos/). The `hammy-base.Dockerfile' is an image with the following features
- Python is installed with CuPy
- Yandex Cloud Monitoring is enabled, which allows you to monitor CPU and memory usage in real time.
- the logs of the main script (which must be named as `main.sh` in the descendant images) are redirected to Yandex Cloud Logging using the `log_yc` bash function
- The main script is executed in a wrapper `hammy_entrypoint.sh`, which at the end of the main script's execution either deletes the machine (default) or shuts it down (do-nothing option is also available).
- A `run_nice` bash function is available which sets a lower priority for the main script, to ensure that the monitoring and wrapper processes continue to work even if the main script exhausts memory or CPU.

An example of a `hammy-base` descendant image is in `chquery.Dockerfile`. It runs a Clickhouse instance connected to Yandex Cloud Storage as an S3 bucker, and passes a query from the container's `CMD` argument (allowing different queries to be run using the same image). Clickhouse can take the data from the S3 bucker and return query results to it. This allows Clickhouse to be used as part of a data processing pipeline. The script `create_query_machine.sh` runs a virtual machine in Yandex Cloud with this image and passes the content of the given file as `CMD` argument.

Sometimes when writing queries you may want to use variable substitution or cycles (for example, if you have a table with many columns `position_1`, `position_2`, ...). The `process_jinja.py` script allows you to write queries as [Jinja](https://jinja.palletsprojects.com/) templates and the script will convert them into regular SQL files.

The `create_hammy_machine.sh` starts a virtual machine with GPU support from a Docker image for the given experiment name.
# Experiment list
## Naming conventions