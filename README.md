# hammy
Experiments with random walks and pipeline to run them.

## Pipeline
![Pipeline diagram](docs/pipeline.svg)

## Results file naming
    <tag>.<experiment_number>.<hypotesis_number>-<implementation_number>.YYYYMMDDHHMMSS.<thread_id>

For example:
walk.2.3-1.20220301201003.d670460

## Action plan
- [x] Make hello world simulator
- [x] Calculate statistics (# of iterations, run time)
- [ ] Save results to file
- [ ] Save results to Clickhouse
- [ ] Save statistics to file
- [ ] Save statistics to Clickhouse
