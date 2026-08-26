# Benchmarking guide

`benchmark.py` is a deterministic synthetic simulator for regression testing the routing and learning behavior. It is deliberately different from SAGE's prediction model, but it is not a substitute for real executions.

## Reproduce the published table

```bash
python benchmark.py
```

The default suite runs 500 tasks for each of five seeds: `3,7,11,19,23`.

## Run a smaller smoke test

```bash
python benchmark.py --seeds 3,7 --tasks-per-seed 25
```

## Export machine-readable results

```bash
python benchmark.py --json benchmark-results.json
```

The JSON document includes the simulator identifier, seeds, task counts, mean and population standard deviation for every metric, learned route distribution, and online-model updates per seed.

## Interpreting metrics

- `quality` is the external simulator's nonlinear artifact-quality score.
- `utility` applies the same external quality-cost-latency objective to every strategy.
- `cost` is realized cost divided by task budget.
- `latency` is realized latency divided by deadline.
- `misses` is the percentage of tasks whose realized latency exceeds the deadline.

Online SAGE can spend more than static SAGE to achieve higher simulated quality. Report both sides of that trade-off and never call the synthetic table production evidence.

## Requirements for a publishable evaluation

A stronger study should add real heterogeneous endpoints, learned-routing baselines, confidence intervals over repeated executions, task-segment analysis, calibration curves, logged propensities, off-policy evaluation, marketplace trace replay, churn, adversarial bids, security tests, and end-to-end cost accounting.
