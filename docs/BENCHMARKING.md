# Benchmarking guide

`benchmark.py` is a deterministic synthetic regression and ablation suite. The
held-out environment is implemented separately in `benchmark_evaluator.py` and
does not call SAGE scoring helpers.

It is structurally different from the router:

- latent capabilities are independently specified and include ranking reversals;
- quality uses a weighted geometric mean plus a bottleneck term, not noisy-OR;
- pair compatibility is multiplicative rather than an additive synergy reward;
- realized cost uses nonlinear assigned workload plus a startup fee;
- realized latency and handoff loss use equations distinct from SAGE.

This separation reduces evaluator-policy alignment. It does not make an
author-designed simulator independent real-world evidence.

## Reproduce the v0.3 table

```bash
python benchmark.py
```

The default suite runs 500 tasks for each of five seeds: `3,7,11,19,23`.
Values below are mean ± population standard deviation across seeds.

| Strategy | Quality | Common utility | Cost / budget | Deadline miss |
|---|---:|---:|---:|---:|
| Incumbent only | 0.332 ± 0.009 | 0.134 ± 0.008 | 0.245 ± 0.005 | 38.0% |
| Advertised-skill solo | 0.445 ± 0.009 | 0.242 ± 0.009 | 0.302 ± 0.004 | 31.1% |
| Hidden feasible solo oracle | 0.473 ± 0.007 | 0.316 ± 0.008 | 0.280 ± 0.004 | 21.8% |
| Random team | 0.307 ± 0.007 | 0.111 ± 0.011 | 0.331 ± 0.012 | 32.2% |
| Greedy team | 0.577 ± 0.007 | 0.387 ± 0.009 | 0.435 ± 0.011 | 23.8% |
| Static SAGE | 0.518 ± 0.012 | 0.363 ± 0.013 | 0.394 ± 0.011 | **13.9%** |
| Learned SAGE, no exploration | **0.627 ± 0.006** | **0.447 ± 0.010** | 0.441 ± 0.014 | 20.0% |
| Learned SAGE, exploration | 0.622 ± 0.006 | 0.444 ± 0.009 | 0.437 ± 0.011 | 19.8% |
| Learned SAGE, random prior | 0.430 ± 0.045 | 0.291 ± 0.041 | **0.288 ± 0.029** | 16.0% |

The greedy team baseline spends almost the same normalized budget as learned
SAGE (0.435 versus 0.441), so the held-out quality and utility differences are
not explained by SAGE being the only policy allowed to form teams. Static SAGE
has the lowest deadline-miss rate, making the learning/resource trade-off
visible rather than declaring one strategy universally best.

## Learning and exploration ablations

The suite holds the task stream fixed and reports:

- static informed prior, no updates, no exploration;
- informed prior with updates, no exploration;
- informed prior with updates and Thompson-style exploration;
- weak random prior with updates, no exploration.

For the first versus last 100 tasks per seed:

| Strategy | Quality: first → last | Utility: first → last |
|---|---:|---:|
| Learned, no exploration | 0.634 → 0.637 | 0.454 → 0.460 |
| Learned, exploration | 0.616 → 0.638 | 0.438 → 0.460 |
| Learned, random prior | 0.398 → 0.509 | 0.259 → 0.371 |

The informed/no-exploration policy already starts strong, confirming that its
prior matters. Exploration does not explain the aggregate gain: the
no-exploration learned policy slightly exceeds the exploration variant in this
suite. The random-prior curve improves substantially but remains below the
informed prior, so v0.3 does not claim that online data has washed out prior
design.

## Smaller runs and JSON

```bash
python benchmark.py --seeds 3,7 --tasks-per-seed 200
python benchmark.py --json benchmark-results.json
```

The schema-v2 JSON includes every strategy, route mixes, model-update counts,
ablation definitions, and early/late learning windows.

## Routing latency

```bash
python benchmark_scaling.py
python benchmark_scaling.py --compare-unfiltered
```

The script reports median decision time for a six-requirement route at growing
registry sizes. The default `candidate_limit=12` isolates bounded-search cost;
the optional unfiltered comparison can be expensive. Results are
machine-dependent and exclude registry/network latency.

One Apple Silicon development run measured top-12 medians of 78.4, 78.9, and
80.1 ms for 20, 40, and 80 registered agents (three repeats). A separate
single-repeat comparison measured approximately 228, 460, and 924 ms without
prefiltering. These measurements demonstrate the scaling mechanism; they are
not production latency guarantees.

## Requirements for publishable evaluation

A stronger study should add real heterogeneous endpoints, public or auditable
task traces, more learned and optimization baselines, confidence intervals over
repeated executions, task-segment analysis, calibration curves, logged
propensities, off-policy evaluation, marketplace trace replay, churn,
adversarial bids, security tests, and end-to-end cost accounting.
