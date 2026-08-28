# Benchmarking guide

The repository has three deterministic synthetic studies with separate
questions and JSON schemas.  None is a public benchmark or evidence of
production superiority.

## 1. Checkpoint trajectory replay

```bash
python benchmark_dynamic.py
python benchmark_dynamic.py \
  --seeds 3,7 \
  --trajectories-per-seed 50 \
  --sweep-cases-per-seed 10 \
  --json dynamic-results.json \
  --svg progress-curve.svg
```

`benchmark_dynamic_evaluator.py` does not call SAGE scoring helpers.  A
checkpoint contains:

- completed DAG artifacts with producer, quality, and portability;
- one in-flight requirement with owner, completed fraction, partial quality,
  and portability;
- elapsed cost and latency;
- active route and optional failure evidence.

For a retained owner, the evaluator reuses all completed in-flight work.  For
a new owner, it reuses only the portable fraction and executes the remainder.
That concrete remaining work drives added cost and recovery latency.  Final
quality uses a weighted geometric mean plus a bottleneck term, and pair
compatibility is multiplicative.  No scalar-progress handoff equation is used.

The common action space includes:

- progress-aware SAGE;
- identical SAGE with in-flight progress masked to zero;
- always continue and always hand off;
- static greedy team formation;
- bounded static coalition/assignment enumeration with DAG-edge cost;
- a hidden-state dynamic oracle upper bound.

Default trajectory results are mean ± population standard deviation over five
seeds and 200 trajectories per seed:

| Strategy | Utility | Added cost / budget | Recovery latency / deadline | Wasted work | Switch rate | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|
| **Progress-aware SAGE** | **0.298 ± 0.021** | **0.191 ± 0.006** | 0.425 ± 0.015 | **0.059 ± 0.004** | 58.4% | **23.3%** |
| Progress-masked SAGE | 0.291 ± 0.021 | 0.204 ± 0.005 | 0.434 ± 0.018 | 0.104 ± 0.010 | 80.1% | 23.6% |
| Always continue | 0.085 ± 0.025 | 0.159 ± 0.005 | 0.627 ± 0.036 | 0.017 ± 0.003 | 0.0% | 34.4% |
| Always hand off | 0.290 ± 0.017 | 0.237 ± 0.008 | 0.508 ± 0.034 | 0.130 ± 0.010 | 100.0% | 30.5% |
| Static greedy team | 0.239 ± 0.031 | 0.248 ± 0.009 | 0.541 ± 0.035 | 0.080 ± 0.007 | 63.4% | 27.7% |
| Static coalition enumeration | 0.286 ± 0.026 | 0.265 ± 0.005 | 0.487 ± 0.034 | 0.104 ± 0.004 | 94.0% | 27.4% |
| Hidden-state dynamic oracle | 0.375 ± 0.019 | 0.243 ± 0.010 | **0.409 ± 0.014** | 0.085 ± 0.005 | 90.0% | 19.6% |

### Controlled progress intervention

The intervention uses a near-substitutable incumbent and research specialist.
For every generated case it holds task, registry, difficulty, artifact
portability, budget, and deadline fixed and changes only the in-flight fraction.

| In-flight fraction | Aware − masked utility |
|---:|---:|
| 0.0 | +0.0000 |
| 0.1 | +0.0114 |
| 0.2 | +0.0307 |
| 0.3 | +0.0375 |
| 0.4 | +0.0418 |
| 0.5 | +0.0463 |
| 0.6 | +0.0511 |
| 0.7 | +0.0562 |
| 0.8 | +0.0618 |
| 0.9 | +0.0684 |

This curve tests a specific continuation/handoff boundary.  Large capability
gaps remain in the trajectory suite because they can justify switching even
late in execution.

## 2. Requirement-conditioned trust convergence

```bash
python benchmark_trust.py
python benchmark_trust.py --seeds 3,7 --observations 250 --json trust-results.json
```

The per-requirement and single-reputation models receive exactly the same
exogenous round-robin evidence.  There is no exploration bonus and no
policy-dependent data collection.  Reported Brier score evaluates the full
agent/requirement matrix; selection regret measures the latent success gap from
the best agent for each requirement.

After 500 observations per seed in the heterogeneous specialist scenario:

| Trust model | Last-window Brier | Last-window selection regret | Mean identification observation |
|---|---:|---:|---:|
| **Per-requirement** | **0.0125 ± 0.0018** | **0.0094 ± 0.0008** | **20.0** |
| Single reputation | 0.0355 ± 0.0008 | 0.0310 ± 0.0359 | 38.2 |

The homogeneous negative control gives both models zero selection regret.  The
per-requirement representation has higher Brier error there because it has more
parameters and no specialization to learn; this is a useful negative result,
not hidden from the report.

## 3. Independent-task regression and learning ablations

```bash
python benchmark.py
python benchmark.py --seeds 3,7 --tasks-per-seed 200
python benchmark.py --json benchmark-results.json
```

`benchmark_evaluator.py` is structurally different from the router:

- independently specified latent capabilities include ranking reversals;
- quality uses a weighted geometric mean plus bottleneck, not assigned-owner
  coverage;
- pair compatibility is multiplicative rather than an additive synergy reward;
- realized cost uses nonlinear assigned workload plus startup cost;
- realized latency and handoff quality use evaluator-specific equations.

Default results cover five seeds and 500 tasks per seed:

| Strategy | Quality | Common utility | Cost / budget | Deadline miss |
|---|---:|---:|---:|---:|
| Incumbent only | 0.332 ± 0.009 | 0.134 ± 0.008 | 0.245 ± 0.005 | 38.0% |
| Advertised-skill solo | 0.445 ± 0.009 | 0.242 ± 0.009 | 0.302 ± 0.004 | 31.1% |
| Hidden feasible solo oracle | 0.473 ± 0.007 | 0.316 ± 0.008 | 0.280 ± 0.004 | 21.8% |
| Random team | 0.301 ± 0.006 | 0.103 ± 0.008 | 0.334 ± 0.010 | 32.3% |
| Greedy team | 0.577 ± 0.007 | 0.387 ± 0.009 | 0.435 ± 0.011 | 23.8% |
| Static SAGE | 0.491 ± 0.005 | 0.334 ± 0.008 | 0.327 ± 0.009 | 23.6% |
| **Learned SAGE, no exploration** | **0.613 ± 0.005** | **0.433 ± 0.008** | 0.428 ± 0.011 | 21.7% |
| Learned SAGE, exploration | 0.606 ± 0.005 | 0.427 ± 0.007 | 0.422 ± 0.012 | 21.3% |
| Learned SAGE, random prior | 0.447 ± 0.052 | 0.308 ± 0.049 | **0.297 ± 0.034** | **15.2%** |

The informed no-exploration policy still starts strong, so its prior matters.
Exploration does not explain the aggregate gain, and random-prior learning does
not catch the informed policy within 500 tasks.  The JSON report contains the
exact first/last-window values.

## 4. Routing latency

```bash
python benchmark_scaling.py
python benchmark_scaling.py --compare-unfiltered
```

The default `candidate_limit=12` bounds local combinatorial search.  Results
exclude registry/network latency and are machine-dependent.  One Apple Silicon
development run measured top-12 medians of 78.4, 78.9, and 80.1 ms at 20, 40,
and 80 registered agents; an unfiltered single-repeat comparison measured
approximately 228, 460, and 924 ms.

## Requirements for publishable evaluation

A stronger study needs checkpointed real executions, public or auditable task
traces, heterogeneous endpoints, repeated stochastic runs, an independently
governed artifact evaluator, calibration curves, logged propensities, causal
off-policy evaluation, churn and adversarial bids, security tests, and
end-to-end cost accounting.
