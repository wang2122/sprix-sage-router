# Changelog

## Unreleased

### Checkpoint-aware rerouting

- add explicit in-flight requirement ownership, completion, observed partial
  quality, and per-artifact portability to `ExecutionState`;
- estimate candidate capability, cost, and latency from reusable work, while
  charging reconfiguration loss once instead of duplicating it in the success
  model and utility;
- replace team noisy-OR coverage with assigned-owner checkpoint-adjusted
  coverage so unassigned teammates do not receive quality credit.

### Focused research evaluation

- add a controlled progress intervention and 1,000-event trajectory replay
  with progress-masked, always-continue, always-handoff, static greedy,
  static-coalition, and dynamic-oracle baselines;
- add a held-out artifact/remaining-work evaluator that does not call router
  switch-loss helpers;
- add a shared-observation convergence study for per-requirement trust versus
  one reputation score, including a homogeneous negative control;
- narrow the documented contribution to checkpoint-aware mid-execution
  rerouting and explicitly demote standard coalition, utility, search, belief,
  and online-learning components.

## 0.3.0 - 2026-08-28

### Benchmark rigor

- separate the synthetic evaluator from router scoring and use independently
  specified latent skills, geometric bottleneck quality, multiplicative team
  compatibility, nonlinear workload cost, and distinct handoff/latency models;
- add random-team and greedy-team baselines;
- split static, learned-no-exploration, learned-exploration, and random-prior
  policies, with first/last learning windows in schema-v2 JSON output;
- remove the stale v0.2 benchmark figure and publish exact five-seed v0.3
  results with explicit synthetic-evidence limits.

### Router behavior and scalability

- add a deterministic configurable candidate prefilter (top 12 by default) and
  request-local calibrated skill/cost/latency caches;
- add workload-sensitive cost with a configurable activation fraction;
- return auditable best-effort budget/deadline violations while preserving hard
  permission, failure, and availability filters;
- update pair synergy only from explicit pair-level evidence.

### Engineering and release

- split public data types and online learning into `sprix_types.py` and
  `sprix_learning.py`;
- add ruff and mypy CI gates, package-build checks, and a routing-scaling tool;
- add source-linked related-work comparisons and a tokenless PyPI Trusted
  Publishing workflow, gated until the PyPI publisher is configured.
