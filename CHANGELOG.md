# Changelog

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
