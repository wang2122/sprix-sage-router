# Contributing to Sprix SAGE Router

Thank you for helping improve Sprix SAGE Router. The project welcomes focused contributions to routing algorithms, evaluation, A2A integration, reliability, documentation, and security.

## Before opening a pull request

1. Open an issue for substantial algorithm or API changes so the design can be discussed first.
2. Keep changes small, reviewable, and scoped to one concern.
3. Add or update tests for behavioral changes.
4. Do not include private task traces, credentials, proprietary Agent Cards, or personal data.

## Development

The reference implementation supports Python 3.10+ and has no runtime dependencies.

```bash
git clone https://github.com/wang2122/sprix-sage-router.git
cd sprix-sage-router
python -m unittest -v
python benchmark.py
python -m examples.a2a_execution_plan
```

Before submitting, run:

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy
python -m compileall -q sprix_sage.py sprix_learning.py sprix_types.py sprix_a2a.py demo.py benchmark.py benchmark_evaluator.py benchmark_dynamic.py benchmark_dynamic_evaluator.py benchmark_trust.py benchmark_scaling.py examples test_*.py
python -m unittest -v
python benchmark.py --seeds 3 --tasks-per-seed 25 --json benchmark-smoke.json
python benchmark_dynamic.py --seeds 3 --trajectories-per-seed 5 --sweep-cases-per-seed 2 --json dynamic-smoke.json --svg progress-smoke.svg
python benchmark_trust.py --seeds 3 --observations 50 --json trust-smoke.json
python benchmark_scaling.py --agent-counts 5,20 --repeats 1
```

Use a focused branch such as `fix/permission-filter` or `feat/a2a-adapter`. Keep generated benchmark JSON, private traces, local environments, and credentials out of commits.

## Pull request expectations

- Explain the user or research problem being solved.
- Describe changes to utility, constraints, calibration, or update rules.
- Report test results and any benchmark movement without overstating synthetic evidence.
- Document backward-incompatible API changes.
- Confirm that the contribution is compatible with the MIT License.

## Areas for contribution

- candidate retrieval and capability normalization;
- A2A discovery, execution, streaming, and cancellation clients;
- offline replay and stronger routing baselines;
- calibration, drift, regret, and causal credit assignment;
- adversarial bids, churn, privacy, and policy enforcement;
- observability, persistence backends, and approval workflows;
- documentation, reproducible examples, and accessibility.

## Pre-submission checklist

Before opening a pull request, confirm that the change is focused and reproducible:

- Run the compile and unit-test commands above, and include the result in the pull request description.
- If you run the benchmark smoke test, inspect the output locally and remove the generated JSON before committing.
- Add a regression test or a runnable example when the change alters routing, planning, persistence, or integration behavior.
- Keep documentation claims tied to an executable command, a checked-in fixture, or a clearly labeled synthetic result.

By participating, you agree to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).
