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
```

Before submitting, run:

```bash
python -m py_compile sprix_sage.py demo.py benchmark.py test_sprix_sage.py
python -m unittest -v
```

## Pull request expectations

- Explain the user or research problem being solved.
- Describe changes to utility, constraints, calibration, or update rules.
- Report test results and any benchmark movement without overstating synthetic evidence.
- Document backward-incompatible API changes.
- Confirm that the contribution is compatible with the MIT License.

By participating, you agree to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).
