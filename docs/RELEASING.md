# Release guide

## Local verification

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy
python -m unittest -v
python benchmark.py --seeds 3 --tasks-per-seed 25
python -m build
python -m twine check dist/*
```

The project version in `pyproject.toml` and `CITATION.cff` must match the Git
tag. A GitHub release triggers the distribution build workflow.

## PyPI trusted publishing

The repository contains an OIDC-based PyPI workflow and stores no API token.
Before the first upload, a PyPI project owner must configure a Trusted Publisher
for:

- owner: `wang2122`
- repository: `sprix-sage-router`
- workflow: `publish-pypi.yml`
- environment: `pypi`

Then create the protected GitHub `pypi` environment and set the repository
variable `PYPI_PUBLISH_ENABLED=true`. Until both sides are configured, GitHub
releases still build and validate artifacts, while the upload job is skipped
instead of reporting a false successful PyPI publication.
