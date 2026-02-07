# Development Guide

This page contains notes for developers contributing to LogPSplinePSD.

## Version Control and Releases

### Commit Types for Releases

**Release-triggering commits:**

- `feat:` - New features → minor version bump (0.1.0 → 0.2.0)
- `fix:` - Bug fixes → patch version bump (0.1.0 → 0.1.1)
- `perf:` - Performance improvements → patch version bump
- `feat!:`, `fix!:`, `perf!:` - Breaking changes → major version bump (0.1.0 → 1.0.0)

**Non-release commits:**

- `docs:` - Documentation only
- `style:` - Code formatting, whitespace
- `refactor:` - Code restructuring without behavior change
- `test:` - Adding or updating tests
- `build:` - Build system changes
- `ci:` - CI configuration changes
- `chore:` - Maintenance tasks

## Building Documentation

To build the documentation locally:

```bash
source .venv/bin/activate
cd docs
# The docs in this repo are RST/Sphinx-based and require Jupyter Book 1.x.
# Ensure you have installed the dev extras (or at least `jupyter-book<2.0`).
../.venv/bin/jupyter-book build .
```

The built HTML documentation will be in `docs/_build/html/`.

## Running Tests

```bash
.venv/bin/python -m pytest tests/
```

## Typechecking (Jaxtyping + Beartype)

Install dev extras so `jaxtyping` and `beartype` are available:

```bash
.venv/bin/python -m pip install -e '.[dev,typecheck]'
```

Run static type checking across the package:

```bash
.venv/bin/python -m mypy --config-file pyproject.toml src/log_psplines
```

The mypy configuration in `pyproject.toml` includes scoped overrides for a
small set of external libraries without stubs. The package source under
`src/log_psplines` is checked end-to-end without per-module suppressions.

Runtime checks are enabled by default when dependencies are installed. You can
disable runtime enforcement with:

```bash
LOG_PSPLINES_RUNTIME_TYPECHECK=0 .venv/bin/python -m pytest tests/test_runtime_typecheck.py
```
