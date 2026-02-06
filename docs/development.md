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
cd docs
jupyter-book build .
```

The built HTML documentation will be in `docs/_build/html/`.

## Running Tests

```bash
pytest tests/
```
