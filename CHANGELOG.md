# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-10

### Added
- Modern packaging with `pyproject.toml` (PEP 517/518).
- `LICENSE` file (MIT).
- Linting configuration with `ruff` in CI.
- `CHANGELOG.md`.

### Changed
- Package renamed from `src` to `nnlib` (imports: `from nnlib import NeuralNetwork`).
- Removed `sys.path.insert` from tests and examples (no longer needed with `pip install -e .`).
- CI updated: Python 3.9/3.10/3.11/3.12 matrix, install with `pip install -e ".[dev]"`.
- `siamese_network.py` moved from `src/` to `examples/`.
- Python 3.8 dropped from support (EOL).

### Removed
- `setup.py` (replaced by `pyproject.toml`).
