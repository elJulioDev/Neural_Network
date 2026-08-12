# Contributing to NeuralNetwork

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/elJulioDev/Neural_Network.git
cd Neural_Network
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Before Submitting

1. **Lint**: `ruff check nnlib/ tests/ examples/ main.py`
2. **Type check**: `mypy nnlib/ --ignore-missing-imports`
3. **Tests**: `python -m unittest discover tests -v`
4. All 68 tests must pass.

## Code Style

* Follow existing patterns — layers are stateless w.r.t. trainable state, caches returned from `forward()`.
* Use `logging` module, not `print()`.
* Type hints on all public methods.
* Docstrings on all public methods.
* No pickle — persistence is `topology.json` + `weights.npz`.

## Pull Requests

* Keep PRs focused on a single change.
* Add tests for new features.
* Update `CHANGELOG.md` under an `[Unreleased]` section.
* Do not commit `AGENTS.md` — it's a local instruction file only.
