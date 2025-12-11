# Repository Guidelines

## Project Structure & Module Organization
- Package code lives in `pybromo/` (core modules, PSF, diffusion, IO in `pybromo/utils/`).
- Tests live in `pybromo/tests/` (pytest style: `test_*.py`).
- Data assets in `pybromo/psf_data/` (MAT files used by PSF routines).
- Notebooks in `notebooks/` for examples and theory notes.
- Packaging: `pyproject.toml`, `setup.py`, `versioneer.py` manage builds and versioning.

## Build, Test, and Development Commands
- Using uv (preferred if available)
  - Install: `uv sync` (or `uv sync --group dev` for dev extras)
  - Run tests: `uv run pytest -q`
  - Lint: `uv run ruff lint .`
  - Format: `uv run ruff format .`
- Using standard Python
  - Editable install: `python -m pip install -e .`
  - Run tests: `python -m pytest -q`
  - Lint: `ruff lint .`
  - Format: `ruff format .`

## Coding Style & Naming Conventions
- Python 3.13+; follow PEP 8.
- Formatting & linting: Ruff (`ruff format`, `ruff lint`).
- Indentation: 4 spaces; line length: 89. Configure in Ruff (`pyproject.toml`: `[tool.ruff] line-length = 89`).
- Naming: modules/functions `snake_case`, classes `CapWords`, constants `UPPER_SNAKE`.
- Keep public API imports in `pybromo/__init__.py` minimal and explicit.
- Prefer small, pure functions; document behavior and units in docstrings.

## Testing Guidelines
- Framework: `pytest` with simple, deterministic tests.
- Location/naming: put tests in `pybromo/tests/` named `test_*.py` (e.g., `pybromo/tests/test_diffusion.py`).
- Add tests for new features and bug fixes; include edge cases and random-state reproducibility where applicable.
- Run full suite locally before PRs: `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (<= 72 chars), body explaining what/why.
  - Example: `fix(diffusion): correct wrap_mirror boundary handling`
- Reference issues with `#123` and include minimal repro or affected module names.
- PRs must include: clear description, linked issues, test results, and any performance/behavioral notes; add screenshots only for UI/plots when relevant (e.g., notebook outputs).

## Security & Configuration Tips
- Do not commit large generated files (`.hdf5`, `dist/`, `build/`); `.gitignore` already covers common artifacts.
- Keep dataset paths configurable; avoid hard-coded user-specific paths in notebooks or modules.
- Versioning is handled by Versioneer (`pybromo/_version.py`); do not hand-edit that file.
