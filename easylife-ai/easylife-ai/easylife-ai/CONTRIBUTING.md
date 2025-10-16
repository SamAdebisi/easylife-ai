# Contributing Guidelines

Thanks for helping build EasyLife AI. The project is staged by roadmap phase (NLP, CV, Forecasting, Recsys, MLOps, Monitoring, Generative AI), so keep contributions narrow and phase-focused to maintain clear history.

## Local Setup

1. Create a virtual environment (`python -m venv .venv`) and activate it.
2. Install development dependencies: `pip install -r requirements-dev.txt`.
3. Install pre-commit hooks: `pre-commit install`.
4. Start the infra stack when you need MLflow/MinIO/Prometheus/Grafana: `make up`.

## Branching & Commits

- Use descriptive branch names, e.g., `phase-1/data-pipeline`.
- Keep commits scoped and message them in imperative form (`Add TF-IDF baseline`).
- Avoid mixing multiple roadmap phases in one PR.

## Coding Standards

- Follow `black` + `isort` style (`make fmt`).
- `flake8` and `pytest` must pass locally (`make lint test`).
- Prefer small, composable functions with type hints and docstrings where helpful.
- Add unit/integration tests alongside new features (`tests/` folders live under each service).

## Data & DVC

- Run `make dvc-init` once to configure the MinIO remote.
- Use `dvc repro` to execute pipelines defined in `dvc.yaml`. Keep large datasets out of Git.
- Store credentials in `.dvc/config.local` (auto-created by `dvc remote modify --local`) instead of committing secrets.

## Pull Requests

- Ensure GitHub Actions succeed (lint + tests). Fix any broken checks before requesting review.
- Provide a concise description, screenshots/logs, and MLflow run IDs if relevant.
- Document new behaviour in `docs/` (architecture updates, model cards, or playbooks).

## Questions

Open a GitHub Discussion or file an issue if you need clarity on roadmap alignment, dataset sources, or infrastructure changes.
