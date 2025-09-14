# EasyLife AI

Monorepo for production-grade ML services. See `docs/ARCHITECTURE.md` for layout.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
make up   # start infra: MLflow, MinIO, Prometheus, Grafana
