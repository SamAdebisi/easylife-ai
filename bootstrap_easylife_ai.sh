#!/usr/bin/env bash
set -euo pipefail

REPO="easylife-ai"
PY_VERSION="3.11"

# 1) Structure
mkdir -p "$REPO" && cd "$REPO"
mkdir -p data/raw data/processed data/external \
         nlp_service/{app,tests} cv_service/{app,tests} \
         ts_forecasting/{app,tests} recsys_service/{app,tests} \
         mlops pipelines docker k8s docs/docs img grafana/provisioning/{dashboards,datasources}

# 2) Python deps
cat > requirements.txt <<'EOF'
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2
numpy==2.1.1
pandas==2.2.2
scikit-learn==1.5.2
mlflow==2.16.0
EOF

cat > requirements-dev.txt <<'EOF'
-r requirements.txt
pytest==8.3.2
pytest-cov==5.0.0
black==24.8.0
isort==5.13.2
flake8==7.1.1
pre-commit==3.8.0
dvc[s3]==3.58.0
EOF

# 3) Pre-commit config
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
      - id: end-of-file-fixer
      - id: trailing-whitespace
EOF

# 4) Basic configs
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.env
.venv/
venv/
.env*
*.log

# DVC
/.dvc/
*.dvc
/data/*
!/data/.gitkeep

# MLflow
mlruns/

# OS
.DS_Store

# Byte-compiled
*.so
EOF

touch data/.gitkeep

# 5) README and docs
cat > README.md <<'EOF'
# EasyLife AI

Monorepo for production-grade ML services. See `docs/ARCHITECTURE.md` for layout.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
make up   # start infra: MLflow, MinIO, Prometheus, Grafana
