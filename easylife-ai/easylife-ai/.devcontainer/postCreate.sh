#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
pip install -r requirements-dev.txt
pre-commit install || true
