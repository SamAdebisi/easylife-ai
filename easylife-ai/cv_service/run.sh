#!/usr/bin/env bash
set -euo pipefail

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Mark executable: chmod +x run.sh.

Devcontainer setup

Create .devcontainer/devcontainer.json.

{
  "name": "easylife-ai",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements-dev.txt && pre-commit install",
  "customizations":
                    {
    "vscode":
                {
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-toolsai.jupyter"
      ]
    }
  }
}
