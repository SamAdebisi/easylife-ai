# EasyLife AI

EasyLife AI is a monorepo that showcases how to design, deploy, and operate production-grade machine learning services. The repo is structured around incremental phases (NLP, CV, TS forecasting, recommender systems, MLOps, monitoring, generative AI) so you can build, review, and ship one capability at a time.

## Quickstart

### 1. Prerequisites
- Python 3.11 (use `pyenv`, `asdf`, or the system interpreter)
- Docker Desktop or Colima with Docker Compose v2
- `make` and `python -m venv`

### 2. Bootstrap the Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```

### 3. Start the local infra stack
```bash
make up        # MLflow + MinIO + Prometheus + Grafana
make logs      # follow container logs
make down      # tear everything down
```

Services:
- MLflow (`http://localhost:5000`) with MinIO-backed artifact storage
- MinIO console (`http://localhost:9001`, user `minio`, password `minio123`)
- Prometheus (`http://localhost:9090`)
- Grafana (`http://localhost:3000`, user `admin`, password `admin`)

### 4. Wire up DVC (optional, run once per machine)
```bash
make dvc-init      # configures the MinIO remote and bootstraps data folders
```
The command expects Docker infra to be up so that MinIO is reachable. Credentials are stored in `.dvc/config.local`, keeping secrets out of Git.

## Repository Layout

| Path              | Purpose |
|-------------------|---------|
| `data/`           | DVC-managed datasets (`raw/`, `processed/`, `external/`) |
| `nlp_service/`    | FastAPI service + training code for the NLP phase |
| `cv_service/`     | Computer vision service (placeholder until Phase 2) |
| `ts_forecasting/` | Time-series forecasting service (Phase 3) |
| `recsys_service/` | Recommendation system components (Phase 4) |
| `mlops/`          | CI/CD, automation, experiment tracking assets |
| `docker/`         | Local infra Docker Compose and Prometheus config |
| `grafana/`        | Provisioned data sources/dashboards |
| `docs/`           | Architecture notes, model cards, playbooks |
| `k8s/`            | Kubernetes manifests (per-service deployments) |

See `docs/ARCHITECTURE.md` for a deeper view of each service and shared components.

## Common Tasks

- `make fmt` – auto-format the codebase (isort + black)
- `make lint` – run flake8 checks
- `make test` – execute the Python test suite
- `make precommit` – run the full pre-commit hook stack

## Phase 1 – NLP Sentiment Service

1. Generate the lightweight sentiment dataset and baseline model:
   ```bash
   dvc repro nlp-preprocess nlp-train
   ```
   This runs the data pipeline and logs metrics/model artifacts to MLflow.
2. Launch the FastAPI service locally:
   ```bash
   uvicorn nlp_service.app.main:app --reload
   ```
   or use `./nlp_service/run_dev.sh` for an env-aware script.
3. Exercise the endpoint and inspect telemetry:
   ```bash
   curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "Fantastic service and quick responses!"}'
   ```
   - Prometheus metrics: `http://127.0.0.1:8000/metrics`
   - MLflow runs: `http://127.0.0.1:5000` (experiments `nlp_sentiment` + `nlp_service_inference`)
4. Capture dashboards following `docs/observability/dashboard_capture.md` and store images under `docs/assets/`.

## Phase 2 – Computer Vision QC Service

1. Ingest data (or generate fallbacks), build manifest, and train models:
   ```bash
   dvc repro cv-ingest cv-preprocess cv-train cv-train-cnn
   ```
   This logs Laplacian and CNN metrics to MLflow (`cv_blur_detection`, `cv_blur_detection_cnn`) and writes artifacts to `cv_service/artifacts/`.
   - Configure ingestion via `configs/cv_ingest.yaml` (supports `copy` or `symlink` strategies and custom glob patterns).
2. Launch the FastAPI service locally:
   ```bash
   uvicorn cv_service.app.main:app --reload --port 8002
   ```
   or use `./cv_service/run_dev.sh` (listens on `${PORT:-8002}`).
3. Send a test image for blur assessment:
   ```bash
   curl -s -X POST http://127.0.0.1:8002/predict_image \
     -F "file=@img/sample_sharp.png"
   ```
   - Prometheus metrics: `http://127.0.0.1:8002/metrics`
   - MLflow runs: `http://127.0.0.1:5000` (experiments `cv_blur_detection` + `cv_service_inference`)
4. Follow `docs/observability/dashboard_capture.md` to snapshot Grafana panels (e.g., `cv_predictions_total`, `sharp_probability`).
5. Switch inference between threshold and CNN variants via `CV_MODEL_VARIANT` environment variable (defaults to `threshold`). Set `CV_MODEL_VARIANT=cnn` to exercise the TorchScript model.

## Phase 3 – Time-series Forecasting Service

1. Generate the synthetic KPI dataset and baseline Holt-Winters model:
   ```bash
   dvc repro ts-prepare ts-train
   ```
   Metrics are logged to MLflow under the `ts_forecasting` experiment; artifacts land in `ts_forecasting/artifacts/`.
2. Launch the FastAPI forecasting API:
   ```bash
   ./ts_forecasting/run_dev.sh   # defaults to port 8003
   ```
3. Request a forecast and inspect Prometheus metrics:
   ```bash
   curl -s http://127.0.0.1:8003/forecast -H "Content-Type: application/json" \
     -d '{"horizon": 14}'
   ```
   The service emits `ts_forecast_requests_total` and `ts_forecast_horizon` metrics for Grafana dashboards.

## Phase 4 – Recommendation Service

1. Synthesize interaction data and train the collaborative-filtering baseline:
   ```bash
   dvc repro recsys-prepare recsys-train
   ```
   Training logs `recall_at_10` and `ndcg_at_10` to the `recsys_collaborative_filtering` MLflow experiment and stores artifacts in `recsys_service/artifacts/`.
2. Start the recommender API:
   ```bash
   ./recsys_service/run_dev.sh   # listens on ${PORT:-8004}
   ```
3. Fetch personalised recommendations and similar items:
   ```bash
   curl -s -X POST http://127.0.0.1:8004/recommendations \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user-0", "top_k": 5}'
   curl -s http://127.0.0.1:8004/items/item-0/similar?top_k=5
   ```
   Prometheus exposes `recsys_recommendation_requests_total` and `recsys_topk_requested` for dashboarding.

## Contributing

1. Create a feature branch.
2. Keep changes scoped to a single roadmap phase (e.g., `phase-1-nlp`).
3. Run `make fmt lint test precommit`.
4. Push and open a PR. The GitHub Actions workflow will lint and run the unit tests.

Advanced contribution details (branching, DVC data flow, release cadence) live in `CONTRIBUTING.md`.

## Next Steps

1. Complete Phase 1 (NLP module) with data ingestion, training, and service endpoints.
2. Track metrics and artifacts in MLflow to verify the observability stack.
3. Build Phase 2+ modules side by side while keeping each phase production-ready.
