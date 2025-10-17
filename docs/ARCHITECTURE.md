# Architecture Overview

EasyLife AI is organised as a monorepo so that shared tooling, infrastructure, and documentation can evolve alongside each machine learning service. Each roadmap phase lives in its own top-level directory and can be iterated on independently while reusing the common platform pieces.

```
repo
├── data/             # DVC tracked datasets (raw, processed, external)
├── docker/           # Docker Compose stack: MinIO, MLflow, Prometheus, Grafana
├── grafana/          # Provisioned datasources/dashboards for observability
├── nlp_service/      # Phase 1 – NLP training + FastAPI inference service
├── cv_service/       # Phase 2 – Computer vision APIs
├── ts_forecasting/   # Phase 3 – Forecasting pipelines
├── recsys_service/   # Phase 4 – Recommendation engines
├── mlops/            # CI/CD, automation, workflows
├── k8s/              # Kubernetes manifests consumed per service
└── docs/             # Architecture notes, model cards, playbooks
```

## Core Platform Components

- **Storage & Artifacts** – MinIO exposes an S3-compatible endpoint (`s3://mlflow`) for model artifacts, datasets, and experiment outputs.
- **Experiment Tracking** – MLflow Server runs with an embedded SQLite backend; artifacts are shipped to MinIO. Each phase should log parameters, metrics, and models to this instance.
- **Observability** – Prometheus scrapes metrics from platform services, while Grafana ships with an auto-provisioned Prometheus datasource. Service-specific dashboards live under `grafana/`.
- **Data Version Control** – DVC manages dataset snapshots inside `data/`. Directories are created via the `bootstrap-data-dirs` pipeline stage. Remotes default to MinIO.
  - NLP phase adds `nlp-preprocess` and `nlp-train` stages that generate a lightweight sentiment dataset and persist a TF-IDF + Logistic Regression pipeline.

## Service Lifecycle

1. **Data acquisition** – Stored in `data/raw/` (tracked with DVC) and transformed into `data/processed/`.
2. **Training pipelines** – Each service owns a training script or notebook that logs runs to MLflow.
3. **Model packaging** – Models and metadata are registered via MLflow and stored in MinIO.
4. **Serving** – FastAPI applications (one per service) expose `/health`, `/metrics`, and domain-specific endpoints. Container images live beside service code.
5. **Deployment** – Dockerfiles support local testing; Kubernetes manifests in `k8s/` provide production deployment examples with configmaps, secrets, and autoscaling hooks.
6. **Monitoring** – Prometheus/Grafana capture infrastructure metrics. Model-specific monitoring (e.g., drift) will be added in later phases.

## Phase Breakdown

- **Phase 0** – Platform bootstrap (this commit): repo scaffolding, reproducible local stack, DVC wiring, documentation.
- **Phase 1** – NLP sentiment classification service with data pipelines, TF-IDF baseline, HuggingFace upgrade, FastAPI endpoints.
- **Phase 2** – Computer vision quality control (blur/defect detection) with OpenCV baselines and PyTorch upgrades.
- **Phase 3** – Time-series forecasting service with backtesting, Prophet/ARIMA baselines, and orchestration via Airflow + CronJobs.
- **Phase 4** – Recommendation system with offline/online workflows, feature store, and evaluation metrics (NDCG, Recall).
- **Phase 5–9** – MLOps automation, monitoring & explainability, generative add-ons, resiliency, and portfolio polish.

Each phase should ship with:
- Reproducible training scripts and data pipelines (DVC stages, notebooks, or ETL code in `pipelines/`).
- Containerised FastAPI microservice with contract tests.
- Kubernetes manifests and CI/CD hooks.
- Documentation in `docs/model_cards/`, `docs/playbooks/`, or service-specific READMEs.

## Phase 1 Snapshot

- **Data Pipeline** – `pipelines/nlp_prepare_data.py` synthesises balanced reviews and writes `data/processed/nlp/{train,test}.csv` via the `nlp-preprocess` DVC stage.
- **Training** – `nlp_service/train.py` fits a TF-IDF + Logistic Regression classifier, logs metrics to MLflow (`nlp_sentiment` experiment), and stores artifacts in `nlp_service/artifacts/`.
- **Serving** – `nlp_service/app/main.py` exposes `/predict`, `/health`, and `/metrics` (Prometheus). It auto-loads the stored model or backfills one if missing, and streams inference telemetry to MLflow (`nlp_service_inference` experiment).
- **Monitoring** – Prometheus counter `nlp_predictions_total` is emitted automatically; Grafana dashboards can be captured following `docs/observability/dashboard_capture.md`.

## Phase 2 Snapshot

- **Data Pipeline** – `pipelines/cv_prepare_data.py` generates geometric patterns with and without Gaussian blur, producing images in `data/raw/cv/` and manifest metadata in `data/processed/cv/labels.csv` via the `cv-preprocess` stage.
- **Data Pipeline** – `pipelines/cv_ingest_data.py` copies real data (or synthesises fallbacks) into `data/raw/cv/`, while `pipelines/cv_prepare_data.py` produces the manifest leveraged by training stages.
- **Training** –
  - Baseline threshold: `cv_service/train.py` computes variance-of-Laplacian scores, identifies an optimal threshold, logs metrics to MLflow (`cv_blur_detection`), and persists artifacts in `cv_service/artifacts/`.
  - CNN upgrade: `cv_service/train_cnn.py` trains a Torch CNN, exports a TorchScript model, and logs results under `cv_blur_detection_cnn`.
- **Serving** – `cv_service/app/main.py` exposes `/predict_image`, `/health`, and `/metrics`. Requests accept multipart images, emit `cv_predictions_total` Prometheus counters, and log blur scores to MLflow (`cv_service_inference`).
- **Model Variants** – The service can load either the threshold or CNN model via the `CV_MODEL_VARIANT` environment variable, enabling rapid A/B comparisons.
- **Testing** – `cv_service/tests/test_predict_image.py` covers inference, metrics exposure, and ensures the fallback training path keeps the model ready for local runs.

## Phase 3 Snapshot

- **Data Pipeline** – `pipelines/ts_prepare_data.py` synthesises a seasonal KPI series and stores it under `data/raw/ts/` and `data/processed/ts/series.csv`.
- **Training** – `ts_forecasting/train.py` fits an additive Holt-Winters baseline, logs metrics (`mae`, `rmse`, `mape`) to MLflow (`ts_forecasting` experiment), and writes artifacts (model, metadata) to `ts_forecasting/artifacts/`.
- **Serving** – `ts_forecasting/app/main.py` exposes `/forecast` alongside `/health` and `/metrics`, instrumented with Prometheus counters/histograms for horizon tracking.
- **Testing** – `ts_forecasting/tests/test_forecast.py` validates the API contract and ensures forecasts are returned for valid horizons.

## Phase 4 Snapshot

- **Data Pipeline** – `pipelines/recsys_prepare_data.py` builds a synthetic user–item interaction dataset plus item catalog into `data/raw/recsys/` and `data/processed/recsys/`.
- **Training** – `recsys_service/train.py` trains a truncated-SVD collaborative filtering model, logs `recall@10` and `ndcg@10` to MLflow (`recsys_collaborative_filtering`), and persists artifacts under `recsys_service/artifacts/`.
- **Serving** – `recsys_service/app/main.py` serves personalised `/recommendations` and `/items/{item_id}/similar` endpoints with Prometheus metrics for request volume and top-k distribution.
- **Testing** – `recsys_service/tests/test_health.py` (extended) seeds artifacts, exercises recommendation and similarity endpoints, and verifies fallbacks for unseen users/items.

## Phase 5 Snapshot

- **Explainability** – `nlp_service/explain.py` surfaces token-level contributions for the sentiment classifier; `/explain` logs the payload to MLflow for post-hoc analysis.
- **Monitoring Automation** – `mlops/collect_metrics.py` queries Prometheus and exports per-metric aggregates so cron/DVC jobs can monitor traffic patterns offline.
- **Testing** – `mlops/tests/test_collect_metrics.py` mocks Prometheus responses to guarantee the collector generates correct CSV summaries.
