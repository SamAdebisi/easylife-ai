
Architecture

repo
├── data/                 # versioned via DVC
├── docker/               # docker-compose infra
├── grafana/              # provisioning
├── nlp_service/          # FastAPI microservice
├── cv_service/
├── ts_forecasting/
├── recsys_service/
├── mlops/                # pipelines, ci-cd docs
├── k8s/                  # manifests
└── docs/                 # docs

	•	Artifact store: MinIO s3://mlflow/
	•	Tracking: MLflow server with SQLite backend by default.
	•	Observability: Prometheus + Grafana.