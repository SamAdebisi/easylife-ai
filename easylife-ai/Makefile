.PHONY: up down logs lint test fmt precommit dvc-init e2e

COMPOSE := docker compose -f docker/docker-compose.yml

e2e:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f --tail=100

lint:
	flake8

fmt:
	isort .
	black .

test:
	pytest -q

precommit:
	pre-commit run --all-files

dvc-init:
	dvc init -q
	dvc remote add -d minio s3://datasets --force
	dvc remote modify minio endpointurl http://127.0.0.1:9000
	dvc remote modify --local minio access_key_id minio
	dvc remote modify --local minio secret_access_key minio123
	dvc repro bootstrap-data-dirs

check-docker:
	@command -v docker >/dev/null || { echo "Install Docker Desktop (Option A) or Colima (Option B) first."; exit 1; }

up: check-docker
	$(COMPOSE) up -d --build
