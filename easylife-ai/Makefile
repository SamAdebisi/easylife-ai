.PHONY: up down logs lint test fmt precommit dvc-init

up:
docker compose -f docker/docker-compose.yml up -d –build

e2e:
docker compose -f docker/docker-compose.yml up –build

down:
docker compose -f docker/docker-compose.yml down -v

logs:
docker compose -f docker/docker-compose.yml logs -f –tail=100

lint:
flake8

fmt:
isort . && black .

test:
pytest -q

precommit:
pre-commit run –all-files

dvc-init:
dvc init -q && dvc remote add -d minio s3://datasets –force
–endpointurl http://127.0.0.1:9000 &&
dvc remote modify minio access_key_id minio &&
dvc remote modify minio secret_access_key minio123 &&
echo ‘data/raw’ | dvc add -q -


# .PHONY: up down logs fmt lint test
# up:        ## start services
# 	docker compose up -d
# down:      ## stop services
# 	docker compose down
# logs:      ## tail logs
# 	docker compose logs -f
# fmt:
# 	black .
# 	isort .
# lint:
# 	flake8 .
# 	mypy || true
# test:
# 	pytest -q
