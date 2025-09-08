.PHONY: help install dev-install test lint format type-check run clean docker-build docker-run

# Variables
PYTHON = python3
PIP = pip
POETRY = poetry
APP_NAME = mmm-application
DOCKER_IMAGE = mmm-app
DOCKER_TAG = latest

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src/mmm --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting
	flake8 src/ tests/
	isort --check-only src/ tests/
	black --check src/ tests/

format: ## Format code
	isort src/ tests/
	black src/ tests/

type-check: ## Run type checking
	mypy src/mmm

quality: lint type-check ## Run all quality checks

fix: format ## Fix code formatting issues
	@echo "Code formatted"

run: ## Run the application
	$(PYTHON) -m mmm

run-dev: ## Run the application in development mode
	$(PYTHON) -m mmm --reload --log-level DEBUG

run-docker: ## Run the application in Docker
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Docker commands
docker-build: ## Build Docker image for production
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) --target production .

docker-build-dev: ## Build Docker image for development
	docker build -t $(DOCKER_IMAGE):dev --target development .

docker-run: ## Run Docker container (standalone)
	docker run -d \
		-p 8000:8000 \
		-v mmm_data:/app/data \
		-v mmm_logs:/app/logs \
		-v mmm_uploads:/app/static/uploads \
		--name mmm-container \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-stop: ## Stop and remove Docker container
	docker stop mmm-container || true
	docker rm mmm-container || true

# Docker Compose commands
docker-up: ## Start services with Docker Compose (development)
	docker-compose up -d

docker-up-build: ## Build and start services with Docker Compose
	docker-compose up -d --build

docker-down: ## Stop and remove Docker Compose services
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

docker-ps: ## Show running containers
	docker-compose ps

# Production Docker Compose
docker-prod-up: ## Start production services
	docker-compose -f docker-compose.prod.yml up -d

docker-prod-build: ## Build and start production services
	docker-compose -f docker-compose.prod.yml up -d --build

docker-prod-down: ## Stop production services
	docker-compose -f docker-compose.prod.yml down

docker-prod-logs: ## View production logs
	docker-compose -f docker-compose.prod.yml logs -f

# Docker maintenance
docker-clean: ## Clean up Docker images and containers
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

docker-reset: ## Reset all Docker data (WARNING: destroys all data)
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.prod.yml down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Container shell access
docker-shell: ## Access running container shell
	docker-compose exec mmm-app /bin/bash

docker-redis-cli: ## Access Redis CLI
	docker-compose exec redis redis-cli

setup: dev-install ## Initial setup for development
	mkdir -p static/uploads logs
	cp .env.example .env
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head
	@echo "Setup complete! Edit .env file as needed and run 'make run-dev'"

db-init: ## Initialize database with migrations
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head

db-migrate: ## Create new migration
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply migrations
	alembic upgrade head

db-reset: ## Reset database (WARNING: destroys all data)
	rm -f mmm_app.db
	alembic upgrade head

# CI/CD targets
ci-test: ## Run tests for CI
	pytest tests/ -v --cov=src/mmm --cov-report=xml --junit-xml=test-results.xml

ci-quality: ## Run quality checks for CI
	flake8 src/ tests/ --format=json --output-file=flake8-report.json || true
	mypy src/mmm --junit-xml=mypy-report.xml || true
	
deploy-staging: ## Deploy to staging (placeholder)
	@echo "Deploying to staging environment..."
	
deploy-prod: ## Deploy to production (placeholder)
	@echo "Deploying to production environment..."