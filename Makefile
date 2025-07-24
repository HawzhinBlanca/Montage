# Montage Build and Test Makefile
# Provides automation for development, testing, and deployment tasks

.PHONY: help vault-smoke install test lint security deploy clean

# Default target
help:
	@echo "Montage Build System"
	@echo "===================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install Python dependencies"
	@echo "  vault-smoke  Test Vault secret integration"
	@echo "  billing-check Run billing verification for API key exposure"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  security     Run security checks"
	@echo "  deploy       Deploy to production"
	@echo "  clean        Clean build artifacts"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install hvac  # Vault client for P0-04

vault-smoke:
	@python3 scripts/vault-smoke-test.py

billing-check:
	@python3 scripts/billing_check.py

test:
	pytest -m "not integration" tests/ -v --cov=montage --cov-report=term-missing --cov-report=html --cov-fail-under=10

lint:
	ruff montage tests

security:
	bandit -r montage -ll

deadcode:
	vulture montage --min-confidence 80

complexity:
	radon cc montage -na | grep -E "\b[C-F]\b" && { echo "Complexity too high"; exit 1; } || true

precommit:
	pre-commit run --all-files

deploy:
	@echo "🚀 Deploying Montage..."
	docker-compose -f docker-compose.yml up -d --build
	@echo "✅ Deployment complete"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf dist/ build/ *.egg-info/