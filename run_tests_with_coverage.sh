#!/bin/bash

# Export required environment variables
export JWT_SECRET_KEY="test-jwt-secret-key-for-testing"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/test_db"
export REDIS_URL="redis://localhost:6379/0"
export ANTHROPIC_API_KEY="test-key"
export DEEPGRAM_API_KEY="test-key"
export GEMINI_API_KEY="test-key"
export HUGGINGFACE_TOKEN="test-token"
export OPENAI_API_KEY="test-key"

# Run tests with coverage
echo "Running tests with coverage..."
coverage run --source=montage -m pytest tests/test_security.py -v
coverage report
coverage html

echo "Coverage report generated in htmlcov/"