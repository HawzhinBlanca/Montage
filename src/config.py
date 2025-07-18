#!/usr/bin/env python3
"""
Configuration module - Load env vars, expose DATABASE_URL, REDIS_URL, API keys, MAX_COST_USD
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:pass@localhost:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Cost limits
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "5.00"))

# Validation
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY not set - premium features will be disabled")
if not ANTHROPIC_API_KEY:
    print("⚠️  ANTHROPIC_API_KEY not set - premium features will be disabled")
if not DEEPGRAM_API_KEY:
    print("⚠️  DEEPGRAM_API_KEY not set - using only local ASR")

print(f"✅ Config loaded - MAX_COST_USD: ${MAX_COST_USD}")