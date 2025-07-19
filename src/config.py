#!/usr/bin/env python3
"""
Configuration module - Load env vars with explicit path
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'

print(f"Loading .env from: {env_path}")
load_dotenv(env_path, override=True)  # Force override any existing env vars

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:pass@localhost:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Cost limits
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "5.00"))

# Debug: Show loaded keys (first 20 chars only)
if OPENAI_API_KEY:
    print(f"✅ OpenAI key loaded: {OPENAI_API_KEY[:20]}...")
else:
    print("⚠️  OPENAI_API_KEY not set - premium features will be disabled")
    
if ANTHROPIC_API_KEY:
    print(f"✅ Anthropic key loaded: {ANTHROPIC_API_KEY[:20]}...")
else:
    print("⚠️  ANTHROPIC_API_KEY not set - premium features will be disabled")
    
if DEEPGRAM_API_KEY:
    print(f"✅ Deepgram key loaded: {DEEPGRAM_API_KEY[:20]}...")
else:
    print("⚠️  DEEPGRAM_API_KEY not set - using only local ASR")

print(f"✅ Config loaded - MAX_COST_USD: ${MAX_COST_USD}")
