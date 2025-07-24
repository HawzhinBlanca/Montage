#!/usr/bin/env python3
"""
Rate limiting configuration for different environments and deployment scenarios.

This module provides environment-specific configurations to optimize rate limiting
for development, testing, staging, and production environments.
"""

import logging
import os
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

from .rate_limiter import RateLimitConfig

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


def get_current_environment() -> Environment:
    """Detect current environment from environment variables"""
    env_name = os.getenv("MONTAGE_ENV", "development").lower()

    try:
        return Environment(env_name)
    except ValueError:
        logger.warning(f"Unknown environment '{env_name}', defaulting to development")
        return Environment.DEVELOPMENT


class EnvironmentConfig:
    """Configuration manager for different environments"""

    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment or get_current_environment()
        logger.info(
            f"Rate limiting configured for {self.environment.value} environment"
        )

    def get_deepgram_config(self) -> RateLimitConfig:
        """Get Deepgram rate limiting config based on environment"""

        configs = {
            Environment.DEVELOPMENT: RateLimitConfig(
                max_tokens=10,  # Generous for development
                refill_rate=2.0,  # Fast refill for testing
                burst_capacity=5,  # Allow bursts
                max_cost_per_minute=Decimal("5.00"),  # Higher budget for dev
                cost_per_request=Decimal("0.003"),
                failure_threshold=5,  # More tolerant
                recovery_timeout=30,  # Quick recovery
                max_queue_size=100,
                request_timeout=600,
                min_interval=0.1,  # Fast for dev
                max_interval=5.0,
            ),
            Environment.TESTING: RateLimitConfig(
                max_tokens=5,  # Limited for tests
                refill_rate=1.0,  # Moderate refill
                burst_capacity=2,  # Small burst
                max_cost_per_minute=Decimal("1.00"),  # Low budget for tests
                cost_per_request=Decimal("0.003"),
                failure_threshold=3,  # Quick failure detection
                recovery_timeout=15,  # Fast recovery for tests
                max_queue_size=20,  # Small queue
                request_timeout=120,  # Short timeout
                min_interval=0.5,
                max_interval=10.0,
            ),
            Environment.STAGING: RateLimitConfig(
                max_tokens=3,  # Conservative
                refill_rate=0.8,  # Slower refill
                burst_capacity=2,
                max_cost_per_minute=Decimal("2.00"),  # Moderate budget
                cost_per_request=Decimal("0.003"),
                failure_threshold=3,
                recovery_timeout=45,
                max_queue_size=50,
                request_timeout=300,
                min_interval=0.8,
                max_interval=15.0,
            ),
            Environment.PRODUCTION: RateLimitConfig(
                max_tokens=2,  # Very conservative
                refill_rate=0.5,  # Slow refill
                burst_capacity=1,  # Minimal burst
                max_cost_per_minute=Decimal("1.50"),  # Strict budget
                cost_per_request=Decimal("0.003"),
                failure_threshold=2,  # Low tolerance
                recovery_timeout=60,  # Longer recovery
                max_queue_size=30,  # Controlled queue
                request_timeout=600,
                min_interval=1.0,  # Slower pace
                max_interval=30.0,  # Longer backoff
            ),
        }

        return configs[self.environment]

    def get_openai_config(self) -> RateLimitConfig:
        """Get OpenAI rate limiting config based on environment"""

        configs = {
            Environment.DEVELOPMENT: RateLimitConfig(
                max_tokens=5,
                refill_rate=1.0,
                burst_capacity=2,
                max_cost_per_minute=Decimal("3.00"),
                cost_per_request=Decimal("0.02"),
                failure_threshold=5,
                recovery_timeout=30,
                max_queue_size=50,
                request_timeout=300,
                min_interval=0.5,
                max_interval=10.0,
            ),
            Environment.TESTING: RateLimitConfig(
                max_tokens=2,
                refill_rate=0.5,
                burst_capacity=1,
                max_cost_per_minute=Decimal("0.50"),
                cost_per_request=Decimal("0.02"),
                failure_threshold=2,
                recovery_timeout=15,
                max_queue_size=10,
                request_timeout=120,
                min_interval=1.0,
                max_interval=30.0,
            ),
            Environment.STAGING: RateLimitConfig(
                max_tokens=3,
                refill_rate=0.5,
                burst_capacity=1,
                max_cost_per_minute=Decimal("1.50"),
                cost_per_request=Decimal("0.02"),
                failure_threshold=3,
                recovery_timeout=45,
                max_queue_size=20,
                request_timeout=180,
                min_interval=1.5,
                max_interval=45.0,
            ),
            Environment.PRODUCTION: RateLimitConfig(
                max_tokens=2,
                refill_rate=0.33,  # 1 every 3 seconds
                burst_capacity=1,
                max_cost_per_minute=Decimal("1.00"),
                cost_per_request=Decimal("0.02"),
                failure_threshold=2,
                recovery_timeout=60,
                max_queue_size=15,
                request_timeout=300,
                min_interval=2.0,
                max_interval=60.0,
            ),
        }

        return configs[self.environment]

    def get_anthropic_config(self) -> RateLimitConfig:
        """Get Anthropic rate limiting config based on environment"""

        configs = {
            Environment.DEVELOPMENT: RateLimitConfig(
                max_tokens=3,
                refill_rate=0.5,
                burst_capacity=1,
                max_cost_per_minute=Decimal("4.00"),
                cost_per_request=Decimal("0.05"),
                failure_threshold=5,
                recovery_timeout=30,
                max_queue_size=20,
                request_timeout=300,
                min_interval=1.0,
                max_interval=15.0,
            ),
            Environment.TESTING: RateLimitConfig(
                max_tokens=1,
                refill_rate=0.2,  # 1 every 5 seconds
                burst_capacity=0,
                max_cost_per_minute=Decimal("0.20"),
                cost_per_request=Decimal("0.05"),
                failure_threshold=2,
                recovery_timeout=15,
                max_queue_size=5,
                request_timeout=120,
                min_interval=3.0,
                max_interval=60.0,
            ),
            Environment.STAGING: RateLimitConfig(
                max_tokens=2,
                refill_rate=0.33,
                burst_capacity=1,
                max_cost_per_minute=Decimal("2.00"),
                cost_per_request=Decimal("0.05"),
                failure_threshold=2,
                recovery_timeout=45,
                max_queue_size=10,
                request_timeout=300,
                min_interval=2.0,
                max_interval=45.0,
            ),
            Environment.PRODUCTION: RateLimitConfig(
                max_tokens=1,
                refill_rate=0.25,  # 1 every 4 seconds
                burst_capacity=0,
                max_cost_per_minute=Decimal("1.25"),
                cost_per_request=Decimal("0.05"),
                failure_threshold=1,  # Very low tolerance
                recovery_timeout=90,  # Long recovery
                max_queue_size=5,  # Small queue
                request_timeout=300,
                min_interval=3.0,
                max_interval=120.0,  # Long backoff
            ),
        }

        return configs[self.environment]

    def get_gemini_config(self) -> RateLimitConfig:
        """Get Gemini rate limiting config based on environment"""

        configs = {
            Environment.DEVELOPMENT: RateLimitConfig(
                max_tokens=20,  # Generous for cheap API
                refill_rate=5.0,
                burst_capacity=10,
                max_cost_per_minute=Decimal("1.00"),
                cost_per_request=Decimal("0.0001"),
                failure_threshold=10,
                recovery_timeout=20,
                max_queue_size=200,
                request_timeout=120,
                min_interval=0.05,
                max_interval=2.0,
            ),
            Environment.TESTING: RateLimitConfig(
                max_tokens=10,
                refill_rate=2.0,
                burst_capacity=5,
                max_cost_per_minute=Decimal("0.20"),
                cost_per_request=Decimal("0.0001"),
                failure_threshold=5,
                recovery_timeout=15,
                max_queue_size=50,
                request_timeout=60,
                min_interval=0.1,
                max_interval=5.0,
            ),
            Environment.STAGING: RateLimitConfig(
                max_tokens=15,
                refill_rate=3.0,
                burst_capacity=7,
                max_cost_per_minute=Decimal("0.75"),
                cost_per_request=Decimal("0.0001"),
                failure_threshold=8,
                recovery_timeout=25,
                max_queue_size=100,
                request_timeout=90,
                min_interval=0.1,
                max_interval=3.0,
            ),
            Environment.PRODUCTION: RateLimitConfig(
                max_tokens=10,
                refill_rate=2.0,
                burst_capacity=5,
                max_cost_per_minute=Decimal("0.50"),
                cost_per_request=Decimal("0.0001"),
                failure_threshold=5,
                recovery_timeout=30,
                max_queue_size=100,
                request_timeout=120,
                min_interval=0.1,
                max_interval=5.0,
            ),
        }

        return configs[self.environment]

    def get_all_configs(self) -> Dict[str, RateLimitConfig]:
        """Get all service configurations for current environment"""
        return {
            "deepgram": self.get_deepgram_config(),
            "openai": self.get_openai_config(),
            "anthropic": self.get_anthropic_config(),
            "gemini": self.get_gemini_config(),
        }


# Global configuration instance
env_config = EnvironmentConfig()


def get_service_config(service_name: str) -> RateLimitConfig:
    """Get rate limiting config for a specific service"""
    config_methods = {
        "deepgram": env_config.get_deepgram_config,
        "openai": env_config.get_openai_config,
        "anthropic": env_config.get_anthropic_config,
        "gemini": env_config.get_gemini_config,
    }

    if service_name in config_methods:
        return config_methods[service_name]()
    else:
        logger.warning(f"No specific config for service {service_name}, using default")
        return RateLimitConfig()  # Default config


def override_environment_config(service_name: str, **overrides) -> RateLimitConfig:
    """Override specific configuration values for a service"""

    base_config = get_service_config(service_name)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
            logger.info(f"Override {service_name}.{key} = {value}")
        else:
            logger.warning(f"Unknown config parameter: {key}")

    return base_config


def validate_environment_setup():
    """Validate that environment is properly configured"""

    current_env = get_current_environment()
    issues = []

    # Check environment variables
    required_vars = {
        Environment.PRODUCTION: ["MONTAGE_ENV", "DATABASE_URL", "DEEPGRAM_API_KEY"],
        Environment.STAGING: ["MONTAGE_ENV", "DATABASE_URL"],
        Environment.TESTING: ["MONTAGE_ENV"],
        Environment.DEVELOPMENT: [],
    }

    for var in required_vars.get(current_env, []):
        if not os.getenv(var):
            issues.append(f"Missing environment variable: {var}")

    # Check budget configurations
    configs = env_config.get_all_configs()
    total_budget = sum(float(config.max_cost_per_minute) for config in configs.values())

    if current_env == Environment.PRODUCTION and total_budget > 5.0:
        issues.append(f"Total budget ${total_budget:.2f}/min too high for production")

    # Log results
    if issues:
        logger.error(f"Environment validation failed for {current_env.value}:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info(f"Environment validation passed for {current_env.value}")
        logger.info(f"Total budget: ${total_budget:.2f}/minute")
        return True


# Special configurations for specific scenarios
class ScenarioConfig:
    """Special configurations for specific use cases"""

    @staticmethod
    def get_batch_processing_config(service_name: str) -> RateLimitConfig:
        """Configuration optimized for batch processing"""
        base_config = get_service_config(service_name)

        # Adjust for batch processing
        base_config.max_tokens = min(base_config.max_tokens, 3)
        base_config.refill_rate *= 0.5  # Slower refill
        base_config.burst_capacity = 0  # No bursts
        base_config.max_queue_size *= 3  # Larger queue
        base_config.request_timeout *= 2  # Longer timeout
        base_config.min_interval *= 2  # Slower minimum

        return base_config

    @staticmethod
    def get_interactive_config(service_name: str) -> RateLimitConfig:
        """Configuration optimized for interactive use"""
        base_config = get_service_config(service_name)

        # Adjust for interactive use
        base_config.max_tokens *= 2  # More tokens
        base_config.refill_rate *= 1.5  # Faster refill
        base_config.burst_capacity += 2  # Allow bursts
        base_config.max_queue_size = min(base_config.max_queue_size, 10)  # Small queue
        base_config.request_timeout = min(
            base_config.request_timeout, 60
        )  # Quick timeout

        return base_config

    @staticmethod
    def get_emergency_config(service_name: str) -> RateLimitConfig:
        """Emergency configuration with minimal limits"""
        return RateLimitConfig(
            max_tokens=1,
            refill_rate=0.1,  # Very slow
            burst_capacity=0,
            max_cost_per_minute=Decimal("0.10"),  # Very low budget
            cost_per_request=Decimal("0.001"),
            failure_threshold=1,
            recovery_timeout=300,  # Long recovery
            max_queue_size=5,
            request_timeout=60,
            min_interval=5.0,  # Very slow
            max_interval=300.0,  # Very long backoff
        )


# Initialize and validate on import
validate_environment_setup()
