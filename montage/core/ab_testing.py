"""
A/B Testing Harness - Tasks.md Step 0
10% experimental traffic routing with Redis storage
"""
import hashlib
from dataclasses import dataclass
from typing import Dict

import redis


@dataclass
class ExperimentConfig:
    """Configuration for A/B experiment"""
    name: str
    traffic_pct: float  # Percentage of traffic to experiment (10.0 = 10%)
    enabled: bool = True

class ABTestingHarness:
    """A/B testing harness with Redis backend for variant assignment"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.experiments: Dict[str, ExperimentConfig] = {}

    def add_experiment(self, name: str, traffic_pct: float = 10.0) -> None:
        """Add experiment with specified traffic percentage"""
        self.experiments[name] = ExperimentConfig(name=name, traffic_pct=traffic_pct)
        # Store in Redis for persistence
        self.redis.hset(f"experiment:{name}", mapping={
            "name": name,
            "traffic_pct": traffic_pct,
            "enabled": "true"
        })

    def assign_variant(self, job_id: str, experiment_name: str) -> str:
        """
        Assign variant for job_id in experiment
        Returns 'control' or 'exp-{experiment_name}'
        Uses consistent hashing for stable assignment
        """
        if experiment_name not in self.experiments:
            return "control"

        config = self.experiments[experiment_name]
        if not config.enabled:
            return "control"

        # Check if already assigned
        existing = self.redis.get(f"variant:{job_id}:{experiment_name}")
        if existing:
            return existing.decode()

        # Hash job_id for consistent assignment
        hash_value = int(hashlib.md5(f"{job_id}:{experiment_name}".encode()).hexdigest(), 16)
        hash_pct = (hash_value % 100) + 1  # 1-100

        # Assign to experiment if within traffic percentage
        if hash_pct <= config.traffic_pct:
            variant = f"exp-{experiment_name}"
        else:
            variant = "control"

        # Store assignment in Redis with 24h expiry
        self.redis.setex(f"variant:{job_id}:{experiment_name}", 86400, variant)

        # Record assignment for analytics
        self.redis.hincrby("assignment_counts", variant, 1)

        return variant

    def get_experiment_stats(self, experiment_name: str) -> Dict[str, int]:
        """Get assignment statistics for experiment"""
        control_key = "control"
        experiment_key = f"exp-{experiment_name}"

        control_count = int(self.redis.hget("assignment_counts", control_key) or 0)
        exp_count = int(self.redis.hget("assignment_counts", experiment_key) or 0)

        return {
            "control": control_count,
            "experiment": exp_count,
            "total": control_count + exp_count
        }

    def disable_experiment(self, experiment_name: str) -> None:
        """Disable experiment - all traffic goes to control"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].enabled = False
            self.redis.hset(f"experiment:{experiment_name}", "enabled", "false")

    def is_experiment_active(self, experiment_name: str) -> bool:
        """Check if experiment is currently active - Tasks.md Step 1D"""
        if experiment_name not in self.experiments:
            # Try to load from Redis
            experiment_data = self.redis.hgetall(f"experiment:{experiment_name}")
            if experiment_data:
                self.experiments[experiment_name] = ExperimentConfig(
                    name=experiment_name,
                    traffic_pct=float(experiment_data.get("traffic_pct", 0)),
                    enabled=experiment_data.get("enabled", "false") == "true"
                )
            else:
                return False

        config = self.experiments[experiment_name]
        return config.enabled and config.traffic_pct > 0

    def update_experiment_traffic(self, experiment_name: str, traffic_pct: float) -> None:
        """Update experiment traffic percentage - Tasks.md Step 1D"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].traffic_pct = traffic_pct

            # Update in Redis
            self.redis.hset(f"experiment:{experiment_name}", "traffic_pct", traffic_pct)

            # If traffic is 0, effectively disable the experiment
            if traffic_pct <= 0:
                self.experiments[experiment_name].enabled = False
                self.redis.hset(f"experiment:{experiment_name}", "enabled", "false")
        else:
            # Create new experiment with this traffic percentage
            self.add_experiment(experiment_name, traffic_pct)

    def enable_experiment(self, experiment_name: str) -> None:
        """Enable experiment - Tasks.md Step 1D"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].enabled = True
            self.redis.hset(f"experiment:{experiment_name}", "enabled", "true")

    def get_all_experiments(self) -> Dict[str, ExperimentConfig]:
        """Get all experiments from Redis - Tasks.md Step 1D"""
        # Load experiments from Redis
        experiment_keys = self.redis.keys("experiment:*")

        for key in experiment_keys:
            experiment_name = key.split(":", 1)[1] if isinstance(key, str) else key.decode().split(":", 1)[1]

            if experiment_name not in self.experiments:
                experiment_data = self.redis.hgetall(key)
                if experiment_data:
                    # Handle both str and bytes keys/values from Redis
                    if isinstance(list(experiment_data.keys())[0], bytes):
                        experiment_data = {k.decode(): v.decode() for k, v in experiment_data.items()}

                    self.experiments[experiment_name] = ExperimentConfig(
                        name=experiment_name,
                        traffic_pct=float(experiment_data.get("traffic_pct", 0)),
                        enabled=experiment_data.get("enabled", "false") == "true"
                    )

        return self.experiments
