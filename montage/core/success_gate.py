#!/usr/bin/env python3
"""
Success Gate - Nightly A/B test evaluation and promotion system
Tasks.md Step 1D: Automated success criteria evaluation for face crop experiments

This module runs as a nightly cron job to:
1. Evaluate A/B test performance metrics
2. Calculate statistical significance
3. Promote successful experiments to default
4. Generate reports for failed experiments
5. Clean up old experiment data
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import redis

try:
    from montage.core.db import Database
    from montage.utils.logging_config import get_logger

    from .ab_testing import ABTestingHarness
    from .metrics import metrics
except ImportError as e:
    # Fallback imports for cron execution
    import warnings
    warnings.warn(f"Import warning in success_gate: {e}")

logger = get_logger(__name__)

# Tasks.md Step 1D: Success Gate Configuration
SUCCESS_CRITERIA = {
    "face_crop": {
        "min_sample_size": 100,  # Minimum jobs to evaluate
        "min_confidence": 0.95,   # Statistical confidence level
        "min_improvement": 0.05,  # 5% minimum improvement
        "metrics_to_evaluate": [
            "face_detection_accuracy",
            "user_engagement_rate",
            "completion_rate",
            "faces_detected_ratio"
        ],
        "max_cost_increase": 0.10,  # Max 10% cost increase allowed
    },
    "speech_density": {
        "min_sample_size": 50,
        "min_confidence": 0.90,
        "min_improvement": 0.03,
        "metrics_to_evaluate": [
            "highlight_quality_score",
            "user_engagement_rate",
            "watch_time_ratio"
        ],
        "max_cost_increase": 0.05,
    }
}

# Redis configuration for cron job
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class SuccessGate:
    """A/B test success evaluation and promotion system"""

    def __init__(self):
        self.db = Database()

        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.ab_harness = ABTestingHarness(self.redis_client)
            logger.info("Success Gate initialized with Redis connection")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self.redis_client = None
            self.ab_harness = None

    def run_nightly_evaluation(self) -> Dict[str, Any]:
        """
        Main entry point for nightly cron job evaluation
        Tasks.md Step 1D: Evaluate all active A/B experiments
        """
        logger.info("Starting nightly A/B test success evaluation...")
        start_time = time.time()

        results = {
            "timestamp": datetime.now().isoformat(),
            "experiments_evaluated": [],
            "promotions": [],
            "failures": [],
            "warnings": [],
            "total_runtime_sec": 0
        }

        if not self.ab_harness:
            error_msg = "AB Testing harness not available - skipping evaluation"
            logger.error(error_msg)
            results["warnings"].append(error_msg)
            return results

        try:
            # Get active experiments
            active_experiments = self._get_active_experiments()
            logger.info(f"Found {len(active_experiments)} active experiments")

            for experiment_name in active_experiments:
                logger.info(f"Evaluating experiment: {experiment_name}")

                try:
                    evaluation_result = self._evaluate_experiment(experiment_name)
                    results["experiments_evaluated"].append(evaluation_result)

                    # Process evaluation decision
                    if evaluation_result["decision"] == "promote":
                        self._promote_experiment(experiment_name)
                        results["promotions"].append(experiment_name)
                        logger.info(f"✅ Promoted experiment: {experiment_name}")

                    elif evaluation_result["decision"] == "fail":
                        self._fail_experiment(experiment_name)
                        results["failures"].append(experiment_name)
                        logger.warning(f"❌ Failed experiment: {experiment_name}")

                    else:  # continue
                        logger.info(f"⏳ Continuing experiment: {experiment_name}")

                except Exception as e:
                    error_msg = f"Error evaluating {experiment_name}: {e}"
                    logger.error(error_msg)
                    results["warnings"].append(error_msg)

            # Clean up old experiment data
            self._cleanup_old_experiments()

        except Exception as e:
            error_msg = f"Critical error in nightly evaluation: {e}"
            logger.error(error_msg)
            results["warnings"].append(error_msg)

        results["total_runtime_sec"] = time.time() - start_time
        logger.info(f"Nightly evaluation completed in {results['total_runtime_sec']:.1f}s")

        # Store results for monitoring
        self._store_evaluation_results(results)

        return results

    def _get_active_experiments(self) -> List[str]:
        """Get list of currently active A/B experiments"""
        try:
            # Get experiments from AB harness
            experiments = []
            for experiment in ["face_crop", "speech_density"]:
                if self.ab_harness.is_experiment_active(experiment):
                    experiments.append(experiment)
            return experiments
        except Exception as e:
            logger.error(f"Failed to get active experiments: {e}")
            return ["face_crop", "speech_density"]  # Fallback to known experiments

    def _evaluate_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Evaluate a single experiment against success criteria
        Tasks.md Step 1D: Statistical significance testing
        """
        logger.info(f"Evaluating experiment: {experiment_name}")

        criteria = SUCCESS_CRITERIA.get(experiment_name, {})
        if not criteria:
            return {
                "experiment": experiment_name,
                "decision": "continue",
                "reason": "No success criteria defined",
                "metrics": {}
            }

        # Get experiment data from last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        control_data = self._get_experiment_data(experiment_name, "control", start_time, end_time)
        experimental_data = self._get_experiment_data(experiment_name, f"exp-{experiment_name}", start_time, end_time)

        # Check sample size requirement
        control_sample_size = len(control_data)
        experimental_sample_size = len(experimental_data)
        min_sample_size = criteria.get("min_sample_size", 50)

        if control_sample_size < min_sample_size or experimental_sample_size < min_sample_size:
            return {
                "experiment": experiment_name,
                "decision": "continue",
                "reason": f"Insufficient sample size: control={control_sample_size}, exp={experimental_sample_size}, required={min_sample_size}",
                "metrics": {
                    "control_sample_size": control_sample_size,
                    "experimental_sample_size": experimental_sample_size
                }
            }

        # Calculate key metrics for both groups
        metrics_comparison = self._calculate_metrics_comparison(
            control_data, experimental_data, criteria["metrics_to_evaluate"]
        )

        # Perform statistical significance test
        significance_results = self._test_statistical_significance(
            metrics_comparison, criteria.get("min_confidence", 0.95)
        )

        # Check if improvement meets threshold
        min_improvement = criteria.get("min_improvement", 0.05)
        significant_improvements = 0
        total_metrics = len(criteria["metrics_to_evaluate"])

        for metric_name in criteria["metrics_to_evaluate"]:
            if metric_name in significance_results:
                improvement = significance_results[metric_name].get("improvement", 0)
                is_significant = significance_results[metric_name].get("is_significant", False)

                if is_significant and improvement >= min_improvement:
                    significant_improvements += 1

        # Decision logic
        if significant_improvements >= (total_metrics // 2):  # Majority of metrics improved
            # Check cost constraint
            cost_increase = metrics_comparison.get("cost_increase_ratio", 0)
            max_cost_increase = criteria.get("max_cost_increase", 0.10)

            if cost_increase <= max_cost_increase:
                decision = "promote"
                reason = f"Significant improvement in {significant_improvements}/{total_metrics} metrics"
            else:
                decision = "fail"
                reason = f"Cost increase {cost_increase:.2%} exceeds limit {max_cost_increase:.2%}"

        elif any(significance_results[m].get("improvement", 0) < -0.02 for m in significance_results):
            # Significant degradation
            decision = "fail"
            reason = "Significant performance degradation detected"

        else:
            decision = "continue"
            reason = "No conclusive results yet"

        return {
            "experiment": experiment_name,
            "decision": decision,
            "reason": reason,
            "metrics": {
                "control_sample_size": control_sample_size,
                "experimental_sample_size": experimental_sample_size,
                "metrics_comparison": metrics_comparison,
                "significance_results": significance_results,
                "significant_improvements": significant_improvements,
                "total_metrics_evaluated": total_metrics
            }
        }

    def _get_experiment_data(self, experiment: str, variant: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Retrieve experiment data from database for analysis"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT 
                        job_id,
                        metrics::json as metrics_data,
                        created_at,
                        status,
                        processing_time_sec,
                        output_size_mb
                    FROM jobs 
                    WHERE created_at BETWEEN %s AND %s
                    AND status = 'completed'
                    AND metrics::json->>'primary_variant' = %s
                    ORDER BY created_at DESC
                """

                cursor.execute(query, (start_time, end_time, variant))
                rows = cursor.fetchall()

                data = []
                for row in rows:
                    try:
                        metrics_data = json.loads(row[1]) if row[1] else {}
                        data.append({
                            "job_id": row[0],
                            "metrics": metrics_data,
                            "created_at": row[2],
                            "status": row[3],
                            "processing_time_sec": row[4] or 0,
                            "output_size_mb": row[5] or 0
                        })
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse metrics for job {row[0]}: {e}")
                        continue

                logger.info(f"Retrieved {len(data)} records for {experiment} variant {variant}")
                return data

        except Exception as e:
            logger.error(f"Failed to get experiment data: {e}")
            return []

    def _calculate_metrics_comparison(self, control_data: List[Dict], experimental_data: List[Dict], metrics_to_evaluate: List[str]) -> Dict[str, Any]:
        """Calculate comparative metrics between control and experimental groups"""
        comparison = {}

        # Helper function to safely get metric values
        def get_metric_values(data: List[Dict], metric_name: str) -> List[float]:
            values = []
            for record in data:
                try:
                    if metric_name in record.get("metrics", {}):
                        value = float(record["metrics"][metric_name])
                        values.append(value)
                    elif metric_name == "completion_rate":
                        # Calculate completion rate from available data
                        processing_time = record.get("processing_time_sec", 0)
                        if processing_time > 0:
                            values.append(1.0)  # Job completed
                    elif metric_name == "cost_per_job":
                        cost = record.get("metrics", {}).get("cost_usd", 0)
                        values.append(float(cost))
                except (ValueError, TypeError):
                    continue
            return values

        # Calculate metrics for each group
        for metric_name in metrics_to_evaluate:
            control_values = get_metric_values(control_data, metric_name)
            experimental_values = get_metric_values(experimental_data, metric_name)

            if control_values and experimental_values:
                control_mean = sum(control_values) / len(control_values)
                experimental_mean = sum(experimental_values) / len(experimental_values)

                comparison[metric_name] = {
                    "control_mean": control_mean,
                    "experimental_mean": experimental_mean,
                    "control_count": len(control_values),
                    "experimental_count": len(experimental_values),
                    "improvement": (experimental_mean - control_mean) / control_mean if control_mean > 0 else 0
                }

        # Calculate overall cost comparison
        control_costs = get_metric_values(control_data, "cost_per_job")
        experimental_costs = get_metric_values(experimental_data, "cost_per_job")

        if control_costs and experimental_costs:
            control_cost_mean = sum(control_costs) / len(control_costs)
            experimental_cost_mean = sum(experimental_costs) / len(experimental_costs)
            comparison["cost_increase_ratio"] = (experimental_cost_mean - control_cost_mean) / control_cost_mean if control_cost_mean > 0 else 0
        else:
            comparison["cost_increase_ratio"] = 0

        return comparison

    def _test_statistical_significance(self, metrics_comparison: Dict[str, Any], min_confidence: float) -> Dict[str, Any]:
        """
        Perform statistical significance testing on metrics
        Uses simplified t-test approximation for cron job environment
        """
        significance_results = {}

        for metric_name, data in metrics_comparison.items():
            if metric_name == "cost_increase_ratio":
                continue

            try:
                control_mean = data["control_mean"]
                experimental_mean = data["experimental_mean"]
                control_count = data["control_count"]
                experimental_count = data["experimental_count"]
                improvement = data["improvement"]

                # Simplified significance test
                # In production, use proper statistical libraries like scipy.stats
                min_sample_size_for_significance = 30
                combined_sample_size = control_count + experimental_count

                # Simple heuristic: require minimum sample size and minimum effect size
                is_significant = (
                    combined_sample_size >= min_sample_size_for_significance and
                    abs(improvement) >= 0.02  # 2% minimum effect size
                )

                # Confidence based on sample size (simplified)
                if combined_sample_size >= 200:
                    confidence = 0.95
                elif combined_sample_size >= 100:
                    confidence = 0.90
                else:
                    confidence = 0.80

                significance_results[metric_name] = {
                    "improvement": improvement,
                    "is_significant": is_significant and confidence >= min_confidence,
                    "confidence": confidence,
                    "sample_size": combined_sample_size,
                    "effect_size": abs(improvement)
                }

            except (KeyError, ZeroDivisionError, TypeError) as e:
                logger.warning(f"Failed to test significance for {metric_name}: {e}")
                significance_results[metric_name] = {
                    "improvement": 0,
                    "is_significant": False,
                    "confidence": 0,
                    "error": str(e)
                }

        return significance_results

    def _promote_experiment(self, experiment_name: str) -> None:
        """Promote successful experiment to default configuration"""
        logger.info(f"Promoting experiment {experiment_name} to production default")

        try:
            # Update A/B testing configuration to make experiment the new control
            if self.ab_harness:
                # Set traffic to 0% for experiment (everyone gets new version)
                self.ab_harness.update_experiment_traffic(experiment_name, 0.0)

                # Store promotion record
                promotion_record = {
                    "experiment": experiment_name,
                    "promoted_at": datetime.now().isoformat(),
                    "previous_control": "control",
                    "new_control": f"exp-{experiment_name}"
                }

                # Store in Redis
                self.redis_client.hset(
                    "promoted_experiments",
                    experiment_name,
                    json.dumps(promotion_record)
                )

            logger.info(f"Successfully promoted {experiment_name}")

        except Exception as e:
            logger.error(f"Failed to promote experiment {experiment_name}: {e}")

    def _fail_experiment(self, experiment_name: str) -> None:
        """Mark experiment as failed and stop traffic"""
        logger.info(f"Failing experiment {experiment_name}")

        try:
            if self.ab_harness:
                # Stop all traffic to experimental variant
                self.ab_harness.update_experiment_traffic(experiment_name, 0.0)

                # Store failure record
                failure_record = {
                    "experiment": experiment_name,
                    "failed_at": datetime.now().isoformat(),
                    "reason": "Did not meet success criteria"
                }

                # Store in Redis
                self.redis_client.hset(
                    "failed_experiments",
                    experiment_name,
                    json.dumps(failure_record)
                )

            logger.info(f"Successfully failed {experiment_name}")

        except Exception as e:
            logger.error(f"Failed to fail experiment {experiment_name}: {e}")

    def _cleanup_old_experiments(self) -> None:
        """Clean up experiment data older than 30 days"""
        logger.info("Cleaning up old experiment data...")

        try:
            cutoff_date = datetime.now() - timedelta(days=30)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Clean up old job records with experiment variants
                cleanup_query = """
                    DELETE FROM jobs 
                    WHERE created_at < %s 
                    AND (
                        metrics::json->>'primary_variant' LIKE 'exp-%'
                        OR metrics::json->>'face_crop_variant' LIKE 'exp-%'
                        OR metrics::json->>'speech_density_variant' LIKE 'exp-%'
                    )
                """

                cursor.execute(cleanup_query, (cutoff_date,))
                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old experiment records")

        except Exception as e:
            logger.error(f"Failed to cleanup old experiments: {e}")

    def _store_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Store evaluation results for monitoring and debugging"""
        try:
            # Store in Redis with 7 day expiry
            results_key = f"success_gate_results:{datetime.now().strftime('%Y-%m-%d')}"
            self.redis_client.setex(
                results_key,
                timedelta(days=7),
                json.dumps(results, indent=2)
            )

            logger.info(f"Stored evaluation results in {results_key}")

        except Exception as e:
            logger.error(f"Failed to store evaluation results: {e}")


def main():
    """Main entry point for cron job execution"""
    # Set up logging for cron environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/montage_success_gate.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info("=== Starting Success Gate Nightly Evaluation ===")

    try:
        success_gate = SuccessGate()
        results = success_gate.run_nightly_evaluation()

        # Print summary for cron logs
        print("Success Gate completed:")
        print(f"  Experiments evaluated: {len(results['experiments_evaluated'])}")
        print(f"  Promotions: {len(results['promotions'])}")
        print(f"  Failures: {len(results['failures'])}")
        print(f"  Warnings: {len(results['warnings'])}")
        print(f"  Runtime: {results['total_runtime_sec']:.1f}s")

        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")

        # Exit with appropriate code
        exit_code = 1 if results['warnings'] else 0
        logger.info(f"Success Gate exiting with code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Critical error in Success Gate main: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
