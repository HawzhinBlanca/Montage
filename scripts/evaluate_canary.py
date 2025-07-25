#!/usr/bin/env python3
"""
Evaluate canary metrics against SLO matrix for Phase 2 dual-import migration.

SLO Requirements (per Tasks.md):
- p99 latency ≤ +20% from baseline
- 5xx error rate < 1%
- ImportError count = 0
- CPU utilization ≤ 80%
- Memory utilization ≤ 85%
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime


class CanaryEvaluator:
    """Evaluates canary deployment metrics against defined SLOs."""
    
    def __init__(self, metrics_file: str = "canary_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.slo_thresholds = {
            "p99_latency_increase_pct": 20.0,  # Max 20% increase
            "error_rate_5xx_pct": 1.0,         # Max 1% 5xx errors
            "import_error_count": 0,           # Zero ImportErrors allowed
            "cpu_utilization_pct": 80.0,       # Max 80% CPU
            "memory_utilization_pct": 85.0     # Max 85% memory
        }
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load canary metrics from JSON file."""
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def evaluate_latency(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate p99 latency against baseline + 20% threshold."""
        baseline_p99 = metrics.get("baseline_p99_ms", 0)
        current_p99 = metrics.get("current_p99_ms", 0)
        
        if baseline_p99 == 0:
            return False, "Missing baseline p99 latency data"
        
        increase_pct = ((current_p99 - baseline_p99) / baseline_p99) * 100
        threshold = self.slo_thresholds["p99_latency_increase_pct"]
        
        passed = increase_pct <= threshold
        status = f"p99 latency: {current_p99}ms vs baseline {baseline_p99}ms ({increase_pct:+.1f}%) - {'PASS' if passed else 'FAIL'}"
        
        return passed, status
    
    def evaluate_error_rate(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate 5xx error rate against 1% threshold."""
        total_requests = metrics.get("total_requests", 0)
        error_5xx_count = metrics.get("error_5xx_count", 0)
        
        if total_requests == 0:
            return False, "No request data available"
        
        error_rate_pct = (error_5xx_count / total_requests) * 100
        threshold = self.slo_thresholds["error_rate_5xx_pct"]
        
        passed = error_rate_pct < threshold
        status = f"5xx error rate: {error_5xx_count}/{total_requests} ({error_rate_pct:.2f}%) - {'PASS' if passed else 'FAIL'}"
        
        return passed, status
    
    def evaluate_import_errors(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate ImportError count (must be 0)."""
        import_errors = metrics.get("import_error_count", 0)
        threshold = self.slo_thresholds["import_error_count"]
        
        passed = import_errors <= threshold
        status = f"ImportErrors: {import_errors} - {'PASS' if passed else 'FAIL'}"
        
        return passed, status
    
    def evaluate_cpu_utilization(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate CPU utilization against 80% threshold."""
        cpu_pct = metrics.get("avg_cpu_utilization_pct", 0)
        threshold = self.slo_thresholds["cpu_utilization_pct"]
        
        passed = cpu_pct <= threshold
        status = f"CPU utilization: {cpu_pct:.1f}% - {'PASS' if passed else 'FAIL'}"
        
        return passed, status
    
    def evaluate_memory_utilization(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate memory utilization against 85% threshold."""
        memory_pct = metrics.get("avg_memory_utilization_pct", 0)
        threshold = self.slo_thresholds["memory_utilization_pct"]
        
        passed = memory_pct <= threshold
        status = f"Memory utilization: {memory_pct:.1f}% - {'PASS' if passed else 'FAIL'}"
        
        return passed, status
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Run all SLO evaluations and return comprehensive results."""
        try:
            metrics = self.load_metrics()
        except Exception as e:
            return {
                "overall_status": "ERROR",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "recommendation": "BLOCK - Unable to load metrics"
            }
        
        evaluations = [
            ("latency", self.evaluate_latency(metrics)),
            ("error_rate", self.evaluate_error_rate(metrics)),
            ("import_errors", self.evaluate_import_errors(metrics)),
            ("cpu", self.evaluate_cpu_utilization(metrics)),
            ("memory", self.evaluate_memory_utilization(metrics))
        ]
        
        results = {}
        all_passed = True
        failed_checks = []
        
        for check_name, (passed, status) in evaluations:
            results[check_name] = {"passed": passed, "status": status}
            if not passed:
                all_passed = False
                failed_checks.append(check_name)
        
        # Determine overall status
        if all_passed:
            overall_status = "PASS"
            recommendation = "PROCEED with Phase 2 completion"
        elif len(failed_checks) <= 1 and "import_errors" not in failed_checks:
            overall_status = "PARTIAL"
            recommendation = f"CAUTION - {len(failed_checks)} SLO violation(s): {', '.join(failed_checks)}"
        else:
            overall_status = "FAIL"
            recommendation = f"BLOCK - {len(failed_checks)} SLO violation(s): {', '.join(failed_checks)}"
        
        return {
            "overall_status": overall_status,
            "recommendation": recommendation,
            "timestamp": datetime.utcnow().isoformat(),
            "canary_duration_hours": metrics.get("duration_hours", 0),
            "slo_evaluations": results,
            "metrics_summary": {
                "total_requests": metrics.get("total_requests", 0),
                "p99_latency_ms": metrics.get("current_p99_ms", 0),
                "error_rate_pct": (metrics.get("error_5xx_count", 0) / max(metrics.get("total_requests", 1), 1)) * 100,
                "import_errors": metrics.get("import_error_count", 0),
                "cpu_avg_pct": metrics.get("avg_cpu_utilization_pct", 0),
                "memory_avg_pct": metrics.get("avg_memory_utilization_pct", 0)
            }
        }
    
    def write_evaluation_report(self, output_file: str = "evaluate_canary.out"):
        """Generate and write the canary evaluation report."""
        results = self.evaluate_all()
        
        report_lines = [
            "# Phase 2 Dual-Import Canary Evaluation Report",
            f"Generated: {results['timestamp']}",
            f"Duration: {results.get('canary_duration_hours', 0)} hours",
            "",
            f"Overall Status: {results['overall_status']}",
            f"Recommendation: {results['recommendation']}",
            "",
            "## SLO Evaluation Results"
        ]
        
        if "slo_evaluations" in results:
            for check_name, check_result in results["slo_evaluations"].items():
                status_icon = "✅" if check_result["passed"] else "❌"
                report_lines.append(f"{status_icon} {check_result['status']}")
        
        report_lines.extend([
            "",
            "## Metrics Summary",
            f"Total Requests: {results.get('metrics_summary', {}).get('total_requests', 0):,}",
            f"p99 Latency: {results.get('metrics_summary', {}).get('p99_latency_ms', 0):.1f}ms",
            f"5xx Error Rate: {results.get('metrics_summary', {}).get('error_rate_pct', 0):.2f}%",
            f"ImportErrors: {results.get('metrics_summary', {}).get('import_errors', 0)}",
            f"CPU Utilization: {results.get('metrics_summary', {}).get('cpu_avg_pct', 0):.1f}%",
            f"Memory Utilization: {results.get('metrics_summary', {}).get('memory_avg_pct', 0):.1f}%"
        ])
        
        if results["overall_status"] == "ERROR":
            report_lines.extend([
                "",
                "## Error Details",
                f"Error: {results.get('error', 'Unknown error')}"
            ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return results


def main():
    """Main entry point for canary evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate canary deployment metrics")
    parser.add_argument("--metrics", default="canary_metrics.json", 
                       help="Path to canary metrics JSON file")
    parser.add_argument("--output", default="evaluate_canary.out",
                       help="Path to output evaluation report")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON instead of text report")
    
    args = parser.parse_args()
    
    evaluator = CanaryEvaluator(args.metrics)
    
    try:
        if args.json:
            results = evaluator.evaluate_all()
            print(json.dumps(results, indent=2))
        else:
            results = evaluator.write_evaluation_report(args.output)
            print(f"Evaluation complete: {results['overall_status']}")
            print(f"Report written to: {args.output}")
            
            # Exit with appropriate code
            if results["overall_status"] == "PASS":
                return 0
            elif results["overall_status"] == "PARTIAL":
                return 1
            else:
                return 2
                
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())