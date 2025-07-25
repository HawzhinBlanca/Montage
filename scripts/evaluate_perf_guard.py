#!/usr/bin/env python3
"""
Phase 7 Performance Guard Evaluator
Validates performance metrics against baseline thresholds
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


class PerformanceGuard:
    """Evaluates performance metrics against baseline thresholds"""

    def __init__(self, baseline_file: str = "perf_base.json"):
        """Initialize with baseline metrics"""
        self.baseline_file = Path(baseline_file)
        self.baseline = self._load_baseline()
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "violations": [],
            "status": "UNKNOWN"
        }

    def _load_baseline(self) -> Dict:
        """Load baseline metrics from file"""
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")

        with open(self.baseline_file) as f:
            return json.load(f)

    def evaluate_latency(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if P95 latency is within threshold (baseline * 1.15)
        """
        baseline_p95 = self.baseline['latency_baseline']['p95_ms']
        threshold = self.baseline['baseline_thresholds']['latency_p95_threshold_ms']
        current_p95 = current_metrics.get('latency_p95_ms', 0)

        if current_p95 == 0:
            return False, "Missing P95 latency data"

        passed = current_p95 <= threshold
        increase_pct = ((current_p95 - baseline_p95) / baseline_p95) * 100

        status = (
            f"P95 latency: {current_p95:.2f}ms vs baseline {baseline_p95:.2f}ms "
            f"(threshold {threshold:.2f}ms, {increase_pct:+.1f}%) - "
            f"{'PASS' if passed else 'FAIL'}"
        )

        self.results['checks']['latency'] = {
            'passed': passed,
            'baseline_ms': baseline_p95,
            'current_ms': current_p95,
            'threshold_ms': threshold,
            'increase_percent': increase_pct
        }

        if not passed:
            self.results['violations'].append(f"Latency violation: {current_p95:.2f}ms > {threshold:.2f}ms")

        return passed, status

    def evaluate_cpu(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if CPU usage is below 80% threshold
        """
        threshold = self.baseline['baseline_thresholds']['cpu_threshold_percent']
        current_cpu = current_metrics.get('cpu_avg_percent', 0)
        peak_cpu = current_metrics.get('cpu_max_percent', 0)

        passed = peak_cpu < threshold

        status = (
            f"CPU usage: avg {current_cpu:.1f}%, peak {peak_cpu:.1f}% "
            f"(threshold {threshold}%) - {'PASS' if passed else 'FAIL'}"
        )

        self.results['checks']['cpu'] = {
            'passed': passed,
            'avg_percent': current_cpu,
            'peak_percent': peak_cpu,
            'threshold_percent': threshold
        }

        if not passed:
            self.results['violations'].append(f"CPU violation: peak {peak_cpu:.1f}% > {threshold}%")

        return passed, status

    def evaluate_memory_growth(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if RSS growth is within 10% threshold
        """
        baseline_rss = self.baseline['resource_baseline']['memory_rss_mb']['avg']
        threshold_pct = self.baseline['baseline_thresholds']['memory_growth_threshold_percent']

        initial_rss = current_metrics.get('memory_initial_mb', baseline_rss)
        final_rss = current_metrics.get('memory_final_mb', initial_rss)

        growth_mb = final_rss - initial_rss
        growth_pct = (growth_mb / initial_rss) * 100 if initial_rss > 0 else 0

        passed = growth_pct <= threshold_pct

        status = (
            f"Memory growth: {growth_mb:.1f}MB ({growth_pct:.1f}%) "
            f"from {initial_rss:.1f}MB to {final_rss:.1f}MB "
            f"(threshold {threshold_pct}%) - {'PASS' if passed else 'FAIL'}"
        )

        self.results['checks']['memory_growth'] = {
            'passed': passed,
            'initial_mb': initial_rss,
            'final_mb': final_rss,
            'growth_mb': growth_mb,
            'growth_percent': growth_pct,
            'threshold_percent': threshold_pct
        }

        if not passed:
            self.results['violations'].append(f"Memory growth violation: {growth_pct:.1f}% > {threshold_pct}%")

        return passed, status

    def evaluate_error_rate(self, current_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if error rate is below 1% threshold
        """
        threshold = self.baseline['baseline_thresholds']['error_rate_threshold_percent']
        error_rate = current_metrics.get('error_rate_percent', 0)

        passed = error_rate < threshold

        status = (
            f"Error rate: {error_rate:.2f}% "
            f"(threshold {threshold}%) - {'PASS' if passed else 'FAIL'}"
        )

        self.results['checks']['error_rate'] = {
            'passed': passed,
            'rate_percent': error_rate,
            'threshold_percent': threshold
        }

        if not passed:
            self.results['violations'].append(f"Error rate violation: {error_rate:.2f}% > {threshold}%")

        return passed, status

    def evaluate_all(self, perf_metrics_file: str) -> Dict:
        """
        Run all performance guard checks
        """
        # Load current metrics
        try:
            with open(perf_metrics_file) as f:
                current_metrics = json.load(f)
        except Exception as e:
            self.results['status'] = 'ERROR'
            self.results['error'] = f"Failed to load metrics: {str(e)}"
            return self.results

        # Run all evaluations
        checks = [
            ('latency', self.evaluate_latency(current_metrics)),
            ('cpu', self.evaluate_cpu(current_metrics)),
            ('memory', self.evaluate_memory_growth(current_metrics)),
            ('errors', self.evaluate_error_rate(current_metrics))
        ]

        all_passed = True
        print("# Phase 7 Performance Guard Evaluation")
        print(f"Baseline: {self.baseline_file}")
        print(f"Current: {perf_metrics_file}")
        print()

        for _check_name, (passed, status) in checks:
            print(f"{'✅' if passed else '❌'} {status}")
            if not passed:
                all_passed = False

        # Overall result
        self.results['status'] = 'PASS' if all_passed else 'FAIL'
        self.results['metrics_file'] = perf_metrics_file
        self.results['baseline_file'] = str(self.baseline_file)

        print()
        print(f"Overall Status: {self.results['status']}")

        if self.results['violations']:
            print("\nViolations:")
            for violation in self.results['violations']:
                print(f"  - {violation}")

        print(f"\nRecommendation: {'PROCEED with deployment' if all_passed else 'INVESTIGATE performance regressions'}")

        return self.results

    def write_report(self, output_file: str = "eval_perf.out"):
        """Write evaluation report to file"""
        with open(output_file, 'w') as f:
            f.write("# Phase 7 Performance Guard Report\n")
            f.write(f"Generated: {self.results['timestamp']}\n")
            f.write(f"Status: {self.results['status']}\n\n")

            f.write("## Check Results\n")
            for check_name, check_data in self.results['checks'].items():
                status = "PASS" if check_data['passed'] else "FAIL"
                f.write(f"- {check_name}: {status}\n")

            if self.results['violations']:
                f.write("\n## Violations\n")
                for violation in self.results['violations']:
                    f.write(f"- {violation}\n")

            f.write("\n## Recommendation\n")
            f.write("PROCEED with deployment\n" if self.results['status'] == 'PASS'
                   else "INVESTIGATE performance regressions\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate performance metrics against baseline")
    parser.add_argument("metrics_file", help="Current performance metrics JSON file")
    parser.add_argument("--baseline", default="perf_base.json", help="Baseline metrics file")
    parser.add_argument("--output", default="eval_perf.out", help="Output report file")

    args = parser.parse_args()

    # Run evaluation
    guard = PerformanceGuard(baseline_file=args.baseline)
    results = guard.evaluate_all(args.metrics_file)
    guard.write_report(args.output)

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'PASS' else 1)


if __name__ == "__main__":
    # If no args provided, run a self-test
    if len(sys.argv) == 1:
        print("Running self-test with simulated metrics...")

        # Create test metrics
        test_metrics = {
            "latency_p95_ms": 82.1,  # Within threshold of 85.84
            "cpu_avg_percent": 42.3,
            "cpu_max_percent": 71.2,  # Below 80%
            "memory_initial_mb": 1050,
            "memory_final_mb": 1145,  # ~9% growth, below 10%
            "error_rate_percent": 0.3  # Below 1%
        }

        with open("test_perf_metrics.json", "w") as f:
            json.dump(test_metrics, f)

        guard = PerformanceGuard()
        results = guard.evaluate_all("test_perf_metrics.json")
        guard.write_report()

        print("\nSelf-test complete. See eval_perf.out for report.")
    else:
        main()
