#!/usr/bin/env python3
"""
CLI tool for managing and monitoring the rate limiting system.

Usage:
  python -m src.cli.rate_limit_cli status
  python -m src.cli.rate_limit_cli monitor
  python -m src.cli.rate_limit_cli emergency --enable
  python -m src.cli.rate_limit_cli export --format json
"""

import argparse
import sys
import time

try:
    from ..core.api_wrappers import get_all_service_status
    from ..core.rate_limit_config import env_config, validate_environment_setup
    from ..core.rate_limit_monitor import (
        RateLimitManager,
        export_rate_limit_metrics,
        get_rate_limit_health,
        global_monitor,
        log_alert_handler,
    )
except ImportError as e:
    print(f"Error importing rate limiting modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def print_status():
    """Print current rate limiting status"""
    print("=" * 60)
    print("MONTAGE RATE LIMITING STATUS")
    print("=" * 60)

    # Environment info
    print(f"Environment: {env_config.environment.value}")
    print()

    # Health summary
    health = get_rate_limit_health()
    status_color = {
        "healthy": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "degraded": "\033[91m",  # Red
        "critical": "\033[95m",  # Magenta
    }
    reset_color = "\033[0m"

    color = status_color.get(health["overall_status"], "")
    print(f"Overall Status: {color}{health['overall_status'].upper()}{reset_color}")
    print(f"Health Score: {health['health_score']:.1f}%")
    print(f"Services: {health['healthy_services']}/{health['total_services']} healthy")

    if health["recent_alerts_count"] > 0:
        print(
            f"Recent Alerts: {health['critical_alerts_count']} critical, "
            f"{health['high_alerts_count']} high, {health['recent_alerts_count']} total"
        )
    print()

    # Service details
    print("SERVICE DETAILS:")
    print("-" * 60)

    service_statuses = get_all_service_status()
    for service_name, status in service_statuses.items():
        circuit_state = status.get("circuit_breaker_state", "UNKNOWN")
        success_rate = status.get("success_rate", 0.0)
        queue_size = status.get("queue_size", 0)
        tokens = status.get("available_tokens", 0)
        failures = status.get("consecutive_failures", 0)
        cost_rate = status.get("current_cost_rate", 0.0)

        # Color coding for circuit state
        state_color = {
            "CLOSED": "\033[92m",  # Green
            "HALF_OPEN": "\033[93m",  # Yellow
            "OPEN": "\033[91m",  # Red
        }.get(circuit_state, "")

        print(
            f"{service_name.upper():<12} | "
            f"Circuit: {state_color}{circuit_state:<9}{reset_color} | "
            f"Success: {success_rate:5.1%} | "
            f"Queue: {queue_size:3d} | "
            f"Tokens: {tokens:5.1f} | "
            f"Failures: {failures:2d} | "
            f"Cost: ${cost_rate:6.3f}/min"
        )

    print()


def monitor_continuously():
    """Monitor rate limiting in real-time"""
    print("Starting continuous monitoring (Press Ctrl+C to stop)...")

    # Start monitoring
    global_monitor.alert_callback = log_alert_handler
    global_monitor.start_monitoring()

    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")

            print_status()

            # Show recent alerts
            recent_alerts = global_monitor.get_recent_alerts(5)  # Last 5 minutes
            if recent_alerts:
                print("RECENT ALERTS:")
                print("-" * 60)
                for alert in recent_alerts[-10:]:  # Last 10 alerts
                    timestamp = alert.timestamp.strftime("%H:%M:%S")
                    severity_colors = {
                        "critical": "\033[95m",
                        "high": "\033[91m",
                        "medium": "\033[93m",
                        "low": "\033[37m",
                    }
                    color = severity_colors.get(alert.severity, "")
                    print(
                        f"{timestamp} | {color}{alert.severity.upper():<8}\033[0m | "
                        f"{alert.service:<12} | {alert.message}"
                    )

            print(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        global_monitor.stop_monitoring()


def export_data(format: str, output_file: str = None):
    """Export rate limiting data"""
    try:
        data = export_rate_limit_metrics()

        if output_file:
            with open(output_file, "w") as f:
                f.write(data)
            print(f"Data exported to {output_file}")
        else:
            print(data)

    except Exception as e:
        print(f"Error exporting data: {e}")
        sys.exit(1)


def manage_emergency_mode(enable: bool):
    """Enable or disable emergency mode"""
    manager = RateLimitManager()

    if enable:
        print("Enabling emergency mode...")
        manager.enable_emergency_mode()
        print(
            "Emergency mode enabled. All rate limits have been reduced to minimum values."
        )
        print("This should only be used during cost emergencies or API failures.")
    else:
        print("Disabling emergency mode...")
        manager.disable_emergency_mode()
        print("Emergency mode disabled. Normal rate limits restored.")


def validate_setup():
    """Validate rate limiting setup"""
    print("Validating rate limiting setup...")

    try:
        is_valid = validate_environment_setup()

        if is_valid:
            print("✅ Setup validation passed")
            return True
        else:
            print("❌ Setup validation failed")
            return False

    except Exception as e:
        print(f"❌ Setup validation error: {e}")
        return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Montage Rate Limiting Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show current rate limiting status"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Start continuous monitoring"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics data")
    export_parser.add_argument(
        "--format", choices=["json"], default="json", help="Export format"
    )
    export_parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    # Emergency command
    emergency_parser = subparsers.add_parser("emergency", help="Manage emergency mode")
    emergency_group = emergency_parser.add_mutually_exclusive_group(required=True)
    emergency_group.add_argument(
        "--enable", action="store_true", help="Enable emergency mode"
    )
    emergency_group.add_argument(
        "--disable", action="store_true", help="Disable emergency mode"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate setup")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute commands
    try:
        if args.command == "status":
            print_status()

        elif args.command == "monitor":
            monitor_continuously()

        elif args.command == "export":
            export_data(args.format, args.output)

        elif args.command == "emergency":
            manage_emergency_mode(args.enable)

        elif args.command == "validate":
            success = validate_setup()
            sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
