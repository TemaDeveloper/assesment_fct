#!/usr/bin/env python3
"""
AIOps Agent - Main Entry Point

An AI-driven autonomous operations agent that monitors system metrics,
detects anomalies using ML (Isolation Forest) and statistical methods (Z-Score),
diagnoses root causes through pattern matching and correlation analysis,
and executes autonomous remediation actions.

Usage:
    python main.py              # Run agent in CLI mode
    python main.py --dashboard  # Run agent with web dashboard
    python main.py --cycles 50  # Run for 50 cycles then stop
    python main.py --fast       # Run with 1-second intervals
"""

import argparse
import sys
import signal

from src.agent.aiops_agent import AIOpsAgent
from src.utils.logger import setup_logger

logger = setup_logger("Main")


def main():
    parser = argparse.ArgumentParser(
        description="AIOps Agent - AI-driven autonomous operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Run agent with default settings
  python main.py --dashboard         Run with web dashboard on port 5050
  python main.py --cycles 100        Run for 100 monitoring cycles
  python main.py --fast              Use 1-second monitoring interval
  python main.py --fast --dashboard  Fast mode with dashboard
        """,
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the web dashboard alongside the agent",
    )
    parser.add_argument(
        "--cycles", type=int, default=None,
        help="Number of monitoring cycles to run (default: unlimited)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use 1-second monitoring interval instead of configured value",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Initialize agent
    agent = AIOpsAgent(config_path=args.config)

    if args.fast:
        agent.interval = 1

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutdown signal received...")
        agent.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_banner()

    if args.dashboard:
        # Run agent in background, dashboard in foreground
        from src.dashboard.app import create_app

        agent.run_in_background()
        app = create_app(agent)
        dashboard_cfg = agent.config["dashboard"]
        logger.info(f"Dashboard: http://localhost:{dashboard_cfg['port']}")
        app.run(
            host=dashboard_cfg["host"],
            port=dashboard_cfg["port"],
            debug=dashboard_cfg["debug"],
        )
    else:
        # Run agent in foreground (CLI mode)
        agent.run(max_cycles=args.cycles)


def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════╗
    ║          AIOps Agent v1.0                        ║
    ║   AI-Driven Autonomous Operations                ║
    ║                                                  ║
    ║   Components:                                    ║
    ║   ├─ Metric Simulator (synthetic data)           ║
    ║   ├─ Anomaly Detector (Isolation Forest+Z-Score) ║
    ║   ├─ Root Cause Analyzer (pattern + correlation) ║
    ║   └─ Remediation Engine (autonomous actions)     ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    main()
