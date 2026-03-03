#!/usr/bin/env python3
"""
AIOps Agent - Main Entry Point

An AI-driven autonomous operations agent that monitors system metrics,
detects anomalies using ML (Isolation Forest) and statistical methods (Z-Score),
diagnoses root causes through pattern matching and correlation analysis,
and executes autonomous remediation actions.

Usage:
    python main.py                    # Run agent in CLI mode
    python main.py --dashboard        # Run agent with web dashboard
    python main.py --cli-dashboard    # Run agent with Rich terminal dashboard
    python main.py --cycles 50        # Run for 50 cycles then stop
    python main.py --fast             # Run with 1-second intervals
"""

import argparse
import sys
import signal
import time

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
  python main.py --cli-dashboard     Run with Rich terminal dashboard
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
        "--cli-dashboard", action="store_true",
        help="Launch a Rich terminal live dashboard",
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

    if args.cli_dashboard:
        run_cli_dashboard(agent, max_cycles=args.cycles)
    elif args.dashboard:
        print_banner()
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
        print_banner()
        # Run agent in foreground (CLI mode)
        agent.run(max_cycles=args.cycles)


def run_cli_dashboard(agent, max_cycles=None):
    """Run agent with a Rich live terminal dashboard."""
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.columns import Columns
    from rich import box

    agent.run_in_background()

    min_samples = agent.config["detector"]["min_training_samples"]

    def build_display():
        state = agent.get_state()
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        # Header
        status_color = "green" if state["running"] else "red"
        status_text = "RUNNING" if state["running"] else "STOPPED"
        header_text = Text()
        header_text.append("  AIOps Agent ", style="bold cyan")
        header_text.append(f"[{status_text}]", style=f"bold {status_color}")
        header_text.append(f"  |  Cycle: {state['cycle_count']}", style="white")
        header_text.append(f"  |  Anomalies: {state['anomaly_count']}", style="yellow")
        header_text.append(f"  |  Remediations: {state['remediation_count']}", style="green")
        health = state.get("health_score", 100)
        h_color = "green" if health >= 75 else ("yellow" if health >= 50 else "red")
        header_text.append(f"  |  Health: {health}", style=f"bold {h_color}")
        ml_label = "Trained" if state["detector_trained"] else f"Learning ({state['history_size']}/{min_samples})"
        ml_color = "green" if state["detector_trained"] else "yellow"
        header_text.append(f"  |  ML: {ml_label}", style=ml_color)
        layout["header"].update(Panel(header_text, style="dim"))

        # Left: Metrics table
        metrics = state["recent_metrics"]
        metric_table = Table(
            title="Current Metrics",
            box=box.SIMPLE_HEAVY,
            title_style="bold cyan",
            header_style="bold",
            show_lines=False,
            expand=True,
        )
        metric_table.add_column("Metric", style="white", min_width=16)
        metric_table.add_column("Value", justify="right", min_width=10)
        metric_table.add_column("Status", justify="center", min_width=8)

        if metrics:
            last = metrics[-1]
            defs = [
                ("CPU Usage", last.get("cpu_usage", 0), "%", 70, 90),
                ("Memory Usage", last.get("memory_usage", 0), "%", 70, 90),
                ("Disk I/O", last.get("disk_io", 0), " MB/s", 50, 80),
                ("Network Latency", last.get("network_latency", 0), " ms", 50, 100),
                ("Error Rate", last.get("error_rate", 0), " /min", 10, 20),
                ("Request Rate", last.get("request_rate", 0), " /s", None, None),
            ]
            for name, val, unit, warn, crit in defs:
                val_str = f"{val:.1f}{unit}"
                if crit is not None and val >= crit:
                    style = "bold red"
                    status = "[red]CRITICAL[/red]"
                elif warn is not None and val >= warn:
                    style = "bold yellow"
                    status = "[yellow]WARNING[/yellow]"
                else:
                    style = "green"
                    status = "[green]OK[/green]"
                metric_table.add_row(name, f"[{style}]{val_str}[/{style}]", status)

        layout["left"].update(Panel(metric_table, border_style="dim"))

        # Right: Event log
        events = [e for e in state["recent_events"] if e["event_type"] != "metric"]
        last_events = events[-12:] if events else []
        last_events.reverse()

        event_text = Text()
        for e in last_events:
            etype = e["event_type"]
            color_map = {
                "anomaly": "red",
                "diagnosis": "yellow",
                "remediation": "green",
                "info": "blue",
            }
            c = color_map.get(etype, "white")
            ts = e["timestamp"].split("T")[1].split(".")[0] if "T" in e["timestamp"] else ""
            event_text.append(f" {ts} ", style="dim")
            event_text.append(f"[{etype.upper():^12s}]", style=f"bold {c}")
            event_text.append(f" {e['summary']}\n", style="white")

        if not last_events:
            event_text.append("  Waiting for events...", style="dim")

        layout["right"].update(
            Panel(event_text, title="Event Log", title_align="left",
                  border_style="dim", subtitle="latest first")
        )

        # Footer: ML training progress
        trained_pct = min(100, int((state["history_size"] / min_samples) * 100)) if min_samples > 0 else 100
        rate = state["remediation_stats"]["autonomous_success_rate"]
        footer_text = Text()
        footer_text.append(f"  ML Training: ", style="dim")
        bar_len = 20
        filled = int(bar_len * trained_pct / 100)
        footer_text.append("[" + "=" * filled + " " * (bar_len - filled) + "]", style="cyan")
        footer_text.append(f" {trained_pct}%", style="bold cyan" if trained_pct >= 100 else "yellow")
        footer_text.append(f"  |  Success Rate: {rate:.0%}", style="green")
        footer_text.append(f"  |  Interval: {agent.interval}s", style="dim")
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    try:
        with Live(build_display(), refresh_per_second=1, screen=True) as live:
            cycle = 0
            while agent.running:
                if max_cycles and cycle >= max_cycles:
                    # Wait until agent finishes the cycles
                    if agent.cycle_count >= max_cycles:
                        break
                time.sleep(1)
                live.update(build_display())
                cycle += 1
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        time.sleep(0.5)
        # Print final summary outside of Live
        state = agent.get_state()
        print("\n" + "=" * 60)
        print(f"  AIOps Agent Session Summary")
        print(f"  Cycles: {state['cycle_count']}  |  Anomalies: {state['anomaly_count']}  |  Remediations: {state['remediation_count']}")
        print(f"  Success Rate: {state['remediation_stats']['autonomous_success_rate']:.0%}  |  ML Trained: {state['detector_trained']}")
        print("=" * 60)


def print_banner():
    banner = r"""
    +==================================================+
    |          AIOps Agent v1.0                         |
    |   AI-Driven Autonomous Operations                 |
    |                                                   |
    |   Components:                                     |
    |   +-- Metric Simulator (synthetic data)           |
    |   +-- Anomaly Detector (Isolation Forest+Z-Score) |
    |   +-- Root Cause Analyzer (pattern + correlation) |
    |   +-- Remediation Engine (autonomous actions)     |
    +==================================================+
    """
    print(banner)


if __name__ == "__main__":
    main()
