"""
AIOps Agent Orchestrator - The central intelligence that ties all components together.

Implements a continuous monitoring loop following the MAPE-K architecture:
- Monitor: Collect metrics from the simulator (or real sources)
- Analyze: Detect anomalies using ML and statistical methods
- Plan: Identify root causes and select remediation strategies
- Execute: Carry out autonomous remediation actions
- Knowledge: Learn from outcomes to improve future decisions

The agent maintains a complete event log and exposes its state for the dashboard.
"""

import time
import threading
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import yaml

from src.simulator.metric_simulator import MetricSimulator, MetricSnapshot
from src.detector.anomaly_detector import AnomalyDetector, AnomalyReport
from src.analyzer.root_cause_analyzer import RootCauseAnalyzer, RootCauseReport
from src.remediation.remediation_engine import RemediationEngine, RemediationAction
from src.utils.logger import setup_logger

logger = setup_logger("Agent")


@dataclass
class AgentEvent:
    """A single event in the agent's lifecycle."""
    timestamp: datetime
    event_type: str  # metric, anomaly, diagnosis, remediation, info
    summary: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "summary": self.summary,
            "details": self.details,
        }


class AIOpsAgent:
    """
    The main AIOps Agent that orchestrates monitoring, detection, analysis,
    and remediation in an autonomous loop.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.name = self.config["agent"]["name"]
        self.interval = self.config["agent"]["monitoring_interval"]

        # Initialize components
        self.simulator = MetricSimulator(self.config)
        self.detector = AnomalyDetector(self.config)
        self.analyzer = RootCauseAnalyzer(self.config)
        self.remediator = RemediationEngine(self.config)

        # Agent state
        self.running = False
        self.cycle_count = 0
        self.events: deque = deque(maxlen=500)
        self.recent_metrics: deque = deque(maxlen=100)
        self.anomaly_count = 0
        self.remediation_count = 0
        self._lock = threading.Lock()

        logger.info(f"AIOps Agent '{self.name}' initialized")

    def _log_event(self, event_type: str, summary: str, details: dict = None):
        """Record an event in the agent's event log."""
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            summary=summary,
            details=details or {},
        )
        with self._lock:
            self.events.append(event)

    def _run_cycle(self):
        """Execute a single monitoring cycle: collect -> detect -> analyze -> remediate."""
        self.cycle_count += 1

        # Step 1: MONITOR - Collect metrics
        snapshot = self.simulator.generate()
        with self._lock:
            self.recent_metrics.append(snapshot.to_dict())

        self._log_event(
            "metric",
            f"Cycle {self.cycle_count}: Collected metrics",
            snapshot.to_dict(),
        )

        # Step 2: ANALYZE - Detect anomalies
        anomaly_report = self.detector.detect(snapshot)

        if not anomaly_report.is_anomaly:
            return  # System healthy, nothing to do

        self.anomaly_count += 1
        self._log_event(
            "anomaly",
            f"Anomaly detected via {anomaly_report.method} "
            f"(confidence: {anomaly_report.confidence:.2f})",
            anomaly_report.to_dict(),
        )

        # Step 3: PLAN - Root cause analysis
        rca_report = self.analyzer.analyze(anomaly_report)
        self._log_event(
            "diagnosis",
            f"Root cause: {rca_report.identified_cause or 'unknown'} "
            f"(confidence: {rca_report.confidence:.2f})",
            rca_report.to_dict(),
        )

        # Step 4: EXECUTE - Remediation
        action = self.remediator.remediate(rca_report)
        if action:
            self.remediation_count += 1
            status = "SUCCESS" if action.success else "FAILED"
            self._log_event(
                "remediation",
                f"Action: {action.action_name} - {status}",
                action.to_dict(),
            )

            # If remediation succeeded, reset simulator anomaly state
            if action.success and action.action_name not in ("alert_only", "escalate_to_human"):
                self.simulator.reset_anomaly_state()

    def run(self, max_cycles: Optional[int] = None):
        """
        Start the agent's monitoring loop.

        Args:
            max_cycles: Optional limit on number of cycles (None = run forever)
        """
        self.running = True
        logger.info(f"Agent '{self.name}' starting monitoring loop (interval={self.interval}s)")
        self._log_event("info", f"Agent started - monitoring interval: {self.interval}s")

        try:
            cycle = 0
            while self.running:
                if max_cycles and cycle >= max_cycles:
                    logger.info(f"Reached max cycles ({max_cycles}), stopping")
                    break

                self._run_cycle()
                cycle += 1
                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
        finally:
            self.running = False
            self._log_event("info", "Agent stopped")
            self._print_summary()

    def run_in_background(self):
        """Start the agent in a background thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        logger.info("Agent running in background thread")
        return thread

    def stop(self):
        """Stop the agent's monitoring loop."""
        self.running = False
        logger.info("Agent stop requested")

    def restart(self):
        """Stop the agent, reset transient state, and start a new background loop."""
        self.stop()
        time.sleep(self.interval + 0.5)  # wait for current cycle to finish
        self.cycle_count = 0
        self.anomaly_count = 0
        self.remediation_count = 0
        with self._lock:
            self.events.clear()
            self.recent_metrics.clear()
        self._log_event("info", "Agent restarted")
        logger.info("Agent state reset, starting new background loop")
        return self.run_in_background()

    def get_health_score(self) -> int:
        """Compute a 0-100 health score from recent metrics.

        Uses weighted distance from danger thresholds:
        - cpu_usage > 90, memory_usage > 90, disk_io > 80,
          network_latency > 100, error_rate > 20, request_rate < 100 (too low = unhealthy)
        """
        with self._lock:
            metrics = list(self.recent_metrics)

        if not metrics:
            return 100  # no data yet, assume healthy

        # Use the last 10 data points for a responsive score
        window = metrics[-10:]
        thresholds = {
            "cpu_usage": (90, True),        # above 90 is bad
            "memory_usage": (90, True),     # above 90 is bad
            "disk_io": (80, True),          # above 80 is bad
            "network_latency": (100, True), # above 100 is bad
            "error_rate": (20, True),       # above 20 is bad
        }
        weights = {
            "cpu_usage": 0.25,
            "memory_usage": 0.25,
            "disk_io": 0.15,
            "network_latency": 0.15,
            "error_rate": 0.20,
        }

        total_score = 0.0
        for metric_name, (threshold, higher_is_bad) in thresholds.items():
            values = [m.get(metric_name, 0) for m in window]
            avg = sum(values) / len(values) if values else 0
            if higher_is_bad:
                ratio = avg / threshold  # 0..1+ where >1 means over threshold
                metric_score = max(0, 1.0 - ratio) * 100
            else:
                metric_score = min(100, (avg / threshold) * 100)
            total_score += metric_score * weights[metric_name]

        return max(0, min(100, int(total_score)))

    def get_state(self) -> dict:
        """Get the current agent state for the dashboard."""
        with self._lock:
            recent_events = [e.to_dict() for e in list(self.events)[-50:]]
            recent_metrics = list(self.recent_metrics)

        return {
            "agent_name": self.name,
            "running": self.running,
            "cycle_count": self.cycle_count,
            "anomaly_count": self.anomaly_count,
            "remediation_count": self.remediation_count,
            "recent_events": recent_events,
            "recent_metrics": recent_metrics[-20:],
            "remediation_stats": self.remediator.get_stats(),
            "detector_trained": bool(self.detector.is_trained),
            "history_size": int(len(self.detector.history)),
            "health_score": self.get_health_score(),
            "min_training_samples": self.config["detector"]["min_training_samples"],
        }

    def _print_summary(self):
        """Print a summary of the agent's activity."""
        stats = self.remediator.get_stats()
        logger.info("=" * 60)
        logger.info(f"Agent Session Summary for '{self.name}'")
        logger.info(f"  Total cycles:           {self.cycle_count}")
        logger.info(f"  Anomalies detected:     {self.anomaly_count}")
        logger.info(f"  Remediations attempted: {self.remediation_count}")
        logger.info(f"  Auto success rate:      {stats['autonomous_success_rate']:.1%}")
        logger.info(f"  ML model trained:       {self.detector.is_trained}")
        logger.info("=" * 60)
