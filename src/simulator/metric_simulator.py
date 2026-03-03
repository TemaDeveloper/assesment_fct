"""
Metric Simulator - Generates synthetic system metrics with realistic patterns.

Simulates a cloud environment producing CPU, memory, disk I/O, network latency,
error rates, and request throughput. Anomalies are injected probabilistically
to mimic real-world incidents like CPU spikes, memory leaks, and cascading failures.
"""

import random
import time
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("Simulator")


@dataclass
class MetricSnapshot:
    """A single point-in-time capture of all system metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    error_rate: float
    request_rate: float
    anomaly_type: Optional[str] = None
    is_anomaly: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": float(round(self.cpu_usage, 2)),
            "memory_usage": float(round(self.memory_usage, 2)),
            "disk_io": float(round(self.disk_io, 2)),
            "network_latency": float(round(self.network_latency, 2)),
            "error_rate": float(round(self.error_rate, 2)),
            "request_rate": float(round(self.request_rate, 2)),
            "anomaly_type": self.anomaly_type,
            "is_anomaly": bool(self.is_anomaly),
        }

    def metric_values(self) -> dict:
        """Return only the numeric metric values."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_io": self.disk_io,
            "network_latency": self.network_latency,
            "error_rate": self.error_rate,
            "request_rate": self.request_rate,
        }


class MetricSimulator:
    """
    Generates realistic system metrics with optional anomaly injection.

    Uses Gaussian noise around configured means to simulate normal operations.
    A sinusoidal component adds time-of-day load variation (diurnal pattern).
    Anomalies are injected based on configured probability, simulating real
    operational incidents.
    """

    def __init__(self, config: dict):
        self.config = config
        self.metrics_config = config["simulator"]["metrics"]
        self.anomaly_config = config["simulator"]["anomaly"]
        self.anomaly_probability = self.anomaly_config["probability"]
        self.tick = 0
        self._memory_leak_active = False
        self._memory_leak_accumulator = 0.0
        self._cascading_failure_active = False
        self._cascading_ticks = 0

    def _diurnal_factor(self) -> float:
        """Simulate time-of-day load variation using a sine wave."""
        return 1.0 + 0.2 * math.sin(self.tick * 0.05)

    def _generate_normal_metric(self, metric_name: str) -> float:
        """Generate a single metric value under normal conditions."""
        cfg = self.metrics_config[metric_name]
        base = cfg["normal_mean"] * self._diurnal_factor()
        noise = np.random.normal(0, cfg["normal_std"])
        value = base + noise
        # Clamp to reasonable bounds
        if metric_name in ("cpu_usage", "memory_usage"):
            value = max(0, min(100, value))
        else:
            value = max(0, value)
        return value

    def _inject_anomaly(self, snapshot: MetricSnapshot) -> MetricSnapshot:
        """Inject an anomaly into the metric snapshot."""
        anomaly_type = random.choice(self.anomaly_config["types"])
        snapshot.is_anomaly = True
        snapshot.anomaly_type = anomaly_type

        if anomaly_type == "cpu_spike":
            snapshot.cpu_usage = min(100, snapshot.cpu_usage + random.uniform(30, 50))
            snapshot.request_rate *= 0.7  # degraded throughput

        elif anomaly_type == "memory_leak":
            self._memory_leak_active = True
            self._memory_leak_accumulator += random.uniform(3, 8)
            snapshot.memory_usage = min(100, snapshot.memory_usage + self._memory_leak_accumulator)

        elif anomaly_type == "disk_saturation":
            snapshot.disk_io = snapshot.disk_io + random.uniform(50, 100)
            snapshot.network_latency *= 1.5  # disk contention affects I/O

        elif anomaly_type == "network_degradation":
            snapshot.network_latency = snapshot.network_latency + random.uniform(80, 200)
            snapshot.error_rate += random.uniform(5, 15)  # timeouts cause errors

        elif anomaly_type == "error_burst":
            snapshot.error_rate = snapshot.error_rate + random.uniform(20, 50)
            snapshot.request_rate *= 0.5  # error storm reduces throughput

        elif anomaly_type == "cascading_failure":
            self._cascading_failure_active = True
            snapshot.cpu_usage = min(100, snapshot.cpu_usage + random.uniform(20, 40))
            snapshot.memory_usage = min(100, snapshot.memory_usage + random.uniform(15, 30))
            snapshot.error_rate += random.uniform(15, 30)
            snapshot.network_latency += random.uniform(40, 100)
            snapshot.request_rate *= 0.3

        return snapshot

    def generate(self) -> MetricSnapshot:
        """Generate the next metric snapshot, possibly with an anomaly."""
        self.tick += 1

        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=self._generate_normal_metric("cpu_usage"),
            memory_usage=self._generate_normal_metric("memory_usage"),
            disk_io=self._generate_normal_metric("disk_io"),
            network_latency=self._generate_normal_metric("network_latency"),
            error_rate=self._generate_normal_metric("error_rate"),
            request_rate=self._generate_normal_metric("request_rate"),
        )

        # Handle ongoing memory leak
        if self._memory_leak_active:
            snapshot.memory_usage = min(100, snapshot.memory_usage + self._memory_leak_accumulator)
            if random.random() < 0.3:  # 30% chance to resolve naturally
                self._memory_leak_active = False
                self._memory_leak_accumulator = 0.0
                logger.info("Memory leak condition resolved naturally")

        # Handle cascading failure continuation
        if self._cascading_failure_active:
            self._cascading_ticks += 1
            snapshot.cpu_usage = min(100, snapshot.cpu_usage + 15)
            snapshot.error_rate += 10
            if self._cascading_ticks > 5:
                self._cascading_failure_active = False
                self._cascading_ticks = 0

        # Probabilistic anomaly injection
        if random.random() < self.anomaly_probability:
            snapshot = self._inject_anomaly(snapshot)
            logger.warning(f"Anomaly injected: {snapshot.anomaly_type}")

        return snapshot

    def reset_anomaly_state(self):
        """Reset persistent anomaly states (used after remediation)."""
        self._memory_leak_active = False
        self._memory_leak_accumulator = 0.0
        self._cascading_failure_active = False
        self._cascading_ticks = 0
