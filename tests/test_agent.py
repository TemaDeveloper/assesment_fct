"""
Tests for the AIOps Agent components.
"""

import pytest
import yaml
import numpy as np

from src.simulator.metric_simulator import MetricSimulator, MetricSnapshot
from src.detector.anomaly_detector import AnomalyDetector, AnomalyReport
from src.analyzer.root_cause_analyzer import RootCauseAnalyzer, RootCauseReport
from src.remediation.remediation_engine import RemediationEngine


@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


class TestMetricSimulator:
    def test_generate_returns_snapshot(self, config):
        sim = MetricSimulator(config)
        snapshot = sim.generate()
        assert isinstance(snapshot, MetricSnapshot)
        assert snapshot.timestamp is not None

    def test_metrics_in_reasonable_range(self, config):
        sim = MetricSimulator(config)
        # Generate many samples - normal ones should be in range
        for _ in range(50):
            snap = sim.generate()
            assert 0 <= snap.cpu_usage <= 100
            assert 0 <= snap.memory_usage <= 100
            assert snap.disk_io >= 0
            assert snap.network_latency >= 0
            assert snap.error_rate >= 0
            assert snap.request_rate >= 0

    def test_to_dict(self, config):
        sim = MetricSimulator(config)
        snapshot = sim.generate()
        d = snapshot.to_dict()
        assert "timestamp" in d
        assert "cpu_usage" in d
        assert "memory_usage" in d
        assert "is_anomaly" in d

    def test_anomaly_injection(self, config):
        # Force high anomaly probability
        config["simulator"]["anomaly"]["probability"] = 1.0
        sim = MetricSimulator(config)
        snapshot = sim.generate()
        assert snapshot.is_anomaly
        assert snapshot.anomaly_type is not None

    def test_reset_anomaly_state(self, config):
        sim = MetricSimulator(config)
        sim._memory_leak_active = True
        sim._memory_leak_accumulator = 50.0
        sim.reset_anomaly_state()
        assert not sim._memory_leak_active
        assert sim._memory_leak_accumulator == 0.0


class TestAnomalyDetector:
    def test_initialization(self, config):
        detector = AnomalyDetector(config)
        assert not detector.is_trained
        assert len(detector.history) == 0

    def test_detect_returns_report(self, config):
        detector = AnomalyDetector(config)
        sim = MetricSimulator(config)
        snapshot = sim.generate()
        report = detector.detect(snapshot)
        assert isinstance(report, AnomalyReport)

    def test_trains_after_enough_samples(self, config):
        config["detector"]["min_training_samples"] = 20
        detector = AnomalyDetector(config)
        sim = MetricSimulator(config)
        # Disable anomalies for clean training data
        config["simulator"]["anomaly"]["probability"] = 0.0
        sim_clean = MetricSimulator(config)

        for _ in range(25):
            snapshot = sim_clean.generate()
            detector.detect(snapshot)

        assert detector.is_trained

    def test_detects_obvious_anomaly(self, config):
        config["simulator"]["anomaly"]["probability"] = 0.0
        detector = AnomalyDetector(config)
        sim = MetricSimulator(config)

        # Train on normal data
        for _ in range(40):
            snapshot = sim.generate()
            detector.detect(snapshot)

        # Inject an extreme anomaly manually
        from datetime import datetime
        anomalous_snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=99.0,
            memory_usage=98.0,
            disk_io=200.0,
            network_latency=500.0,
            error_rate=100.0,
            request_rate=50.0,
        )

        report = detector.detect(anomalous_snapshot)
        assert report.is_anomaly
        assert report.confidence > 0.3


class TestRootCauseAnalyzer:
    def test_initialization(self, config):
        analyzer = RootCauseAnalyzer(config)
        assert len(analyzer.metric_history) == 0

    def test_no_anomaly_returns_normal(self, config):
        analyzer = RootCauseAnalyzer(config)
        normal_report = AnomalyReport(
            is_anomaly=False, confidence=0.0, method="test",
            anomalous_metrics=[], details={},
        )
        result = analyzer.analyze(normal_report)
        assert result.identified_cause is None
        assert "normally" in result.description

    def test_identifies_cpu_spike(self, config):
        analyzer = RootCauseAnalyzer(config)
        from datetime import datetime
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=95.0, memory_usage=50.0, disk_io=30.0,
            network_latency=20.0, error_rate=5.0, request_rate=300.0,
        )
        anomaly_report = AnomalyReport(
            is_anomaly=True, confidence=0.8, method="test",
            anomalous_metrics=["cpu_usage", "request_rate"],
            details={}, snapshot=snapshot,
        )
        result = analyzer.analyze(anomaly_report)
        assert result.identified_cause == "cpu_spike"
        assert result.confidence > 0.5

    def test_identifies_cascading_failure(self, config):
        analyzer = RootCauseAnalyzer(config)
        from datetime import datetime
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=95.0, memory_usage=95.0, disk_io=30.0,
            network_latency=200.0, error_rate=50.0, request_rate=100.0,
        )
        anomaly_report = AnomalyReport(
            is_anomaly=True, confidence=0.9, method="test",
            anomalous_metrics=["cpu_usage", "memory_usage", "error_rate", "network_latency", "request_rate"],
            details={}, snapshot=snapshot,
        )
        result = analyzer.analyze(anomaly_report)
        assert result.identified_cause == "cascading_failure"


class TestRemediationEngine:
    def test_initialization(self, config):
        engine = RemediationEngine(config)
        assert len(engine.action_history) == 0

    def test_no_cause_skips_remediation(self, config):
        engine = RemediationEngine(config)
        report = RootCauseReport(
            identified_cause=None, confidence=0.0,
            description="", correlated_metrics=[],
            recommendations=[], details={},
        )
        result = engine.remediate(report)
        assert result is None

    def test_low_confidence_alerts_only(self, config):
        config["remediation"]["confidence_threshold"] = 0.8
        engine = RemediationEngine(config)
        report = RootCauseReport(
            identified_cause="cpu_spike", confidence=0.3,
            description="test", correlated_metrics=[],
            recommendations=[], details={},
        )
        result = engine.remediate(report)
        assert result.action_name == "alert_only"

    def test_high_confidence_takes_action(self, config):
        engine = RemediationEngine(config)
        report = RootCauseReport(
            identified_cause="cpu_spike", confidence=0.9,
            description="test", correlated_metrics=[],
            recommendations=[], details={},
        )
        result = engine.remediate(report)
        assert result is not None
        assert result.action_name != "alert_only"

    def test_stats_tracking(self, config):
        engine = RemediationEngine(config)
        stats = engine.get_stats()
        assert stats["total_actions"] == 0
        assert stats["success_rate"] == 0

    def test_disabled_autonomous_mode(self, config):
        config["agent"]["autonomous_mode"] = False
        engine = RemediationEngine(config)
        report = RootCauseReport(
            identified_cause="cpu_spike", confidence=0.9,
            description="test", correlated_metrics=[],
            recommendations=[], details={},
        )
        result = engine.remediate(report)
        assert result.action_name == "alert_only"
        assert "autonomous_mode_disabled" in result.details.get("reason", "")
