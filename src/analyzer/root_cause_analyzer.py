"""
Root Cause Analyzer - Correlates anomalous metrics to identify probable root causes.

Uses a knowledge-based approach combined with metric correlation analysis:
1. Pattern Matching: Maps combinations of anomalous metrics to known incident patterns
2. Correlation Analysis: Computes pairwise Pearson correlation between metrics
   over a sliding window to detect co-occurring anomalies
3. Confidence Scoring: Weights each diagnosis by how well the observed pattern
   matches known failure signatures

This enables the agent to go beyond "something is wrong" to "here's WHY it's wrong."
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.detector.anomaly_detector import AnomalyReport, METRIC_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("Analyzer")


# Knowledge base: maps metric combinations to likely root causes
FAILURE_SIGNATURES = {
    "cpu_spike": {
        "primary_metrics": {"cpu_usage"},
        "secondary_metrics": {"request_rate"},
        "description": "CPU usage spike, likely from compute-intensive workload or runaway process",
    },
    "memory_leak": {
        "primary_metrics": {"memory_usage"},
        "secondary_metrics": {"cpu_usage", "error_rate"},
        "description": "Gradually increasing memory usage indicating a memory leak",
    },
    "disk_saturation": {
        "primary_metrics": {"disk_io"},
        "secondary_metrics": {"network_latency"},
        "description": "Disk I/O saturation causing performance degradation",
    },
    "network_degradation": {
        "primary_metrics": {"network_latency"},
        "secondary_metrics": {"error_rate"},
        "description": "Network latency increase leading to timeouts and errors",
    },
    "error_burst": {
        "primary_metrics": {"error_rate"},
        "secondary_metrics": {"request_rate"},
        "description": "Sudden burst of errors, possibly from a bad deployment or dependency failure",
    },
    "cascading_failure": {
        "primary_metrics": {"cpu_usage", "memory_usage", "error_rate"},
        "secondary_metrics": {"network_latency", "request_rate"},
        "description": "Multiple correlated failures indicating a cascading system failure",
    },
}


@dataclass
class RootCauseReport:
    """Result of root cause analysis."""
    identified_cause: Optional[str]
    confidence: float
    description: str
    correlated_metrics: list
    recommendations: list
    details: dict

    def to_dict(self) -> dict:
        return {
            "identified_cause": self.identified_cause,
            "confidence": float(round(self.confidence, 3)),
            "description": self.description,
            "correlated_metrics": self.correlated_metrics,
            "recommendations": self.recommendations,
            "details": self.details,
        }


class RootCauseAnalyzer:
    """
    Analyzes detected anomalies to identify their root cause.

    Maintains a history of metric values to compute correlations and matches
    observed anomaly patterns against a knowledge base of known failure signatures.
    """

    def __init__(self, config: dict):
        self.config = config
        analyzer_cfg = config["analyzer"]
        self.correlation_threshold = analyzer_cfg["correlation_threshold"]
        self.history_window = config["agent"]["history_window"]
        self.metric_history: deque = deque(maxlen=self.history_window)

    def _compute_correlations(self) -> dict:
        """Compute pairwise Pearson correlations between metrics."""
        if len(self.metric_history) < 10:
            return {}

        data = np.array(list(self.metric_history))
        n_metrics = data.shape[1]
        correlations = {}

        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                std_i = np.std(data[:, i])
                std_j = np.std(data[:, j])
                if std_i == 0 or std_j == 0:
                    continue
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                if abs(corr) > self.correlation_threshold:
                    pair = (METRIC_NAMES[i], METRIC_NAMES[j])
                    correlations[pair] = round(float(corr), 3)

        return correlations

    def _match_failure_signature(
        self, anomalous_metrics: list
    ) -> tuple[Optional[str], float, str]:
        """Match observed anomalous metrics to known failure signatures."""
        anomalous_set = set(anomalous_metrics)
        best_match = None
        best_score = 0.0
        best_desc = "Unknown anomaly pattern"

        for cause, signature in FAILURE_SIGNATURES.items():
            primary = signature["primary_metrics"]
            secondary = signature["secondary_metrics"]

            # Score based on how many primary/secondary metrics match
            primary_match = len(primary & anomalous_set) / len(primary)
            secondary_match = (
                len(secondary & anomalous_set) / len(secondary) if secondary else 0
            )

            # Primary metrics are weighted 70%, secondary 30%
            score = 0.7 * primary_match + 0.3 * secondary_match

            # Bonus for signatures that explain more anomalous metrics
            # (prefer broader explanations like cascading_failure)
            all_sig_metrics = primary | secondary
            coverage = len(all_sig_metrics & anomalous_set) / max(len(anomalous_set), 1)
            score += 0.1 * coverage

            if score > best_score and primary_match > 0:
                best_match = cause
                best_score = score
                best_desc = signature["description"]

        return best_match, best_score, best_desc

    def _generate_recommendations(self, cause: Optional[str]) -> list:
        """Generate actionable recommendations based on identified cause."""
        recommendations = {
            "cpu_spike": [
                "Investigate top CPU-consuming processes",
                "Consider horizontal scaling if load-related",
                "Check for infinite loops or inefficient queries",
            ],
            "memory_leak": [
                "Profile application memory allocation",
                "Restart the affected service as immediate mitigation",
                "Check recent code changes for memory management issues",
            ],
            "disk_saturation": [
                "Identify large files or growing logs",
                "Implement log rotation if not configured",
                "Consider provisioning additional storage",
            ],
            "network_degradation": [
                "Check network connectivity and DNS resolution",
                "Investigate for packet loss or bandwidth saturation",
                "Consider enabling failover to backup network",
            ],
            "error_burst": [
                "Review application logs for error details",
                "Check recent deployments for regressions",
                "Verify external dependency health",
            ],
            "cascading_failure": [
                "Immediately isolate failing components",
                "Activate circuit breakers on affected services",
                "Consider rolling back recent changes",
                "Prepare for potential full system restart",
            ],
        }
        return recommendations.get(cause, ["Monitor the situation closely"])

    def analyze(self, anomaly_report: AnomalyReport) -> RootCauseReport:
        """
        Perform root cause analysis on a detected anomaly.

        Combines pattern matching against known failure signatures with
        metric correlation analysis to produce a diagnosis.
        """
        # Record the metric values
        if anomaly_report.snapshot:
            values = anomaly_report.snapshot.metric_values()
            self.metric_history.append([values[m] for m in METRIC_NAMES])

        if not anomaly_report.is_anomaly:
            return RootCauseReport(
                identified_cause=None,
                confidence=0.0,
                description="No anomaly detected - system operating normally",
                correlated_metrics=[],
                recommendations=[],
                details={},
            )

        # Step 1: Match against known failure signatures
        cause, match_confidence, description = self._match_failure_signature(
            anomaly_report.anomalous_metrics
        )

        # Step 2: Compute metric correlations
        correlations = self._compute_correlations()
        correlated_pairs = [
            {"metrics": list(pair), "correlation": corr}
            for pair, corr in correlations.items()
        ]

        # Step 3: Adjust confidence based on correlation evidence
        if correlated_pairs:
            correlation_boost = min(0.15, len(correlated_pairs) * 0.05)
            match_confidence = min(1.0, match_confidence + correlation_boost)

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(cause)

        report = RootCauseReport(
            identified_cause=cause,
            confidence=match_confidence,
            description=description,
            correlated_metrics=correlated_pairs,
            recommendations=recommendations,
            details={
                "anomalous_metrics": anomaly_report.anomalous_metrics,
                "detection_method": anomaly_report.method,
                "detection_confidence": anomaly_report.confidence,
            },
        )

        logger.info(
            f"Root cause: {cause or 'unknown'} (confidence={match_confidence:.2f}) - {description}"
        )

        return report
