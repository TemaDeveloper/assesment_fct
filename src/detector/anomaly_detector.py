"""
Anomaly Detector - Multi-method anomaly detection using ML and statistical approaches.

Implements two complementary detection strategies:
1. Isolation Forest (unsupervised ML) - Learns normal patterns and flags outliers
2. Z-Score Analysis (statistical) - Flags metrics exceeding configured standard deviation thresholds

The ensemble approach provides robust detection: ML catches complex multi-dimensional
anomalies while Z-score catches obvious single-metric spikes immediately.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.simulator.metric_simulator import MetricSnapshot
from src.utils.logger import setup_logger

logger = setup_logger("Detector")

METRIC_NAMES = [
    "cpu_usage", "memory_usage", "disk_io",
    "network_latency", "error_rate", "request_rate",
]


@dataclass
class AnomalyReport:
    """Result of anomaly detection analysis."""
    is_anomaly: bool
    confidence: float                       # 0.0 to 1.0
    method: str                             # which detection method triggered
    anomalous_metrics: list                 # which specific metrics are abnormal
    details: dict                           # extra diagnostic info
    snapshot: Optional[MetricSnapshot] = None

    def to_dict(self) -> dict:
        return {
            "is_anomaly": bool(self.is_anomaly),
            "confidence": float(round(self.confidence, 3)),
            "method": self.method,
            "anomalous_metrics": list(self.anomalous_metrics),
            "details": self.details,
        }


class AnomalyDetector:
    """
    Ensemble anomaly detector combining Isolation Forest and Z-Score methods.

    The detector maintains a sliding window of historical metrics for training
    the Isolation Forest model. Before enough data is collected, it relies
    solely on Z-Score detection. Once trained, both methods vote and the
    result with higher confidence wins.
    """

    def __init__(self, config: dict):
        self.config = config
        detector_cfg = config["detector"]

        # Isolation Forest settings
        if_cfg = detector_cfg["isolation_forest"]
        self.contamination = if_cfg["contamination"]
        self.n_estimators = if_cfg["n_estimators"]
        self.random_state = if_cfg["random_state"]

        # Z-Score settings
        self.zscore_threshold = detector_cfg["zscore_threshold"]
        self.min_training_samples = detector_cfg["min_training_samples"]

        # Historical data for training
        self.history: deque = deque(maxlen=config["agent"]["history_window"])
        self.scaler = StandardScaler()
        self.isolation_forest: Optional[IsolationForest] = None
        self.is_trained = False

    def _snapshot_to_array(self, snapshot: MetricSnapshot) -> np.ndarray:
        """Convert a metric snapshot to a feature vector."""
        values = snapshot.metric_values()
        return np.array([values[m] for m in METRIC_NAMES])

    def _train_model(self):
        """Train/retrain the Isolation Forest on accumulated history."""
        if len(self.history) < self.min_training_samples:
            return

        X = np.array(list(self.history))
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        logger.info(f"Isolation Forest trained on {len(self.history)} samples")

    def _detect_zscore(self, features: np.ndarray) -> AnomalyReport:
        """Statistical anomaly detection using Z-scores."""
        if len(self.history) < 5:
            return AnomalyReport(
                is_anomaly=False, confidence=0.0, method="zscore",
                anomalous_metrics=[], details={"reason": "insufficient_data"},
            )

        history_arr = np.array(list(self.history))
        means = history_arr.mean(axis=0)
        stds = history_arr.std(axis=0)
        # Avoid division by zero
        stds = np.where(stds == 0, 1e-6, stds)

        zscores = np.abs((features - means) / stds)

        anomalous = []
        for i, (name, zscore) in enumerate(zip(METRIC_NAMES, zscores)):
            if zscore > self.zscore_threshold:
                anomalous.append({"metric": name, "zscore": round(float(zscore), 2)})

        if anomalous:
            max_zscore = max(a["zscore"] for a in anomalous)
            # Confidence scales with how extreme the z-score is
            confidence = float(min(1.0, max_zscore / (self.zscore_threshold * 2)))
            return AnomalyReport(
                is_anomaly=True,
                confidence=confidence,
                method="zscore",
                anomalous_metrics=[a["metric"] for a in anomalous],
                details={"zscores": anomalous, "threshold": self.zscore_threshold},
            )

        return AnomalyReport(
            is_anomaly=False, confidence=0.0, method="zscore",
            anomalous_metrics=[], details={},
        )

    def _detect_isolation_forest(self, features: np.ndarray) -> AnomalyReport:
        """ML-based anomaly detection using Isolation Forest."""
        if not self.is_trained:
            return AnomalyReport(
                is_anomaly=False, confidence=0.0, method="isolation_forest",
                anomalous_metrics=[], details={"reason": "model_not_trained"},
            )

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.isolation_forest.predict(X_scaled)[0]
        # score_samples returns negative anomaly scores; more negative = more anomalous
        anomaly_score = self.isolation_forest.score_samples(X_scaled)[0]

        is_anomaly = bool(prediction == -1)
        # Convert anomaly score to confidence (score is typically between -0.5 and 0.5)
        confidence = float(max(0.0, min(1.0, -anomaly_score)))

        details = {
            "raw_score": round(float(anomaly_score), 4),
            "prediction": int(prediction),
        }

        # Identify which metrics contributed most to the anomaly
        anomalous_metrics = []
        if is_anomaly:
            history_arr = np.array(list(self.history))
            means = history_arr.mean(axis=0)
            stds = history_arr.std(axis=0)
            stds = np.where(stds == 0, 1e-6, stds)
            deviations = np.abs((features - means) / stds)

            for i, (name, dev) in enumerate(zip(METRIC_NAMES, deviations)):
                if dev > 1.5:  # Moderately deviant
                    anomalous_metrics.append(name)

        return AnomalyReport(
            is_anomaly=is_anomaly,
            confidence=confidence,
            method="isolation_forest",
            anomalous_metrics=anomalous_metrics,
            details=details,
        )

    def detect(self, snapshot: MetricSnapshot) -> AnomalyReport:
        """
        Run ensemble anomaly detection on a metric snapshot.

        Combines results from Z-Score and Isolation Forest detectors.
        The method with the higher confidence score wins.
        """
        features = self._snapshot_to_array(snapshot)
        self.history.append(features)

        # Retrain periodically (every 20 new samples)
        if len(self.history) % 20 == 0:
            self._train_model()

        # Run both detectors
        zscore_result = self._detect_zscore(features)
        iforest_result = self._detect_isolation_forest(features)

        # Ensemble: pick the result with higher confidence, preferring detection
        if zscore_result.is_anomaly and iforest_result.is_anomaly:
            # Both agree - high confidence
            best = zscore_result if zscore_result.confidence >= iforest_result.confidence else iforest_result
            best.confidence = min(1.0, (zscore_result.confidence + iforest_result.confidence) / 1.5)
            best.method = "ensemble(zscore+isolation_forest)"
            # Merge anomalous metrics
            all_metrics = set(zscore_result.anomalous_metrics + iforest_result.anomalous_metrics)
            best.anomalous_metrics = list(all_metrics)
        elif zscore_result.is_anomaly:
            best = zscore_result
        elif iforest_result.is_anomaly:
            best = iforest_result
        else:
            best = AnomalyReport(
                is_anomaly=False, confidence=0.0, method="ensemble",
                anomalous_metrics=[], details={},
            )

        best.snapshot = snapshot

        if best.is_anomaly:
            logger.warning(
                f"ANOMALY DETECTED [{best.method}] confidence={best.confidence:.2f} "
                f"metrics={best.anomalous_metrics}"
            )

        return best
