"""
Remediation Engine - Autonomous decision-making and action execution.

Implements an intelligent remediation strategy:
1. Selects the best remediation action from a configured set based on the root cause
2. Uses a confidence threshold to decide between autonomous action vs. alerting
3. Tracks action history to avoid repeating failed remediations
4. Simulates action execution with configurable success rates

The engine follows the Observe-Orient-Decide-Act (OODA) loop pattern:
- Observe: Receive anomaly and root cause data
- Orient: Evaluate available actions and past outcomes
- Decide: Select the best action based on expected success
- Act: Execute the remediation (simulated)
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.analyzer.root_cause_analyzer import RootCauseReport
from src.utils.logger import setup_logger

logger = setup_logger("Remediation")


@dataclass
class RemediationAction:
    """A single remediation action with its outcome."""
    timestamp: datetime
    cause: str
    action_name: str
    description: str
    success: bool
    confidence: float
    execution_time: float  # simulated execution time in seconds
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cause": self.cause,
            "action_name": self.action_name,
            "description": self.description,
            "success": self.success,
            "confidence": round(self.confidence, 3),
            "execution_time": round(self.execution_time, 2),
            "details": self.details,
        }


class RemediationEngine:
    """
    Autonomous remediation engine that selects and executes corrective actions.

    Maintains a history of actions taken and their outcomes to learn from
    past successes and failures. Uses a confidence threshold to determine
    whether to act autonomously or defer to human operators.
    """

    def __init__(self, config: dict):
        self.config = config
        remediation_cfg = config["remediation"]
        self.actions_config = remediation_cfg["actions"]
        self.confidence_threshold = remediation_cfg["confidence_threshold"]
        self.max_retries = remediation_cfg["max_retries"]
        self.autonomous_mode = config["agent"]["autonomous_mode"]

        # Track remediation history per cause
        self.action_history: list[RemediationAction] = []
        self.failure_counts: defaultdict = defaultdict(int)
        # Track success rates learned from experience
        self.learned_success_rates: dict = {}

    def _select_best_action(self, cause: str) -> Optional[dict]:
        """
        Select the best remediation action for the given root cause.

        Prioritizes actions with higher success rates. If an action has
        failed recently, its effective priority is reduced.
        """
        if cause not in self.actions_config:
            logger.warning(f"No remediation actions configured for cause: {cause}")
            return None

        available_actions = self.actions_config[cause]
        scored_actions = []

        for action in available_actions:
            name = action["name"]
            base_success_rate = action["success_rate"]

            # Adjust based on learned experience
            key = f"{cause}:{name}"
            if key in self.learned_success_rates:
                effective_rate = (base_success_rate + self.learned_success_rates[key]) / 2
            else:
                effective_rate = base_success_rate

            # Penalize actions that have failed recently
            recent_failures = sum(
                1 for a in self.action_history[-10:]
                if a.action_name == name and not a.success
            )
            penalty = recent_failures * 0.15
            effective_rate = max(0.1, effective_rate - penalty)

            scored_actions.append((action, effective_rate))

        # Sort by effective success rate (descending)
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        best_action, best_rate = scored_actions[0]
        logger.info(
            f"Selected action: {best_action['name']} "
            f"(effective success rate: {best_rate:.2f})"
        )
        return best_action

    def _execute_action(self, action: dict, cause: str) -> RemediationAction:
        """
        Simulate executing a remediation action.

        In a real system, this would call APIs, run scripts, etc.
        Here we simulate with configurable success rates.
        """
        start_time = time.time()

        # Simulate execution delay
        execution_delay = random.uniform(0.5, 2.0)
        time.sleep(execution_delay)

        # Simulate success/failure based on configured rate
        success = random.random() < action["success_rate"]
        execution_time = time.time() - start_time

        result = RemediationAction(
            timestamp=datetime.now(),
            cause=cause,
            action_name=action["name"],
            description=action["description"],
            success=success,
            confidence=action["success_rate"],
            execution_time=execution_time,
            details={
                "autonomous": self.autonomous_mode,
                "attempt": self.failure_counts.get(f"{cause}:{action['name']}", 0) + 1,
            },
        )

        # Update learning
        key = f"{cause}:{action['name']}"
        if success:
            self.failure_counts[key] = 0
            self.learned_success_rates[key] = min(
                1.0, self.learned_success_rates.get(key, action["success_rate"]) + 0.05
            )
            logger.info(f"Remediation SUCCESS: {action['name']} for {cause}")
        else:
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
            self.learned_success_rates[key] = max(
                0.1, self.learned_success_rates.get(key, action["success_rate"]) - 0.1
            )
            logger.warning(f"Remediation FAILED: {action['name']} for {cause}")

        self.action_history.append(result)
        return result

    def remediate(self, rca_report: RootCauseReport) -> Optional[RemediationAction]:
        """
        Decide and execute remediation based on root cause analysis.

        Decision logic:
        - If confidence < threshold: log alert only, no autonomous action
        - If confidence >= threshold and autonomous mode: execute best action
        - If max retries exceeded for a cause: escalate to human operator
        """
        if not rca_report.identified_cause:
            logger.info("No identified cause - skipping remediation")
            return None

        cause = rca_report.identified_cause

        # Check confidence threshold
        if rca_report.confidence < self.confidence_threshold:
            logger.info(
                f"Confidence ({rca_report.confidence:.2f}) below threshold "
                f"({self.confidence_threshold}) - alerting only, no autonomous action"
            )
            return RemediationAction(
                timestamp=datetime.now(),
                cause=cause,
                action_name="alert_only",
                description="Confidence too low for autonomous action; alerting operators",
                success=True,
                confidence=rca_report.confidence,
                execution_time=0.0,
                details={"reason": "below_confidence_threshold"},
            )

        if not self.autonomous_mode:
            logger.info("Autonomous mode disabled - alerting only")
            return RemediationAction(
                timestamp=datetime.now(),
                cause=cause,
                action_name="alert_only",
                description="Autonomous mode disabled; operator intervention required",
                success=True,
                confidence=rca_report.confidence,
                execution_time=0.0,
                details={"reason": "autonomous_mode_disabled"},
            )

        # Select and execute the best action
        action = self._select_best_action(cause)
        if not action:
            return None

        # Check retry limits
        key = f"{cause}:{action['name']}"
        if self.failure_counts.get(key, 0) >= self.max_retries:
            logger.warning(
                f"Max retries ({self.max_retries}) exceeded for {action['name']} - "
                f"escalating to human operator"
            )
            return RemediationAction(
                timestamp=datetime.now(),
                cause=cause,
                action_name="escalate_to_human",
                description=f"Max retries exceeded for {action['name']}; escalating",
                success=True,
                confidence=rca_report.confidence,
                execution_time=0.0,
                details={"reason": "max_retries_exceeded", "failed_action": action["name"]},
            )

        return self._execute_action(action, cause)

    def get_stats(self) -> dict:
        """Return remediation statistics."""
        total = len(self.action_history)
        successes = sum(1 for a in self.action_history if a.success)
        actions_taken = [a for a in self.action_history if a.action_name not in ("alert_only", "escalate_to_human")]
        auto_actions = len(actions_taken)
        auto_successes = sum(1 for a in actions_taken if a.success)

        return {
            "total_actions": total,
            "total_successes": successes,
            "success_rate": round(successes / total, 3) if total > 0 else 0,
            "autonomous_actions": auto_actions,
            "autonomous_success_rate": round(auto_successes / auto_actions, 3) if auto_actions > 0 else 0,
            "learned_rates": dict(self.learned_success_rates),
        }
