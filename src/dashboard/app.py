"""
Flask Web Dashboard - Real-time monitoring interface for the AIOps Agent.

Provides:
- Live metric charts updated via polling
- Event timeline showing anomalies, diagnoses, and remediations
- Agent statistics and health overview
- REST API endpoints for programmatic access
"""

import numpy as np
from flask import Flask, render_template, jsonify


def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_app(agent):
    """Create and configure the Flask dashboard app."""
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/state")
    def api_state():
        """Return the current agent state as JSON."""
        return jsonify(sanitize_for_json(agent.get_state()))

    @app.route("/api/metrics")
    def api_metrics():
        """Return recent metrics for charting."""
        state = agent.get_state()
        return jsonify(sanitize_for_json(state["recent_metrics"]))

    @app.route("/api/events")
    def api_events():
        """Return recent events."""
        state = agent.get_state()
        return jsonify(sanitize_for_json(state["recent_events"]))

    @app.route("/api/stats")
    def api_stats():
        """Return remediation statistics."""
        state = agent.get_state()
        return jsonify(sanitize_for_json({
            "cycle_count": state["cycle_count"],
            "anomaly_count": state["anomaly_count"],
            "remediation_count": state["remediation_count"],
            "remediation_stats": state["remediation_stats"],
            "detector_trained": state["detector_trained"],
            "history_size": state["history_size"],
        }))

    return app
