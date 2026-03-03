# AIOps Agent - AI-Driven Autonomous Operations

An intelligent AIOps agent that monitors system metrics, detects anomalies using machine learning, diagnoses root causes through pattern matching and correlation analysis, and executes autonomous remediation actions.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     AIOps Agent (MAPE-K Loop)                    │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ MONITOR  │→ │   ANALYZE    │→ │   PLAN   │→ │   EXECUTE   │ │
│  │          │  │              │  │          │  │             │ │
│  │ Metric   │  │ Anomaly      │  │ Root     │  │ Remediation │ │
│  │Simulator │  │ Detector     │  │ Cause    │  │ Engine      │ │
│  │          │  │              │  │ Analyzer │  │             │ │
│  │ - CPU    │  │ - Isolation  │  │          │  │ - Auto      │ │
│  │ - Memory │  │   Forest     │  │ - Pattern│  │   scaling   │ │
│  │ - Disk   │  │ - Z-Score    │  │   Match  │  │ - Service   │ │
│  │ - Network│  │ - Ensemble   │  │ - Metric │  │   restart   │ │
│  │ - Errors │  │   Voting     │  │   Corr.  │  │ - Circuit   │ │
│  │ - RPS    │  │              │  │          │  │   breaker   │ │
│  └──────────┘  └──────────────┘  └──────────┘  └─────────────┘ │
│                                                                  │
│                    ┌──────────┐                                   │
│                    │KNOWLEDGE │ ← Learns from remediation         │
│                    │  BASE    │   outcomes to improve future      │
│                    └──────────┘   decisions                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   Web Dashboard (Flask)                     │  │
│  │  Real-time charts, event log, agent stats                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

The agent follows the **MAPE-K** (Monitor, Analyze, Plan, Execute, Knowledge) architecture pattern, a standard reference model for autonomous computing systems.

## AI Techniques & Logic

### 1. Anomaly Detection (Ensemble Method)

**Isolation Forest (Unsupervised ML)**
- Trained on a sliding window of historical metric data
- Learns the "normal" operating profile of the system
- Flags multi-dimensional outliers that deviate from learned patterns
- Self-retrains every 20 new data points to adapt to changing baselines

**Z-Score Statistical Analysis**
- Computes standard deviation-based thresholds per metric
- Provides immediate detection of single-metric spikes (no training needed)
- Acts as a fallback when the ML model hasn't collected enough training data

**Ensemble Voting**: Both methods run on every data point. When both agree on an anomaly, confidence is boosted. The method with higher confidence wins in case of disagreement.

### 2. Root Cause Analysis

**Failure Signature Pattern Matching**
- A knowledge base maps metric combinations to known failure types (CPU spike, memory leak, disk saturation, network degradation, error burst, cascading failure)
- Signatures have primary and secondary metrics, weighted 70/30 in scoring
- Coverage bonus favors diagnoses that explain more anomalous metrics

**Pearson Correlation Analysis**
- Computes pairwise correlation between metrics over the historical window
- Identifies co-occurring metric changes (e.g., high CPU + low request rate)
- Correlation evidence boosts diagnostic confidence

### 3. Autonomous Remediation (OODA Loop)

**Decision Logic**
- Confidence threshold gating: only acts autonomously above configurable threshold (default 0.7)
- Below threshold: generates alerts for human operators
- Selects best action based on effective success rates
- Tracks past outcomes and adjusts future action selection (reinforcement-like learning)

**Adaptive Learning**
- Success rates are updated after each action (+5% for success, -10% for failure)
- Actions that failed recently receive a penalty in selection scoring
- Max retry limits prevent infinite retry loops; triggers escalation to human operators

### 4. Synthetic Data Generation

- Gaussian noise around configurable means for realistic baseline variation
- Sinusoidal diurnal pattern simulates time-of-day load changes
- Probabilistic anomaly injection with multiple failure types
- Persistent state anomalies (memory leaks, cascading failures) that evolve over time

## Project Structure

```
aiops-agent/
├── main.py                          # Entry point (CLI + dashboard)
├── config/
│   └── config.yaml                  # All agent configuration
├── src/
│   ├── agent/
│   │   └── aiops_agent.py           # Agent orchestrator (MAPE-K loop)
│   ├── simulator/
│   │   └── metric_simulator.py      # Synthetic metric generation
│   ├── detector/
│   │   └── anomaly_detector.py      # Isolation Forest + Z-Score
│   ├── analyzer/
│   │   └── root_cause_analyzer.py   # Pattern matching + correlation
│   ├── remediation/
│   │   └── remediation_engine.py    # Autonomous action engine
│   ├── dashboard/
│   │   ├── app.py                   # Flask web dashboard
│   │   └── templates/index.html     # Real-time monitoring UI
│   └── utils/
│       └── logger.py                # Logging configuration
├── tests/
│   └── test_agent.py                # Unit tests (19 tests)
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd aiops-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

**CLI Mode** (metrics and events printed to terminal):
```bash
python main.py
```

**With Web Dashboard** (real-time charts at http://localhost:5050):
```bash
python main.py --dashboard
```

**Fast Mode** (1-second intervals instead of 5):
```bash
python main.py --fast
```

**Limited Cycles** (run for N cycles then stop):
```bash
python main.py --cycles 50
```

**Combined**:
```bash
python main.py --dashboard --fast --cycles 100
```

Press `Ctrl+C` to stop the agent gracefully. A session summary will be printed.

### Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings are in `config/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `agent` | Monitoring interval, history window size, autonomous mode toggle |
| `simulator` | Normal metric ranges, anomaly probability, anomaly types |
| `detector` | Isolation Forest parameters, Z-score threshold, minimum training samples |
| `analyzer` | Correlation threshold, correlation time window |
| `remediation` | Actions per failure type, confidence threshold, max retries |
| `dashboard` | Host, port |

## Dashboard

The web dashboard provides:
- **Architecture visualization** of the MAPE-K pipeline
- **Live statistics**: cycle count, anomaly count, remediation count, ML model status
- **Real-time charts**: CPU/Memory, Network/Errors, Disk/Requests, Anomaly timeline
- **Event log**: Color-coded events showing anomalies (red), diagnoses (yellow), remediations (green)

## Assumptions

1. **Simulated Environment**: Metrics are synthetically generated rather than collected from real infrastructure. In production, the simulator would be replaced with real metric collectors (Prometheus, CloudWatch, etc.)
2. **Simulated Remediation**: Actions are simulated with configurable success rates. In production, these would execute real operations (API calls, kubectl commands, etc.)
3. **Knowledge-Based RCA**: Root cause analysis uses a predefined knowledge base. A production system would incorporate more sophisticated ML-based causal inference.
4. **Single-Agent Architecture**: The current design is a single agent. Production AIOps would use multi-agent architectures with specialized agents per domain.

## Future Improvements

1. **Real Metric Integration**: Connect to Prometheus, Datadog, or CloudWatch for live metrics
2. **Deep Learning Models**: Use LSTMs or Transformer-based models for time-series anomaly detection
3. **Causal Inference**: Replace pattern matching with causal discovery algorithms (e.g., Granger causality, PC algorithm) for more accurate root cause analysis
4. **Multi-Agent System**: Implement specialized agents (monitoring agent, diagnosis agent, remediation agent) communicating via message queues
5. **LLM Integration**: Use large language models for natural-language incident summarization and runbook generation
6. **Reinforcement Learning**: Train a RL agent for remediation action selection, optimizing for MTTR (Mean Time to Resolution)
7. **Persistent Storage**: Add database backend for historical incident data, enabling trend analysis and post-incident review
8. **Alerting Integration**: Connect to PagerDuty, Slack, or email for operator notifications
9. **Kubernetes Integration**: Deploy as a Kubernetes operator that can directly manage pods, deployments, and services
10. **A/B Testing for Remediations**: Test multiple remediation strategies in parallel to identify the most effective approach

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| scikit-learn | Isolation Forest anomaly detection |
| NumPy / Pandas | Numerical computation and data handling |
| Flask | Web dashboard and REST API |
| Plotly.js | Real-time interactive charts |
| PyYAML | Configuration management |
| Rich | Terminal formatting |
| pytest | Unit testing framework |
