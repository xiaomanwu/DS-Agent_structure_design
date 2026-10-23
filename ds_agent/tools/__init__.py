"""Business tools the agent can call.

Each module is deliberately independent of the agent loop so it can be imported
and unit-tested on its own:

* ``sql``          - guarded DuckDB execution + template NL-to-SQL
* ``anomaly``      - seasonal decomposition, robust z-scores, CUSUM, episodes
* ``contribution`` - additive / ratio decomposition and multi-dimension attribution
* ``experiment``   - experiment archetypes, sizing, readout
* ``causal``       - DiD, event study, PSM, IPW, AIPW, synthetic control
* ``report``       - Markdown + standalone HTML report builder
"""

from .sql import SQLTool, Warehouse, TimeWindow, parse_time_window, sanitize_sql
from .anomaly import AnomalyConfig, AnomalyDetector, detect_metric_anomalies, scan_segment_anomalies
from .contribution import attribute_change, contribution_table, decompose_change
from .experiment import EXPERIMENT_TYPES, ExperimentDesign, analyze_experiment, design_experiment, recommend_experiment_type
from .causal import (
    aipw_effect,
    compare_estimators,
    difference_in_differences,
    event_study,
    ipw_effect,
    propensity_score_matching,
    synthetic_control,
)
from .report import Report, Section

__all__ = [
    "SQLTool", "Warehouse", "TimeWindow", "parse_time_window", "sanitize_sql",
    "AnomalyConfig", "AnomalyDetector", "detect_metric_anomalies", "scan_segment_anomalies",
    "attribute_change", "contribution_table", "decompose_change",
    "EXPERIMENT_TYPES", "ExperimentDesign", "analyze_experiment", "design_experiment",
    "recommend_experiment_type",
    "aipw_effect", "compare_estimators", "difference_in_differences", "event_study",
    "ipw_effect", "propensity_score_matching", "synthetic_control",
    "Report", "Section",
]
