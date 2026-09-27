"""Evaluation: metrics, bootstrap CIs, paired comparison, gates, reports (pure)."""

from bci_platform.evaluation.comparison import ComparisonResult, paired_compare
from bci_platform.evaluation.evaluator import (
    EvaluationResult,
    GateDecision,
    TargetScale,
    check_gates,
    evaluate,
)
from bci_platform.evaluation.metrics import (
    bootstrap_ci,
    coverage,
    mae,
    nll_gaussian,
    r2,
    regression_metrics,
    rmse,
)

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "GateDecision",
    "TargetScale",
    "bootstrap_ci",
    "check_gates",
    "coverage",
    "evaluate",
    "mae",
    "nll_gaussian",
    "paired_compare",
    "r2",
    "regression_metrics",
    "rmse",
]
