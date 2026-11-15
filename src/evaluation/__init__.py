"""Evaluation module: retrieval and abstention metrics, and ablation benchmark runner."""

from .metrics import (
    abstention_auroc,
    compute_all_metrics,
    expected_calibration_error,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from .run_benchmark import (
    evaluate_system,
    format_ablation_markdown,
    main,
    parse_args,
    run_benchmark,
)

__all__ = [
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "abstention_auroc",
    "expected_calibration_error",
    "compute_all_metrics",
    "run_benchmark",
    "evaluate_system",
    "format_ablation_markdown",
    "parse_args",
    "main",
]
