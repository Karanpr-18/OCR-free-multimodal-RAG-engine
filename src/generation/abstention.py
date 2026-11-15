"""
Confidence-calibrated abstention for multimodal RAG.

Implements logistic confidence scoring based on retrieval score magnitude,
retrieval margin, and VLM prediction confidence:
    s̃₁ = s₁ / N_q
    Δs = (s₁ - s₂) / max(s₁, ε)
    C(q) = σ(w₁·s̃₁ + w₂·Δs + w₃·log(p / (1 - p)) + b)

Includes threshold calibration for target precision, AUROC, ECE,
and risk-coverage curve computation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _sigmoid(z: float) -> float:
    """Compute numerically stable sigmoid."""
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(z)
    return exp_pos / (1.0 + exp_pos)


def _trapezoid_integral(x_points: Sequence[float], y_points: Sequence[float]) -> float:
    """Compute integral using the trapezoidal rule in pure Python."""
    if len(x_points) < 2 or len(x_points) != len(y_points):
        return 0.0
    integral = 0.0
    for i in range(1, len(x_points)):
        dx = x_points[i] - x_points[i - 1]
        avg_y = 0.5 * (y_points[i] + y_points[i - 1])
        integral += dx * avg_y
    return integral


class AbstentionCalibrator:
    """
    Confidence calibrator and selective abstention gate.

    Computes composite calibrated confidence score using:
    - Normalized top retrieval score: s̃₁ = s₁ / N_q
    - Retrieval top-1 vs top-2 relative margin: Δs = (s₁ - s₂) / max(s₁, ε)
    - VLM generation logit: log(p / (1 - p))
    - Logistic combination: C(q) = σ(w₁·s̃₁ + w₂·Δs + w₃·logit(p) + b)
    """

    def __init__(
        self,
        weights: Optional[Tuple[float, float, float]] = None,
        bias: float = 0.0,
        default_threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the abstention calibrator.

        Args:
            weights: Tuple of (w1, w2, w3) coefficients for (s̃₁, Δs, logit(p)).
                     Defaults to (1.0, 1.0, 1.0).
            bias: Intercept term b in logistic formula.
            default_threshold: Default confidence cutoff below which to abstain.
            eps: Epsilon parameter for numerical stability.
        """
        self.weights = weights if weights is not None else (1.0, 1.0, 1.0)
        self.bias = bias
        self.default_threshold = default_threshold
        self.eps = eps

    def compute_confidence(
        self,
        retrieval_scores: Sequence[float],
        query_token_count: int = 10,
        vlm_confidence: float = 0.85,
    ) -> float:
        """
        Compute calibrated confidence score C(q) for a query.

        Formula:
            s̃₁ = s₁ / N_q
            Δs = (s₁ - s₂) / max(s₁, ε)
            logit(p) = log(p / (1 - p))
            C(q) = σ(w₁·s̃₁ + w₂·Δs + w₃·logit(p) + b)

        Args:
            retrieval_scores: List of retrieval scores (e.g. MaxSim), ordered or unordered.
            query_token_count: Number of query tokens N_q (minimum 1).
            vlm_confidence: VLM reader probability p in [0.0, 1.0].

        Returns:
            Calibrated confidence score C(q) in [0.0, 1.0].
        """
        w1, w2, w3 = self.weights
        b = self.bias
        eps = self.eps

        # Handle empty retrieval scores
        if not retrieval_scores:
            p_clamped = min(max(float(vlm_confidence), eps), 1.0 - eps)
            logit_p = math.log(p_clamped / (1.0 - p_clamped))
            z = (w3 * logit_p) + b
            return round(_sigmoid(z), 4)

        # Sort scores descending: s₁ >= s₂ >= ...
        sorted_scores = sorted([float(s) for s in retrieval_scores], reverse=True)
        s1 = sorted_scores[0]
        s2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Query token count N_q
        n_q = max(int(query_token_count), 1)

        # 1. Normalized top score
        s1_norm = s1 / float(n_q)

        # 2. Score margin
        denom = max(s1, eps)
        delta_s = (s1 - s2) / denom

        # 3. VLM confidence logit
        p_clamped = min(max(float(vlm_confidence), eps), 1.0 - eps)
        logit_p = math.log(p_clamped / (1.0 - p_clamped))

        # 4. Logistic combination
        z = (w1 * s1_norm) + (w2 * delta_s) + (w3 * logit_p) + b
        confidence = _sigmoid(z)

        return round(float(confidence), 4)

    def extract_features(
        self,
        retrieval_scores: Sequence[float],
        query_token_count: int = 10,
        vlm_confidence: float = 0.85,
    ) -> list[float]:
        """
        Extract the 3 raw features [s̃₁, Δs, logit(p)] for a query.

        Args:
            retrieval_scores: Sequence of retrieval scores.
            query_token_count: Query token count.
            vlm_confidence: VLM reader probability.

        Returns:
            List of 3 floats: [s1_norm, delta_s, logit_p].
        """
        eps = self.eps
        if not retrieval_scores:
            s1_norm = 0.0
            delta_s = 0.0
        else:
            sorted_scores = sorted([float(s) for s in retrieval_scores], reverse=True)
            s1 = sorted_scores[0]
            s2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            n_q = max(int(query_token_count), 1)
            s1_norm = s1 / float(n_q)
            delta_s = (s1 - s2) / max(s1, eps)

        p_clamped = min(max(float(vlm_confidence), eps), 1.0 - eps)
        logit_p = math.log(p_clamped / (1.0 - p_clamped))

        return [s1_norm, delta_s, logit_p]

    def fit(
        self,
        retrieval_scores_list: Sequence[Sequence[float]],
        query_token_counts: Sequence[int],
        vlm_confidences: Sequence[float],
        labels: Sequence[int],
    ) -> "AbstentionCalibrator":
        """
        Fit logistic regression weights (w1, w2, w3, b) on calibration data.

        Args:
            retrieval_scores_list: List of retrieval score sequences.
            query_token_counts: List of token counts.
            vlm_confidences: List of VLM confidences.
            labels: Binary labels (1 = answerable/correct, 0 = unanswerable/wrong).

        Returns:
            Self (calibrator with updated weights and bias).
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning(
                "scikit-learn is required to fit calibration weights. Leaving weights unchanged."
            )
            return self

        features = [
            self.extract_features(r_scores, n_q, p)
            for r_scores, n_q, p in zip(
                retrieval_scores_list, query_token_counts, vlm_confidences
            )
        ]
        if np is not None:
            X = np.array(features, dtype=float)
            y = np.array(labels, dtype=int)
            unique_classes = np.unique(y)
        else:
            X = features
            y = list(labels)
            unique_classes = set(y)

        if len(unique_classes) < 2:
            logger.warning("Fit requires at least 2 distinct classes in labels. Skipping fit.")
            return self

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(X, y)

        coefs = lr.coef_[0]
        self.weights = (float(coefs[0]), float(coefs[1]), float(coefs[2]))
        self.bias = float(lr.intercept_[0])
        logger.info(
            "Calibrator fitted: weights=%s, bias=%.4f", self.weights, self.bias
        )
        return self

    def calibrate_threshold(
        self,
        val_scores: Sequence[float],
        val_labels: Sequence[int],
        target_precision: float = 0.9,
    ) -> float:
        """
        Calibrate decision threshold to guarantee a target precision on answered queries.

        Finds lowest threshold tau such that Precision(answered queries) >= target_precision,
        thereby maximizing coverage (recall).

        Args:
            val_scores: Validation confidence scores in [0.0, 1.0].
            val_labels: Binary validation labels (1 = correct, 0 = incorrect).
            target_precision: Desired minimum precision (e.g. 0.90).

        Returns:
            Calibrated optimal threshold tau in [0.0, 1.0].
        """
        scores = [float(s) for s in val_scores]
        labels = [int(l) for l in val_labels]

        if not scores:
            return self.default_threshold

        candidates = sorted(list(set(scores)))

        best_threshold: Optional[float] = None
        best_coverage = -1.0
        fallback_threshold = candidates[0]
        best_precision = 0.0
        n_total = len(scores)

        for thresh in candidates:
            answered_indices = [i for i, s in enumerate(scores) if s >= thresh]
            n_answered = len(answered_indices)

            if n_answered == 0:
                continue

            n_correct = sum(1 for i in answered_indices if labels[i] == 1)
            precision = n_correct / float(n_answered)
            coverage = n_answered / float(n_total)

            if precision >= target_precision:
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_threshold = thresh

            if precision > best_precision:
                best_precision = precision
                fallback_threshold = thresh

        chosen = best_threshold if best_threshold is not None else fallback_threshold
        self.default_threshold = round(float(chosen), 4)
        logger.info(
            "Calibrated threshold: %.4f (target_precision=%.2f, achieved_coverage=%.2f)",
            self.default_threshold,
            target_precision,
            max(best_coverage, 0.0),
        )
        return self.default_threshold

    def should_abstain(
        self, confidence: float, threshold: Optional[float] = None
    ) -> bool:
        """
        Determine whether the model should abstain from answering.

        Args:
            confidence: Query confidence score.
            threshold: Cutoff threshold (defaults to calibrated self.default_threshold).

        Returns:
            True if confidence < threshold (abstain), False otherwise.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        return float(confidence) < float(thresh)

    def compute_auroc(
        self, confidences: Sequence[float], labels: Sequence[int]
    ) -> float:
        """
        Compute Area Under ROC Curve (AUROC) for abstention/answerability detection.

        Args:
            confidences: Predicted confidence scores.
            labels: Ground truth binary labels (1 = answerable, 0 = unanswerable).

        Returns:
            AUROC score in [0.0, 1.0].
        """
        c_list = [float(c) for c in confidences]
        y_list = [int(y) for y in labels]

        n_pos = sum(1 for y in y_list if y == 1)
        n_neg = len(y_list) - n_pos

        if n_pos == 0 or n_neg == 0:
            logger.warning("AUROC requires both positive and negative labels. Returning 0.5.")
            return 0.5

        if roc_auc_score is not None:
            try:
                score = roc_auc_score(y_list, c_list)
                return round(float(score), 4)
            except Exception as err:
                logger.warning("sklearn roc_auc_score failed (%s); using exact Wilcoxon.", err)

        # Exact Wilcoxon-Mann-Whitney U statistic:
        pos_scores = [c for c, y in zip(c_list, y_list) if y == 1]
        neg_scores = [c for c, y in zip(c_list, y_list) if y == 0]

        u_stat = 0.0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    u_stat += 1.0
                elif p == n:
                    u_stat += 0.5

        auroc = u_stat / (n_pos * n_neg)
        return round(float(auroc), 4)

    def compute_ece(
        self,
        confidences: Sequence[float],
        labels: Sequence[int],
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Partitions predictions into equal-width bins and computes:
            ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

        Args:
            confidences: Confidence values in [0.0, 1.0].
            labels: Binary labels in {0, 1}.
            n_bins: Number of equal-width bins.

        Returns:
            ECE metric in [0.0, 1.0].
        """
        c_list = [float(c) for c in confidences]
        y_list = [int(y) for y in labels]
        n = len(c_list)

        if n == 0:
            return 0.0

        bin_width = 1.0 / n_bins
        ece = 0.0

        for i in range(n_bins):
            bin_lower = i * bin_width
            bin_upper = (i + 1) * bin_width

            if i == n_bins - 1:
                in_bin_indices = [
                    idx for idx, c in enumerate(c_list) if bin_lower <= c <= bin_upper
                ]
            else:
                in_bin_indices = [
                    idx for idx, c in enumerate(c_list) if bin_lower <= c < bin_upper
                ]

            count = len(in_bin_indices)
            if count > 0:
                acc_in_bin = sum(y_list[idx] for idx in in_bin_indices) / count
                conf_in_bin = sum(c_list[idx] for idx in in_bin_indices) / count
                weight = count / n
                ece += weight * abs(acc_in_bin - conf_in_bin)

        return round(float(ece), 4)

    def get_risk_coverage_curve(
        self,
        confidences: Sequence[float],
        labels: Sequence[int],
        num_thresholds: int = 101,
    ) -> dict[str, Any]:
        """
        Compute the Risk-Coverage trade-off curve across confidence thresholds.

        - Coverage: fraction of queries answered (confidence >= tau).
        - Risk: error rate among answered queries (1 - precision).
        - AURC: Area Under Risk-Coverage curve.

        Args:
            confidences: Model confidence scores.
            labels: Binary ground-truth labels (1 = correct, 0 = incorrect).
            num_thresholds: Number of threshold evaluation points.

        Returns:
            Dict with 'thresholds', 'coverages', 'risks', and 'aurc'.
        """
        c_list = [float(c) for c in confidences]
        y_list = [int(y) for y in labels]
        n = len(c_list)

        if n == 0:
            return {
                "thresholds": [],
                "coverages": [],
                "risks": [],
                "aurc": 0.0,
            }

        step = 1.0 / max(num_thresholds - 1, 1)
        thresholds = [i * step for i in range(num_thresholds)]
        coverages: list[float] = []
        risks: list[float] = []

        last_risk = 0.0

        for tau in thresholds:
            answered_indices = [i for i, c in enumerate(c_list) if c >= tau]
            n_answered = len(answered_indices)

            if n_answered == 0:
                coverage = 0.0
                risk = last_risk
            else:
                coverage = n_answered / float(n)
                error_count = sum(1 for i in answered_indices if y_list[i] == 0)
                risk = error_count / float(n_answered)
                last_risk = risk

            coverages.append(round(coverage, 4))
            risks.append(round(risk, 4))

        # Sort by coverage ascending for trapezoidal integration
        sorted_pairs = sorted(zip(coverages, risks), key=lambda x: x[0])
        cov_sorted = [p[0] for p in sorted_pairs]
        risk_sorted = [p[1] for p in sorted_pairs]

        aurc = _trapezoid_integral(cov_sorted, risk_sorted)

        return {
            "thresholds": [round(float(t), 4) for t in thresholds],
            "coverages": coverages,
            "risks": risks,
            "aurc": round(aurc, 4),
        }
