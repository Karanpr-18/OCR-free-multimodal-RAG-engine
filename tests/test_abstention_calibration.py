"""
Unit tests for confidence-calibrated abstention and selective prediction.

Tests:
  - Confidence computation C(q) = σ(w₁·s̃₁ + w₂·Δs + w₃·logit(p) + b)
  - Mathematical correctness with known manual inputs and expected values
  - Edge cases: empty scores, single score, extreme probabilities (0 and 1)
  - Selective prediction threshold gating (should_abstain)
  - Threshold calibration for target precision and maximum coverage
  - Area Under the ROC Curve (AUROC) computation and fallbacks
  - Expected Calibration Error (ECE) across prediction bins
  - Risk-Coverage curve and AURC computation
  - Logistic regression parameter fitting
"""

from __future__ import annotations

import math
from typing import List

import pytest

from src.generation.abstention import AbstentionCalibrator, _sigmoid, _trapezoid_integral


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestAbstentionCalibrator:

    def test_sigmoid_numerical_stability(self):
        """Test sigmoid function with positive, zero, negative, and extreme inputs."""
        assert math.isclose(_sigmoid(0.0), 0.5)
        assert math.isclose(_sigmoid(2.0), 1.0 / (1.0 + math.exp(-2.0)))
        assert math.isclose(_sigmoid(-2.0), 1.0 / (1.0 + math.exp(2.0)))
        # Extremes should not overflow or throw OverflowError
        assert math.isclose(_sigmoid(100.0), 1.0, abs_tol=1e-7)
        assert math.isclose(_sigmoid(-100.0), 0.0, abs_tol=1e-7)

    def test_compute_confidence_known_analytical_values(self):
        """
        Verify confidence C(q) with known manual inputs.
        Weights = (1.0, 1.0, 1.0), bias = 0.0
        retrieval_scores = [20.0, 10.0]
        query_token_count = 10
        vlm_confidence = 0.5

        Calculations:
          s1 = 20.0, s2 = 10.0
          s_norm = 20.0 / 10 = 2.0
          delta_s = (20.0 - 10.0) / 20.0 = 0.5
          logit(0.5) = ln(0.5 / 0.5) = 0.0
          z = (1.0 * 2.0) + (1.0 * 0.5) + (1.0 * 0.0) + 0.0 = 2.5
          C(q) = sigmoid(2.5) = 1 / (1 + exp(-2.5)) = 0.9241418... -> 0.9241
        """
        calibrator = AbstentionCalibrator(weights=(1.0, 1.0, 1.0), bias=0.0)
        conf = calibrator.compute_confidence(
            retrieval_scores=[20.0, 10.0],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert math.isclose(conf, 0.9241, abs_tol=1e-4)

    def test_compute_confidence_equal_retrieval_scores(self):
        """
        When top 2 scores are identical (delta_s = 0.0):
        Scores: [10.0, 10.0], N_q = 10, p = 0.5
          s1_norm = 1.0, delta_s = 0.0, logit(p) = 0.0
          z = 1.0 -> sigmoid(1.0) = 0.731058... -> 0.7311
        """
        calibrator = AbstentionCalibrator(weights=(1.0, 1.0, 1.0), bias=0.0)
        conf = calibrator.compute_confidence(
            retrieval_scores=[10.0, 10.0],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert math.isclose(conf, 0.7311, abs_tol=1e-4)

    def test_compute_confidence_with_custom_weights_and_bias(self):
        """Verify confidence calculation with custom weights and non-zero bias."""
        calibrator = AbstentionCalibrator(weights=(0.5, 0.5, 0.5), bias=-1.0)
        # Scores: [10.0, 5.0], N_q = 10, p = 0.5
        # s1_norm = 1.0, delta_s = 0.5, logit(p) = 0.0
        # z = (0.5 * 1.0) + (0.5 * 0.5) + 0 - 1.0 = 0.75 - 1.0 = -0.25
        # sigmoid(-0.25) = 1 / (1 + exp(0.25)) = 0.4378
        conf = calibrator.compute_confidence(
            retrieval_scores=[10.0, 5.0],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert math.isclose(conf, 0.4378, abs_tol=1e-4)

    def test_compute_confidence_edge_case_empty_scores(self):
        """Empty retrieval scores should fallback to reader probability logit."""
        calibrator = AbstentionCalibrator(weights=(1.0, 1.0, 1.0), bias=0.0)
        # p = 0.5 -> logit(0.5) = 0 -> sigmoid(0) = 0.5
        conf = calibrator.compute_confidence(
            retrieval_scores=[],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert math.isclose(conf, 0.5, abs_tol=1e-4)

    def test_compute_confidence_edge_case_single_score(self):
        """Single retrieval score implies s2 = 0.0 and delta_s = 1.0."""
        calibrator = AbstentionCalibrator(weights=(1.0, 1.0, 1.0), bias=0.0)
        # s1 = 10.0, s2 = 0.0 -> s1_norm = 1.0, delta_s = 1.0, logit(0.5) = 0
        # z = 2.0 -> sigmoid(2.0) = 0.8808
        conf = calibrator.compute_confidence(
            retrieval_scores=[10.0],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert math.isclose(conf, 0.8808, abs_tol=1e-4)

    def test_compute_confidence_boundary_probabilities(self):
        """Boundary values p=0.0 and p=1.0 should be clamped without math domain errors."""
        calibrator = AbstentionCalibrator(weights=(1.0, 1.0, 1.0), bias=0.0)

        # p = 0.0
        conf_zero = calibrator.compute_confidence(
            retrieval_scores=[10.0, 5.0],
            query_token_count=10,
            vlm_confidence=0.0,
        )
        assert 0.0 <= conf_zero <= 1.0

        # p = 1.0
        conf_one = calibrator.compute_confidence(
            retrieval_scores=[10.0, 5.0],
            query_token_count=10,
            vlm_confidence=1.0,
        )
        assert 0.0 <= conf_one <= 1.0
        assert conf_one > conf_zero

    def test_should_abstain(self):
        """Verify decision threshold gating."""
        calibrator = AbstentionCalibrator(default_threshold=0.6)

        # Below threshold -> abstain (True)
        assert calibrator.should_abstain(0.59) is True
        # At or above threshold -> do not abstain (False)
        assert calibrator.should_abstain(0.60) is False
        assert calibrator.should_abstain(0.75) is False

        # Explicit threshold override
        assert calibrator.should_abstain(0.55, threshold=0.50) is False
        assert calibrator.should_abstain(0.55, threshold=0.70) is True

    def test_calibrate_threshold_target_precision(self):
        """
        Test threshold calibration with known candidate pool:
          Scores:  [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
          Labels:  [0,   0,   1,   0,   1,   1]

        Precision at thresholds:
          tau=0.1: ans=6, correct=3, prec = 0.50
          tau=0.3: ans=4, correct=3, prec = 0.75
          tau=0.7: ans=3, correct=2, prec = 0.67
          tau=0.8: ans=2, correct=2, prec = 1.00 >= 0.90 (coverage = 2/6 = 0.33)
          tau=0.9: ans=1, correct=1, prec = 1.00 >= 0.90 (coverage = 1/6 = 0.17)

        Optimal threshold maximizing coverage at precision >= 0.90 is 0.8.
        """
        calibrator = AbstentionCalibrator()
        val_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        val_labels = [0, 0, 1, 0, 1, 1]

        optimal_tau = calibrator.calibrate_threshold(
            val_scores=val_scores,
            val_labels=val_labels,
            target_precision=0.90,
        )
        assert math.isclose(optimal_tau, 0.8, abs_tol=1e-3)
        assert calibrator.default_threshold == optimal_tau

    def test_calibrate_threshold_empty_fallback(self):
        """Empty validation set retains default threshold."""
        calibrator = AbstentionCalibrator(default_threshold=0.55)
        tau = calibrator.calibrate_threshold([], [], target_precision=0.9)
        assert tau == 0.55

    def test_compute_auroc_perfect_and_random(self):
        """Test AUROC computation for perfect, random, and inverse rankings."""
        calibrator = AbstentionCalibrator()

        # 1. Perfect discrimination: all positives scored higher than all negatives
        pos_scores = [0.95, 0.85, 0.75]
        neg_scores = [0.30, 0.20, 0.10]
        scores = pos_scores + neg_scores
        labels = [1, 1, 1, 0, 0, 0]

        auroc_perfect = calibrator.compute_auroc(scores, labels)
        assert math.isclose(auroc_perfect, 1.0, abs_tol=1e-4)

        # 2. Inverted discrimination: all negatives scored higher than positives
        inv_scores = [0.10, 0.20, 0.30, 0.75, 0.85, 0.95]
        inv_labels = [1, 1, 1, 0, 0, 0]
        auroc_inverted = calibrator.compute_auroc(inv_scores, inv_labels)
        assert math.isclose(auroc_inverted, 0.0, abs_tol=1e-4)

        # 3. Single class edge case returns 0.5 safely
        assert calibrator.compute_auroc([0.8, 0.9], [1, 1]) == 0.5
        assert calibrator.compute_auroc([0.2, 0.3], [0, 0]) == 0.5

    def test_compute_ece(self):
        """Test Expected Calibration Error calculation."""
        calibrator = AbstentionCalibrator()

        # 1. Perfect calibration: 10 samples in bin [0.8, 0.9), all confidence=0.8, 8 positives, 2 negatives
        # Accuracy = 8/10 = 0.8 == Confidence = 0.8 -> error = 0.0
        confidences = [0.85] * 10
        labels = [1] * 8 + [0] * 2  # acc = 0.80, conf = 0.85 -> diff = 0.05
        ece = calibrator.compute_ece(confidences, labels, n_bins=10)
        assert math.isclose(ece, 0.05, abs_tol=1e-4)

        # 2. Completely uncalibrated: high confidence (0.95), zero accuracy (all labels 0)
        bad_confidences = [0.95] * 10
        bad_labels = [0] * 10
        bad_ece = calibrator.compute_ece(bad_confidences, bad_labels, n_bins=10)
        assert math.isclose(bad_ece, 0.95, abs_tol=1e-4)

        # 3. Empty inputs return 0.0
        assert calibrator.compute_ece([], []) == 0.0

    def test_risk_coverage_curve(self):
        """Verify properties of risk-coverage curve."""
        calibrator = AbstentionCalibrator()
        confidences = [0.1, 0.3, 0.6, 0.8, 0.95]
        labels = [0, 0, 1, 1, 1]

        curve = calibrator.get_risk_coverage_curve(confidences, labels, num_thresholds=11)

        assert "thresholds" in curve
        assert "coverages" in curve
        assert "risks" in curve
        assert "aurc" in curve

        # At tau=0.0, coverage should be 1.0 (all 5 samples answered)
        assert math.isclose(curve["coverages"][0], 1.0)
        # At tau=0.0, 3 correct out of 5 -> risk = 2/5 = 0.4
        assert math.isclose(curve["risks"][0], 0.4)

        # Coverage must be non-increasing with increasing threshold
        coverages = curve["coverages"]
        for i in range(1, len(coverages)):
            assert coverages[i] <= coverages[i - 1]

        assert curve["aurc"] >= 0.0

    def test_trapezoid_integral(self):
        """Verify trapezoidal integration helper."""
        # Integral of y = 2x from 0 to 1 is 1.0
        x = [0.0, 0.5, 1.0]
        y = [0.0, 1.0, 2.0]
        area = _trapezoid_integral(x, y)
        assert math.isclose(area, 1.0, abs_tol=1e-4)

        # Edge cases
        assert _trapezoid_integral([], []) == 0.0
        assert _trapezoid_integral([1.0], [1.0]) == 0.0

    def test_extract_features(self):
        """Verify raw feature vector extraction."""
        calibrator = AbstentionCalibrator()
        features = calibrator.extract_features(
            retrieval_scores=[20.0, 10.0],
            query_token_count=10,
            vlm_confidence=0.5,
        )
        assert len(features) == 3
        s1_norm, delta_s, logit_p = features
        assert math.isclose(s1_norm, 2.0)
        assert math.isclose(delta_s, 0.5)
        assert math.isclose(logit_p, 0.0, abs_tol=1e-5)

    def test_fit_calibrator(self):
        """Test calibrator weight fitting if scikit-learn is present."""
        calibrator = AbstentionCalibrator()
        retrieval_scores_list = [
            [25.0, 10.0],
            [22.0, 8.0],
            [5.0, 4.8],
            [3.0, 2.9],
        ]
        query_token_counts = [10, 10, 10, 10]
        vlm_confidences = [0.9, 0.85, 0.2, 0.15]
        labels = [1, 1, 0, 0]

        # Fit should update weights without crashing
        calibrator.fit(
            retrieval_scores_list=retrieval_scores_list,
            query_token_counts=query_token_counts,
            vlm_confidences=vlm_confidences,
            labels=labels,
        )
        assert len(calibrator.weights) == 3
        assert isinstance(calibrator.bias, float)
