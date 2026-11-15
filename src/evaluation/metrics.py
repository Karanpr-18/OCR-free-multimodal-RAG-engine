"""
Evaluation metrics for retrieval, ranking, and abstention calibration.

Includes standard Information Retrieval (IR) metrics:
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@k)
And selective classification / uncertainty quantification metrics:
- Abstention AUROC
- Expected Calibration Error (ECE)
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Sequence[Any], set[Any]],
    k: int,
) -> float:
    """
    Compute Recall@k for a ranked list of retrieved items.

    Recall@k = |Retrieved[:k] ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ranked list of retrieved item identifiers.
        relevant_ids: Ground truth relevant item identifiers.
        k: Rank cutoff (must be >= 1).

    Returns:
        Recall@k as a float in [0.0, 1.0]. Returns 0.0 for empty or invalid inputs.
    """
    if k <= 0 or not relevant_ids or not retrieved_ids:
        return 0.0

    target_set = set(relevant_ids)
    if not target_set:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    hits = len(retrieved_k & target_set)
    return float(hits / len(target_set))


def mrr(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Sequence[Any], set[Any]],
) -> float:
    """
    Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of the first relevant retrieved item (1-indexed).
    Returns 0.0 if no relevant item appears in retrieved_ids.

    Args:
        retrieved_ids: Ranked list of retrieved item identifiers.
        relevant_ids: Ground truth relevant item identifiers.

    Returns:
        Reciprocal rank as a float in [0.0, 1.0].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    target_set = set(relevant_ids)
    if not target_set:
        return 0.0

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in target_set:
            return float(1.0 / rank)

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Union[Sequence[Any], set[Any]],
    k: int = 5,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k (nDCG@k) with binary relevance.

    DCG@k = sum_{i=1}^min(k, |retrieved|) rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^min(k, |relevant|) 1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: Ranked list of retrieved item identifiers.
        relevant_ids: Ground truth relevant item identifiers.
        k: Cutoff rank (default: 5).

    Returns:
        nDCG@k as a float in [0.0, 1.0]. Returns 0.0 for empty inputs.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    target_set = set(relevant_ids)
    if not target_set:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in target_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(target_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def _compute_auroc_rank_fallback(
    confidences: Sequence[float],
    is_correct: Sequence[int | bool],
) -> float:
    """
    Mann-Whitney U statistic calculation of AUROC.
    Handles tied scores using average fractional ranks.
    """
    pairs = sorted(
        zip([float(c) for c in confidences], [1 if bool(x) else 0 for x in is_correct]),
        key=lambda p: p[0],
    )
    n = len(pairs)
    if n < 2:
        return 0.5

    n_pos = sum(p[1] for p in pairs)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and pairs[j][0] == pairs[j + 1][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average rank
        for k_idx in range(i, j + 1):
            ranks[k_idx] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[idx] for idx in range(n) if pairs[idx][1] == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auroc)))


def abstention_auroc(
    confidences: Sequence[float],
    is_correct: Sequence[int | bool],
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic (AUROC) curve
    measuring how well confidence scores discriminate correct from incorrect predictions.

    Args:
        confidences: Predicted confidence scores in [0.0, 1.0].
        is_correct: Ground truth binary correctness indicators (1/True = correct, 0/False = incorrect).

    Returns:
        AUROC score as a float in [0.0, 1.0].
        Returns 0.5 for degenerate edge cases (single class, empty, single item).
    """
    if len(confidences) == 0 or len(is_correct) == 0:
        return 0.0

    if len(confidences) != len(is_correct):
        raise ValueError(
            f"Length mismatch: {len(confidences)} confidences vs {len(is_correct)} labels"
        )

    if len(confidences) < 2:
        return 0.5

    y_true = [1 if bool(x) else 0 for x in is_correct]
    unique_classes = set(y_true)
    if len(unique_classes) < 2:
        # Cannot discriminate if only one class exists in ground truth
        return 0.5

    if roc_auc_score is not None:
        try:
            score = roc_auc_score(y_true, confidences)
            return float(score)
        except ValueError:
            return 0.5

    return _compute_auroc_rank_fallback(confidences, y_true)


def expected_calibration_error(
    confidences: Sequence[float],
    accuracies: Sequence[Union[int, bool, float]],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) across confidence bins.

    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: Model confidence scores in [0.0, 1.0].
        accuracies: Binary correctness (1 or 0) or empirical accuracy for each prediction.
        n_bins: Number of equal-width bins in [0, 1]. Default: 10.

    Returns:
        ECE as a float in [0.0, 1.0]. Returns 0.0 for empty inputs.
    """
    if len(confidences) == 0 or len(accuracies) == 0:
        return 0.0

    if len(confidences) != len(accuracies):
        raise ValueError(
            f"Length mismatch: {len(confidences)} confidences vs {len(accuracies)} accuracies"
        )

    if n_bins <= 0:
        raise ValueError("n_bins must be greater than 0")

    total_samples = len(confidences)
    confs = [float(c) for c in confidences]
    accs = [float(a) for a in accuracies]

    bin_boundaries = [i / float(n_bins) for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        bin_confs: list[float] = []
        bin_accs: list[float] = []

        for c, a in zip(confs, accs):
            if i == 0:
                in_bin = (c >= bin_lower) and (c <= bin_upper)
            else:
                in_bin = (c > bin_lower) and (c <= bin_upper)

            if in_bin:
                bin_confs.append(c)
                bin_accs.append(a)

        bin_count = len(bin_confs)
        if bin_count > 0:
            bin_acc = sum(bin_accs) / bin_count
            bin_conf = sum(bin_confs) / bin_count
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(max(0.0, min(1.0, ece)))


def compute_all_metrics(
    results_list: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate evaluation metrics across a list of query evaluation dictionaries.

    Expected keys per item (with fallback aliases):
    - 'retrieved_ids' / 'retrieved' / 'ranked_doc_ids': Sequence of retrieved IDs
    - 'relevant_ids' / 'relevant' / 'gold_doc_ids': Sequence of ground truth IDs
    - 'confidence' / 'confidence_score' / 'conf': float in [0, 1] (optional)
    - 'is_correct' / 'correct' / 'accuracy': bool or 0/1 (optional)
    - 'latency_ms' / 'search_ms' / 'latency': float in ms (optional)

    Returns:
        Dictionary mapping metric names to aggregate float scores:
        - recall@1
        - recall@5
        - mrr
        - ndcg@5
        - abstention_auroc (if confidence and correctness present)
        - ece (if confidence and correctness present)
        - p50_latency_ms (if latency present)
        - p95_latency_ms (if latency present)
        - mean_latency_ms (if latency present)
        - num_samples
    """
    if not results_list:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "abstention_auroc": 0.5,
            "ece": 0.0,
            "num_samples": 0.0,
        }

    recalls_1: list[float] = []
    recalls_5: list[float] = []
    mrrs: list[float] = []
    ndcgs_5: list[float] = []
    confidences: list[float] = []
    correctness: list[int] = []
    latencies: list[float] = []

    for item in results_list:
        retrieved = (
            item.get("retrieved_ids")
            or item.get("retrieved")
            or item.get("ranked_doc_ids")
            or []
        )
        relevant = (
            item.get("relevant_ids")
            or item.get("relevant")
            or item.get("gold_doc_ids")
            or []
        )

        recalls_1.append(recall_at_k(retrieved, relevant, k=1))
        recalls_5.append(recall_at_k(retrieved, relevant, k=5))
        mrrs.append(mrr(retrieved, relevant))
        ndcgs_5.append(ndcg_at_k(retrieved, relevant, k=5))

        conf = (
            item.get("confidence")
            if item.get("confidence") is not None
            else item.get("confidence_score")
        )
        if conf is None:
            conf = item.get("conf")

        corr = (
            item.get("is_correct")
            if item.get("is_correct") is not None
            else item.get("correct")
        )
        if corr is None:
            corr = item.get("accuracy")

        if conf is not None and corr is not None:
            confidences.append(float(conf))
            correctness.append(1 if bool(corr) else 0)

        lat = (
            item.get("latency_ms")
            if item.get("latency_ms") is not None
            else item.get("search_ms")
        )
        if lat is None:
            lat = item.get("latency")
        if lat is not None:
            latencies.append(float(lat))

    n = len(results_list)

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    output: dict[str, float] = {
        "recall@1": _mean(recalls_1),
        "recall@5": _mean(recalls_5),
        "mrr": _mean(mrrs),
        "ndcg@5": _mean(ndcgs_5),
        "num_samples": float(n),
    }

    if confidences and correctness:
        output["abstention_auroc"] = abstention_auroc(confidences, correctness)
        output["ece"] = expected_calibration_error(confidences, correctness)
    else:
        output["abstention_auroc"] = 0.5
        output["ece"] = 0.0

    if latencies:
        sorted_lat = sorted(latencies)
        output["mean_latency_ms"] = _mean(latencies)
        idx_50 = min(int(0.50 * len(sorted_lat)), len(sorted_lat) - 1)
        idx_95 = min(int(0.95 * len(sorted_lat)), len(sorted_lat) - 1)
        output["p50_latency_ms"] = float(sorted_lat[idx_50])
        output["p95_latency_ms"] = float(sorted_lat[idx_95])

    return output
