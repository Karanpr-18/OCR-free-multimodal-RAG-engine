"""
HuggingFace ViDoRe dataset loader and evaluation split builder.

Supports loading visual document retrieval datasets (e.g. TabFQuad, DocVQA, InfoVQA)
and constructing calibrated counterfactual unanswerable query splits for testing
RAG abstention mechanisms.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SUPPORTED_VIDORE_DATASETS = [
    "vidore/tabfquad_test_subsampled",
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
    "vidore/arxivqa_test_subsampled",
]

_COUNTERFACTUAL_TEMPLATES = [
    "What was the projected figure for the fiscal year 2049 in {entity}?",
    "According to the document, why was {entity} relocated to Mars in 2050?",
    "What was the total expenditure for {entity} during the 19th century?",
    "What percentage decrease occurred in {entity} according to Appendix Z?",
    "How many quantum processors were listed under {entity} in Table 99?",
]


def load_vidore_dataset(
    dataset_name: str = "vidore/tabfquad_test_subsampled",
    split: str = "test",
) -> Any:
    """
    Load a ViDoRe benchmark dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset repository id.
        split: Dataset split to load (e.g. 'test', 'train').

    Returns:
        HuggingFace Dataset instance.

    Raises:
        ImportError: If 'datasets' library is not installed.
        RuntimeError: If loading from HuggingFace fails.
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        logger.error("HuggingFace 'datasets' library is required to load ViDoRe datasets")
        raise ImportError(
            "The 'datasets' package is required. Install via `pip install datasets`."
        ) from err

    logger.info("Loading ViDoRe dataset '%s' (split='%s')", dataset_name, split)
    try:
        ds = load_dataset(dataset_name, split=split)
        logger.info(
            "Successfully loaded '%s' [%s] with %d examples",
            dataset_name,
            split,
            len(ds),
        )
        return ds
    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", dataset_name, e)
        raise RuntimeError(
            f"Failed to load ViDoRe dataset '{dataset_name}' with split '{split}': {e}"
        ) from e


def _generate_counterfactual_query(
    original_query: str, rng: random.Random
) -> Tuple[str, str]:
    """
    Generate a counterfactual unanswerable variant of an answerable query.

    Uses entity perturbation, temporal shifting, or synthetic impossibility templates.
    """
    # 1. Temporal perturbation: replace 4-digit years with futuristic / out-of-range years
    years = re.findall(r"\b(19\d\d|20[0-2]\d)\b", original_query)
    if years and rng.random() < 0.6:
        chosen_year = rng.choice(years)
        perturbed_year = str(rng.choice([2045, 2055, 2088, 2099]))
        perturbed = original_query.replace(chosen_year, perturbed_year, 1)
        return perturbed, "temporal_counterfactual"

    # 2. Numeric / percentage perturbation
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", original_query)
    if numbers and rng.random() < 0.5:
        chosen_num = rng.choice(numbers)
        replacement = "999,999" if "%" not in chosen_num else "99.9%"
        perturbed = original_query.replace(chosen_num, replacement, 1)
        return perturbed, "numeric_counterfactual"

    # 3. Noun/Entity extract fallback with template injection
    words = [w for w in original_query.split() if len(w) > 4 and w.isalpha()]
    entity = rng.choice(words) if words else "the organization"
    template = rng.choice(_COUNTERFACTUAL_TEMPLATES)
    return template.format(entity=entity), "template_counterfactual"


def build_unanswerable_split(
    answerable_queries: Union[List[Dict[str, Any]], Any],
    corpus_pages: Optional[Union[List[Dict[str, Any]], Any]] = None,
    ratio: float = 0.25,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Construct counterfactual unanswerable queries for abstention calibration.

    Args:
        answerable_queries: List of dicts or HuggingFace Dataset containing answerable query items.
        corpus_pages: Optional collection of corpus documents/pages.
        ratio: Desired ratio of unanswerable to answerable queries (default 0.25 -> 20% total).
        seed: Random seed for deterministic counterfactual generation.

    Returns:
        List of unanswerable query dictionaries tagged with is_answerable=False.
    """
    rng = random.Random(seed)

    # Convert Dataset to list of records if needed
    if hasattr(answerable_queries, "to_list"):
        items = answerable_queries.to_list()
    elif hasattr(answerable_queries, "__iter__") and not isinstance(
        answerable_queries, list
    ):
        items = list(answerable_queries)
    else:
        items = list(answerable_queries)

    if not items:
        logger.warning("No answerable queries provided to build_unanswerable_split")
        return []

    target_count = max(1, int(len(items) * ratio))
    sampled_indices = rng.sample(range(len(items)), min(target_count, len(items)))

    unanswerable_queries: List[Dict[str, Any]] = []

    for rank, idx in enumerate(sampled_indices):
        item = items[idx]
        orig_query = item.get("query", "")
        if not orig_query and "question" in item:
            orig_query = item["question"]

        perturbed_query, perturbation_type = _generate_counterfactual_query(
            orig_query, rng
        )

        h = hashlib.sha256(
            f"{perturbed_query}:{seed}:{rank}".encode("utf-8")
        ).hexdigest()[:12]
        query_id = f"unans_{rank}_{h}"

        record: Dict[str, Any] = {
            "query_id": query_id,
            "query": perturbed_query,
            "original_query": orig_query,
            "is_answerable": False,
            "gold_answers": [],
            "relevant_page_ids": [],
            "unanswerable_type": perturbation_type,
        }

        # Carry over source metadata if present
        if "dataset" in item:
            record["dataset"] = item["dataset"]

        unanswerable_queries.append(record)

    logger.info(
        "Built %d counterfactual unanswerable queries from %d base queries (ratio=%.2f)",
        len(unanswerable_queries),
        len(items),
        ratio,
    )
    return unanswerable_queries


def get_evaluation_splits(
    dataset_name: str = "vidore/tabfquad_test_subsampled",
    val_ratio: float = 0.2,
    unanswerable_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load a dataset and partition into train/validation/test splits with unanswerable queries.

    Args:
        dataset_name: HuggingFace ViDoRe dataset name.
        val_ratio: Fraction of answerable queries reserved for confidence calibration.
        unanswerable_ratio: Ratio of synthetic counterfactual queries to generate.
        seed: Random seed for deterministic splitting.

    Returns:
        Dictionary containing:
          - 'val': List of queries for calibration (answerable + unanswerable)
          - 'test': List of queries for evaluation (answerable + unanswerable)
          - 'test_answerable': Answerable test queries only
          - 'test_unanswerable': Unanswerable test queries only
          - 'corpus': Unique corpus pages mapping (page_id -> image/metadata)
          - 'summary': Statistics dict
    """
    raw_dataset = load_vidore_dataset(dataset_name, split="test")

    rng = random.Random(seed)
    indices = list(range(len(raw_dataset)))
    rng.shuffle(indices)

    # Standardize item structure and build corpus index
    corpus: Dict[str, Any] = {}
    normalized_queries: List[Dict[str, Any]] = []

    for idx, raw_item in enumerate(raw_dataset):
        # Determine unique page identifier
        image = raw_item.get("image")
        image_name = raw_item.get(
            "image_filename", raw_item.get("image_path", f"page_{idx}.png")
        )

        h = hashlib.sha256(f"{image_name}:{idx}".encode("utf-8")).hexdigest()[:16]
        page_id = f"doc_{h}"

        if page_id not in corpus and image is not None:
            corpus[page_id] = {
                "page_id": page_id,
                "image": image,
                "filename": image_name,
            }

        q_text = raw_item.get("query", raw_item.get("question", ""))
        answers = raw_item.get("answers", raw_item.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        q_id = f"q_{idx}_{h[:8]}"
        normalized_queries.append(
            {
                "query_id": q_id,
                "query": q_text,
                "is_answerable": True,
                "gold_answers": answers,
                "relevant_page_ids": [page_id],
                "dataset": dataset_name,
            }
        )

    # Partition answerable into validation and test
    n_total = len(normalized_queries)
    n_val = int(n_total * val_ratio)

    val_answerable = normalized_queries[:n_val]
    test_answerable = normalized_queries[n_val:]

    # Build unanswerable splits for both validation and test
    val_unanswerable = build_unanswerable_split(
        val_answerable, ratio=unanswerable_ratio, seed=seed
    )
    test_unanswerable = build_unanswerable_split(
        test_answerable, ratio=unanswerable_ratio, seed=seed + 1
    )

    val_combined = val_answerable + val_unanswerable
    rng.shuffle(val_combined)

    test_combined = test_answerable + test_unanswerable
    rng.shuffle(test_combined)

    summary = {
        "dataset_name": dataset_name,
        "total_corpus_pages": len(corpus),
        "total_answerable_queries": n_total,
        "val_answerable_count": len(val_answerable),
        "val_unanswerable_count": len(val_unanswerable),
        "test_answerable_count": len(test_answerable),
        "test_unanswerable_count": len(test_unanswerable),
        "val_total": len(val_combined),
        "test_total": len(test_combined),
    }

    logger.info(
        "Splits prepared for '%s': %d corpus pages, %d val queries (%d unans), %d test queries (%d unans)",
        dataset_name,
        len(corpus),
        len(val_combined),
        len(val_unanswerable),
        len(test_combined),
        len(test_unanswerable),
    )

    return {
        "val": val_combined,
        "test": test_combined,
        "test_answerable": test_answerable,
        "test_unanswerable": test_unanswerable,
        "corpus": corpus,
        "summary": summary,
    }
