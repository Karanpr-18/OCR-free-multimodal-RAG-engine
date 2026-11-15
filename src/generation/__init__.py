"""Generation module: VLM grounded reader and calibrated abstention."""

from .abstention import AbstentionCalibrator
from .vlm_reader import VLMReader

__all__ = [
    "VLMReader",
    "AbstentionCalibrator",
]
