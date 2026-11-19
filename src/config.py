"""
Centralized configuration loading from environment variables and YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class ModelSettings(BaseSettings):
    """Vision-language and embedding model settings."""

    vlm_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Model ID for the VLM reader",
    )
    vlm_api_endpoint: Optional[str] = Field(
        default=None,
        description="Optional OpenAI-compatible endpoint URL for VLM reader",
    )
    vlm_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for VLM endpoint",
    )
    colqwen_model: str = Field(
        default="vidore/colqwen2.5-v0.2",
        description="HuggingFace model ID for the ColQwen visual encoder",
    )
    device: str = Field(
        default="auto",
        description="Device for model inference (auto, cuda, cpu, mps)",
    )
    dtype: str = Field(default="bfloat16")
    max_new_tokens: int = Field(default=1024)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline settings."""

    top_k: int = Field(default=2, description="Number of pages to retrieve")
    collection_name: str = Field(default="multimodal_docs")
    embedding_dim: int = Field(default=128)
    similarity: str = Field(default="maxsim")
    binary_quantization: bool = Field(default=True)
    rescore: bool = Field(default=True)


class QdrantSettings(BaseSettings):
    """Qdrant vector-store connection settings."""

    mode: str = Field(
        default="local", description="'local' for embedded, 'remote' for server"
    )
    url: str = Field(default="http://localhost:6333")
    local_path: str = Field(default="./qdrant_storage")

    model_config = SettingsConfigDict(env_prefix="QDRANT_")


class AbstentionSettings(BaseSettings):
    """Abstention / confidence calibration settings."""

    confidence_threshold: float = Field(default=0.5)
    calibration_method: str = Field(default="logistic")


class Settings(BaseSettings):
    """
    Root application settings.

    Priority (highest → lowest):
      1. Environment variables / .env
      2. configs/default.yaml
      3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Top-level env overrides ---
    qdrant_mode: str = Field(default="local")
    qdrant_url: str = Field(default="http://localhost:6333")
    vlm_model: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    vlm_api_endpoint: Optional[str] = Field(default=None)
    vlm_api_key: Optional[str] = Field(default=None)
    colqwen_model: str = Field(default="vidore/colqwen2.5-v0.2")
    device: str = Field(default="auto")
    top_k: int = Field(default=2)
    confidence_threshold: float = Field(default=0.5)

    # --- Nested settings (populated from YAML) ---
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    abstention: AbstentionSettings = Field(default_factory=AbstentionSettings)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings by merging YAML defaults with env overrides."""
        yaml_path = path or _DEFAULT_CONFIG
        yaml_data = _load_yaml(yaml_path)

        # Build nested settings from YAML sections
        model_cfg = ModelSettings(**(yaml_data.get("model", {})))
        retrieval_cfg = RetrievalSettings(**(yaml_data.get("retrieval", {})))
        qdrant_cfg = QdrantSettings(**(yaml_data.get("qdrant", {})))
        abstention_cfg = AbstentionSettings(**(yaml_data.get("abstention", {})))

        return cls(
            model=model_cfg,
            retrieval=retrieval_cfg,
            qdrant=qdrant_cfg,
            abstention=abstention_cfg,
        )


# Module-level singleton — import `settings` anywhere.
settings = Settings.from_yaml()
