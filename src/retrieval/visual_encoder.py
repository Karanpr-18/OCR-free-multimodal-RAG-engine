"""
ColQwen2.5 / ColPali visual encoder module for OCR-free document retrieval.

Wraps multi-vector vision-language models (ColQwen2.5, ColPali) via colpali-engine
to encode document page images into multi-vector patch representations (N_patches x 128)
and queries into multi-vector token representations (N_tokens x 128).
Includes automatic device fallback (CUDA -> MPS -> CPU) and low-VRAM fallback.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve compute device with automatic CUDA -> MPS -> CPU fallback."""
    if torch is None:
        return "cpu"

    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if requested_device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"

    return requested_device


def resolve_dtype(
    device: str, requested_dtype: Optional[Union[str, Any]] = None
) -> Any:
    """Resolve torch dtype ensuring CPU compatibility with bfloat16."""
    if torch is None:
        return None

    if requested_dtype is None:
        return torch.float32 if device == "cpu" else torch.bfloat16

    if isinstance(requested_dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved = dtype_map.get(requested_dtype.lower(), torch.float32)
    else:
        resolved = requested_dtype

    # Fallback to float32 on CPU if bfloat16 is requested but unsupported
    if device == "cpu" and resolved in (torch.bfloat16, torch.float16):
        logger.info("Using float32 for CPU inference instead of %s.", resolved)
        return torch.float32

    return resolved


class VisualEncoder:
    """
    ColQwen2.5 / ColPali visual document encoder.

    Encodes:
      - Document page images into (N_patches x 128) multi-vector patch embeddings.
      - Query text into (N_tokens x 128) multi-vector token embeddings.
    """

    DEFAULT_MODEL = "vidore/colqwen2.5-v0.2"
    LOW_VRAM_FALLBACK = "vidore/colpali-v1.3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: Optional[Union[str, Any]] = None,
        low_vram_fallback: bool = True,
    ) -> None:
        """
        Initialize the multi-vector visual encoder.

        Args:
            model_name: HuggingFace model identifier (e.g. vidore/colqwen2.5-v0.2).
            device: 'auto', 'cuda', 'mps', or 'cpu'.
            dtype: torch.dtype or string ('bfloat16', 'float16', 'float32').
            low_vram_fallback: If True, falls back to colpali-v1.3 on CUDA OOM.
        """
        self.model_name = model_name
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.low_vram_fallback = low_vram_fallback
        self.embedding_dim: int = 128

        self.model: Any = None
        self.processor: Any = None
        self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> None:
        """Load the model and processor with error handling and fallback."""
        if torch is None:
            logger.warning("torch is not installed; running in mock/stub mode.")
            return

        try:
            self._init_engine(model_name)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() and self.low_vram_fallback:
                logger.warning(
                    "CUDA OOM loading %s. Falling back to lower-VRAM model %s.",
                    model_name,
                    self.LOW_VRAM_FALLBACK,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model_name = self.LOW_VRAM_FALLBACK
                self._init_engine(self.LOW_VRAM_FALLBACK)
            else:
                raise

    def _init_engine(self, model_name: str) -> None:
        """Instantiate model and processor via colpali-engine or transformers."""
        name_lower = model_name.lower()
        logger.info(
            "Loading multi-vector model '%s' on %s (%s)...",
            model_name,
            self.device,
            self.dtype,
        )

        try:
            if "colqwen" in name_lower:
                from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

                self.model = ColQwen2_5.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                ).to(self.device)
                self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
            elif "colpali" in name_lower:
                from colpali_engine.models import ColPali, ColPaliProcessor

                self.model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                ).to(self.device)
                self.processor = ColPaliProcessor.from_pretrained(model_name)
            else:
                # Generic fallback using colpali Auto classes or transformers
                try:
                    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

                    self.model = ColQwen2_5.from_pretrained(
                        model_name,
                        torch_dtype=self.dtype,
                    ).to(self.device)
                    self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
                except Exception:
                    from transformers import AutoModel, AutoProcessor

                    self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=self.dtype,
                    ).to(self.device)
                    self.processor = AutoProcessor.from_pretrained(model_name)
        except ImportError as e:
            logger.warning(
                "colpali_engine not found or failed to import (%s). "
                "VisualEncoder loaded without active weights.",
                e,
            )
            return

        self.model.eval()
        logger.info("Successfully loaded %s on %s.", model_name, self.device)

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        """
        Encode a list of PIL Images into multi-vector patch embeddings.

        Args:
            images: List of PIL Image instances representing document pages.
            batch_size: Inference batch size.

        Returns:
            List of numpy arrays, each of shape (N_patches, 128), dtype np.float32.
        """
        if not images:
            return []

        if self.model is None or self.processor is None:
            logger.warning(
                "VisualEncoder model/processor not loaded (running stub/lightweight mode). "
                "Generating synthetic 128-d multi-vector embeddings for %d images.",
                len(images),
            )
            # Produce 10 synthetic patch vectors per image normalized to unit length
            rng = np.random.default_rng(seed=42)
            stub_embeddings: List[np.ndarray] = []
            for _ in images:
                vecs = rng.standard_normal((10, self.embedding_dim), dtype=np.float32)
                vecs = self._l2_normalize(vecs)
                stub_embeddings.append(vecs)
            return stub_embeddings

        embeddings_list: List[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch_raw = images[i : i + batch_size]
            batch_imgs = [
                img.convert("RGB") if img.mode != "RGB" else img
                for img in batch_raw
            ]

            batch_inputs = self.processor.process_images(batch_imgs)
            batch_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                emb_tensor = self._extract_embeddings(outputs)

                attn_mask = batch_inputs.get("attention_mask", None)
                for j in range(emb_tensor.shape[0]):
                    sample_emb = emb_tensor[j]
                    if attn_mask is not None and j < attn_mask.shape[0]:
                        mask_j = attn_mask[j].bool()
                        if mask_j.sum() > 0 and mask_j.shape[0] == sample_emb.shape[0]:
                            sample_emb = sample_emb[mask_j]

                    # Normalize vectors to unit length for cosine/MaxSim similarity
                    sample_np = sample_emb.detach().cpu().to(torch.float32).numpy()
                    sample_np = self._l2_normalize(sample_np)
                    embeddings_list.append(sample_np)

        return embeddings_list

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        """
        Encode a list of text queries into multi-vector token embeddings.

        Args:
            queries: List of query strings.
            batch_size: Inference batch size.

        Returns:
            List of numpy arrays, each of shape (N_tokens, 128), dtype np.float32.
        """
        if not queries:
            return []

        if self.model is None or self.processor is None:
            logger.warning(
                "VisualEncoder model/processor not loaded (running stub/lightweight mode). "
                "Generating synthetic 128-d multi-vector query embeddings for %d queries.",
                len(queries),
            )
            rng = np.random.default_rng(seed=42)
            stub_embeddings: List[np.ndarray] = []
            for _ in queries:
                vecs = rng.standard_normal((5, self.embedding_dim), dtype=np.float32)
                vecs = self._l2_normalize(vecs)
                stub_embeddings.append(vecs)
            return stub_embeddings

        embeddings_list: List[np.ndarray] = []

        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            batch_inputs = self.processor.process_queries(batch_q)
            batch_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                emb_tensor = self._extract_embeddings(outputs)

                attn_mask = batch_inputs.get("attention_mask", None)
                for j in range(emb_tensor.shape[0]):
                    sample_emb = emb_tensor[j]
                    if attn_mask is not None and j < attn_mask.shape[0]:
                        mask_j = attn_mask[j].bool()
                        if mask_j.sum() > 0 and mask_j.shape[0] == sample_emb.shape[0]:
                            sample_emb = sample_emb[mask_j]

                    sample_np = sample_emb.detach().cpu().to(torch.float32).numpy()
                    sample_np = self._l2_normalize(sample_np)
                    embeddings_list.append(sample_np)

        return embeddings_list

    @staticmethod
    def _extract_embeddings(outputs: Any) -> Any:
        """Extract embeddings tensor from model output."""
        if hasattr(outputs, "embeddings"):
            return outputs.embeddings
        elif isinstance(outputs, torch.Tensor):
            return outputs
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2-normalize multi-vector embeddings along the feature dimension."""
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.maximum(norm, eps)

    @staticmethod
    def compute_maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """
        Compute late-interaction MaxSim similarity score:
            S(q, p) = sum_{i=1}^{N_q} max_{j in {1..N_p}} <E_q^(i), E_p^(j)>

        Args:
            query_emb: (N_q, 128) array of query token embeddings.
            doc_emb: (N_p, 128) array of document patch embeddings.

        Returns:
            Scalar MaxSim similarity score.
        """
        if query_emb.size == 0 or doc_emb.size == 0:
            return 0.0

        # Cosine similarity matrix between all query tokens and patch embeddings
        sim_matrix = np.matmul(query_emb, doc_emb.T)  # (N_q, N_p)
        max_per_query_token = np.max(sim_matrix, axis=1)  # (N_q,)
        return float(np.sum(max_per_query_token))
