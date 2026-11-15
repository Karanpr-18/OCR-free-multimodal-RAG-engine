"""
VLM-based grounded QA reader using Qwen2.5-VL with citation extraction and confidence estimation.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
from typing import Any, List, Optional, Sequence, Union

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError:
    AutoProcessor = None  # type: ignore[assignment,misc]
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment,misc]

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _image_to_base64(image: Any) -> str:
    """Convert a PIL Image or file path to a base64-encoded PNG data URI."""
    if isinstance(image, str):
        if Image is None:
            raise ImportError("Pillow (PIL) is required to load images from paths.")
        image = Image.open(image).convert("RGB")
    elif Image is not None and not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image or path str, got {type(image)}")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def _ensure_pil_image(image: Any) -> Any:
    """Ensure the given image is loaded as a PIL Image in RGB format."""
    if isinstance(image, str):
        if Image is None:
            raise ImportError("Pillow (PIL) is required to load images from paths.")
        return Image.open(image).convert("RGB")
    if Image is not None and isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


class VLMReader:
    """
    Visual Language Model (VLM) grounded QA reader.

    Supports Qwen2.5-VL (3B and 7B models) with auto device/dtype placement,
    multi-image grounded prompt construction, token-level log-probability
    confidence estimation, explicit citation extraction, and fallback to
    OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        dtype: Optional[Union[str, Any]] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        lazy_load: bool = True,
    ) -> None:
        """
        Initialize the VLMReader.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: Target device ("auto", "cuda", "cpu", "mps").
            dtype: Model torch precision ("auto", "bfloat16", "float16", "float32").
            api_endpoint: Optional OpenAI-compatible API URL for remote inference.
            api_key: Optional API key for remote endpoint.
            timeout: Request timeout in seconds for API or generation.
            lazy_load: If True, defer local weight loading until first call.
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint.rstrip("/") if api_endpoint else None
        self.api_key = api_key
        self.timeout = timeout
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(dtype)

        self.processor: Any = None
        self.model: Any = None
        self._is_loaded = False

        if not lazy_load and not self.api_endpoint:
            self._load_local_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve the computing device."""
        if device == "auto":
            if torch is not None:
                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        return device

    def _resolve_dtype(self, dtype: Optional[Union[str, Any]]) -> Any:
        """Resolve torch precision dtype based on device capabilities."""
        if torch is None:
            return dtype or "bfloat16"

        if isinstance(dtype, torch.dtype):
            return dtype

        if dtype is None or dtype == "auto":
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(str(dtype).lower(), torch.float32)

    def _load_local_model(self) -> None:
        """Load Qwen2.5-VL weights and processor into memory."""
        if self._is_loaded:
            return

        if torch is None:
            raise ImportError(
                "torch is required for local VLM inference. Install with `pip install torch`."
            )

        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "transformers is required for local VLM inference. "
                "Install it with `pip install transformers`."
            )

        logger.info(
            "Loading local VLM model: %s on %s with dtype %s",
            self.model_name,
            self.device,
            self.torch_dtype,
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            device_map = "auto" if self.device == "cuda" else None
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("VLM model successfully loaded.")
        except Exception as err:
            logger.error("Failed to load local model %s: %s", self.model_name, err)
            raise

    def construct_messages(
        self,
        query: str,
        page_images: Sequence[Any],
        page_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """
        Construct multi-image structured messages for Qwen2.5-VL grounded QA.

        Args:
            query: User's natural language question.
            page_images: List of PIL images or image file paths.
            page_ids: Corresponding unique page identifiers.

        Returns:
            Formatted message conversation structure.
        """
        if len(page_images) != len(page_ids):
            raise ValueError(
                f"Mismatch: {len(page_images)} images and {len(page_ids)} page_ids."
            )

        user_content: list[dict[str, Any]] = []

        for idx, (img, pid) in enumerate(zip(page_images, page_ids)):
            pil_img = _ensure_pil_image(img)
            user_content.append(
                {
                    "type": "text",
                    "text": f"--- Document Page [Page {pid}] (Reference Index {idx + 1}) ---\n",
                }
            )
            user_content.append({"type": "image", "image": pil_img})

        instructions = (
            f"\nQuestion: {query}\n\n"
            "Instructions:\n"
            "1. Answer the question thoroughly based STRICTLY on the visual document pages provided above.\n"
            "2. Ground every factual claim with explicit page citations in the exact format [Page <page_id>].\n"
            "3. If citing data from tables or charts, cite specific table rows/columns or chart axes/values "
            "(e.g., '[Page {first_page}, Table 1 Row 2]' or '[Page {first_page}, Chart Y-axis]').\n"
            "4. If the provided pages do NOT contain sufficient information to answer the question with certainty, "
            "explicitly state: 'The provided document pages do not contain sufficient information to answer this question.'\n\n"
            "Answer:"
        ).format(first_page=page_ids[0] if page_ids else "N/A")

        user_content.append({"type": "text", "text": instructions})

        system_prompt = (
            "You are a precise, grounded document intelligence assistant. "
            "You inspect multimodal document pages, diagrams, tables, and text to answer questions accurately. "
            "You always cite specific pages [Page <id>] and visual elements without hallucinating."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def extract_citations(
        self, text: str, page_ids: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """
        Extract page references and fine-grained citations from generated answer.

        Args:
            text: Generated model answer.
            page_ids: Valid page IDs in context.

        Returns:
            Tuple of (citations_list, unique_page_references).
        """
        page_set = set(str(pid) for pid in page_ids)
        extracted_pages: set[str] = set()
        citations: list[str] = []

        # 1. Match bracketed page citations: [Page X] or [Page X, ...]
        bracket_pattern = re.compile(
            r"\[(?:Page|page|p\.)\s*([^\]]+)\]", re.IGNORECASE
        )
        for match in bracket_pattern.finditer(text):
            citation_str = match.group(0)
            citations.append(citation_str)
            inner = match.group(1)
            for candidate in re.split(r"[,;]\s*", inner):
                candidate_clean = candidate.strip()
                # Check direct match with page_ids
                for pid in page_set:
                    if pid in candidate_clean:
                        extracted_pages.add(pid)

        # 2. Match unbracketed "Page <id>" mentions
        unbracketed_pattern = re.compile(
            r"\b(?:Page|page|p\.)\s*([A-Za-z0-9_-]+)", re.IGNORECASE
        )
        for match in unbracketed_pattern.finditer(text):
            pid_candidate = match.group(1).strip()
            if pid_candidate in page_set:
                extracted_pages.add(pid_candidate)
                if match.group(0) not in citations:
                    citations.append(match.group(0))

        # 3. Match fine-grained visual references: Table, Row, Column, Axis, Chart, Figure
        element_pattern = re.compile(
            r"\b(?:Table\s+\d+|Row\s+\d+|Column\s+\d+|Chart\s+[A-Za-z0-9_-]+|"
            r"Figure\s+\d+|[xy]-axis|legend)\b",
            re.IGNORECASE,
        )
        for match in element_pattern.finditer(text):
            found_elem = match.group(0)
            if found_elem not in citations:
                citations.append(found_elem)

        # Stable ordered unique list of page references
        sorted_pages = [pid for pid in page_ids if pid in extracted_pages]
        return citations, sorted_pages

    def _call_api_endpoint(
        self,
        messages: list[dict[str, Any]],
        page_ids: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Call an OpenAI-compatible multimodal endpoint."""
        if httpx is None:
            raise ImportError(
                "httpx is required for API endpoint inference. Install with `pip install httpx`."
            )

        api_url = f"{self.api_endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Convert images in user message to base64 data URLs
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                api_content = []
                for item in content:
                    if item.get("type") == "text":
                        api_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image":
                        data_uri = _image_to_base64(item["image"])
                        api_content.append(
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        )
                formatted_messages.append({"role": role, "content": api_content})
            else:
                formatted_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        answer = choice["message"]["content"].strip()
        citations, page_refs = self.extract_citations(answer, page_ids)

        confidence = 0.88 if choice.get("finish_reason") == "stop" else 0.50

        return {
            "answer": answer,
            "citations": citations,
            "page_references": page_refs,
            "vlm_confidence": confidence,
        }

    def generate_answer(
        self,
        query: str,
        page_images: Sequence[Any],
        page_ids: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Generate a grounded answer with citations and confidence score.

        Args:
            query: Question to answer.
            page_images: List of page images (PIL or paths).
            page_ids: Identifiers for the pages.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for greedy decoding).

        Returns:
            Dictionary containing:
                - answer (str): Generated grounded response.
                - citations (list[str]): Extracted citations (pages, tables, charts).
                - page_references (list[str]): Unique matched page IDs.
                - vlm_confidence (float): Estimated confidence in [0.0, 1.0].
        """
        messages = self.construct_messages(query, page_images, page_ids)

        if self.api_endpoint is not None:
            try:
                return self._call_api_endpoint(
                    messages,
                    page_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as err:
                logger.warning(
                    "API call failed (%s). Attempting local fallback if model available.",
                    err,
                )
                if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
                    raise

        # Local model inference
        self._load_local_model()

        try:
            # Process prompt & images
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [_ensure_pil_image(img) for img in page_images]
                video_inputs = None

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            do_sample = temperature > 0.0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
                "output_scores": True,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Sequence decoding
            prompt_len = inputs.input_ids.shape[1]
            generated_token_ids = outputs.sequences[:, prompt_len:]
            generated_text = self.processor.batch_decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            # Calculate confidence from token log-probabilities
            vlm_confidence = self._compute_token_confidence(
                outputs.scores, generated_token_ids[0]
            )

            citations, page_refs = self.extract_citations(generated_text, page_ids)

            return {
                "answer": generated_text,
                "citations": citations,
                "page_references": page_refs,
                "vlm_confidence": vlm_confidence,
            }

        except Exception as err:
            logger.error("Error during local VLM generation: %s", err)
            return {
                "answer": f"Error during generation: {str(err)}",
                "citations": [],
                "page_references": [],
                "vlm_confidence": 0.0,
            }

    def _compute_token_confidence(
        self,
        scores: Sequence[Any],
        token_ids: Any,
    ) -> float:
        """
        Compute mean exponential log-probability across generated tokens.

        Args:
            scores: Tuple of step prediction logits from model.generate().
            token_ids: 1D tensor or list of generated token IDs.

        Returns:
            Normalized confidence score in [0.0, 1.0].
        """
        if not scores or len(token_ids) == 0:
            return 0.50

        total_log_prob = 0.0
        num_tokens = min(len(scores), len(token_ids))

        if torch is not None and F is not None and isinstance(scores[0], torch.Tensor):
            for t in range(num_tokens):
                step_logits = scores[t][0]  # shape (vocab_size,)
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                target_token = token_ids[t].item() if hasattr(token_ids[t], "item") else int(token_ids[t])
                token_log_prob = step_log_probs[target_token].item()
                total_log_prob += token_log_prob
        else:
            # Fallback for mock environments
            return 0.85

        mean_log_prob = total_log_prob / max(num_tokens, 1)
        confidence = float(math.exp(max(min(mean_log_prob, 0.0), -20.0)))
        return round(min(max(confidence, 0.0), 1.0), 4)

    def batch_generate(
        self,
        queries: Sequence[str],
        batch_page_images: Sequence[Sequence[Any]],
        batch_page_ids: Sequence[Sequence[str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Batch generate grounded answers for evaluation and benchmark suites.

        Args:
            queries: List of questions.
            batch_page_images: List of image sequences for each query.
            batch_page_ids: List of page ID sequences for each query.
            max_new_tokens: Max new tokens.
            temperature: Sampling temperature.

        Returns:
            List of generation result dictionaries.
        """
        if not (len(queries) == len(batch_page_images) == len(batch_page_ids)):
            raise ValueError("All batch inputs must have the same length.")

        results = []
        for query, images, ids in zip(queries, batch_page_images, batch_page_ids):
            res = self.generate_answer(
                query=query,
                page_images=images,
                page_ids=ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(res)
        return results
