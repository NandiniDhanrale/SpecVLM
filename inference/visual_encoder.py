"""
Visual Encoder — Vision Pipeline for VLMs

The visual encoder extracts visual embeddings from images using a
vision tower (typically CLIP ViT or SigLIP). These embeddings are
then projected into the LLM's embedding space.

This is a CRITICAL optimization target because:
1. Vision encoding is ~30% of total inference time for VLMs
2. Visual tokens dominate the prefix length (576 tokens vs ~20 text tokens)
3. The vision tower output is IDENTICAL for draft and target models
   when they share the same vision backbone
4. Visual embeddings can be PRE-COMPUTED and CACHED

Architecture:
┌──────────┐    ┌──────────────┐    ┌────────────────┐
│  Image   │───▶│  Vision Tower │───▶│  Projection    │
│ Input    │    │  (ViT/CLIP)   │    │  Layer (MLP)   │
└──────────┘    └──────────────┘    └────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Visual Embedding │
                                    │ Cache (Redis)    │
                                    └──────────────────┘

Production considerations:
- Cache visual embeddings keyed by (model_id, image_hash)
- Use 4-bit quantized ViT for draft model vision tower
- Precompute embeddings during upload (async) to hide latency
- Use tensor parallelism for vision tower when model > 7B
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VisualEmbeddingCache:
    """
    Cache for pre-computed visual embeddings.

    In production, this would be backed by Redis/Memcached with
    TTL-based eviction. For development, uses in-memory dict.
    """

    max_size: int = 1000
    _cache: dict = field(default_factory=dict)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)

    def set(self, key: str, value: torch.Tensor) -> None:
        if len(self._cache) >= self.max_size:
            # Simple LRU: remove first item
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


class VisualEncoder:
    """
    Handles image → visual embedding pipeline.

    Supports:
    - Multiple vision tower architectures (CLIP, SigLIP, etc.)
    - Image preprocessing (resize, normalize, etc.)
    - Shared embedding extraction for draft/target models
    - Embedding caching with content-addressable keys
    - Mixed precision encoding
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_size: int = 1000,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        # Vision tower (lazy-loaded from the first model that needs it)
        self._vision_tower = None
        self._processor = None

        # Embedding cache
        self.cache = VisualEmbeddingCache(max_size=cache_size)

        # Performance tracking
        self.total_encodings = 0
        self.cache_hits = 0
        self.total_encoding_time_ms = 0.0

    def _get_image_hash(self, image_path: str) -> str:
        """Compute content-addressable hash for an image file."""
        try:
            with open(image_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            return f"img:{file_hash}"
        except Exception:
            return f"img:{os.path.basename(image_path)}"

    def encode_image(
        self,
        image_path: str,
        model_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Encode a single image into visual embeddings.

        Args:
            image_path: Path or URL to image
            model_id: Model identifier (for cache key composition)
            use_cache: Whether to check/update cache

        Returns:
            Tensor of shape (1, num_visual_tokens, hidden_dim)

        The encoding pipeline:
        1. Load and preprocess image (resize, normalize)
        2. Forward through vision tower (ViT)
        3. Apply projection layer to LLM embedding space
        4. Return visual embeddings
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{model_id}:{self._get_image_hash(image_path)}" if model_id else self._get_image_hash(image_path)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                logger.debug(f"Visual embedding cache HIT for {cache_key}")
                return cached

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        visual_embeds = self._encode_pil_image(image)

        # Update cache
        if use_cache:
            self.cache.set(cache_key, visual_embeds)

        elapsed_ms = (time.time() - start_time) * 1000
        self.total_encodings += 1
        self.total_encoding_time_ms += elapsed_ms

        return visual_embeds

    def encode_images(
        self,
        image_paths: list[str],
        model_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Encode multiple images. Returns concatenated visual embeddings.

        Multiple images are concatenated along the sequence dimension:
        shape: (1, total_visual_tokens, hidden_dim)
        """
        all_embeds = []
        for path in image_paths:
            embeds = self.encode_image(path, model_id, use_cache)
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=1) if all_embeds else torch.empty(0)

    def _encode_pil_image(self, image) -> torch.Tensor:
        """
        Encode a PIL image through the vision pipeline.

        The actual vision tower is loaded lazily from the processor
        attached to the first model that uses this encoder.
        """
        if self._processor is None or self._vision_tower is None:
            raise RuntimeError(
                "Vision tower not initialized. Call attach_model() first."
            )

        # Preprocess image for ViT
        inputs = self._processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Forward through vision tower
        with torch.no_grad():
            if hasattr(self._vision_tower, "forward"):
                # Standard ViT forward
                vision_outputs = self._vision_tower(
                    inputs.pixel_values.to(dtype=self.dtype),
                    output_hidden_states=True,
                )
                # Use intermediate features (typically layer -2)
                visual_embeds = vision_outputs.hidden_states[-2]
            else:
                # Direct pixel_values → embeddings
                visual_embeds = self._vision_tower(
                    inputs.pixel_values.to(dtype=self.dtype)
                )

        return visual_embeds

    def attach_model(self, model) -> None:
        """
        Attach a model's vision tower and processor.

        This allows the visual encoder to share the same vision tower
        across draft and target models when they have the same architecture.
        """
        if hasattr(model, "vision_tower"):
            self._vision_tower = model.vision_tower
        elif hasattr(model, "get_vision_embeds"):
            # Generic interface
            pass

        if hasattr(model, "processor"):
            self._processor = model.processor

    @property
    def cache_stats(self) -> dict:
        """Return cache performance statistics."""
        return {
            "total_encodings": self.total_encodings,
            "cache_hits": self.cache_hits,
            "cache_miss_rate": (
                1 - (self.cache_hits / self.total_encodings)
                if self.total_encodings > 0
                else 0
            ),
            "avg_encoding_time_ms": (
                self.total_encoding_time_ms / self.total_encodings
                if self.total_encodings > 0
                else 0
            ),
            "cache_size": len(self.cache._cache),
        }
