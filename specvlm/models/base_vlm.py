"""
Base VLM (Vision-Language Model) Interface

Abstract base class defining the contract for all VLM implementations.
Supports multiple backends: vLLM, HuggingFace Transformers, SGLang.

The key abstraction layers in a VLM:
1. Vision Tower (ViT/CLIP) — encodes images into visual embeddings
2. Projection Layer — maps visual embeddings into LLM embedding space
3. Language Model — autoregressive decoder that consumes text + visual tokens
4. KV Cache — stores key/value tensors for attention reuse across decode steps
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import torch


@dataclass
class VLMInput:
    """Structured input to a VLM model."""

    # Text prompt (without image tokens)
    prompt: str

    # Image path(s) — can be local file paths or URLs
    image_paths: list[str] = field(default_factory=list)

    # Pre-computed visual embeddings (for cache reuse)
    visual_embeds: Optional[torch.Tensor] = None

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_strings: list[str] = field(default_factory=lambda: ["<|end|>", "<|im_end|>"])

    # Speculative decoding parameters
    num_speculative_tokens: int = 5

    # Request metadata
    request_id: str = ""
    priority: int = 0
    client_id: str = ""


@dataclass
class VLMOutput:
    """Structured output from a VLM model."""

    # Generated text
    text: str = ""

    # Token-level details
    tokens: list[int] = field(default_factory=list)
    token_texts: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    # Performance metrics
    ttft_ms: float = 0.0  # Time to first token (milliseconds)
    tokens_per_second: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0

    # KV cache stats
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0

    # Speculative decoding stats
    draft_tokens_generated: int = 0
    draft_tokens_accepted: int = 0
    speculation_acceptance_rate: float = 0.0

    # Raw logits (for analysis)
    draft_logits: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None

    # Visual embeddings (for cache reuse in subsequent calls)
    visual_embeds: Optional[torch.Tensor] = None


class BaseVLM(ABC):
    """
    Abstract base class for all VLM models.

    Every VLM must implement:
    - load(): Load model weights onto device
    - generate(): Synchronous text generation
    - generate_stream(): Async streaming generation
    - encode_image(): Convert image → visual embeddings
    - get_kv_cache_stats(): Report cache utilization

    Design philosophy:
    - All implementations share the same interface so the speculative
      decoder can swap draft/target models transparently
    - CUDA graphs are used for the decode phase to reduce kernel launch overhead
    - Memory pre-allocation minimizes fragmentation during long-running serving
    """

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.model = None
        self.processor = None
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights, processor, and move to device."""

    @abstractmethod
    def generate(self, inputs: VLMInput) -> VLMOutput:
        """Synchronous generation."""

    @abstractmethod
    async def generate_stream(self, inputs: VLMInput) -> AsyncGenerator[VLMOutput, None]:
        """Async streaming generation — yields tokens as they're produced."""

    @abstractmethod
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image into visual embeddings.

        Returns shape: (1, num_visual_tokens, hidden_dim)
        This is the output of the vision tower + projection layer.
        """

    @abstractmethod
    def get_kv_cache_stats(self) -> dict:
        """Return current KV cache utilization statistics."""

    @abstractmethod
    def get_memory_stats(self) -> dict:
        """Return GPU memory allocation and utilization."""

    def num_parameters(self) -> int:
        """Count total and trainable parameters."""
        if self.model is None:
            return 0
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def estimate_inference_memory(self, batch_size: int = 1, seq_len: int = 2048) -> float:
        """
        Estimate GPU memory required for inference in GB.

        Factors:
        - Model weights (FP16: 2 bytes/param, FP32: 4 bytes/param)
        - KV cache: 2 * num_layers * num_heads * head_dim * seq_len * 2 bytes * batch_size
        - Activations: depends on batch_size and seq_len
        - Overhead: CUDA context, intermediate buffers
        """
        if self.model is None:
            return 0.0
        n_params = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 2 if self.dtype == torch.bfloat16 else 4
        weights_gb = n_params * bytes_per_param / 1e9

        # Estimate KV cache size
        if hasattr(self.model, "config"):
            cfg = self.model.config
            n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 32))
            n_heads = getattr(cfg, "num_attention_heads", 32)
            head_dim = getattr(cfg, "hidden_size", 4096) // n_heads
            kv_cache_gb = (
                2  # K and V
                * n_layers
                * n_heads
                * head_dim
                * seq_len
                * batch_size
                * 2  # FP16 bytes
                / 1e9
            )
        else:
            kv_cache_gb = 0.0

        # Activations: roughly 10-20% of weights for inference
        activations_gb = weights_gb * 0.15
        overhead_gb = 0.5  # CUDA context, etc.

        return weights_gb + kv_cache_gb + activations_gb + overhead_gb

    def optimize_memory(self) -> None:
        """
        Apply memory optimization techniques:
        - Clear CUDA cache
        - Enable gradient checkpointing (if training)
        - Optimize attention for inference
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
