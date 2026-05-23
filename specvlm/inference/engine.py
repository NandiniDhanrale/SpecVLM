"""
SpecVLM Inference Engine

The central orchestrator for VLM inference. Coordinates:
- Model loading and lifecycle management
- Request batching and scheduling
- Streaming response generation
- Performance metric collection
- Memory optimization

Architecture:
┌─────────────────────────────────────────────────────────┐
│                    InferenceEngine                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │  Vision  │  │   LLM    │  │     KV Cache          │   │
│  │  Tower   │  │  Decoder │  │     Manager           │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │  Visual  │  │  Token   │  │     Profiler          │   │
│  │  Encoder │  │  Sampler │  │     & Metrics         │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Backend: vLLM | Transformers | SGLang                   │
└─────────────────────────────────────────────────────────┘

Production design:
- Async-first: all public APIs are async
- Backend-agnostic: swap vLLM / Transformers / SGLang
- Memory-aware: tracks GPU memory and triggers GC when needed
- Instrumented: Prometheus metrics on all critical paths
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Optional

import torch

from specvlm.config.settings import settings
from specvlm.inference.visual_encoder import VisualEncoder
from specvlm.models.base_vlm import VLMInput, VLMOutput
from specvlm.models.draft_model import DraftModel
from specvlm.models.target_model import TargetModel

logger = logging.getLogger(__name__)


class EngineMode(Enum):
    """Engine operating modes."""
    BASELINE = "baseline"  # Standard autoregressive decoding
    SPECULATIVE = "speculative"  # Speculative decoding with draft model


class EngineBackend(Enum):
    """Supported inference backends."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    SGLANG = "sglang"


@dataclass
class EngineConfig:
    """Engine configuration."""
    mode: EngineMode = EngineMode.BASELINE
    backend: EngineBackend = EngineBackend.VLLM
    draft_model_id: Optional[str] = None
    target_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_batch_size: int = 64
    max_waiting_ms: int = 10  # Max time to wait for batch filling
    enable_profiling: bool = False
    profile_dir: Optional[str] = None


class InferenceEngine:
    """
    Main inference engine for SpecVLM.

    Features:
    - Async/sync generation with streaming
    - Speculative decoding mode
    - Visual embedding computation and caching
    - Performance instrumentation
    - Memory optimization
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.mode = self.config.mode

        # Model instances (lazy-loaded)
        self._draft_model: Optional[DraftModel] = None
        self._target_model: Optional[TargetModel] = None

        # Visual encoder (shared between draft and target)
        self.visual_encoder = VisualEncoder()

        # Performance tracking
        self.metrics = {
            "requests_total": 0,
            "requests_in_flight": 0,
            "tokens_generated": 0,
            "total_inference_time_ms": 0.0,
            "total_prefill_time_ms": 0.0,
            "total_decode_time_ms": 0.0,
        }

        # Event loop for async operations
        self._loop = None
        self._loaded = False

    async def load(self) -> None:
        """Load models asynchronously."""
        logger.info(f"Loading InferenceEngine in {self.mode.value} mode")

        loop = asyncio.get_event_loop()

        if self.mode == EngineMode.SPECULATIVE:
            if self.config.draft_model_id:
                await loop.run_in_executor(None, self._load_draft_model)
            if self.config.target_model_id:
                await loop.run_in_executor(None, self._load_target_model)
        else:
            if self.config.target_model_id:
                await loop.run_in_executor(None, self._load_target_model)

        self._loaded = True
        logger.info("InferenceEngine loaded successfully")

    def _load_draft_model(self) -> None:
        """Load draft model (CPU-blocking operation)."""
        logger.info(f"Loading draft model: {self.config.draft_model_id}")
        self._draft_model = DraftModel(
            model_id=self.config.draft_model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            backend=self.config.backend.value,
        )
        self._draft_model.load()
        logger.info(
            f"Draft model loaded: {self.config.draft_model_id} "
            f"({self._draft_model.num_parameters()['total'] / 1e9:.2f}B params)"
        )

    def _load_target_model(self) -> None:
        """Load target model (CPU-blocking operation)."""
        logger.info(f"Loading target model: {self.config.target_model_id}")
        self._target_model = TargetModel(
            model_id=self.config.target_model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            backend=self.config.backend.value,
        )
        self._target_model.load()
        logger.info(
            f"Target model loaded: {self.config.target_model_id} "
            f"({self._target_model.num_parameters()['total'] / 1e9:.2f}B params)"
        )

    async def generate(
        self, inputs: VLMInput
    ) -> VLMOutput:
        """
        Synchronous generation (blocking call, but async interface).

        In BASELINE mode: standard autoregressive decoding.
        In SPECULATIVE mode: uses speculative decoding.
        """
        if not self._loaded:
            await self.load()

        self.metrics["requests_total"] += 1
        self.metrics["requests_in_flight"] += 1
        start_time = time.time()

        try:
            if self.mode == EngineMode.SPECULATIVE and self._draft_model:
                result = await self._generate_speculative(inputs)
            else:
                result = await self._generate_baseline(inputs)

            elapsed_ms = (time.time() - start_time) * 1000
            result.ttft_ms = elapsed_ms
            result.tokens_per_second = (
                result.num_output_tokens / (elapsed_ms / 1000)
                if elapsed_ms > 0
                else 0
            )

            self.metrics["tokens_generated"] += result.num_output_tokens
            self.metrics["total_inference_time_ms"] += elapsed_ms

            return result
        finally:
            self.metrics["requests_in_flight"] -= 1

    async def generate_stream(
        self, inputs: VLMInput
    ) -> AsyncGenerator[VLMOutput, None]:
        """
        Streaming generation — yields tokens as they're produced.

        In BASELINE mode: yields each token from standard decoding.
        In SPECULATIVE mode: yields groups of accepted tokens.
        """
        if not self._loaded:
            await self.load()

        self.metrics["requests_total"] += 1
        self.metrics["requests_in_flight"] += 1

        try:
            if self.mode == EngineMode.SPECULATIVE and self._draft_model:
                async for chunk in self._generate_stream_speculative(inputs):
                    yield chunk
            else:
                async for chunk in self._generate_stream_baseline(inputs):
                    yield chunk
        finally:
            self.metrics["requests_in_flight"] -= 1

    async def _generate_baseline(self, inputs: VLMInput) -> VLMOutput:
        """Standard autoregressive generation without speculation."""
        if self._target_model is None:
            raise RuntimeError("Target model not loaded")

        # Encode images if provided
        visual_embeds = None
        if inputs.image_paths and self.visual_encoder:
            visual_embeds = await asyncio.get_event_loop().run_in_executor(
                None, self.visual_encoder.encode_images, inputs.image_paths
            )

        # Store visual embeddings for potential reuse
        inputs.visual_embeds = visual_embeds

        return self._target_model.generate(inputs)

    async def _generate_stream_baseline(
        self, inputs: VLMInput
    ) -> AsyncGenerator[VLMOutput, None]:
        """Streaming baseline generation."""
        if self._target_model is None:
            raise RuntimeError("Target model not loaded")

        async for chunk in self._target_model.generate_stream(inputs):
            self.metrics["tokens_generated"] += 1
            yield chunk

    async def _generate_speculative(self, inputs: VLMInput) -> VLMOutput:
        """Speculative decoding generation."""
        from specvlm.inference.speculative_decoder import SpeculativeDecoder

        decoder = SpeculativeDecoder(
            draft_model=self._draft_model,
            target_model=self._target_model,
        )
        return await decoder.generate(inputs)

    async def _generate_stream_speculative(
        self, inputs: VLMInput
    ) -> AsyncGenerator[VLMOutput, None]:
        """Streaming speculative decoding."""
        from specvlm.inference.speculative_decoder import SpeculativeDecoder

        decoder = SpeculativeDecoder(
            draft_model=self._draft_model,
            target_model=self._target_model,
        )
        async for chunk in decoder.generate_stream(inputs):
            self.metrics["tokens_generated"] += chunk.num_output_tokens
            yield chunk

    def get_stats(self) -> dict:
        """Return engine performance statistics."""
        total_requests = self.metrics["requests_total"]
        total_time = self.metrics["total_inference_time_ms"] / 1000

        stats = {
            "mode": self.mode.value,
            "backend": self.config.backend.value,
            "loaded": self._loaded,
            "requests_total": total_requests,
            "requests_in_flight": self.metrics["requests_in_flight"],
            "tokens_generated": self.metrics["tokens_generated"],
            "avg_tokens_per_request": (
                self.metrics["tokens_generated"] / total_requests
                if total_requests > 0
                else 0
            ),
            "total_runtime_seconds": total_time,
        }

        # Add model-specific stats
        if self._target_model:
            stats["target_model"] = self._target_model.get_kv_cache_stats()

        if self._draft_model:
            stats["draft_model"] = self._draft_model.get_kv_cache_stats()

        # Add GPU memory stats
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }

        return stats

    async def warmup(self) -> None:
        """
        Warm up the engine by running a small inference.

        This loads CUDA kernels and avoids cold-start latency on the
        first real request. Critical for production serving.
        """
        logger.info("Warming up inference engine...")

        warmup_input = VLMInput(
            prompt="Describe what you see in this image.",
            max_new_tokens=10,
            temperature=0.0,  # Deterministic for warmup
        )

        try:
            await self.generate(warmup_input)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down inference engine...")

        if self._target_model and hasattr(self._target_model, "_vllm_engine"):
            del self._target_model._vllm_engine
        if self._draft_model and hasattr(self._draft_model, "_vllm_engine"):
            del self._draft_model._vllm_engine

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._loaded = False
        logger.info("Inference engine shutdown complete")
