"""
Draft Model — Small, Fast VLM for Speculative Decoding

The draft model is a lightweight VLM (e.g., Qwen2-VL-2B or SmolVLM)
that rapidly proposes candidate tokens. It shares the same visual
encoding pipeline as the target model to avoid redundant computation.

Key design points:
- Uses the same vision tower architecture (or a distilled version)
- Runs at 2-5x the speed of the target model
- Lower precision (INT8/FP8) quantization for faster inference
- Shares KV cache prefix with target model for cache hits

In production (OpenAI/Baseten-style):
- Draft model runs on a separate, smaller GPU (e.g., L4 vs A100)
- Draft model uses INT4 quantization for maximum throughput
- Multiple draft model replicas feed into a single target model
"""

import time
from typing import AsyncGenerator, Optional

import torch

from specvlm.models.base_vlm import BaseVLM, VLMInput, VLMOutput


class DraftModel(BaseVLM):
    """
    Lightweight draft model for speculative decoding.

    Wraps a small VLM (2B-3B parameters) optimized for fast token generation.
    Supports multiple backends:
    - vLLM (preferred for production)
    - Transformers (for flexibility)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        backend: str = "vllm",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        quantization: Optional[str] = None,
    ):
        super().__init__(model_id, device, dtype)
        self.backend = backend
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization

        # Performance tracking
        self.total_tokens_generated = 0
        self.total_generation_time_ms = 0.0

    def load(self) -> None:
        """Load the draft model using the selected backend."""
        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        self.is_loaded = True

    def _load_vllm(self) -> None:
        """Load via vLLM — production-grade PagedAttention serving."""
        from vllm import LLM, SamplingParams

        # Configure vLLM engine for draft model
        self._vllm_engine = LLM(
            model=self.model_id,
            trust_remote_code=True,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=self.quantization,
            enforce_eager=False,
            max_num_seqs=256,
            # Enable prefix caching for shared visual embeddings
            enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams

    def _load_transformers(self) -> None:
        """Load via HuggingFace Transformers — more flexible but slower."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, inputs: VLMInput) -> VLMOutput:
        """Synchronous generation — used for batch preprocessing."""
        if self.backend == "vllm":
            return self._generate_vllm(inputs)
        else:
            return self._generate_transformers(inputs)

    def _generate_vllm(self, inputs: VLMInput) -> VLMOutput:
        """Generate using vLLM engine."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=inputs.temperature,
            top_p=inputs.top_p,
            top_k=inputs.top_k,
            max_tokens=inputs.max_new_tokens,
            stop=inputs.stop_strings,
        )

        # Build multimodal data
        multimodal_data = {}
        if inputs.image_paths:
            from vllm.multimodal import MultiModalData

            multimodal_data = {
                "image": inputs.image_paths,
            }

        start_time = time.time()
        outputs = self._vllm_engine.generate(
            inputs={"prompt": inputs.prompt, "multi_modal_data": multimodal_data}
            if multimodal_data
            else inputs.prompt,
            sampling_params=sampling_params,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        result = VLMOutput()
        for output in outputs:
            for out in output.outputs:
                result.text = out.text
                result.tokens = out.token_ids
                result.logprobs = out.cumulative_logprob
                result.num_output_tokens = len(out.token_ids)

        result.ttft_ms = elapsed_ms  # Approximate for sync mode
        result.tokens_per_second = result.num_output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        # Track performance
        self.total_tokens_generated += result.num_output_tokens
        self.total_generation_time_ms += elapsed_ms

        return result

    def _generate_transformers(self, inputs: VLMInput) -> VLMOutput:
        """Generate using Transformers pipeline."""
        from transformers import TextStreamer

        # Process image
        images = []
        if inputs.image_paths:
            from PIL import Image
            images = [Image.open(p) for p in inputs.image_paths]

        # Build conversation template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} if images else None,
                    {"type": "text", "text": inputs.prompt},
                ],
            }
        ]
        conversation = [c for c in conversation if c is not None]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs_processed = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs_processed,
                max_new_tokens=inputs.max_new_tokens,
                temperature=inputs.temperature,
                top_p=inputs.top_p,
                top_k=inputs.top_k,
                do_sample=inputs.temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        elapsed_ms = (time.time() - start_time) * 1000

        # Decode
        generated_ids = generated_ids[0][inputs_processed.input_ids.shape[1]:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)

        result = VLMOutput()
        result.text = text
        result.tokens = generated_ids.tolist() if hasattr(generated_ids, 'tolist') else generated_ids
        result.num_output_tokens = len(result.tokens)
        result.ttft_ms = elapsed_ms
        result.tokens_per_second = result.num_output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        return result

    async def generate_stream(self, inputs: VLMInput) -> AsyncGenerator[VLMOutput, None]:
        """Async streaming generation. Yields tokens as they're produced."""
        if self.backend == "vllm":
            async for chunk in self._generate_stream_vllm(inputs):
                yield chunk
        else:
            # Fall back to sync for transformers backend
            result = self.generate(inputs)
            yield result

    async def _generate_stream_vllm(self, inputs: VLMInput) -> AsyncGenerator[VLMOutput, None]:
        """Stream from vLLM engine."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=inputs.temperature,
            top_p=inputs.top_p,
            top_k=inputs.top_k,
            max_tokens=inputs.max_new_tokens,
            stop=inputs.stop_strings,
        )

        multimodal_data = {}
        if inputs.image_paths:
            multimodal_data = {"image": inputs.image_paths}

        first_token_time = None
        token_count = 0
        stream_start = time.time()

        async for request_output in self._vllm_engine.generate(
            inputs={"prompt": inputs.prompt, "multi_modal_data": multimodal_data}
            if multimodal_data
            else inputs.prompt,
            sampling_params=sampling_params,
            stream=True,
        ):
            if first_token_time is None and request_output.outputs:
                first_token_time = time.time()
                ttft_ms = (first_token_time - stream_start) * 1000

            for out in request_output.outputs:
                token_count += 1
                result = VLMOutput(
                    text=out.text,
                    tokens=out.token_ids,
                    ttft_ms=ttft_ms or 0.0,
                    num_output_tokens=token_count,
                )
                yield result

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode image using the vision tower."""
        from PIL import Image

        if self.backend == "transformers" and self.processor:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Forward through vision tower
                if hasattr(self.model, "vision_tower"):
                    visual_embeds = self.model.vision_tower(inputs.pixel_values)
                elif hasattr(self.model, "get_vision_embeds"):
                    visual_embeds = self.model.get_vision_embeds(inputs.pixel_values)
                else:
                    # Generic vision encoder forward
                    visual_embeds = self.model.get_vision_features(inputs.pixel_values)
            return visual_embeds
        else:
            # vLLM handles vision internally; this is a stub for cache sharing
            return torch.empty(0)

    def get_kv_cache_stats(self) -> dict:
        """Return KV cache statistics for the draft model."""
        stats = {
            "model": self.model_id,
            "backend": self.backend,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_second": (
                self.total_tokens_generated / (self.total_generation_time_ms / 1000)
                if self.total_generation_time_ms > 0
                else 0
            ),
        }

        if self.backend == "vllm":
            try:
                engine_stats = self._vllm_engine.llm_engine.get_cache_stats()
                stats.update(engine_stats)
            except Exception:
                pass

        return stats

    def get_memory_stats(self) -> dict:
        """Report GPU memory usage."""
        stats = {"device": self.device}
        if torch.cuda.is_available() and self.device != "cpu":
            device_id = torch.cuda.current_device()
            stats["allocated_gb"] = torch.cuda.memory_allocated(device_id) / 1e9
            stats["reserved_gb"] = torch.cuda.memory_reserved(device_id) / 1e9
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated(device_id) / 1e9
        return stats
