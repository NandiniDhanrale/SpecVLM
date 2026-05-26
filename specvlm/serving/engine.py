from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .schemas import GenerateIn, MetricsOut


@dataclass
class StreamEvent:
    text: str
    delta: str
    metrics: MetricsOut


class BaseStreamingEngine:
    async def stream(self, req: GenerateIn) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


class MockStreamingEngine(BaseStreamingEngine):
    async def stream(self, req: GenerateIn) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        full = ""
        accepted = 0
        proposed = 0

        # Cheap "tokenizer": stream space-delimited chunks for demo.
        demo_text = (
            "This is a mock SpecVLM stream. "
            "Replace SPECVLM_ENGINE=mock with SPECVLM_ENGINE=vllm to wire vLLM. "
            "Streaming tokens and speculative metrics in real time."
        )
        chunks = [c + " " for c in demo_text.split(" ")]
        for chunk in chunks[: req.sampling_params.max_tokens]:
            await asyncio.sleep(0.03)
            full += chunk
            delta = chunk
            proposed += 1
            # Pretend the draft acceptance rate improves over time.
            accepted += 1 if proposed % 5 != 0 else 0
            elapsed = max(1e-6, time.perf_counter() - start)
            metrics = MetricsOut(
                accepted_draft_tokens=accepted,
                proposed_draft_tokens=proposed,
                acceptance_rate=(accepted / proposed) if proposed else 0.0,
                tokens_per_second=proposed / elapsed,
                t_ms=int(elapsed * 1000),
            )
            yield StreamEvent(text=full, delta=delta, metrics=metrics)


class VllmStreamingEngine(BaseStreamingEngine):
    def __init__(self) -> None:
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.sampling_params import SamplingParams
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "vLLM is not installed (install with `pip install -e .[vllm]`)."
            ) from e

        model = os.environ.get("SPECVLM_MODEL", "")
        if not model:
            raise RuntimeError("Set SPECVLM_MODEL to your target model path/name.")

        speculative_config = os.environ.get("SPECVLM_SPEC_CONFIG")
        engine_args = AsyncEngineArgs(
            model=model,
            speculative_config=speculative_config,
        )
        self._SamplingParams = SamplingParams
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def stream(self, req: GenerateIn) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        request_id = req.request_id or f"req-{int(start * 1e9)}"

        sp = self._SamplingParams(
            max_tokens=req.sampling_params.max_tokens,
            temperature=req.sampling_params.temperature,
            top_p=req.sampling_params.top_p,
        )

        prev_text = ""
        generator = self._engine.generate(req.prompt, sp, request_id)
        async for request_output in generator:
            out = request_output.outputs[0]
            text = out.text
            delta = text[len(prev_text) :] if text.startswith(prev_text) else text
            prev_text = text

            accepted = int(
                getattr(request_output, "draft_tokens_accepted", None)
                or getattr(request_output, "accepted_draft_tokens", 0)
                or 0
            )
            proposed = int(
                getattr(request_output, "draft_tokens_proposed", None)
                or getattr(request_output, "proposed_draft_tokens", 0)
                or 0
            )
            elapsed = max(1e-6, time.perf_counter() - start)
            metrics = MetricsOut(
                accepted_draft_tokens=accepted,
                proposed_draft_tokens=proposed,
                acceptance_rate=(accepted / proposed) if proposed else 0.0,
                tokens_per_second=(
                    int(getattr(out, "num_generated_tokens", 0) or 0) / elapsed
                ),
                t_ms=int(elapsed * 1000),
            )
            yield StreamEvent(text=text, delta=delta, metrics=metrics)


def get_engine() -> BaseStreamingEngine:
    engine = os.environ.get("SPECVLM_ENGINE", "mock").strip().lower()
    if engine == "vllm":
        return VllmStreamingEngine()
    return MockStreamingEngine()
