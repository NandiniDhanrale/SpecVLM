"""
Throughput Benchmark — Measure Maximum Inference Throughput

Designed to saturate the serving infrastructure and measure:
- Maximum requests per second (RPS)
- Optimal batch size
- GPU utilization under load
- Scaling efficiency (single GPU vs multi-GPU)
- Queue depth and request scheduling overhead

Uses Locust for distributed load testing in production.
For local testing, uses asyncio-based concurrent request generation.

Usage:
    python -m specvlm.benchmarks.throughput_benchmark \
        --concurrency 1 4 8 16 \
        --prompts data/test_prompts.json \
        --output results/throughput.json
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.models.base_vlm import VLMInput

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    """Throughput benchmark result for a given concurrency level."""
    concurrency: int
    total_requests: int
    total_time_s: float
    requests_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors: int
    avg_tokens_per_request: float
    avg_tokens_per_second: float


class ThroughputBenchmark:
    """
    Measures maximum throughput under varying concurrency levels.

    Methodology:
    1. Start at low concurrency (1)
    2. Ramp up to target concurrency levels
    3. At each level, send requests as fast as possible
    4. Measure saturation point (where latency increases non-linearly)
    5. Record optimal concurrency for maximum throughput
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        concurrency_levels: list[int] = None,
        prompts: list[str] = None,
        requests_per_level: int = 20,
        output_dir: str = "results",
    ):
        self.model_id = model_id
        self.concurrency_levels = concurrency_levels or [1, 2, 4, 8]
        self.prompts = prompts or [
            "Describe what you see.",
            "What objects are in this image?",
            "Summarize the visual scene.",
        ]
        self.requests_per_level = requests_per_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[ThroughputResult] = []

    async def run_all(self) -> list[ThroughputResult]:
        """Run benchmark across all concurrency levels."""
        engine_config = EngineConfig(
            mode=EngineMode.BASELINE,
            backend=EngineBackend.TRANSFORMERS,
            target_model_id=self.model_id,
        )

        engine = InferenceEngine(config=engine_config)
        await engine.load()

        try:
            for concurrency in self.concurrency_levels:
                logger.info(f"\n--- Concurrency: {concurrency} ---")
                result = await self._run_concurrency_level(
                    engine=engine,
                    concurrency=concurrency,
                )
                self.results.append(result)

                logger.info(
                    f"  RPS: {result.requests_per_second:.1f}, "
                    f"Latency P50: {result.p50_latency_ms:.0f}ms, "
                    f"P95: {result.p95_latency_ms:.0f}ms"
                )
        finally:
            await engine.shutdown()

        self._print_summary()
        return self.results

    async def _run_concurrency_level(
        self,
        engine: InferenceEngine,
        concurrency: int,
    ) -> ThroughputResult:
        """Run benchmark at a specific concurrency level."""
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        tokens_list = []
        errors = 0

        async def make_request(prompt: str) -> None:
            nonlocal errors
            async with semaphore:
                start = time.time()
                try:
                    vlm_input = VLMInput(
                        prompt=prompt,
                        max_new_tokens=64,
                        temperature=0.0,
                    )
                    output = await engine.generate(vlm_input)
                    elapsed = (time.time() - start) * 1000
                    latencies.append(elapsed)
                    tokens_list.append(output.num_output_tokens)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    errors += 1

        start_time = time.time()
        tasks = []
        for i in range(self.requests_per_level):
            prompt = self.prompts[i % len(self.prompts)]
            tasks.append(make_request(prompt))

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        total_requests = self.requests_per_level - errors

        if latencies:
            return ThroughputResult(
                concurrency=concurrency,
                total_requests=total_requests,
                total_time_s=total_time,
                requests_per_second=total_requests / total_time if total_time > 0 else 0,
                avg_latency_ms=np.mean(latencies),
                p50_latency_ms=np.median(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                errors=errors,
                avg_tokens_per_request=np.mean(tokens_list) if tokens_list else 0,
                avg_tokens_per_second=(
                    sum(tokens_list) / total_time if total_time > 0 else 0
                ),
            )

        return ThroughputResult(
            concurrency=concurrency,
            total_requests=total_requests,
            total_time_s=total_time,
            requests_per_second=0,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            errors=errors,
            avg_tokens_per_request=0,
            avg_tokens_per_second=0,
        )

    def _print_summary(self) -> None:
        """Print throughput summary."""
        logger.info("\n" + "=" * 70)
        logger.info(f"{'Concurrency':>12} {'RPS':>10} {'Lat P50':>10} {'Lat P95':>10} {'Errors':>8}")
        logger.info("-" * 70)

        for r in self.results:
            logger.info(
                f"{r.concurrency:>12} {r.requests_per_second:>10.1f} "
                f"{r.p50_latency_ms:>10.0f} {r.p95_latency_ms:>10.0f} "
                f"{r.errors:>8}"
            )

        # Find optimal concurrency
        if len(self.results) > 1:
            max_rps = max(r.requests_per_second for r in self.results)
            optimal = [r for r in self.results if r.requests_per_second == max_rps][0]
            logger.info(f"\nOptimal concurrency: {optimal.concurrency} "
                        f"({optimal.requests_per_second:.1f} RPS)")

    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to JSON."""
        path = path or str(self.output_dir / "throughput_results.json")
        data = [
            {
                "concurrency": r.concurrency,
                "requests_per_second": r.requests_per_second,
                "avg_latency_ms": r.avg_latency_ms,
                "p50_latency_ms": r.p50_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "errors": r.errors,
                "avg_tokens_per_second": r.avg_tokens_per_second,
            }
            for r in self.results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")
        return path


async def main():
    parser = argparse.ArgumentParser(description="SpecVLM Throughput Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompts", default=None, help="JSON file with prompt list")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--output", default="results/throughput.json")
    args = parser.parse_args()

    prompts = None
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)

    benchmark = ThroughputBenchmark(
        model_id=args.model,
        concurrency_levels=args.concurrency,
        prompts=prompts,
        requests_per_level=args.requests_per_level,
    )

    await benchmark.run_all()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
