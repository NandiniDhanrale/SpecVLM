"""
Latency Benchmark — Measure TTFT, TPS, and Token-Level Latency

This benchmark measures the core latency metrics for VLM inference:
- TTFT (Time to First Token): How long before the first token appears
- ITL (Inter-Token Latency): Time between consecutive tokens
- TPS (Tokens Per Second): Overall throughput
- Speculative speedup: Baseline vs speculative decoding comparison

The benchmark generates detailed reports and visualizations for
comparing different configurations.

Usage:
    python -m specvlm.benchmarks.latency_benchmark \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --image data/images/test.jpg \
        --prompt "Describe this image" \
        --warmup 3 --iterations 10
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
from specvlm.monitoring.profiler import InferenceProfiler
from specvlm.models.base_vlm import VLMInput

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    ttft_ms: list[float] = field(default_factory=list)
    inter_token_latency_ms: list[float] = field(default_factory=list)
    tokens_per_second: list[float] = field(default_factory=list)
    total_time_ms: list[float] = field(default_factory=list)
    num_tokens: list[int] = field(default_factory=list)
    acceptance_rates: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        """Compute summary statistics."""
        return {
            "config": self.config_name,
            "ttft_ms": {
                "mean": np.mean(self.ttft_ms),
                "median": np.median(self.ttft_ms),
                "p95": np.percentile(self.ttft_ms, 95),
                "p99": np.percentile(self.ttft_ms, 99),
                "min": min(self.ttft_ms),
                "max": max(self.ttft_ms),
                "std": np.std(self.ttft_ms),
            },
            "tokens_per_second": {
                "mean": np.mean(self.tokens_per_second),
                "median": np.median(self.tokens_per_second),
                "p95": np.percentile(self.tokens_per_second, 95),
                "min": min(self.tokens_per_second),
                "max": max(self.tokens_per_second),
            },
            "inter_token_latency_ms": {
                "mean": np.mean(self.inter_token_latency_ms),
                "median": np.median(self.inter_token_latency_ms),
                "p95": np.percentile(self.inter_token_latency_ms, 95),
            },
            "total_time_ms": {
                "mean": np.mean(self.total_time_ms),
                "std": np.std(self.total_time_ms),
            },
            "avg_tokens_per_request": np.mean(self.num_tokens),
            "avg_acceptance_rate": (
                np.mean(self.acceptance_rates) if self.acceptance_rates else 0.0
            ),
        }


class LatencyBenchmark:
    """
    Measures end-to-end latency for VLM inference.

    Benchmark methodology:
    1. Warmup runs to load CUDA kernels
    2. Multiple iterations with the same prompt
    3. Record TTFT, ITL, and TPS for each iteration
    4. Compare baseline vs speculative decoding
    """

    def __init__(
        self,
        target_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        draft_model_id: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct",
        image_path: Optional[str] = None,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 128,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        output_dir: str = "results",
    ):
        self.target_model_id = target_model_id
        self.draft_model_id = draft_model_id
        self.image_path = image_path
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[str, BenchmarkResult] = {}

    async def run_all(self) -> dict[str, dict]:
        """
        Run all benchmark configurations.

        Configurations:
        1. baseline: Standard autoregressive decoding
        2. speculative: Speculative decoding with draft model
        """
        # 1. Baseline
        logger.info("=" * 60)
        logger.info("Running BASELINE benchmark")
        logger.info("=" * 60)

        baseline_result = await self._run_config(
            config_name="baseline",
            use_speculative=False,
        )
        self.results["baseline"] = baseline_result

        # 2. Speculative
        logger.info("=" * 60)
        logger.info("Running SPECULATIVE benchmark")
        logger.info("=" * 60)

        speculative_result = await self._run_config(
            config_name="speculative",
            use_speculative=True,
        )
        self.results["speculative"] = speculative_result

        # Generate reports
        return self._generate_reports()

    async def _run_config(
        self,
        config_name: str,
        use_speculative: bool,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        result = BenchmarkResult(config_name=config_name)

        # Configure engine
        mode = EngineMode.SPECULATIVE if use_speculative else EngineMode.BASELINE
        engine_config = EngineConfig(
            mode=mode,
            backend=EngineBackend.TRANSFORMERS,  # Use transformers for detailed profiling
            target_model_id=self.target_model_id,
            draft_model_id=self.draft_model_id if use_speculative else None,
            enable_profiling=True,
        )

        engine = InferenceEngine(config=engine_config)
        await engine.load()

        # Create profiler
        profiler = InferenceProfiler(enabled=True)

        # Warmup
        logger.info(f"Warming up ({self.warmup_iterations} iterations)...")
        warmup_input = VLMInput(
            prompt=self.prompt,
            image_paths=[self.image_path] if self.image_path else [],
            max_new_tokens=min(self.max_tokens, 32),  # Short warmup
            temperature=0.0,
        )

        for i in range(self.warmup_iterations):
            with profiler.profile("warmup"):
                await engine.generate(warmup_input)

        # Benchmark iterations
        logger.info(f"Benchmarking ({self.benchmark_iterations} iterations)...")

        benchmark_input = VLMInput(
            prompt=self.prompt,
            image_paths=[self.image_path] if self.image_path else [],
            max_new_tokens=self.max_tokens,
            temperature=0.7,
        )

        for i in range(self.benchmark_iterations):
            start_time = time.time()

            with profiler.profile(f"iteration_{i}"):
                output = await engine.generate(benchmark_input)

            total_ms = (time.time() - start_time) * 1000

            result.ttft_ms.append(output.ttft_ms)
            result.total_time_ms.append(total_ms)
            result.tokens_per_second.append(output.tokens_per_second)
            result.num_tokens.append(output.num_output_tokens)

            # Compute inter-token latency
            if output.num_output_tokens > 1:
                decode_time = total_ms - output.ttft_ms
                itl = decode_time / (output.num_output_tokens - 1)
                result.inter_token_latency_ms.append(itl)

            if use_speculative:
                result.acceptance_rates.append(output.speculation_acceptance_rate)

            logger.info(
                f"  Iter {i+1}: TTFT={output.ttft_ms:.1f}ms, "
                f"TPS={output.tokens_per_second:.1f}, "
                f"Tokens={output.num_output_tokens}"
                + (
                    f", Accept={output.speculation_acceptance_rate:.2f}"
                    if use_speculative
                    else ""
                )
            )

        # Cleanup
        await engine.shutdown()

        return result

    def _generate_reports(self) -> dict[str, dict]:
        """Generate summary reports for all configurations."""
        summaries = {}

        for name, result in self.results.items():
            summary = result.summary()
            summaries[name] = summary

            logger.info(f"\n{'='*60}")
            logger.info(f"  {name.upper()} RESULTS")
            logger.info(f"{'='*60}")
            logger.info(f"  TTFT (ms):          {summary['ttft_ms']['mean']:8.1f} ± {summary['ttft_ms']['std']:5.1f}")
            logger.info(f"  TTFT P95 (ms):      {summary['ttft_ms']['p95']:8.1f}")
            logger.info(f"  TTFT P99 (ms):      {summary['ttft_ms']['p99']:8.1f}")
            logger.info(f"  Tokens/sec:         {summary['tokens_per_second']['mean']:8.1f}")
            logger.info(f"  ITL mean (ms):      {summary['inter_token_latency_ms']['mean']:8.2f}")
            logger.info(f"  Total time (ms):    {summary['total_time_ms']['mean']:8.1f}")
            logger.info(f"  Avg tokens/req:     {summary['avg_tokens_per_request']:8.1f}")

            if name == "speculative":
                logger.info(f"  Acceptance rate:    {summary['avg_acceptance_rate']:8.2f}")

        # Speedup comparison
        if "baseline" in summaries and "speculative" in summaries:
            base_tps = summaries["baseline"]["tokens_per_second"]["mean"]
            spec_tps = summaries["speculative"]["tokens_per_second"]["mean"]
            speedup = spec_tps / base_tps if base_tps > 0 else 0

            base_ttft = summaries["baseline"]["ttft_ms"]["mean"]
            spec_ttft = summaries["speculative"]["ttft_ms"]["mean"]
            ttft_reduction = (1 - spec_ttft / base_ttft) * 100 if base_ttft > 0 else 0

            logger.info(f"\n{'='*60}")
            logger.info(f"  SPECULATIVE SPEEDUP")
            logger.info(f"{'='*60}")
            logger.info(f"  Throughput speedup:  {speedup:.2f}x")
            logger.info(f"  TTFT reduction:     {ttft_reduction:.1f}%")

        return summaries

    def save_results(self, path: Optional[str] = None) -> str:
        """Save benchmark results to JSON."""
        path = path or str(self.output_dir / "latency_benchmark_results.json")
        summaries = {name: r.summary() for name, r in self.results.items()}

        with open(path, "w") as f:
            json.dump(summaries, f, indent=2)

        logger.info(f"Results saved to {path}")
        return path


async def main():
    parser = argparse.ArgumentParser(description="SpecVLM Latency Benchmark")
    parser.add_argument("--target-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="results/latency_benchmark.json")
    args = parser.parse_args()

    benchmark = LatencyBenchmark(
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
        image_path=args.image,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )

    await benchmark.run_all()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
