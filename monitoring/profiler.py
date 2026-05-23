"""
Inference Profiler — PyTorch Profiler for VLM Performance Analysis

Profiling is essential for identifying bottlenecks in the inference pipeline:
- Prefill vs decode time
- Attention computation overhead
- KV cache operations
- Data transfer (CPU→GPU)
- Kernel launch overhead

Supports:
- PyTorch profiler (detailed kernel-level traces)
- torch.profiler (Chrome trace format)
- Memory profiling (allocation tracking)
- CUDA runtime analysis

Use:
    profiler = InferenceProfiler()
    with profiler.profile("prefill"):
        model.prefill(inputs)
    profiler.save_trace("trace.json")
"""

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class InferenceProfiler:
    """
    Performance profiler for VLM inference.

    Provides:
    1. Context manager for profiling code sections
    2. Automatic CUDA synchronization for accurate timing
    3. PyTorch profiler integration for detailed traces
    4. Memory allocation tracking
    5. Report generation with key metrics
    """

    def __init__(self, enabled: bool = False, output_dir: str = "./profiles"):
        self.enabled = enabled
        self.output_dir = output_dir
        self._timers: dict[str, list[float]] = defaultdict(list)
        self._memory_stats: dict[str, list[dict]] = defaultdict(list)
        self._current_section: Optional[str] = None

        # PyTorch profiler
        self._torch_profiler = None
        self._profiler_active = False

    @contextmanager
    def profile(self, section: str):
        """
        Profile a code section.

        Usage:
            with profiler.profile("attention"):
                attention_output = model.attention(x)
        """
        if not self.enabled:
            yield
            return

        self._current_section = section

        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_event:
            start_event.record()

        start_time = time.time()
        mem_before = (
            torch.cuda.memory_allocated() / 1e6
            if torch.cuda.is_available()
            else 0
        )

        try:
            yield
        finally:
            elapsed = time.time() - start_time

            if end_event and start_event:
                end_event.record()
                torch.cuda.synchronize()
                cuda_elapsed = start_event.elapsed_time(end_event)
            else:
                cuda_elapsed = elapsed * 1000

            mem_after = (
                torch.cuda.memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0
            )

            self._timers[section].append({
                "wall_time_s": elapsed,
                "cuda_time_ms": cuda_elapsed,
                "memory_delta_mb": mem_after - mem_before,
            })

            self._memory_stats[section].append({
                "before_mb": mem_before,
                "after_mb": mem_after,
            })

    def start_torch_profiler(self, trace_path: Optional[str] = None) -> None:
        """
        Start the PyTorch profiler for detailed kernel traces.

        The output can be viewed in Chrome's chrome://tracing.
        """
        if not self.enabled:
            return

        self._profiler_active = True
        trace_path = trace_path or os.path.join(self.output_dir, "torch_trace.json")
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.dirname(trace_path),
                worker_name=os.path.basename(trace_path).replace(".json", ""),
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._torch_profiler.__enter__()

    def stop_torch_profiler(self) -> None:
        """Stop the PyTorch profiler and save the trace."""
        if self._torch_profiler and self._profiler_active:
            self._torch_profiler.__exit__(None, None, None)
            self._profiler_active = False

    def generate_report(self) -> dict:
        """
        Generate a comprehensive profiling report.

        Returns:
            dict with timing breakdown, memory usage, and recommendations
        """
        report = {
            "sections": {},
            "totals": {
                "total_wall_time_s": 0.0,
                "total_cuda_time_ms": 0.0,
                "peak_memory_mb": 0.0,
            },
            "bottlenecks": [],
            "recommendations": [],
        }

        for section, timings in self._timers.items():
            wall_times = [t["wall_time_s"] for t in timings]
            cuda_times = [t["cuda_time_ms"] for t in timings]
            mem_deltas = [t["memory_delta_mb"] for t in timings]

            section_report = {
                "calls": len(timings),
                "avg_wall_time_s": sum(wall_times) / len(wall_times),
                "total_wall_time_s": sum(wall_times),
                "avg_cuda_time_ms": sum(cuda_times) / len(cuda_times),
                "total_cuda_time_ms": sum(cuda_times),
                "avg_memory_mb": sum(mem_deltas) / len(mem_deltas) if mem_deltas else 0,
                "peak_memory_mb": max(mem_deltas) if mem_deltas else 0,
                "pct_of_total": 0.0,  # Computed below
            }

            report["sections"][section] = section_report
            report["totals"]["total_wall_time_s"] += section_report["total_wall_time_s"]
            report["totals"]["total_cuda_time_ms"] += section_report["total_cuda_time_ms"]
            report["totals"]["peak_memory_mb"] = max(
                report["totals"]["peak_memory_mb"],
                section_report["peak_memory_mb"],
            )

        # Compute percentages
        for section in report["sections"]:
            if report["totals"]["total_wall_time_s"] > 0:
                report["sections"][section]["pct_of_total"] = (
                    report["sections"][section]["total_wall_time_s"]
                    / report["totals"]["total_wall_time_s"]
                    * 100
                )

        # Identify bottlenecks
        threshold = 20.0  # Sections taking >20% of time are bottlenecks
        for section, data in sorted(
            report["sections"].items(),
            key=lambda x: x[1]["total_wall_time_s"],
            reverse=True,
        ):
            if data["pct_of_total"] > threshold:
                report["bottlenecks"].append(section)
                report["recommendations"].append(
                    self._get_recommendation(section, data["pct_of_total"])
                )

        return report

    def _get_recommendation(self, section: str, pct: float) -> str:
        """Generate optimization recommendation based on profiling data."""
        recommendations = {
            "prefill": (
                "Visual encoding is {pct:.0f}% of total time. "
                "Consider: (1) cache visual embeddings, (2) use FP8 vision tower, "
                "(3) reduce image resolution, (4) async prefill during decode"
            ),
            "attention": (
                "Attention computation is {pct:.0f}% of total time. "
                "Consider: (1) use FlashAttention-2, (2) reduce KV cache size, "
                "(3) use sliding window attention for draft model"
            ),
            "decode": (
                "Autoregressive decode is {pct:.0f}% of total time. "
                "Consider: (1) enable speculative decoding, "
                "(2) use CUDA graphs for decode kernel, "
                "(3) use vLLM PagedAttention"
            ),
            "verification": (
                "Token verification is {pct:.0f}% of total time. "
                "Consider: (1) increase speculation length, "
                "(2) use stricter acceptance criterion, "
                "(3) batch verification across requests"
            ),
            "visual_encoding": (
                "Visual encoding is {pct:.0f}% of total time. "
                "Consider: (1) pre-compute and cache embeddings, "
                "(2) use a smaller vision tower for draft model, "
                "(3) pipeline image loading with LLM decode"
            ),
        }
        for key, template in recommendations.items():
            if key in section.lower():
                return template.format(pct=pct)
        return f"{section} is {pct:.0f}% of total time. Profile further to identify root cause."

    def save_trace(self, path: str) -> None:
        """Save profiling data as JSON for external analysis."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report = self.generate_report()

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Profiling report saved to {path}")
