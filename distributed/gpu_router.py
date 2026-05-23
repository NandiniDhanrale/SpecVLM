"""
GPU Router — Intelligent Request-to-GPU Assignment

The GPU router decides which GPU(s) should handle each request based on:
1. Current GPU utilization (memory + compute)
2. KV cache locality (prefer GPUs with cached data)
3. Tensor parallelism requirements (models sharded across GPUs)
4. Pipeline parallelism stage (prefer GPUs in the same pipeline)

Routing strategies:
- LEAST_LOADED: Send to GPU with most free memory (default)
- CACHE_AWARE: Prefer GPU with matching cache entries
- ROUND_ROBIN: Simple fair distribution
- AFFINITY: Prefer specific GPUs for specific models

Architecture:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Request     │────▶│  GPU Router  │────▶│  GPU 0 (Draft)│
│  Dispatcher  │     │              │     ├──────────────┤
│              │     │  Strategies: │────▶│  GPU 1 (Target)│
│              │     │  • Load      │     ├──────────────┤
│              │     │  • Cache     │────▶│  GPU 2+ (TP) │
│              │     │  • Round     │     └──────────────┘
└──────────────┘     └──────────────┘
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    LEAST_LOADED = "least_loaded"
    CACHE_AWARE = "cache_aware"
    ROUND_ROBIN = "round_robin"
    AFFINITY = "affinity"


@dataclass
class GPUInfo:
    """Runtime information about a GPU."""
    gpu_id: int
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    utilization_pct: float = 0.0
    active_requests: int = 0
    max_requests: int = 4
    models_loaded: list[str] = field(default_factory=list)
    cache_keys: set[str] = field(default_factory=set)
    is_healthy: bool = True


class GPURouter:
    """
    Routes inference requests to the optimal GPU.

    Uses runtime metrics (memory, utilization, cache state) to make
    intelligent routing decisions that maximize throughput.
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        gpu_count: int = 1,
    ):
        self.strategy = strategy
        self._round_robin_counter = 0

        # GPU registry
        self._gpus: dict[int, GPUInfo] = {}
        for i in range(gpu_count):
            self._gpus[i] = GPUInfo(gpu_id=i)

    def register_gpu(self, gpu_id: int, info: GPUInfo) -> None:
        """Register or update GPU information."""
        self._gpus[gpu_id] = info

    def select_gpu(
        self,
        request_id: str = "",
        preferred_gpu: Optional[int] = None,
        cache_key: str = "",
    ) -> int:
        """
        Select the best GPU for a request.

        Args:
            request_id: Unique request identifier
            preferred_gpu: Force a specific GPU
            cache_key: Cache key for cache-aware routing

        Returns:
            Selected GPU ID
        """
        if preferred_gpu is not None and preferred_gpu in self._gpus:
            if self._gpus[preferred_gpu].is_healthy:
                return preferred_gpu

        # Filter healthy GPUs
        healthy_gpus = {
            gid: gpu
            for gid, gpu in self._gpus.items()
            if gpu.is_healthy
        }

        if not healthy_gpus:
            raise RuntimeError("No healthy GPUs available")

        if self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(healthy_gpus)
        elif self.strategy == RoutingStrategy.CACHE_AWARE:
            return self._select_cache_aware(healthy_gpus, cache_key)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_gpus)
        else:
            return self._select_least_loaded(healthy_gpus)

    def _select_least_loaded(self, gpus: dict[int, GPUInfo]) -> int:
        """Select GPU with most free memory / least active requests."""
        def load_score(gpu: GPUInfo) -> float:
            load = gpu.active_requests / max(gpu.max_requests, 1)
            memory_pct = 1.0 - (gpu.free_memory_gb / max(gpu.total_memory_gb, 1))
            return load * 0.6 + memory_pct * 0.4

        return min(gpus.items(), key=lambda x: load_score(x[1]))[0]

    def _select_cache_aware(
        self,
        gpus: dict[int, GPUInfo],
        cache_key: str,
    ) -> int:
        """
        Select GPU with matching cache entries.

        If no matches, fall back to least-loaded.
        """
        # Check for cache hits
        for gpu_id, gpu in gpus.items():
            if cache_key in gpu.cache_keys:
                return gpu_id

        return self._select_least_loaded(gpus)

    def _select_round_robin(self, gpus: dict[int, GPUInfo]) -> int:
        """Simple round-robin across healthy GPUs."""
        gpu_ids = sorted(gpus.keys())
        selected = gpu_ids[self._round_robin_counter % len(gpu_ids)]
        self._round_robin_counter += 1
        return selected

    def mark_request_started(self, gpu_id: int) -> None:
        """Track that a request started on a GPU."""
        if gpu_id in self._gpus:
            self._gpus[gpu_id].active_requests += 1

    def mark_request_completed(self, gpu_id: int) -> None:
        """Track that a request completed on a GPU."""
        if gpu_id in self._gpus:
            self._gpus[gpu_id].active_requests = max(
                0, self._gpus[gpu_id].active_requests - 1
            )

    def get_all_gpu_status(self) -> dict:
        """Return status of all GPUs."""
        return {
            str(gid): {
                "gpu_id": gpu.gpu_id,
                "total_memory_gb": gpu.total_memory_gb,
                "free_memory_gb": gpu.free_memory_gb,
                "utilization_pct": gpu.utilization_pct,
                "active_requests": gpu.active_requests,
                "max_requests": gpu.max_requests,
                "models_loaded": gpu.models_loaded,
                "is_healthy": gpu.is_healthy,
            }
            for gid, gpu in self._gpus.items()
        }
